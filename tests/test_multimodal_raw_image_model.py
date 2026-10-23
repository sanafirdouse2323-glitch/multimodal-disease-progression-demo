import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from train_multimodal_raw_image_model import (
    RawImageTemporalFusionModel, collate_sequences, main, make_sequences,
    fit_feature_stats, validate_manifest,
)
from predict_multimodal_raw_image_model import main as predict_main


class RawImageTemporalFusionTests(unittest.TestCase):
    def test_fusion_model_returns_one_logit_vector_per_visit(self):
        model = RawImageTemporalFusionModel(n_features=2)
        batch = {
            "images": torch.zeros(2, 2, 1, 16, 16, 16),
            "features": torch.zeros(2, 2, 2),
            "feature_mask": torch.ones(2, 2, 2),
            "time": torch.tensor([[0., 1.], [0., 0.]]),
            "labels": torch.tensor([[0, 1], [2, -100]]),
            "padding_mask": torch.tensor([[False, False], [False, True]]),
        }
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 2, 4))
        self.assertTrue(torch.isfinite(logits).all())

    def test_feature_stats_use_training_rows_and_missingness_is_preserved(self):
        frame = pd.DataFrame({"patient_id": ["A", "B", "C"], "visit_idx": [0, 0, 0],
                              "image_path": ["a", "b", "c"], "stage": [0, 1, 2],
                              "time_months": [0.0, 0.0, 0.0],
                              "lab1": [2.0, np.nan, 1000.0]})
        stats = fit_feature_stats(frame.iloc[:2], ["lab1"])
        self.assertEqual(stats["lab1"]["mean"], 2.0)
        sequences = make_sequences(frame.iloc[:2], ["lab1"], stats)
        self.assertEqual(sequences[0]["feature_mask"][0, 0], 1.0)
        self.assertEqual(sequences[1]["feature_mask"][0, 0], 0.0)
        self.assertEqual(sequences[1]["features"][0, 0], 0.0)

    def test_manifest_requires_tabular_modality_and_unique_patient_visits(self):
        base = pd.DataFrame({"patient_id": ["A"], "image_path": ["scan.nii.gz"],
                             "stage": [0], "visit_idx": [0]})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least one supported numeric feature"):
                validate_manifest(base, Path(directory))
            base["lab1"] = [1.0]
            duplicated = pd.concat([base, base], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "at most one scan row per visit_idx"):
                validate_manifest(duplicated, Path(directory))

    def test_end_to_end_fusion_training_smoke_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            rng = np.random.default_rng(31)
            for i in range(10):
                path = root / f"scan_{i:02d}.nii.gz"
                arr = rng.normal(loc=float(i % 4), scale=0.5, size=(16, 16, 16)).astype(np.float32)
                sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))
                rows.append({
                    "patient_id": f"P{i:02d}", "image_path": path.name, "stage": i % 4,
                    "visit_idx": 0, "time_months": 0.0, "lab1": float(i),
                    "steps": float(8000 - i * 100),
                })
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            output = root / "output"
            argv = [
                "train_multimodal_raw_image_model.py", "--manifest", str(manifest),
                "--epochs", "1", "--batch-size", "2", "--image-size", "16",
                "--target-spacing-mm", "1.0", "--output-dir", str(output),
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            checkpoint = torch.load(output / "multimodal_raw_model.pt", map_location="cpu", weights_only=False)
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["architecture"], "RawImageTemporalFusionModel")
            self.assertIn("lab1", checkpoint["feature_columns"])
            self.assertIn("steps", checkpoint["feature_columns"])
            self.assertEqual(summary["n_scans"], 10)
            self.assertEqual(summary["n_patients"], 10)
            self.assertIn("macro_f1", summary["test_metrics"])
            self.assertTrue((output / "training_history.csv").is_file())

            prediction_manifest = root / "prediction_manifest.csv"
            pd.DataFrame(rows).drop(columns=["stage"]).to_csv(prediction_manifest, index=False)
            prediction_path = root / "predictions.csv"
            predict_argv = [
                "predict_multimodal_raw_image_model.py",
                "--checkpoint", str(output / "multimodal_raw_model.pt"),
                "--manifest", str(prediction_manifest),
                "--output", str(prediction_path),
                "--batch-size", "2",
                "--device", "cpu",
            ]
            with patch("sys.argv", predict_argv), contextlib.redirect_stdout(io.StringIO()):
                predict_main()
            predictions = pd.read_csv(prediction_path)
            self.assertEqual(len(predictions), 10)
            self.assertTrue(predictions["predicted_stage"].isin([0, 1, 2, 3]).all())
            probability_cols = [f"prob_stage_{i}" for i in range(4)]
            np.testing.assert_allclose(predictions[probability_cols].sum(axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
