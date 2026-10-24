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
    fit_feature_stats, validate_manifest, fit_feature_schema, encode_feature_frame,
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

    def test_group_prefixed_real_data_features_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = root / "scan.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.float32)), str(scan))
            frame = pd.DataFrame({
                "patient_id": ["P1"], "image_path": [scan.name], "stage": [0], "visit_idx": [0],
                "C_hemoglobin": [12.5], "W_steps": [4200.0], "T_duration_days": [14.0],
                "M_dose_mg": [5.0],
            })
            _, feature_cols = validate_manifest(frame, root)
            self.assertEqual(
                set(feature_cols),
                {"C_hemoglobin", "W_steps", "T_duration_days", "M_dose_mg"},
            )
            duplicated_scan = frame.copy()
            duplicated_scan.loc[1] = duplicated_scan.loc[0]
            duplicated_scan.loc[1, "patient_id"] = "P2"
            duplicated_scan.loc[1, "stage"] = 1
            with self.assertRaisesRegex(ValueError, "assigned to multiple patient_id"):
                validate_manifest(duplicated_scan, root)

    def test_categorical_text_is_encoded_and_unseen_values_are_handled(self):
        frame = pd.DataFrame({
            "patient_id": ["P1", "P2", "P3"],
            "image_path": ["a", "b", "c"],
            "stage": [0, 1, 2],
            "visit_idx": [0, 0, 0],
            "time_months": [0.0, 0.0, 0.0],
            "C_age": [45, 51, 63],
            "C_sex": ["Female", "male", "Other"],
            "M_medicine_name": ["Metformin", "Insulin", "NewMedicine"],
        })
        schema = fit_feature_schema(frame.iloc[:2], ["C_age", "C_sex", "M_medicine_name"])
        encoded, columns = encode_feature_frame(frame, schema)
        self.assertIn("C_age", columns)
        self.assertIn("C_sex__category_0", columns)
        self.assertIn("C_sex__unknown", columns)
        self.assertIn("M_medicine_name__unknown", columns)
        sex_category_cols = [col for col in columns if col.startswith("C_sex__category_")]
        medicine_category_cols = [col for col in columns if col.startswith("M_medicine_name__category_")]
        self.assertEqual(float(encoded.loc[2, sex_category_cols].sum()), 0.0)
        self.assertEqual(float(encoded.loc[2, medicine_category_cols].sum()), 0.0)
        self.assertEqual(float(encoded.loc[2, "C_sex__unknown"]), 0.0)
        self.assertEqual(float(encoded.loc[2, "M_medicine_name__unknown"]), 0.0)
        self.assertEqual(float(encoded.loc[0, sex_category_cols].sum()), 1.0)
        self.assertEqual(float(encoded.loc[0, medicine_category_cols].sum()), 1.0)

    def test_categorical_feature_absent_from_training_stays_missing(self):
        frame = pd.DataFrame({
            "patient_id": ["P1", "P2"],
            "image_path": ["a", "b"],
            "stage": [0, 1],
            "visit_idx": [0, 0],
            "time_months": [0.0, 0.0],
            "M_medicine_name": [pd.NA, "UnexpectedMedicine"],
        })
        schema = fit_feature_schema(frame.iloc[[0]], ["M_medicine_name"])
        encoded, columns = encode_feature_frame(frame, schema)
        self.assertTrue(columns)
        self.assertTrue(encoded.loc[:, columns].isna().to_numpy().all())

    def test_visit_indices_and_timeline_are_validated(self):
        base = pd.DataFrame({
            "patient_id": ["P1", "P1"],
            "image_path": ["a.nii.gz", "b.nii.gz"],
            "stage": [0, 1],
            "visit_idx": [0, 1],
            "time_months": [0.0, 6.0],
            "C_age": [45.0, 46.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a.nii.gz", "b.nii.gz"):
                sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32)), str(root / name))
            valid, _ = validate_manifest(base, root)
            self.assertEqual(valid["visit_idx"].tolist(), [0, 1])
            invalid_index = base.copy()
            invalid_index.loc[1, "visit_idx"] = 1.5
            with self.assertRaisesRegex(ValueError, "visit_idx"):
                validate_manifest(invalid_index, root)
            invalid_time = base.copy()
            invalid_time["time_months"] = [6.0, 0.0]
            with self.assertRaisesRegex(ValueError, "time_months must not decrease"):
                validate_manifest(invalid_time, root)

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
                    "C_sex": "Female" if i % 2 == 0 else "Male",
                    "M_medicine_name": "Metformin" if i % 3 else "Insulin",
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
            self.assertIn("C_sex", checkpoint["feature_schema"]["categorical_categories"])
            self.assertIn("M_medicine_name", checkpoint["feature_schema"]["categorical_categories"])
            self.assertEqual(summary["n_scans"], 10)
            self.assertEqual(summary["n_patients"], 10)
            self.assertIn("macro_f1", summary["test_metrics"])
            self.assertTrue((output / "training_history.csv").is_file())

            prediction_manifest = root / "prediction_manifest.csv"
            prediction_rows = pd.DataFrame(rows).drop(columns=["stage"])
            # Exercise inference-time handling of a category absent from training and a missing value.
            prediction_rows.loc[0, "M_medicine_name"] = "PreviouslyUnseenMedicine"
            prediction_rows.loc[1, "C_sex"] = np.nan
            prediction_rows.to_csv(prediction_manifest, index=False)
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
