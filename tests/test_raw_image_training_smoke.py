"""End-to-end smoke test for the raw-volume training CLI."""
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

from train_raw_image_model import main


class RawImageTrainingSmokeTest(unittest.TestCase):
    def test_trains_and_writes_outputs_on_tiny_synthetic_nifti_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            rng = np.random.default_rng(9)
            for i in range(10):
                image_path = root / f"scan_{i:02d}.nii.gz"
                # Synthetic test volumes only; these are not patient images.
                arr = rng.normal(loc=float(i % 4), scale=0.5, size=(16, 16, 16)).astype(np.float32)
                image = sitk.GetImageFromArray(arr)
                image.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(image, str(image_path))
                rows.append({"patient_id": f"P{i:02d}", "image_path": image_path.name, "stage": i % 4})
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            output = root / "output"
            argv = [
                "train_raw_image_model.py", "--manifest", str(manifest),
                "--epochs", "1", "--batch-size", "2", "--image-size", "16",
                "--target-spacing-mm", "1.0", "--output-dir", str(output),
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            checkpoint_path = output / "raw_image_3dcnn.pt"
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint.get("artifact_type"),
                             "multimodal-disease-progression.image-only-3dcnn")
            self.assertEqual(checkpoint.get("artifact_version"), 1)
            self.assertEqual(checkpoint.get("architecture"), "Small3DCNN")
            self.assertEqual(checkpoint["preprocessing"]["image_size_voxels"], 16)
            self.assertIn("model_state_dict", checkpoint)
            self.assertTrue((output / "training_history.csv").is_file())
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["task"], "supervised 3D scan classification")
            self.assertEqual(summary["n_scans"], 10)
            self.assertEqual(summary["n_patients"], 10)
            self.assertIn("accuracy", summary["test_metrics"])
            self.assertIn("macro_f1", summary["test_metrics"])

            # Verify the actual training artifact, not only a hand-built test checkpoint.
            prediction_manifest = root / "prediction.csv"
            pd.DataFrame([
                {"patient_id": row["patient_id"], "image_path": row["image_path"]}
                for row in rows[:2]
            ]).to_csv(prediction_manifest, index=False)
            prediction_output = output / "predictions.csv"
            with patch("sys.argv", [
                "predict_raw_image_model.py",
                "--checkpoint", str(checkpoint_path),
                "--manifest", str(prediction_manifest),
                "--output", str(prediction_output),
                "--device", "cpu",
            ]), contextlib.redirect_stdout(io.StringIO()):
                from predict_raw_image_model import main as predict_main
                predict_main()
            self.assertTrue(prediction_output.is_file())
            predictions = pd.read_csv(prediction_output)
            self.assertEqual(len(predictions), 2)
            self.assertTrue(predictions["predicted_stage"].isin([0, 1, 2, 3]).all())
            np.testing.assert_allclose(
                predictions[[f"prob_stage_{i}" for i in range(4)]].sum(axis=1).to_numpy(),
                np.ones(2), atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
