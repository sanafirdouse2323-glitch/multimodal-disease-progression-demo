"""Validation and integration tests for image-only checkpoint inference."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from predict_raw_image_model import (
    ARTIFACT_TYPE,
    main,
    prepare_prediction_manifest,
    validate_checkpoint,
)
from train_raw_image_model import Small3DCNN


def make_checkpoint(image_size: int = 16) -> dict:
    model = Small3DCNN(n_classes=4)
    return {
        "artifact_type": ARTIFACT_TYPE,
        "artifact_version": 1,
        "architecture": "Small3DCNN",
        "model_config": {"n_classes": 4},
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "preprocessing": {
            "modality": "generic",
            "image_size_voxels": image_size,
            "target_spacing_mm": 1.0,
            "orientation": "LPS before physical-space resampling",
            "cube_policy": "center crop/pad after normalization; no anisotropic stretching",
            "ct_window_low": None,
            "ct_window_high": None,
        },
        "class_labels": {"0": "stage_0", "1": "stage_1", "2": "stage_2", "3": "stage_3"},
        "training": {"seed": 7, "epochs_requested": 1, "epochs_completed": 1,
                     "best_validation_loss": 1.0, "selected_checkpoint_by": "lowest validation cross-entropy"},
    }


class RawImagePredictionTests(unittest.TestCase):
    def test_validate_checkpoint_rejects_missing_preprocessing(self):
        checkpoint = make_checkpoint()
        del checkpoint["preprocessing"]
        with self.assertRaisesRegex(ValueError, "preprocessing metadata"):
            validate_checkpoint(checkpoint)

    def test_validate_checkpoint_rejects_unknown_artifact_version(self):
        checkpoint = make_checkpoint()
        checkpoint["artifact_version"] = 999
        with self.assertRaisesRegex(ValueError, "artifact_version"):
            validate_checkpoint(checkpoint)

    def test_manifest_rejects_fractional_visit_and_cross_patient_path_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan.nii.gz").touch()
            base = pd.DataFrame({
                "patient_id": ["P1", "P1"],
                "image_path": ["scan.nii.gz", "scan2.nii.gz"],
                "visit_idx": [0, 1],
                "time_months": [0.0, 6.0],
            })
            invalid_visit = base.copy()
            invalid_visit["visit_idx"] = [0.0, 1.5]
            with self.assertRaisesRegex(ValueError, "visit_idx"):
                prepare_prediction_manifest(invalid_visit, root)

            duplicate_path = pd.DataFrame({
                "patient_id": ["P1", "P2"],
                "image_path": ["scan.nii.gz", "scan.nii.gz"],
            })
            with self.assertRaisesRegex(ValueError, "multiple patient_id"):
                prepare_prediction_manifest(duplicate_path, root)

    def test_cli_predicts_from_checkpoint_and_writes_normalized_probabilities(self):
        with tempfile.TemporaryDirectory(prefix="image_prediction_test_") as directory:
            root = Path(directory)
            scan_path = root / "scan.nii.gz"
            volume = np.random.default_rng(41).normal(size=(16, 16, 16)).astype(np.float32)
            sitk.WriteImage(sitk.GetImageFromArray(volume), str(scan_path))
            checkpoint_path = root / "raw_image_3dcnn.pt"
            torch.save(make_checkpoint(), checkpoint_path)
            manifest_path = root / "prediction.csv"
            pd.DataFrame([{
                "patient_id": "P001", "image_path": "scan.nii.gz",
                "visit_idx": 0, "time_months": 0.0,
            }]).to_csv(manifest_path, index=False)
            output_path = root / "predictions.csv"
            argv = [
                "predict_raw_image_model.py", "--checkpoint", str(checkpoint_path),
                "--manifest", str(manifest_path), "--output", str(output_path), "--device", "cpu",
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()

            self.assertTrue(output_path.is_file())
            result = pd.read_csv(output_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "patient_id"], "P001")
            self.assertIn(int(result.loc[0, "predicted_stage"]), [0, 1, 2, 3])
            probability_columns = [f"prob_stage_{i}" for i in range(4)]
            self.assertTrue(np.isfinite(result[probability_columns].to_numpy()).all())
            self.assertTrue((result[probability_columns].to_numpy() >= 0).all())
            np.testing.assert_allclose(result[probability_columns].sum(axis=1), [1.0], atol=1e-5)
            self.assertAlmostEqual(
                float(result.loc[0, "confidence"]),
                float(result[probability_columns].max(axis=1).iloc[0]),
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
