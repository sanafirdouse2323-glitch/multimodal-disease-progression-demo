"""End-to-end smoke test for a genuinely trained raw-image checkpoint.

All scans and labels are synthetic. This checks training -> checkpoint save ->
safe checkpoint load -> prediction CLI; it is not a model-quality or clinical test.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]


class RawCheckpointIntegrationTest(unittest.TestCase):
    def test_train_checkpoint_then_predict_two_visits(self):
        with tempfile.TemporaryDirectory(prefix="raw_checkpoint_test_") as tmp:
            work = Path(tmp)
            scans = work / "scans"
            scans.mkdir()
            train_rows = []
            rng = np.random.default_rng(2718)

            # Ten separate patients satisfy the repository's patient-level split.
            # Two visits for P000 exercise the temporal prediction path.
            for patient_number in range(10):
                patient_id = f"P{patient_number:03d}"
                n_visits = 2 if patient_number == 0 else 1
                for visit_idx in range(n_visits):
                    image = (rng.normal(size=(16, 16, 16)) + patient_number * 0.05 + visit_idx * 0.02).astype(np.float32)
                    image_path = scans / f"{patient_id}_v{visit_idx}.nii.gz"
                    sitk.WriteImage(sitk.GetImageFromArray(image), str(image_path))
                    train_rows.append({
                        "patient_id": patient_id,
                        "image_path": str(image_path),
                        "stage": (patient_number + visit_idx) % 4,
                        "visit_idx": visit_idx,
                        "time_months": visit_idx * 6,
                        "C_age": 30 + patient_number,
                        "M_medicine_name": "MedicineA" if patient_number % 2 == 0 else "MedicineB",
                    })

            training_manifest = work / "training.csv"
            pd.DataFrame(train_rows).to_csv(training_manifest, index=False)
            checkpoint_dir = work / "checkpoint"
            train_cmd = [
                sys.executable, str(ROOT / "train_multimodal_raw_image_model.py"),
                "--manifest", str(training_manifest),
                "--epochs", "1", "--batch-size", "2", "--image-size", "16",
                "--target-spacing-mm", "2.0", "--output-dir", str(checkpoint_dir),
                "--seed", "7",
            ]
            trained = subprocess.run(
                train_cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, check=False
            )
            self.assertEqual(trained.returncode, 0, msg=trained.stdout[-5000:] + "\n" + trained.stderr[-5000:])
            checkpoint_path = checkpoint_dir / "multimodal_raw_model.pt"
            self.assertTrue(checkpoint_path.is_file(), "training must produce the checkpoint")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint.get("architecture"), "RawImageTemporalFusionModel")
            self.assertIn("feature_schema", checkpoint)
            self.assertIn("M_medicine_name", checkpoint["feature_schema"]["categorical_categories"])

            # Mimic guided form output: no stage labels, one row per visit, scan paths
            # relative to the manifest directory, and one genuinely unseen category.
            prediction_dir = work / "guided"
            prediction_scans = prediction_dir / "scans"
            prediction_scans.mkdir(parents=True)
            prediction_rows = []
            for visit_idx in range(2):
                source = scans / f"P000_v{visit_idx}.nii.gz"
                destination = prediction_scans / f"visit{visit_idx}.nii.gz"
                destination.write_bytes(source.read_bytes())
                prediction_rows.append({
                    "patient_id": "P000",
                    "image_path": f"scans/visit{visit_idx}.nii.gz",
                    "visit_idx": visit_idx,
                    "time_months": visit_idx * 6,
                    "C_age": 30,
                    "M_medicine_name": "PreviouslyUnseenCategory" if visit_idx else "MedicineA",
                })
            prediction_manifest = prediction_dir / "prediction_manifest.csv"
            pd.DataFrame(prediction_rows).to_csv(prediction_manifest, index=False)
            output_path = work / "predictions.csv"
            predict_cmd = [
                sys.executable, str(ROOT / "predict_multimodal_raw_image_model.py"),
                "--checkpoint", str(checkpoint_path),
                "--manifest", str(prediction_manifest),
                "--output", str(output_path), "--device", "cpu",
            ]
            predicted = subprocess.run(
                predict_cmd, cwd=ROOT, capture_output=True, text=True, timeout=900, check=False
            )
            self.assertEqual(predicted.returncode, 0, msg=predicted.stdout[-5000:] + "\n" + predicted.stderr[-5000:])
            self.assertTrue(output_path.is_file(), "inference must produce predictions.csv")
            results = pd.read_csv(output_path)
            self.assertEqual(len(results), 2)
            self.assertEqual(results["patient_id"].tolist(), ["P000", "P000"])
            self.assertEqual(results["visit_idx"].tolist(), [0, 1])
            probability_columns = [f"prob_stage_{i}" for i in range(4)]
            self.assertTrue(set(probability_columns).issubset(results.columns))
            probabilities = results[probability_columns].to_numpy(dtype=float)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
