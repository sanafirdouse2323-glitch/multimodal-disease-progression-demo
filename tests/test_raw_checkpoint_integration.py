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

            # Exercise the actual Streamlit UI in Chromium: upload the trained checkpoint,
            # add two visits through the guided form, prepare CSV+ZIP, and run inference.
            # This is a browser smoke/integration test, not an exhaustive usability audit.
            import os
            import time
            from playwright.sync_api import sync_playwright

            env = os.environ.copy()
            env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
            app_process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                    "--server.headless=true", "--server.port=8765",
                    "--browser.gatherUsageStats=false",
                ],
                cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page()
                    page.set_default_timeout(20_000)
                    deadline = time.time() + 60
                    while True:
                        try:
                            page.goto("http://127.0.0.1:8765", wait_until="domcontentloaded", timeout=3000)
                            break
                        except Exception:
                            if time.time() >= deadline:
                                raise
                            time.sleep(1)

                    page.get_by_text("Choose a view").wait_for()
                    page.get_by_role("radio", name="Run trained model").check()
                    page.get_by_text("Guided entry: add one scan visit at a time").wait_for()
                    file_inputs = page.locator('input[type="file"]')
                    file_inputs.nth(0).set_input_files(str(checkpoint_path))
                    page.get_by_text("Checkpoint recognised.", exact=False).wait_for(timeout=30_000)

                    # Baseline visit.
                    page.get_by_label("Pseudonymous patient ID (same ID for the same person across visits)").fill("P001")
                    page.get_by_label("Visit order (baseline = 0, next = 1, ...)").fill("0")
                    page.get_by_label("Months since baseline (baseline = 0)").fill("0")
                    page.locator('input[type="file"]').nth(1).set_input_files(str(scans / "P000_v0.nii.gz"))
                    page.get_by_label("C_age (number; leave blank if unavailable)").fill("30")
                    page.get_by_label("M_medicine_name (category/text; leave blank if unavailable)").fill("MedicineA")
                    page.get_by_role("button", name="Add this visit to the review table").click()
                    page.get_by_text("Visit added.", exact=False).wait_for(timeout=20_000)

                    # Follow-up visit. The app re-renders the form with a new key after add.
                    page.get_by_label("Pseudonymous patient ID (same ID for the same person across visits)").fill("P001")
                    page.get_by_label("Visit order (baseline = 0, next = 1, ...)").fill("1")
                    page.get_by_label("Months since baseline (baseline = 0)").fill("6")
                    page.locator('input[type="file"]').nth(1).set_input_files(str(scans / "P000_v1.nii.gz"))
                    page.get_by_label("C_age (number; leave blank if unavailable)").fill("30")
                    page.get_by_label("M_medicine_name (category/text; leave blank if unavailable)").fill("PreviouslyUnseenCategory")
                    page.get_by_role("button", name="Add this visit to the review table").click()
                    page.get_by_text("Visit added.", exact=False).wait_for(timeout=20_000)

                    page.get_by_role("button", name="Validate and prepare manifest + scan ZIP").click()
                    page.get_by_text("Files prepared.", exact=False).wait_for(timeout=20_000)
                    page.get_by_role("button", name="Download prepared manifest CSV").wait_for()
                    page.get_by_role("button", name="Download prepared scan ZIP").wait_for()

                    page.get_by_role("radio", name="Use guided entries above").check()
                    page.get_by_role("button", name="Run prediction").click(timeout=10_000)
                    page.get_by_text("Prediction completed for 2 visit(s).", exact=True).wait_for(timeout=180_000)
                    page.get_by_text("P001", exact=False).wait_for()
                    browser.close()
            finally:
                app_process.terminate()
                try:
                    app_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    app_process.kill()
                    app_process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
