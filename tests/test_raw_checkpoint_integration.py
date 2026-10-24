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
            if os.environ.get("SKIP_STREAMLIT_BROWSER_TEST", "").strip().lower() in {"1", "true", "yes"}:
                return

            import time
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright

            env = os.environ.copy()
            env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
            app_log_path = work / "streamlit_app.log"
            app_log_handle = app_log_path.open("w", encoding="utf-8")
            app_process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                    "--server.headless=true", "--server.port=8765",
                    "--browser.gatherUsageStats=false",
                ],
                cwd=ROOT, env=env, stdout=app_log_handle, stderr=subprocess.STDOUT,
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
                    # Streamlit renders custom radio controls whose actual input is
                    # visually hidden; click the visible label instead of the input.
                    page.get_by_text("Run trained model", exact=True).click()
                    page.get_by_text("Guided entry: add one scan visit at a time").wait_for()
                    def uploader_container(label: str):
                        # Locate the uploader from its rendered label text, then query
                        # the nearest component. Streamlit versions can place the label
                        # outside the element carrying data-testid="stFileUploader".
                        label_node = page.get_by_text(label, exact=True)
                        label_node.wait_for(state="visible")
                        container = label_node.locator(
                            "xpath=ancestor::*[@data-testid='stFileUploader'][1]"
                        )
                        if container.count() == 0:
                            container = page.locator(
                                "[data-testid='stFileUploader']"
                            ).filter(has_text=label)
                        self.assertEqual(
                            container.count(), 1,
                            f"Expected one uploader container for {label!r}; "
                            f"containers={page.locator('[data-testid=stFileUploader]').count()}",
                        )
                        return container

                    def uploader_input(label: str):
                        return uploader_container(label).locator('input[type="file"]')

                    def upload_via_ui(label: str, file_path: Path):
                        # Exercise the same visible control a user clicks. The old
                        # hidden-input shortcut left Streamlit's server-side uploader
                        # state empty even though the browser test had set a file path.
                        container = uploader_container(label)
                        # The hidden file input has an implicit button role in the
                        # current Streamlit DOM, so get_by_role("button") can select
                        # that hidden input instead of the visible browse control.
                        visible_buttons = container.locator("button:visible")
                        if visible_buttons.count() > 0:
                            chooser_trigger = visible_buttons.first
                        else:
                            # Fallback for Streamlit versions where the dropzone itself
                            # opens the native file chooser without a child <button>.
                            chooser_trigger = container.locator(
                                "[data-testid='stFileUploaderDropzone']"
                            )
                        self.assertEqual(
                            chooser_trigger.count(), 1,
                            f"Could not find a visible choose-file/dropzone control for {label!r}",
                        )
                        with page.expect_file_chooser(timeout=15_000) as chooser_event:
                            chooser_trigger.click()
                        chooser_event.value.set_files(str(file_path))
                        file_input = uploader_input(label)
                        self.assertEqual(
                            file_input.count(), 1,
                            f"Expected one file input for uploader {label!r}",
                        )
                        self.assertIn(
                            file_path.name,
                            file_input.evaluate("(el) => Array.from(el.files || []).map(file => file.name)"),
                            f"Browser file chooser did not retain {file_path.name!r}",
                        )

                    upload_via_ui("Model checkpoint (.pt)", checkpoint_path)
                    page.get_by_text("Checkpoint recognised.", exact=False).wait_for(timeout=30_000)

                    def wait_for_ui_text(expected: str, timeout_ms: int = 20_000):
                        try:
                            page.get_by_text(expected, exact=False).wait_for(timeout=timeout_ms)
                        except PlaywrightTimeoutError as exc:
                            visible = page.locator("body").inner_text()
                            app_log_handle.flush()
                            try:
                                log_text = app_log_path.read_text(encoding="utf-8", errors="replace")
                            except OSError as log_error:
                                log_text = f"Could not read Streamlit log: {log_error}"
                            raise AssertionError(
                                f"Expected UI state was not reached: {expected!r}.\\n"
                                + "Visible UI:\\n" + visible[-8000:]
                                + "\\nStreamlit server log:\\n" + log_text[-12000:]
                            ) from exc

                    def textbox(label: str):
                        return page.get_by_role("textbox", name=label, exact=True)

                    def spinbutton(label: str):
                        return page.get_by_role("spinbutton", name=label, exact=True)

                    def wait_for_streamlit_idle():
                        # Widget edits trigger script reruns over Streamlit's websocket.
                        # Do not start a file upload while a previous rerun is still active.
                        page.wait_for_function(
                            """() => {
                                const status = document.querySelector('[data-testid="stStatusWidget"]');
                                if (!status) return true;
                                return !/\\b(running|stopping)\\b/i.test(status.innerText || "");
                            }""",
                            timeout=15_000,
                        )
                        # Give the frontend a short settling interval after the status
                        # indicator goes idle; no model work is done during this pause.
                        page.wait_for_timeout(400)

                    # Baseline visit. Stage every field before adding so Streamlit has
                    # applied browser widget state to the current script run.
                    textbox("Pseudonymous patient ID (same ID for the same person across visits)").fill("P001")
                    spinbutton("Visit order (baseline = 0, next = 1, ...)").fill("0")
                    spinbutton("Visit order (baseline = 0, next = 1, ...)").press("Enter")
                    spinbutton("Months since baseline (baseline = 0)").fill("0")
                    spinbutton("Months since baseline (baseline = 0)").press("Enter")
                    wait_for_streamlit_idle()
                    upload_via_ui(
                        "Scan for this visit (3D image file, e.g. .nii or .nii.gz)",
                        scans / "P000_v0.nii.gz",
                    )
                    wait_for_ui_text("Scan attached: P000_v0.nii.gz")
                    textbox("C_age (number; leave blank if unavailable)").fill("30")
                    medicine_label = "M_medicine_name (category/text; leave blank if unavailable)"
                    medicine_box = textbox(medicine_label)
                    if not medicine_box.is_visible():
                        page.get_by_text("Medicine features", exact=True).click()
                        medicine_box.wait_for(state="visible")
                    medicine_box.fill("MedicineA")
                    # Confirm the uploader retained the selected file. Do not rely on
                    # Streamlit's rendered filename text, which is theme/version-dependent.
                    baseline_scan_files = uploader_input(
                        "Scan for this visit (3D image file, e.g. .nii or .nii.gz)"
                    ).evaluate("(el) => Array.from(el.files || []).map(file => file.name)")
                    self.assertIn("P000_v0.nii.gz", baseline_scan_files)
                    page.get_by_role("button", name="Add this visit to the review table").click()
                    wait_for_ui_text("Visits in review table: 1")

                    # Follow-up visit. The app re-renders with a fresh visit-specific upload key after add.
                    textbox("Pseudonymous patient ID (same ID for the same person across visits)").fill("P001")
                    spinbutton("Visit order (baseline = 0, next = 1, ...)").fill("1")
                    spinbutton("Visit order (baseline = 0, next = 1, ...)").press("Enter")
                    spinbutton("Months since baseline (baseline = 0)").fill("6")
                    spinbutton("Months since baseline (baseline = 0)").press("Enter")
                    wait_for_streamlit_idle()
                    upload_via_ui(
                        "Scan for this visit (3D image file, e.g. .nii or .nii.gz)",
                        scans / "P000_v1.nii.gz",
                    )
                    wait_for_ui_text("Scan attached: P000_v1.nii.gz")
                    # Streamlit may not expose the selected filename as ordinary page
                    # text in every version/theme. Verify the actual file input instead.
                    scan_files = uploader_input(
                        "Scan for this visit (3D image file, e.g. .nii or .nii.gz)"
                    ).evaluate("(el) => Array.from(el.files || []).map(file => file.name)")
                    self.assertIn(
                        "P000_v1.nii.gz",
                        scan_files,
                        f"Follow-up scan not selected in the browser file input: {scan_files}",
                    )
                    textbox("C_age (number; leave blank if unavailable)").fill("30")
                    medicine_box = textbox(medicine_label)
                    if not medicine_box.is_visible():
                        page.get_by_text("Medicine features", exact=True).click()
                        medicine_box.wait_for(state="visible")
                    medicine_box.fill("PreviouslyUnseenCategory")
                    self.assertEqual(
                        textbox("Pseudonymous patient ID (same ID for the same person across visits)").input_value(),
                        "P001",
                        "Follow-up patient ID was not retained in the browser control.",
                    )
                    self.assertEqual(
                        medicine_box.input_value(),
                        "PreviouslyUnseenCategory",
                        "Follow-up medicine category was not retained in the browser control.",
                    )
                    # Verify that the browser has the intended follow-up values and file
                    # before submitting the Streamlit form; failures here are easier to
                    # diagnose than an ambiguous stale success message.
                    self.assertEqual(
                        spinbutton("Visit order (baseline = 0, next = 1, ...)").input_value(),
                        "1",
                        "Follow-up visit index was not retained in the browser control.",
                    )
                    self.assertAlmostEqual(
                        float(spinbutton("Months since baseline (baseline = 0)").input_value()),
                        6.0,
                        msg="Follow-up month value was not retained in the browser control.",
                    )
                    scan_files = uploader_input(
                        "Scan for this visit (3D image file, e.g. .nii or .nii.gz)"
                    ).evaluate("(el) => Array.from(el.files || []).map(file => file.name)")
                    self.assertIn("P000_v1.nii.gz", scan_files, f"Follow-up scan not selected: {scan_files}")
                    page.get_by_role("button", name="Add this visit to the review table").click()
                    wait_for_ui_text("Visits in review table: 2")

                    page.get_by_role("button", name="Validate and prepare manifest + scan ZIP").click()
                    wait_for_ui_text("Files prepared for 2 visit(s).")
                    page.get_by_role("button", name="Download prepared manifest CSV").wait_for()
                    page.get_by_role("button", name="Download prepared scan ZIP").wait_for()

                    page.get_by_text("Use guided entries above", exact=True).click()
                    page.get_by_role("button", name="Run prediction").click(timeout=10_000)
                    # Stop on either a success or a visible application-level error.
                    # This avoids waiting the full inference timeout when the UI has
                    # already rendered a useful diagnostic.
                    try:
                        page.wait_for_function(
                            """() => {
                                const body = document.body.innerText;
                                return body.includes("Prediction completed for 2 visit(s).")
                                    || body.includes("Prediction failed.")
                                    || body.includes("The prediction process ended without creating an output CSV.")
                                    || body.includes("Could not run inference:")
                                    || body.includes("Some image_path entries do not match files in the ZIP")
                                    || body.includes("Manifest is missing required columns")
                                    || body.includes("The manifest contains no rows.")
                                    || body.includes("Archive contains an unsafe path")
                                    || body.includes("The scan archive is not a valid ZIP file.");
                            }""",
                            timeout=60_000,
                        )
                    except PlaywrightTimeoutError as exc:
                        visible_text = page.locator("body").inner_text()
                        app_log_handle.flush()
                        try:
                            app_log_text = app_log_path.read_text(encoding="utf-8", errors="replace")
                        except OSError as log_error:
                            app_log_text = f"Could not read Streamlit log: {log_error}"
                        raise AssertionError(
                            "Guided inference produced no success or error within 60 seconds. "
                            "Visible UI:\\n" + visible_text[-8000:]
                            + "\\nStreamlit server log:\\n" + app_log_text[-12000:]
                        ) from exc

                    visible_text = page.locator("body").inner_text()
                    self.assertIn(
                        "Prediction completed for 2 visit(s).",
                        visible_text,
                        msg="Guided Streamlit inference returned an app error. Visible UI:\\n" + visible_text[-8000:],
                    )
                    # This success-only control is rendered after the predictions CSV
                    # has been read and displayed. Do not use a loose "P001" locator:
                    # patient IDs also appear in the manifest review table and scan paths.
                    page.get_by_role("button", name="Download predictions CSV").wait_for()
                    browser.close()
            finally:
                app_process.terminate()
                try:
                    app_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    app_process.kill()
                    app_process.wait(timeout=5)
                app_log_handle.close()


if __name__ == "__main__":
    unittest.main()
