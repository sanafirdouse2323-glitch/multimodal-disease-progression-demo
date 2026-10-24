"""Focused UI-only smoke test for Streamlit's scan uploader.

Uses synthetic metadata and a placeholder .nii.gz file. It checks upload delivery
to Streamlit's session state only; it does not run image parsing or inference.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
FILE_INPUTS = 'input[type="file"]'


class StreamlitScanUploadSmokeTest(unittest.TestCase):
    def test_scan_upload_is_acknowledged_by_streamlit(self):
        with tempfile.TemporaryDirectory(prefix="streamlit_upload_smoke_") as tmp:
            work = Path(tmp)
            checkpoint_path = work / "metadata_only_checkpoint.pt"
            scan_path = work / "upload_only_test.nii.gz"
            log_path = work / "streamlit_app.log"

            # The guided form only needs these safe primitive metadata fields to render.
            # This deliberately is not a prediction-ready checkpoint.
            torch.save(
                {
                    "architecture": "RawImageTemporalFusionModel",
                    "feature_schema": {
                        "numeric_columns": ["C_age"],
                        "categorical_categories": {
                            "M_medicine_name": ["MedicineA", "MedicineB"]
                        },
                    },
                },
                checkpoint_path,
            )
            # The app stores uploaded bytes before any image parsing. Arbitrary synthetic
            # bytes are sufficient for this upload-delivery-only test.
            scan_path.write_bytes(b"synthetic upload-only payload; not a medical image")

            env = os.environ.copy()
            env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
            log_handle = log_path.open("w", encoding="utf-8")
            app_process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                    "--server.headless=true", "--server.port=8766",
                    "--server.address=127.0.0.1", "--browser.gatherUsageStats=false",
                ],
                cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
            )

            browser = None
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch()
                    page = browser.new_page()
                    page.set_default_timeout(10_000)

                    deadline = time.time() + 45
                    last_error = None
                    while True:
                        try:
                            page.goto(
                                "http://127.0.0.1:8766",
                                wait_until="domcontentloaded",
                                timeout=3_000,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            if time.time() >= deadline:
                                raise AssertionError(
                                    f"Streamlit did not start within 45 seconds: {last_error}"
                                ) from exc
                            time.sleep(1)

                    page.get_by_text("Choose a view").wait_for()
                    page.get_by_text("Run trained model", exact=True).click()
                    page.get_by_text(
                        "Guided entry: add one scan visit at a time", exact=True
                    ).wait_for()

                    def inspect_file_inputs():
                        return page.locator(FILE_INPUTS).evaluate_all(
                            """elements => elements.map((element, index) => ({
                                index,
                                accept: element.getAttribute("accept") || "",
                                files: Array.from(element.files || []).map(file => file.name)
                            }))"""
                        )

                    initial_inputs = inspect_file_inputs()
                    checkpoint_candidates = [
                        item for item in initial_inputs if ".pt" in item["accept"].lower()
                    ]
                    self.assertEqual(
                        len(checkpoint_candidates), 1,
                        f"Could not uniquely identify checkpoint input. Inputs: {initial_inputs}",
                    )
                    page.locator(FILE_INPUTS).nth(
                        checkpoint_candidates[0]["index"]
                    ).set_input_files(str(checkpoint_path))
                    page.get_by_text(
                        "Checkpoint recognised.", exact=False
                    ).wait_for(timeout=30_000)

                    # Find the scan uploader from its accepted extensions instead of a
                    # brittle ancestor/label relationship in Streamlit's generated DOM.
                    current_inputs = inspect_file_inputs()
                    scan_candidates = [
                        item for item in current_inputs
                        if any(ext in item["accept"].lower() for ext in ("nii", "gz", "mha", "nrrd"))
                        and not any(ext in item["accept"].lower() for ext in (".pt", "csv", "zip"))
                    ]
                    self.assertEqual(
                        len(scan_candidates), 1,
                        f"Could not uniquely identify scan input. Inputs: {current_inputs}",
                    )
                    scan_index = scan_candidates[0]["index"]
                    page.locator(FILE_INPUTS).nth(scan_index).set_input_files(str(scan_path))

                    try:
                        page.get_by_text(
                            "Scan attached: upload_only_test.nii.gz", exact=False
                        ).wait_for(timeout=15_000)
                    except PlaywrightTimeoutError as exc:
                        page_text = page.locator("body").inner_text()
                        try:
                            log_text = log_path.read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except OSError as log_error:
                            log_text = f"Could not read Streamlit log: {log_error}"
                        raise AssertionError(
                            "Streamlit did not acknowledge the scan upload.\n"
                            f"File inputs after selection: {inspect_file_inputs()}\n"
                            f"Visible UI:\n{page_text[-6000:]}\n"
                            f"Streamlit log:\n{log_text[-6000:]}"
                        ) from exc

                    self.assertIn(
                        "Scan attached: upload_only_test.nii.gz",
                        page.locator("body").inner_text(),
                    )
            finally:
                if browser is not None:
                    browser.close()
                app_process.terminate()
                try:
                    app_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    app_process.kill()
                    app_process.wait(timeout=5)
                log_handle.close()


if __name__ == "__main__":
    unittest.main()
