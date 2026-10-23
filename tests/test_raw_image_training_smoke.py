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
            self.assertTrue((output / "raw_image_3dcnn.pt").is_file())
            self.assertTrue((output / "training_history.csv").is_file())
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["task"], "supervised 3D scan classification")
            self.assertEqual(summary["n_scans"], 10)
            self.assertEqual(summary["n_patients"], 10)
            self.assertIn("accuracy", summary["test_metrics"])
            self.assertIn("macro_f1", summary["test_metrics"])


if __name__ == "__main__":
    unittest.main()
