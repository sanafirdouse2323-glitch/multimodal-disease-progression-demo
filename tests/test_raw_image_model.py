import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_raw_image_model import (
    Small3DCNN, normalize_volume, volume_to_tensor, split_by_patient,
    read_medical_volume, validate_manifest,
)


class RawImagePipelineTests(unittest.TestCase):
    def test_normalize_volume_is_finite(self):
        arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        result = normalize_volume(arr)
        self.assertEqual(result.shape, arr.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_constant_volume_is_finite(self):
        result = normalize_volume(np.ones((4, 4, 4), dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())

    def test_ct_window_clips_before_standardization(self):
        arr = np.array([-1500, -1000, 0, 1000, 1500], dtype=np.float32).reshape(1, 1, 5)
        result = normalize_volume(arr, modality="ct", ct_window_low=-1000, ct_window_high=1000)
        self.assertEqual(result.shape, arr.shape)
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_ct_window_rejected(self):
        with self.assertRaises(ValueError):
            normalize_volume(np.ones((2, 2, 2)), modality="ct", ct_window_low=100, ct_window_high=0)

    def test_volume_tensor_shape(self):
        result = volume_to_tensor(
            np.random.default_rng(1).normal(size=(12, 14, 16)).astype(np.float32), 16
        )
        self.assertEqual(tuple(result.shape), (1, 16, 16, 16))
        self.assertTrue(torch.isfinite(result).all())

    def test_3d_cnn_output_shape(self):
        model = Small3DCNN(n_classes=4)
        result = model(torch.zeros(2, 1, 32, 32, 32))
        self.assertEqual(tuple(result.shape), (2, 4))

    def test_patient_split_has_no_overlap(self):
        frame = pd.DataFrame({
            "patient_id": [f"P{i:02d}" for i in range(20) for _ in range(2)],
            "image_path": [f"scan_{i}_{v}.nii.gz" for i in range(20) for v in range(2)],
            "stage": [i % 4 for i in range(20) for _ in range(2)],
        })
        train, valid, test = split_by_patient(frame, seed=7)
        a, b, c = set(train.patient_id), set(valid.patient_id), set(test.patient_id)
        self.assertTrue(a.isdisjoint(b))
        self.assertTrue(a.isdisjoint(c))
        self.assertTrue(b.isdisjoint(c))
        self.assertEqual(len(train) + len(valid) + len(test), len(frame))

    def test_manifest_resolves_relative_paths_and_rejects_bad_stage(self):
        frame = pd.DataFrame({"patient_id": ["P1"], "image_path": ["scan.nii.gz"], "stage": [2]})
        with tempfile.TemporaryDirectory() as directory:
            validated = validate_manifest(frame, Path(directory))
            self.assertEqual(validated.loc[0, "stage"], 2)
            self.assertEqual(validated.loc[0, "image_path"], str((Path(directory) / "scan.nii.gz").resolve()))
            bad = frame.copy()
            bad["stage"] = [4]
            with self.assertRaises(ValueError):
                validate_manifest(bad, Path(directory))

    def test_reads_a_real_nifti_file_written_by_simpleitk(self):
        import SimpleITK as sitk
        original = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small_volume.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(original), str(path))
            loaded = read_medical_volume(str(path))
            self.assertEqual(loaded.shape, original.shape)
            np.testing.assert_allclose(loaded, original)


if __name__ == "__main__":
    unittest.main()
