import unittest

import numpy as np
import pandas as pd
import torch

import demo_disease_progression as demo


class DemoPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = demo.generate_demo_dataset(n_patients=16, seed=123)

    def test_labels_are_in_expected_range(self):
        self.assertTrue(set(self.raw["stage"].unique()).issubset({0, 1, 2, 3}))

    def test_patient_splits_do_not_overlap(self):
        train_ids, valid_ids, test_ids = demo.split_patient_ids(self.raw, seed=7)
        self.assertFalse(train_ids & valid_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(valid_ids & test_ids)
        self.assertEqual(len(train_ids | valid_ids | test_ids), self.raw["patient_id"].nunique())

    def test_missingness_indicator_survives_imputation(self):
        patient_ids = sorted(self.raw["patient_id"].unique())
        train_ids = set(patient_ids[:10])
        valid_ids = set(patient_ids[10:])
        train = self.raw[self.raw.patient_id.isin(train_ids)].copy()
        valid = self.raw[self.raw.patient_id.isin(valid_ids)].copy()
        row_index = train.index[0]
        train.loc[row_index, "lab1"] = np.nan

        transformed_train, transformed_valid = demo.prepare_features(train, valid)
        self.assertEqual(float(transformed_train.loc[row_index, "lab1__observed"]), 0.0)
        self.assertTrue(np.isfinite(transformed_train["lab1"].to_numpy()).all())
        self.assertTrue(np.isfinite(transformed_valid["lab1"].to_numpy()).all())

    def test_scaling_statistics_come_from_training_split_only(self):
        patient_ids = sorted(self.raw["patient_id"].unique())
        train = self.raw[self.raw.patient_id.isin(patient_ids[:10])].copy()
        other = self.raw[self.raw.patient_id.isin(patient_ids[10:])].copy()
        train_mean = train["lab1"].mean()
        other.loc[:, "lab1"] = 1_000_000.0

        means, stds = demo.fit_feature_stats(train)
        transformed_other = demo.transform_features(other, means, stds)

        self.assertAlmostEqual(float(means["lab1"]), float(train_mean))
        self.assertGreater(float(transformed_other["lab1"].iloc[0]), 1000.0)

    def test_baseline_features_include_missingness_indicators(self):
        patient_ids = sorted(self.raw["patient_id"].unique())
        train = self.raw[self.raw.patient_id.isin(patient_ids[:10])].copy()
        valid = self.raw[self.raw.patient_id.isin(patient_ids[10:])].copy()
        transformed_train, transformed_valid = demo.prepare_features(train, valid)
        features, labels = demo.flatten_visit_features(transformed_valid)
        expected_feature_count = 2 * sum(len(cols) for cols in demo.GROUPS.values())
        self.assertEqual(features.shape[1], expected_feature_count)
        self.assertEqual(features.shape[0], len(labels))
        self.assertTrue(np.isfinite(features).all())

    def test_collation_marks_padding_and_model_outputs_finite_logits(self):
        patient_ids = sorted(self.raw["patient_id"].unique())
        train = self.raw[self.raw.patient_id.isin(patient_ids[:12])].copy()
        valid = self.raw[self.raw.patient_id.isin(patient_ids[12:])].copy()
        transformed_train, transformed_valid = demo.prepare_features(train, valid)
        sequences = demo.make_sequences(transformed_valid)
        batch = demo.collate_sequences(sequences)

        self.assertEqual(batch["labels"].shape, batch["padding_mask"].shape)
        self.assertTrue(torch.equal(batch["padding_mask"], batch["labels"].eq(-100)))

        model = demo.MultimodalTemporalModel()
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        self.assertEqual(logits.shape[:2], batch["labels"].shape)
        self.assertEqual(logits.shape[-1], 4)
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
