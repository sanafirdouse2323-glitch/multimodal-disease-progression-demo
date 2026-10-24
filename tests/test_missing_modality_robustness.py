import random
import unittest

import numpy as np

import demo_disease_progression as demo
from experiments import missing_modality_robustness as experiment


class MissingFeatureGroupExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = demo.generate_demo_dataset(n_patients=12, seed=123)
        train_ids, _, _ = demo.split_patient_ids(raw, seed=123)
        train_raw = raw[raw["patient_id"].isin(train_ids)].copy()
        means, stds = demo.fit_feature_stats(train_raw)
        transformed = demo.transform_features(raw, means, stds)
        cls.sequences = demo.make_sequences(transformed)

    def test_remove_feature_group_zeros_values_and_observation_masks(self):
        original = self.sequences[0]
        original_copy = {key: value.copy() for key, value in original.items()}
        masked = experiment.remove_feature_groups([original], ["wearable"])[0]

        self.assertTrue(np.all(masked["wearable"] == 0))
        self.assertTrue(np.all(masked["wearable_mask"] == 0))
        np.testing.assert_array_equal(masked["labels"], original["labels"])
        np.testing.assert_array_equal(masked["time"], original["time"])
        for name in ("clinical", "treatment", "medicine", "imaging"):
            np.testing.assert_array_equal(masked[name], original[name])
            np.testing.assert_array_equal(masked[f"{name}_mask"], original[f"{name}_mask"])
        for key, value in original_copy.items():
            np.testing.assert_array_equal(original[key], value)

    def test_unknown_feature_group_is_rejected(self):
        with self.assertRaises(ValueError):
            experiment.remove_feature_groups(self.sequences, ["unknown_group"])

    def test_feature_group_dropout_changes_a_copy_without_mutating_source(self):
        source = self.sequences[0]
        source_copy = {key: value.copy() for key, value in source.items()}
        random.seed(10)
        dataset = experiment.FeatureGroupDropoutDataset(self.sequences, probability=1.0)
        dropped = dataset[0]

        changed = [
            name for name in experiment.FEATURE_GROUPS
            if not np.array_equal(dropped[name], source[name])
            or not np.array_equal(dropped[f"{name}_mask"], source[f"{name}_mask"])
        ]
        self.assertGreaterEqual(len(changed), 1)
        self.assertLess(len(changed), len(experiment.FEATURE_GROUPS))
        for name in changed:
            self.assertTrue(np.all(dropped[name] == 0))
            self.assertTrue(np.all(dropped[f"{name}_mask"] == 0))
        for key, value in source_copy.items():
            np.testing.assert_array_equal(source[key], value)

    def test_calibration_metrics_are_finite_and_validate_probability_rows(self):
        y_true = np.array([0, 1, 1, 0], dtype=np.int64)
        probabilities = np.array([
            [0.8, 0.2],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        ece = experiment.expected_calibration_error(y_true, probabilities, n_bins=5)
        brier = experiment.multiclass_brier_score(y_true, probabilities)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)
        self.assertGreaterEqual(brier, 0.0)
        self.assertLessEqual(brier, 2.0)

        with self.assertRaises(ValueError):
            experiment.expected_calibration_error(y_true, probabilities * 2.0)
        with self.assertRaises(ValueError):
            experiment.expected_calibration_error(y_true, np.array([[0.2, 0.2]]))

    def test_evaluation_conditions_include_complete_and_each_single_missing_feature_group(self):
        conditions = experiment.make_conditions(self.sequences)
        self.assertEqual(tuple(conditions), experiment.CONDITION_ORDER)
        self.assertEqual(set(conditions), {
            "complete", "without_clinical", "without_treatment",
            "without_medicine", "without_wearable", "without_imaging"
        })


if __name__ == "__main__":
    unittest.main()
