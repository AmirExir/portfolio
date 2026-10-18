"""Synthetic event contracts establish split behavior, not field performance."""
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from fault_classifier.modeling import FEATURE_COLUMNS, evaluate_independent_events, train_and_evaluate
from fault_classifier.tests.test_modeling import sample_data


def events():
    data = sample_data(60)
    # Each independent synthetic event contains measurements of all three classes.
    data["event_id"] = np.tile(np.arange(20).repeat(3), 3).astype(str)
    return data


def metadata():
    return {"schema_version": 1, "dataset_id": "synthetic-test", "source": "test fixture",
            "acquisition_context": "synthetic only", "label_definition": "G-C-B-A binary",
            "measurement_convention": "synthetic fixture values",
            "units": {name: "synthetic" for name in FEATURE_COLUMNS}}


class EventTests(unittest.TestCase):
    def train(self, data=None):
        return train_and_evaluate(events() if data is None else data, folds=3,
                                  models={"logistic": LogisticRegression()}, data_metadata=metadata())

    def test_events_are_disjoint_in_holdout_and_every_cv_fold(self):
        data = events()
        result = self.train(data)
        split = result.report["split"]
        self.assertFalse(set(split["train_groups"]) & set(split["test_groups"]))
        for fold in split["cv_row_positions"]:
            train = set(data.iloc[fold["train"]].event_id)
            valid = set(data.iloc[fold["validation"]].event_id)
            self.assertFalse(train & valid)
            self.assertFalse((train | valid) & set(split["test_groups"]))
        self.assertEqual(self.train(data).report["split"], split)

    def test_missing_event_ids_fail(self):
        data = events()
        data.loc[0, "event_id"] = None
        with self.assertRaisesRegex(ValueError, "missing"):
            self.train(data)

    def test_external_requires_disjoint_events_and_matching_units(self):
        result = self.train()
        external = events()
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluate_independent_events(result, external, metadata())
        external["event_id"] = "new-" + external.event_id
        with self.assertRaisesRegex(ValueError, "exactly repeat"):
            evaluate_independent_events(result, external, metadata())
        # Distinct synthetic measurements test the contract, not field validity.
        external.loc[:, list(FEATURE_COLUMNS)] += 0.123
        evaluated = evaluate_independent_events(result, external, metadata())
        self.assertEqual(evaluated["metrics"]["samples"], len(external))
        altered = metadata()
        altered["units"]["Ia"] = "other"
        with self.assertRaisesRegex(ValueError, "units"):
            evaluate_independent_events(result, external, altered)

    def test_missing_metadata_never_infers_units(self):
        altered = metadata()
        del altered["units"]["Va"]
        with self.assertRaisesRegex(ValueError, "units"):
            train_and_evaluate(events(), data_metadata=altered)


if __name__ == "__main__":
    unittest.main()
