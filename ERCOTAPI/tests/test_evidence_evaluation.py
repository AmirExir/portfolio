"""Offline evaluation mechanics and fixture provenance."""

import json
import unittest
from pathlib import Path

from ERCOTAPI.rag_ingestion.evaluation import evaluate_predictions, validate_cases


ROOT = Path(__file__).resolve().parents[2]


class EvidenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads((ROOT / "ERCOTAPI/evaluation/synthetic_cases.json").read_text())
        self.predictions = json.loads((ROOT / "ERCOTAPI/evaluation/synthetic_predictions.json").read_text())

    def test_support_detection_is_not_reported_as_semantic_accuracy(self):
        report = evaluate_predictions(self.cases, self.predictions, root=ROOT)
        self.assertEqual(report["ground_truth_status"], "synthetic_fixture")
        self.assertEqual(report["metrics"]["support_review_detection_accuracy"]["value"], 1.0)
        self.assertEqual(report["metrics"]["abstention_accuracy"]["value"], 1.0)
        self.assertIsNone(report["semantic_support_accuracy"])

    def test_missing_predictions_are_failures_not_dropped_rows(self):
        report = evaluate_predictions(self.cases, [], root=ROOT)
        self.assertEqual(report["metrics"]["expected_evidence_recall"]["value"], 0.0)
        self.assertEqual(report["metrics"]["abstention_accuracy"]["value"], 0.0)
        self.assertEqual(report["metrics"]["requested_as_of_match_rate"]["value"], 0.0)

    def test_duplicate_or_unknown_prediction_ids_are_rejected(self):
        for predictions in ([self.predictions[0]] * 2, [{"id": "unknown"}]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_predictions(self.cases, predictions, root=ROOT)

    def test_engineer_reviewed_claim_requires_explicit_provenance(self):
        self.cases["ground_truth_status"] = "engineer_reviewed"
        with self.assertRaisesRegex(ValueError, "reviewer"):
            validate_cases(self.cases, ROOT)

    def test_malformed_labels_and_prediction_structures_fail_clearly(self):
        self.cases["cases"][0]["expected_abstention"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_cases(self.cases, ROOT)
        self.cases["cases"][0]["expected_abstention"] = False
        with self.assertRaisesRegex(ValueError, "JSON list"):
            evaluate_predictions(self.cases, {"id": "changed-quantity"}, root=ROOT)

    def test_real_source_anchors_exist_but_are_not_reviewed_ground_truth(self):
        cases = json.loads((ROOT / "ERCOTAPI/evaluation/repository_cases.json").read_text())
        self.assertEqual(cases["ground_truth_status"], "draft_requires_engineer_review")
        checks = validate_cases(cases, ROOT)
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(item["anchor_found"] for item in checks))


if __name__ == "__main__":
    unittest.main()
