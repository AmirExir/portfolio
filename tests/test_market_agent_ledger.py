from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from market_agent.agent.ledger import (
    DuplicateLedgerRecordError,
    ImmatureOutcomeError,
    LedgerIntegrityError,
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
    UnknownPredictionError,
)


UTC = timezone.utc


def _prediction(
    prediction_id: str = "pred-1",
    *,
    horizon_sessions: int = 2,
    target_session: date = date(2026, 1, 6),
) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        created_at_utc=datetime(2026, 1, 2, 15, tzinfo=UTC),
        data_cutoff_utc=datetime(2026, 1, 2, 14, 59, tzinfo=UTC),
        as_of_session=date(2026, 1, 2),
        target_session=target_session,
        symbol="sndk",
        horizon_sessions=horizon_sessions,
        model_name="ridge",
        model_version="ridge-v1",
        policy_version="policy-v1",
        forecast_return=0.05,
        target_weight=0.25,
        probability_positive=0.70,
        lower_bound_return=-0.02,
        upper_bound_return=0.12,
        feature_set_version="features-v1",
        feature_hash="abc123",
    )


def _outcome(
    prediction_id: str = "pred-1",
    *,
    target_session: date = date(2026, 1, 6),
) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=f"outcome-{prediction_id}",
        prediction_id=prediction_id,
        recorded_at_utc=datetime(2026, 1, 6, 22, tzinfo=UTC),
        target_session=target_session,
        realized_return=0.03,
        benchmark_return=0.01,
        transaction_cost_return=0.0005,
        entry_price=100.0,
        exit_price=103.0,
        max_adverse_excursion=-0.04,
        max_favorable_excursion=0.06,
        stop_breached=False,
        data_source="deterministic-test",
    )


class PredictionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "predictions.jsonl"
        self.ledger = PredictionLedger(self.path)

    def test_prediction_and_matured_outcome_are_append_only(self) -> None:
        prediction = _prediction()
        outcome = _outcome()

        prediction_entry = self.ledger.append_prediction(prediction)
        outcome_entry = self.ledger.append_outcome(outcome)

        self.assertEqual(prediction_entry.sequence, 1)
        self.assertEqual(outcome_entry.sequence, 2)
        self.assertEqual(
            outcome_entry.previous_hash,
            prediction_entry.record_hash,
        )
        completed = self.ledger.completed_predictions(
            date(2026, 1, 6),
            horizon_sessions=2,
        )
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].prediction, prediction)
        self.assertEqual(completed[0].outcome, outcome)
        self.assertAlmostEqual(outcome.net_return, 0.0295)

        with self.assertRaises(DuplicateLedgerRecordError):
            self.ledger.append_prediction(prediction)
        with self.assertRaises(DuplicateLedgerRecordError):
            self.ledger.append_outcome(
                OutcomeRecord(
                    **{
                        **outcome.__dict__,
                        "outcome_id": "different-outcome-id",
                    }
                )
            )

    def test_maturity_and_horizon_filters_are_explicit(self) -> None:
        short = _prediction()
        long = _prediction(
            "pred-30",
            horizon_sessions=30,
            target_session=date(2026, 2, 13),
        )
        self.ledger.append_prediction(short)
        self.ledger.append_prediction(long)

        self.assertEqual(
            self.ledger.matured_predictions(
                date(2026, 1, 5),
                horizon_sessions=2,
            ),
            (),
        )
        self.assertEqual(
            self.ledger.pending_outcomes(
                date(2026, 1, 6),
                horizon_sessions=2,
            ),
            (short,),
        )
        self.assertEqual(
            self.ledger.matured_predictions(
                date(2026, 1, 6),
                horizon_sessions=30,
            ),
            (),
        )

    def test_unknown_or_immature_outcomes_are_rejected(self) -> None:
        with self.assertRaises(UnknownPredictionError):
            self.ledger.append_outcome(_outcome("missing"))

        self.ledger.append_prediction(_prediction())
        with self.assertRaises(ImmatureOutcomeError):
            OutcomeRecord(
                outcome_id="too-early",
                prediction_id="pred-1",
                recorded_at_utc=datetime(2026, 1, 5, 22, tzinfo=UTC),
                target_session=date(2026, 1, 6),
                realized_return=0.01,
                benchmark_return=0.0,
            )

    def test_tampered_record_fails_hash_verification(self) -> None:
        self.ledger.append_prediction(_prediction())
        envelope = json.loads(self.path.read_text(encoding="utf-8"))
        envelope["record"]["forecast_return"] = 9.99
        self.path.write_text(
            json.dumps(envelope, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(LedgerIntegrityError, "hash mismatch"):
            self.ledger.read_entries()

    def test_naive_point_in_time_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            PredictionRecord(
                **{
                    **_prediction().__dict__,
                    "prediction_id": "naive",
                    "created_at_utc": datetime(2026, 1, 2, 15),
                }
            )


if __name__ == "__main__":
    unittest.main()

