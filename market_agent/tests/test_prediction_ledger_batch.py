from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from market_agent.agent.ledger import (
    DuplicateLedgerRecordError,
    OutcomeRecord,
    PredictionLedger,
    PredictionRecord,
    UnknownPredictionError,
)


UTC = timezone.utc


def _prediction(prediction_id: str, symbol: str) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=prediction_id,
        created_at_utc=datetime(2026, 1, 2, 22, tzinfo=UTC),
        data_cutoff_utc=datetime(2026, 1, 2, 21, tzinfo=UTC),
        as_of_session=date(2026, 1, 2),
        target_session=date(2026, 1, 5),
        symbol=symbol,
        horizon_sessions=1,
        model_name="Ridge",
        model_version="ridge-v1",
        forecast_return=0.05,
        target_weight=0.0,
    )


def _outcome(
    prediction: PredictionRecord,
    *,
    outcome_id: str | None = None,
    recorded_at_utc: datetime | None = None,
    target_session: date | None = None,
    target_maturity_utc: datetime | None = None,
) -> OutcomeRecord:
    return OutcomeRecord(
        outcome_id=outcome_id or f"outcome-{prediction.prediction_id}",
        prediction_id=prediction.prediction_id,
        recorded_at_utc=(
            recorded_at_utc or datetime(2026, 1, 5, 22, tzinfo=UTC)
        ),
        target_session=target_session or prediction.target_session,
        target_maturity_utc=(
            target_maturity_utc or prediction.target_maturity_utc
        ),
        realized_return=0.05,
        benchmark_return=0.01,
    )


class PredictionLedgerBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.ledger = PredictionLedger(
            Path(self.temporary_directory.name) / "prediction-ledger.jsonl"
        )

    def test_appends_unique_batch_with_one_verification_and_fsync(self) -> None:
        records = (
            _prediction("pred-a", "SNDK"),
            _prediction("pred-b", "MU"),
            _prediction("pred-c", "NVDA"),
        )

        with (
            mock.patch.object(
                self.ledger,
                "_read_entries_from_handle",
                wraps=self.ledger._read_entries_from_handle,
            ) as read_entries,
            mock.patch("market_agent.agent.ledger.os.fsync") as fsync,
        ):
            result = self.ledger.append_predictions(records)

        self.assertEqual(result.appended_count, 3)
        self.assertEqual(result.duplicate_count, 0)
        self.assertEqual(result.duplicate_prediction_ids, ())
        self.assertEqual(
            [entry.sequence for entry in result.entries],
            [1, 2, 3],
        )
        self.assertIsNone(result.entries[0].previous_hash)
        self.assertEqual(
            result.entries[1].previous_hash,
            result.entries[0].record_hash,
        )
        self.assertEqual(
            result.entries[2].previous_hash,
            result.entries[1].record_hash,
        )
        read_entries.assert_called_once()
        fsync.assert_called_once()
        self.assertEqual(self.ledger.predictions(), records)

    def test_skips_existing_and_in_batch_duplicates_in_candidate_order(self) -> None:
        existing = _prediction("pred-existing", "SNDK")
        first_new = _prediction("pred-new", "MU")
        last_new = _prediction("pred-last", "NVDA")
        self.ledger.append_prediction(existing)

        with (
            mock.patch.object(
                self.ledger,
                "_read_entries_from_handle",
                wraps=self.ledger._read_entries_from_handle,
            ) as read_entries,
            mock.patch("market_agent.agent.ledger.os.fsync") as fsync,
        ):
            result = self.ledger.append_predictions(
                (existing, first_new, first_new, last_new)
            )

        self.assertEqual(result.appended_count, 2)
        self.assertEqual(result.duplicate_count, 2)
        self.assertEqual(
            result.duplicate_prediction_ids,
            ("pred-existing", "pred-new"),
        )
        self.assertEqual(
            [entry.record.prediction_id for entry in result.entries],
            ["pred-new", "pred-last"],
        )
        self.assertEqual(
            [entry.sequence for entry in result.entries],
            [2, 3],
        )
        read_entries.assert_called_once()
        fsync.assert_called_once()
        self.assertEqual(
            [record.prediction_id for record in self.ledger.predictions()],
            ["pred-existing", "pred-new", "pred-last"],
        )

    def test_single_append_still_rejects_duplicates(self) -> None:
        record = _prediction("pred-existing", "SNDK")
        self.ledger.append_prediction(record)

        with self.assertRaises(DuplicateLedgerRecordError):
            self.ledger.append_prediction(record)

    def test_appends_unique_outcome_batch_with_one_verification_and_fsync(
        self,
    ) -> None:
        predictions = (
            _prediction("pred-a", "SNDK"),
            _prediction("pred-b", "MU"),
            _prediction("pred-c", "NVDA"),
        )
        self.ledger.append_predictions(predictions)
        outcomes = tuple(_outcome(prediction) for prediction in predictions)

        with (
            mock.patch.object(
                self.ledger,
                "_read_entries_from_handle",
                wraps=self.ledger._read_entries_from_handle,
            ) as read_entries,
            mock.patch("market_agent.agent.ledger.os.fsync") as fsync,
        ):
            result = self.ledger.append_outcomes(outcomes)

        self.assertEqual(result.appended_count, 3)
        self.assertEqual(result.duplicate_count, 0)
        self.assertEqual(result.failures, ())
        self.assertEqual(
            [entry.sequence for entry in result.entries],
            [4, 5, 6],
        )
        self.assertEqual(
            result.entries[0].previous_hash,
            self.ledger.read_entries()[2].record_hash,
        )
        self.assertEqual(
            result.entries[1].previous_hash,
            result.entries[0].record_hash,
        )
        read_entries.assert_called_once()
        fsync.assert_called_once()
        self.assertEqual(self.ledger.outcomes(), outcomes)

    def test_outcome_batch_reports_existing_and_in_batch_duplicates(self) -> None:
        predictions = (
            _prediction("pred-a", "SNDK"),
            _prediction("pred-b", "MU"),
            _prediction("pred-c", "NVDA"),
        )
        self.ledger.append_predictions(predictions)
        existing = _outcome(predictions[0])
        first_new = _outcome(predictions[1])
        last_new = _outcome(predictions[2])
        self.ledger.append_outcome(existing)

        with mock.patch("market_agent.agent.ledger.os.fsync") as fsync:
            result = self.ledger.append_outcomes(
                (existing, first_new, first_new, last_new)
            )

        self.assertEqual(result.appended_count, 2)
        self.assertEqual(result.duplicate_count, 2)
        self.assertEqual(
            result.duplicate_outcome_ids,
            ("outcome-pred-a", "outcome-pred-b"),
        )
        self.assertEqual(result.failures, ())
        fsync.assert_called_once()
        self.assertEqual(
            [outcome.outcome_id for outcome in self.ledger.outcomes()],
            ["outcome-pred-a", "outcome-pred-b", "outcome-pred-c"],
        )

    def test_outcome_batch_reports_linkage_and_maturity_failures(self) -> None:
        first = _prediction("pred-a", "SNDK")
        wrong_session_prediction = _prediction("pred-b", "MU")
        wrong_maturity_prediction = _prediction("pred-c", "NVDA")
        immature_prediction = _prediction("pred-d", "AMD")
        last = _prediction("pred-e", "INTC")
        self.ledger.append_predictions(
            (
                first,
                wrong_session_prediction,
                wrong_maturity_prediction,
                immature_prediction,
                last,
            )
        )
        existing = _outcome(first)
        self.ledger.append_outcome(existing)
        unknown_prediction = _prediction("pred-unknown", "NVDA")
        unknown = _outcome(unknown_prediction)
        wrong_session = _outcome(
            wrong_session_prediction,
            outcome_id="outcome-wrong-session",
            recorded_at_utc=datetime(2026, 1, 6, 22, tzinfo=UTC),
            target_session=date(2026, 1, 6),
            target_maturity_utc=datetime(2026, 1, 6, 21, tzinfo=UTC),
        )
        wrong_maturity = _outcome(
            wrong_maturity_prediction,
            outcome_id="outcome-wrong-maturity",
            target_maturity_utc=datetime(2026, 1, 5, 20, tzinfo=UTC),
        )
        immature = _outcome(immature_prediction)
        # OutcomeRecord prevents this state at construction; mutate only to
        # regression-test the ledger's independent defensive maturity check.
        object.__setattr__(
            immature,
            "recorded_at_utc",
            datetime(2026, 1, 5, 20, tzinfo=UTC),
        )
        valid = _outcome(last)

        result = self.ledger.append_outcomes(
            (
                existing,
                unknown,
                wrong_session,
                wrong_maturity,
                immature,
                valid,
            )
        )

        self.assertEqual(result.appended_count, 1)
        self.assertEqual(result.duplicate_count, 1)
        self.assertEqual(
            result.duplicate_outcome_ids,
            (existing.outcome_id,),
        )
        self.assertEqual(len(result.failures), 4)
        self.assertIsInstance(
            result.failures[0].error,
            UnknownPredictionError,
        )
        self.assertIn("target_session", str(result.failures[1].error))
        self.assertIn("target_maturity_utc", str(result.failures[2].error))
        self.assertIn("before target maturity", str(result.failures[3].error))
        self.assertEqual(self.ledger.outcomes(), (existing, valid))

    def test_single_outcome_append_still_rejects_unknown_prediction(self) -> None:
        unknown = _outcome(_prediction("pred-unknown", "NVDA"))

        with self.assertRaises(UnknownPredictionError):
            self.ledger.append_outcome(unknown)

    def test_single_outcome_append_still_rejects_duplicates(self) -> None:
        prediction = _prediction("pred-existing", "SNDK")
        outcome = _outcome(prediction)
        self.ledger.append_prediction(prediction)
        self.ledger.append_outcome(outcome)

        with self.assertRaises(DuplicateLedgerRecordError):
            self.ledger.append_outcome(outcome)


if __name__ == "__main__":
    unittest.main()
