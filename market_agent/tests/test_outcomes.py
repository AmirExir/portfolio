from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from market_agent.agent.ledger import PredictionLedger, PredictionRecord
from market_agent.agent.outcomes import append_matured_outcomes


UTC = timezone.utc


class OutcomeMaturityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        ledger_path = Path(self.temporary_directory.name) / "ledger.jsonl"
        self.ledger = PredictionLedger(ledger_path)
        self.prediction = PredictionRecord(
            prediction_id="pred-test",
            created_at_utc=datetime(2026, 1, 2, 22, tzinfo=UTC),
            data_cutoff_utc=datetime(2026, 1, 2, 21, tzinfo=UTC),
            as_of_session=date(2026, 1, 2),
            target_session=date(2026, 1, 6),
            symbol="SNDK",
            horizon_sessions=2,
            model_name="Ridge",
            model_version="ridge-v1",
            forecast_return=0.10,
            target_weight=0.05,
        )
        self.ledger.append_prediction(self.prediction)
        index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        self.frames = {
            "SNDK": pd.DataFrame(
                {"close": [100.0, 95.0, 130.0]},
                index=index,
            ),
            "SPY": pd.DataFrame(
                {"close": [500.0, 502.0, 505.0]},
                index=index,
            ),
        }

    def test_waits_for_close_then_appends_realized_result(self) -> None:
        before_close = append_matured_outcomes(
            self.ledger,
            price_loader=self.frames.__getitem__,
            as_of_utc=datetime(2026, 1, 6, 20, 0, tzinfo=UTC),
        )
        self.assertEqual(before_close.appended, 0)
        self.assertEqual(before_close.pending_not_mature, 1)

        after_close = append_matured_outcomes(
            self.ledger,
            price_loader=self.frames.__getitem__,
            as_of_utc=datetime(2026, 1, 6, 22, 0, tzinfo=UTC),
        )
        self.assertEqual(after_close.appended, 1)
        outcomes = self.ledger.outcomes()
        self.assertEqual(len(outcomes), 1)
        self.assertAlmostEqual(outcomes[0].realized_return, 0.30)
        self.assertAlmostEqual(outcomes[0].benchmark_return, 0.01)
        self.assertAlmostEqual(outcomes[0].max_adverse_excursion or 0.0, -0.05)
        self.assertAlmostEqual(outcomes[0].max_favorable_excursion or 0.0, 0.30)

    def test_delayed_execution_uses_recorded_return_start_session(self) -> None:
        ledger = PredictionLedger(
            Path(self.temporary_directory.name)
            / "delayed-execution-ledger.jsonl"
        )
        prediction = PredictionRecord(
            prediction_id="pred-delayed-execution",
            created_at_utc=datetime(2026, 1, 2, 22, tzinfo=UTC),
            data_cutoff_utc=datetime(2026, 1, 2, 21, tzinfo=UTC),
            as_of_session=date(2026, 1, 2),
            return_start_session=date(2026, 1, 5),
            target_session=date(2026, 1, 6),
            symbol="SNDK",
            horizon_sessions=1,
            model_name="RL Policy",
            model_version="execution-aligned-v3",
            policy_version="rl-shadow-contextual-v3-h30",
            forecast_return=0.0,
            target_weight=0.05,
        )
        ledger.append_prediction(prediction)

        result = append_matured_outcomes(
            ledger,
            price_loader=self.frames.__getitem__,
            as_of_utc=datetime(2026, 1, 6, 22, 0, tzinfo=UTC),
        )

        self.assertEqual(result.appended, 1)
        outcome = ledger.outcomes()[0]
        self.assertAlmostEqual(
            outcome.realized_return,
            130.0 / 95.0 - 1.0,
        )
        self.assertAlmostEqual(
            outcome.benchmark_return,
            505.0 / 502.0 - 1.0,
        )
        self.assertEqual(
            outcome.metadata["return_start_session"],
            "2026-01-05",
        )


if __name__ == "__main__":
    unittest.main()
