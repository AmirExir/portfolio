"""Regression cases for close-time end-of-day research signals and next-open execution."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_agent import risk_alerts
from market_agent.agent.evaluation import PolicyPerformance
from market_agent.evaluate_recovery import execution_targets, market_history, simulate
from market_agent.patterns import rsi
from market_agent.recovery_strategy import PROBE, recovery_path, recovery_snapshot


def closes() -> pd.Series:
    """Generate a rising market, repeated breakdowns and a slow rebound."""
    values = np.r_[
        np.linspace(110, 170, 65), np.linspace(155, 100, 18), np.arange(101, 140)
    ]
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values)))


def test_price_alerts_override_oversold_and_only_completed_sessions_influence_entries() -> (
    None
):
    prices = closes()
    for mode in ("trend", "immediate", "confirmed"):
        path = recovery_path(prices, entry=mode)
        for i in range(49, len(prices)):
            obs = risk_alerts.observe_closes("TEST", prices.iloc[: i + 1], {}, 0)
            assert (path.state.iloc[i] in {"Risk exit", "Oversold / risk"}) == (
                obs["breakdown"] or obs["shock"]
            )
        pd.testing.assert_frame_equal(
            recovery_path(prices.iloc[:92], entry=mode), path.iloc[:92]
        )
        assert path.state.iloc[63] == "Trend"
        assert path.state.iloc[82] == "Oversold / risk"
        assert path.sleeve_fraction.iloc[82] == 0
    early = recovery_path(prices, entry="immediate")
    confirmed = recovery_path(prices)
    assert early.sleeve_fraction.iloc[83] == PROBE
    assert confirmed.sleeve_fraction.iloc[83:91].eq(0).all()
    assert confirmed.confirmation_watch_next.iloc[90]
    assert prices.iloc[91] > confirmed.next_breakout_close.iloc[90]
    assert confirmed.rsi.iloc[91] >= 35
    assert confirmed.state.iloc[91] == "Starter entry"
    assert confirmed.sleeve_fraction.iloc[92] == PROBE
    assert recovery_path(prices, entry="trend").sleeve_fraction.iloc[83:93].eq(0).all()
    assert confirmed.state.iloc[-1] == "Trend"


@pytest.mark.parametrize("endpoint", [83, 95, 64])
def test_implied_rsi_hurdle_reproduces_next_rsi(endpoint: int) -> None:
    c = closes().iloc[:endpoint]
    hurdle = recovery_snapshot(c)["next_rsi_close"]
    next_session = c.index[-1] + pd.offsets.BDay()
    for delta in (-0.01, 0.01):
        scenario = c.copy()
        scenario.loc[next_session] = hurdle + delta
        assert bool(rsi(scenario).iloc[-1] >= 35) == (delta > 0)


def test_starter_expires_and_rsi_handles_one_sided_and_flat_windows() -> None:
    c = closes().iloc[:84]
    c = pd.concat(
        [
            c,
            pd.Series(
                101.0, index=pd.bdate_range(c.index[-1] + pd.offsets.BDay(), periods=31)
            ),
        ]
    )
    path = recovery_path(c, entry="immediate")
    assert path.sleeve_fraction.iloc[83:113].eq(PROBE).all()
    assert path.state.iloc[113] == "Starter expired"
    assert rsi(c).iloc[113] == 50
    assert rsi(c).iloc[63] == 100
    assert rsi(c).iloc[82] == 0
    assert rsi(c).iloc[:14].isna().all()


def test_gap_is_borne_by_previous_position_and_costs_apply_only_to_trades() -> None:
    c = closes()
    frames = {"TEST": pd.DataFrame({"open": c, "close": c})}
    weights = execution_targets(frames, "immediate")
    assert weights.iloc[83, 0] == 0
    assert weights.iloc[84, 0] == PROBE
    dates = pd.bdate_range("2024-04-01", periods=3)
    opens = pd.DataFrame({"TEST": [100.0, 80.0, 88.0]}, index=dates)
    targets = pd.DataFrame({"TEST": [1.0, 0.0, 0.0]}, index=dates)

    def run(cost: float) -> PolicyPerformance:
        return simulate(
            targets, opens, start="2024-04-01", end="2024-04-03", cost_bps=cost
        )

    assert run(0).cumulative_net_return == pytest.approx(-0.2)
    result = run(5)
    assert result.cumulative_net_return == pytest.approx(
        0.8 * (1 - 0.0005) / (1 + 0.0005) - 1
    )


def test_formatter_replays_legacy_warnings_and_keeps_stale_readings_out() -> None:
    c = closes().iloc[:83]
    obs = risk_alerts.observe_closes("TEST", c, {}, 0)
    obs["recovery"] = recovery_snapshot(c)
    row = {
        "Symbol": "TEST",
        "As Of Session": obs["as_of_session"],
        "Last Price": obs["close"],
        "Smart Policy": "Strong Sell / Avoid",
        "All Patterns": "Oversold, Volume Spike",
    }

    def watch(observation: dict) -> dict:
        return risk_alerts.evaluate_watch(
            [row], {"TEST": observation}, {}, ["TEST"], {}
        )

    result = watch(obs)
    message = risk_alerts.format_watch(result)
    assert "Strong Sell / Avoid" in message
    assert "Early rule 0%" in message and "Confirmed rule 0%" in message
    assert "On a one-cent grid, RSI first qualifies at" in message
    assert "Oversold, Volume Spike" in message
    assert f"exceed {c.tail(5).max():.2f}" in message
    del result["warnings"][0]["policy"]
    assert "Sell / Avoid" in risk_alerts.format_watch(result)
    stale = watch(dict(obs, issue="Stale"))
    assert "recovery" not in stale["observations"][0]
    assert "Early rule" not in risk_alerts.format_watch(stale)


def test_overbought_pattern_is_visible_during_a_rising_trend() -> None:
    c = closes().iloc[:64]
    obs = risk_alerts.observe_closes("TEST", c, {}, 0)
    obs["recovery"] = recovery_snapshot(c)
    row = {
        "Symbol": "TEST",
        "As Of Session": obs["as_of_session"],
        "Last Price": obs["close"],
        "All Patterns": "Uptrend, Overbought",
    }
    message = risk_alerts.format_watch(
        risk_alerts.evaluate_watch([row], {"TEST": obs}, {}, ["TEST"], {})
    )
    assert "Overbought readings" in message
    assert "RSI 100.0. Uptrend, Overbought." in message
    assert "Oversold recovery study" not in message


def test_missing_or_misaligned_prices_and_crypto_are_rejected(tmp_path: Path) -> None:
    c = closes().iloc[:5]
    frame = pd.DataFrame({"open": c, "close": c})
    frame.to_csv(tmp_path / "A.csv", index_label="date")
    frame.iloc[1:].to_csv(tmp_path / "B.csv", index_label="date")
    with pytest.raises(ValueError, match="session calendar"):
        market_history(["A", "B"], "2024-01-01", "2024-01-31", tmp_path)
    with pytest.raises(ValueError, match="US stock"):
        market_history(["BTC-USD"], "2024-01-01", "2024-01-31", tmp_path)
    with pytest.raises(ValueError, match="finite"):
        recovery_path(pd.Series([100, np.nan]))
