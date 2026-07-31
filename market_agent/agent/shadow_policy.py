"""Execution-aligned, long-only tabular policy for shadow evaluation.

This module is intentionally isolated from order submission and the existing
forecast model-selection path.  It provides a reproducible policy that maps a
decision-time state to one of four target fractions of a per-name risk budget:
0%, 25%, 50%, or 100%.

Input rows contain only features available at decision time plus two explicit
labels, ``forward_asset_return`` and ``forward_benchmark_return``.  The labels
are used for chronological reward updates and are never part of the state.
Returns and weights are decimal fractions throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping

import numpy as np
import pandas as pd


ACTION_FRACTIONS: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)
FORWARD_ASSET_RETURN = "forward_asset_return"
FORWARD_BENCHMARK_RETURN = "forward_benchmark_return"
OPTIONAL_PORTFOLIO_RISK = "portfolio_risk"

CONTEXT_FEATURES: tuple[str, ...] = (
    "decision_horizon",
    "asset_momentum",
    "asset_volatility",
    "spy_trend",
    "qqq_trend",
    "vix_level",
    "tlt_trend",
    "gap",
    "volume_shock",
    "event_flag",
    "event_score",
    "event_confidence",
    "sector_relative_strength",
    "sector_breadth",
    "forecast_return",
    "forecast_probability_up",
    "forecast_lower_bound",
    "forecast_model_agreement",
    "forecast_uncertainty",
)
OPTIONAL_CONTEXT_FEATURES: tuple[str, ...] = ("portfolio_correlation",)
STATE_COMPONENTS: tuple[str, ...] = (
    "current_exposure",
    "entry_drawdown",
    "holding_period",
    "previous_action",
    *CONTEXT_FEATURES,
    *OPTIONAL_CONTEXT_FEATURES,
)

_TOLERANCE = 1e-12
_MISSING_BIN = -1
_DEFAULT_BIN_EDGES: dict[str, tuple[float, ...]] = {
    # Exposure and previous action are expressed as risk-budget utilization.
    "current_exposure": (0.125, 0.375, 0.75),
    "entry_drawdown": (-0.10, -0.05, -0.02, -0.005),
    "holding_period": (0.5, 2.0, 5.0, 10.0, 20.0, 40.0),
    "previous_action": (0.125, 0.375, 0.75),
    "decision_horizon": (1.5, 5.0, 15.0, 45.0),
    # Momentum/trend/gap/relative-strength inputs are decimal returns.
    "asset_momentum": (-0.10, -0.03, 0.0, 0.03, 0.10),
    # Asset volatility is annualized and expressed as a decimal.
    "asset_volatility": (0.15, 0.25, 0.40, 0.70),
    "spy_trend": (-0.03, 0.0, 0.03),
    "qqq_trend": (-0.03, 0.0, 0.03),
    "vix_level": (15.0, 20.0, 30.0, 40.0),
    "tlt_trend": (-0.03, 0.0, 0.03),
    "gap": (-0.03, -0.01, 0.01, 0.03),
    # Volume shock is current volume / trailing normal volume - 1.
    "volume_shock": (-0.50, 0.0, 0.50, 1.0),
    "event_flag": (0.5,),
    "event_score": (-0.50, -0.10, 0.10, 0.50),
    "event_confidence": (0.25, 0.50, 0.75),
    "sector_relative_strength": (-0.05, -0.01, 0.01, 0.05),
    "sector_breadth": (0.25, 0.50, 0.75),
    "forecast_return": (-0.10, -0.03, 0.0, 0.03, 0.10),
    "forecast_probability_up": (0.35, 0.50, 0.65),
    "forecast_lower_bound": (-0.10, -0.03, 0.0, 0.03, 0.10),
    "forecast_model_agreement": (0.25, 0.50, 0.75),
    "forecast_uncertainty": (0.01, 0.03, 0.07, 0.15),
    "portfolio_correlation": (-0.25, 0.25, 0.60, 0.80),
}


PolicyState = tuple[int, ...]


@dataclass(frozen=True)
class ShadowPolicyConfig:
    """Configuration for chronological tabular Q-learning and abstention."""

    risk_budget_fraction: float = 0.05
    learning_rate: float = 0.15
    discount_factor: float = 0.90
    exploration_rate: float = 0.15
    training_epochs: int = 8
    transaction_cost_bps: float = 10.0
    drawdown_penalty: float = 0.10
    portfolio_risk_penalty: float = 0.01
    min_state_visits: int = 5
    min_action_visits: int = 1
    tie_tolerance: float = 1e-10
    random_seed: int = 1729
    minimum_purge_sessions: int = 30

    def __post_init__(self) -> None:
        """Reject invalid policy or evaluation assumptions."""
        _require_fraction(
            "risk_budget_fraction",
            self.risk_budget_fraction,
            allow_zero=False,
        )
        _require_fraction("learning_rate", self.learning_rate, allow_zero=False)
        _require_fraction("discount_factor", self.discount_factor)
        _require_fraction("exploration_rate", self.exploration_rate)
        _require_nonnegative("transaction_cost_bps", self.transaction_cost_bps)
        _require_nonnegative("drawdown_penalty", self.drawdown_penalty)
        _require_nonnegative("portfolio_risk_penalty", self.portfolio_risk_penalty)
        _require_nonnegative("tie_tolerance", self.tie_tolerance)
        for name, value, minimum in (
            ("training_epochs", self.training_epochs, 1),
            ("min_state_visits", self.min_state_visits, 1),
            ("min_action_visits", self.min_action_visits, 1),
            ("minimum_purge_sessions", self.minimum_purge_sessions, 1),
        ):
            if not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.random_seed, (int, np.integer)):
            raise ValueError("random_seed must be an integer")


@dataclass(frozen=True)
class PolicyPosition:
    """Position-dependent inputs required to make one policy decision."""

    exposure: float = 0.0
    entry_drawdown: float = 0.0
    holding_period: int = 0
    previous_action_fraction: float = 0.0


@dataclass(frozen=True)
class ShadowDecision:
    """One frozen-policy target decision and its abstention diagnostics."""

    action_fraction: float
    target_exposure: float
    state: PolicyState
    state_visits: int
    action_visits: int
    q_values: tuple[float, ...]
    abstained: bool
    reason: str


@dataclass(frozen=True)
class RewardBreakdown:
    """Components of the execution-aligned daily reward."""

    reward: float
    active_return: float
    turnover_cost: float
    drawdown_cost: float
    portfolio_risk_cost: float


@dataclass(frozen=True)
class ShadowEvaluation:
    """Held-out chronological evaluation produced without policy updates."""

    decisions: pd.DataFrame
    cumulative_reward: float
    cumulative_net_return: float
    max_drawdown: float
    turnover: float
    invested_fraction: float
    training_start: Hashable | None = None
    training_end: Hashable | None = None
    evaluation_start: Hashable | None = None
    purge_sessions: int | None = None


class StateDiscretizer:
    """Convert continuous decision-time and position features into a state."""

    def __init__(
        self,
        bin_edges: Mapping[str, tuple[float, ...]] | None = None,
    ) -> None:
        supplied = dict(bin_edges or {})
        unknown = sorted(set(supplied) - set(STATE_COMPONENTS))
        if unknown:
            raise ValueError(f"unknown state bin definitions: {unknown}")

        edges: dict[str, tuple[float, ...]] = {}
        for name in STATE_COMPONENTS:
            raw_edges = supplied.get(name, _DEFAULT_BIN_EDGES[name])
            numeric = tuple(_finite_float(value, f"{name} bin edge") for value in raw_edges)
            if tuple(sorted(numeric)) != numeric or len(set(numeric)) != len(numeric):
                raise ValueError(f"{name} bin edges must be strictly increasing")
            edges[name] = numeric
        self._bin_edges = edges

    @property
    def bin_edges(self) -> dict[str, tuple[float, ...]]:
        """Return a defensive copy suitable for experiment metadata."""
        return dict(self._bin_edges)

    def encode(
        self,
        context: Mapping[str, object],
        position: PolicyPosition,
        *,
        risk_budget_fraction: float,
    ) -> PolicyState:
        """Encode one state without reading return labels."""
        _validate_position(position, risk_budget_fraction)
        utilization = position.exposure / risk_budget_fraction
        values: dict[str, float | None] = {
            "current_exposure": utilization,
            "entry_drawdown": position.entry_drawdown,
            "holding_period": float(position.holding_period),
            "previous_action": position.previous_action_fraction,
        }
        for feature in CONTEXT_FEATURES:
            values[feature] = _context_float(context, feature)

        correlation = context.get("portfolio_correlation")
        if correlation is None or pd.isna(correlation):
            values["portfolio_correlation"] = None
        else:
            correlation_value = _finite_float(
                correlation,
                "portfolio_correlation",
            )
            if correlation_value < -1.0 or correlation_value > 1.0:
                raise ValueError("portfolio_correlation must be between -1 and 1")
            values["portfolio_correlation"] = correlation_value

        event_flag = values["event_flag"]
        if event_flag not in (0.0, 1.0):
            raise ValueError("event_flag must be 0 or 1")
        if not -1.0 <= float(values["event_score"]) <= 1.0:
            raise ValueError("event_score must be between -1 and 1")
        if not 0.0 <= float(values["event_confidence"]) <= 1.0:
            raise ValueError("event_confidence must be between 0 and 1")
        if values["asset_volatility"] is not None and values["asset_volatility"] < 0.0:
            raise ValueError("asset_volatility cannot be negative")
        if values["vix_level"] is not None and values["vix_level"] < 0.0:
            raise ValueError("vix_level cannot be negative")
        if not 0.0 <= float(values["sector_breadth"]) <= 1.0:
            raise ValueError("sector_breadth must be between 0 and 1")
        if not 0.0 <= float(values["forecast_probability_up"]) <= 1.0:
            raise ValueError("forecast_probability_up must be between 0 and 1")
        if not 0.0 <= float(values["forecast_model_agreement"]) <= 1.0:
            raise ValueError("forecast_model_agreement must be between 0 and 1")
        if float(values["forecast_uncertainty"]) < 0.0:
            raise ValueError("forecast_uncertainty cannot be negative")
        if (
            float(values["forecast_lower_bound"])
            > float(values["forecast_return"]) + _TOLERANCE
        ):
            raise ValueError("forecast_lower_bound cannot exceed forecast_return")

        return tuple(
            _MISSING_BIN
            if values[name] is None
            else int(np.digitize(values[name], self._bin_edges[name], right=False))
            for name in STATE_COMPONENTS
        )


class ShadowTargetWeightPolicy:
    """Long-only tabular target-weight policy that never submits orders."""

    def __init__(
        self,
        config: ShadowPolicyConfig | None = None,
        *,
        discretizer: StateDiscretizer | None = None,
    ) -> None:
        self.config = config or ShadowPolicyConfig()
        self.discretizer = discretizer or StateDiscretizer()
        self._q_table: dict[PolicyState, np.ndarray] = {}
        self._state_visits: dict[PolicyState, int] = {}
        self._action_visits: dict[tuple[PolicyState, int], int] = {}
        self._fitted = False
        self._frozen = True
        self.training_start: Hashable | None = None
        self.training_end: Hashable | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether a chronological training pass has completed."""
        return self._fitted

    @property
    def is_frozen(self) -> bool:
        """Whether Q-values are protected from evaluation-time updates."""
        return self._frozen

    def q_table_snapshot(self) -> dict[PolicyState, tuple[float, ...]]:
        """Return immutable numeric values for persistence or audit comparison."""
        return {
            state: tuple(float(value) for value in values)
            for state, values in self._q_table.items()
        }

    def visit_snapshot(
        self,
    ) -> tuple[dict[PolicyState, int], dict[tuple[PolicyState, int], int]]:
        """Return defensive copies of state and state-action visit counts."""
        return dict(self._state_visits), dict(self._action_visits)

    def fit(self, training_frame: pd.DataFrame) -> "ShadowTargetWeightPolicy":
        """Train in timestamp order and freeze the resulting policy.

        Each row must contain decision-time context plus matured one-session
        forward asset and benchmark returns.  Training always starts from a
        clean Q-table and a newly seeded random generator.
        """
        frame = _validate_policy_frame(training_frame)
        self._q_table = {}
        self._state_visits = {}
        self._action_visits = {}
        self._fitted = False
        self._frozen = False
        self.training_start = frame.index[0]
        self.training_end = frame.index[-1]
        rng = np.random.default_rng(self.config.random_seed)
        state_support: dict[PolicyState, set[Hashable]] = {}
        action_support: dict[tuple[PolicyState, int], set[Hashable]] = {}

        for _ in range(self.config.training_epochs):
            episode = _EpisodeState()
            for row_number in range(len(frame)):
                context = frame.iloc[row_number]
                observation_id = frame.index[row_number]
                position = episode.public_position
                state = self.discretizer.encode(
                    context,
                    position,
                    risk_budget_fraction=self.config.risk_budget_fraction,
                )
                action_index = self._training_action(state, rng)
                action_fraction = ACTION_FRACTIONS[action_index]
                transition = _transition(
                    episode,
                    context,
                    action_fraction=action_fraction,
                    config=self.config,
                )

                terminal = row_number == len(frame) - 1
                if terminal:
                    next_best = 0.0
                else:
                    next_context = frame.iloc[row_number + 1]
                    next_state = self.discretizer.encode(
                        next_context,
                        transition.episode.public_position,
                        risk_budget_fraction=self.config.risk_budget_fraction,
                    )
                    next_values = self._q_table.get(
                        next_state,
                        np.zeros(len(ACTION_FRACTIONS), dtype=float),
                    )
                    next_best = float(np.max(next_values))

                values = self._q_table.setdefault(
                    state,
                    np.zeros(len(ACTION_FRACTIONS), dtype=float),
                )
                old_value = float(values[action_index])
                target = transition.reward.reward
                if not terminal:
                    target += self.config.discount_factor * next_best
                values[action_index] = old_value + self.config.learning_rate * (
                    target - old_value
                )
                visit_key = (state, action_index)
                state_support.setdefault(state, set()).add(observation_id)
                action_support.setdefault(visit_key, set()).add(observation_id)
                episode = transition.episode

        # Reliability support counts unique market observations, not repeated
        # optimization epochs.  Multiple passes over one date are not
        # independent evidence for a state or action.
        self._state_visits = {
            state: len(observations)
            for state, observations in state_support.items()
        }
        self._action_visits = {
            key: len(observations)
            for key, observations in action_support.items()
        }
        self._fitted = True
        self._frozen = True
        return self

    def decide(
        self,
        context: Mapping[str, object],
        position: PolicyPosition | None = None,
    ) -> ShadowDecision:
        """Select a frozen target or abstain to zero for weak state evidence."""
        if not self._fitted or not self._frozen:
            raise RuntimeError("policy must be fitted and frozen before decisions")
        position = position or PolicyPosition()
        state = self.discretizer.encode(
            context,
            position,
            risk_budget_fraction=self.config.risk_budget_fraction,
        )
        values = self._q_table.get(state)
        state_visits = self._state_visits.get(state, 0)
        if values is None:
            return self._abstention(state, "unseen state")
        if state_visits < self.config.min_state_visits:
            return self._abstention(
                state,
                f"state visits {state_visits} below {self.config.min_state_visits}",
            )

        maximum = float(np.max(values))
        best = np.flatnonzero(
            np.isclose(
                values,
                maximum,
                rtol=0.0,
                atol=self.config.tie_tolerance,
            )
        )
        if len(best) != 1:
            return self._abstention(state, "top Q-values are tied")

        action_index = int(best[0])
        action_visits = self._action_visits.get((state, action_index), 0)
        if action_visits < self.config.min_action_visits:
            return self._abstention(
                state,
                f"action visits {action_visits} below {self.config.min_action_visits}",
            )

        action_fraction = ACTION_FRACTIONS[action_index]
        return ShadowDecision(
            action_fraction=action_fraction,
            target_exposure=action_fraction * self.config.risk_budget_fraction,
            state=state,
            state_visits=state_visits,
            action_visits=action_visits,
            q_values=tuple(float(value) for value in values),
            abstained=False,
            reason=(
                "cash target selected"
                if action_fraction == 0.0
                else "unique visited Q-value maximum"
            ),
        )

    def evaluate(self, held_out_frame: pd.DataFrame) -> ShadowEvaluation:
        """Evaluate chronologically without changing Q-values or visit counts."""
        if not self._fitted or not self._frozen:
            raise RuntimeError("policy must be fitted and frozen before evaluation")
        frame = _validate_policy_frame(held_out_frame)
        before_q = self.q_table_snapshot()
        before_visits = self.visit_snapshot()
        evaluation = self._evaluate_frozen(frame)
        if before_q != self.q_table_snapshot() or before_visits != self.visit_snapshot():
            raise RuntimeError("frozen held-out evaluation modified policy state")
        return evaluation

    def fit_purged_holdout(
        self,
        frame: pd.DataFrame,
        *,
        train_end: Hashable,
        test_start: Hashable,
        purge_sessions: int | None = None,
    ) -> ShadowEvaluation:
        """Fit before a purge gap, then run one frozen held-out evaluation."""
        validated = _validate_policy_frame(frame)
        train_position = _unique_index_position(validated.index, train_end, "train_end")
        test_position = _unique_index_position(validated.index, test_start, "test_start")
        actual_purge = test_position - train_position - 1
        required_purge = (
            self.config.minimum_purge_sessions
            if purge_sessions is None
            else purge_sessions
        )
        if (
            not isinstance(required_purge, (int, np.integer))
            or isinstance(required_purge, (bool, np.bool_))
        ):
            raise ValueError("purge_sessions must be an integer")
        required_purge = int(required_purge)
        if required_purge < self.config.minimum_purge_sessions:
            raise ValueError(
                "purge_sessions cannot be below configured minimum "
                f"{self.config.minimum_purge_sessions}"
            )
        if actual_purge < required_purge:
            raise ValueError(
                f"held-out split has {actual_purge} purged sessions; "
                f"at least {required_purge} are required"
            )
        training = validated.iloc[: train_position + 1]
        held_out = validated.iloc[test_position:]
        if held_out.empty:
            raise ValueError("held-out evaluation frame cannot be empty")

        self.fit(training)
        result = self.evaluate(held_out)
        return ShadowEvaluation(
            decisions=result.decisions,
            cumulative_reward=result.cumulative_reward,
            cumulative_net_return=result.cumulative_net_return,
            max_drawdown=result.max_drawdown,
            turnover=result.turnover,
            invested_fraction=result.invested_fraction,
            training_start=training.index[0],
            training_end=training.index[-1],
            evaluation_start=held_out.index[0],
            purge_sessions=actual_purge,
        )

    def _training_action(
        self,
        state: PolicyState,
        rng: np.random.Generator,
    ) -> int:
        if rng.random() < self.config.exploration_rate:
            return int(rng.integers(0, len(ACTION_FRACTIONS)))
        values = self._q_table.get(
            state,
            np.zeros(len(ACTION_FRACTIONS), dtype=float),
        )
        best = np.flatnonzero(
            np.isclose(values, np.max(values), rtol=0.0, atol=self.config.tie_tolerance)
        )
        return int(rng.choice(best))

    def _abstention(self, state: PolicyState, reason: str) -> ShadowDecision:
        values = self._q_table.get(
            state,
            np.zeros(len(ACTION_FRACTIONS), dtype=float),
        )
        state_visits = self._state_visits.get(state, 0)
        cash_visits = self._action_visits.get((state, 0), 0)
        return ShadowDecision(
            action_fraction=0.0,
            target_exposure=0.0,
            state=state,
            state_visits=state_visits,
            action_visits=cash_visits,
            q_values=tuple(float(value) for value in values),
            abstained=True,
            reason=reason,
        )

    def _evaluate_frozen(self, frame: pd.DataFrame) -> ShadowEvaluation:
        episode = _EpisodeState()
        records: list[dict[str, object]] = []
        cumulative_reward = 0.0
        total_turnover = 0.0
        invested = 0

        for row_number in range(len(frame)):
            context = frame.iloc[row_number]
            decision = self.decide(context, episode.public_position)
            transition = _transition(
                episode,
                context,
                action_fraction=decision.action_fraction,
                config=self.config,
            )
            reward = transition.reward
            turnover = abs(transition.episode.exposure - episode.exposure)
            cumulative_reward += reward.reward
            total_turnover += turnover
            if decision.target_exposure > _TOLERANCE:
                invested += 1
            records.append(
                {
                    "action_fraction": decision.action_fraction,
                    "target_exposure": decision.target_exposure,
                    "abstained": decision.abstained,
                    "reason": decision.reason,
                    "state_visits": decision.state_visits,
                    "action_visits": decision.action_visits,
                    "reward": reward.reward,
                    "active_return": reward.active_return,
                    "turnover_cost": reward.turnover_cost,
                    "drawdown_cost": reward.drawdown_cost,
                    "portfolio_risk_cost": reward.portfolio_risk_cost,
                    "portfolio_drawdown": transition.episode.portfolio_drawdown,
                    "entry_drawdown": transition.episode.entry_drawdown,
                    "holding_period": transition.episode.holding_period,
                }
            )
            episode = transition.episode

        decisions = pd.DataFrame(records, index=frame.index)
        return ShadowEvaluation(
            decisions=decisions,
            cumulative_reward=float(cumulative_reward),
            cumulative_net_return=float(episode.equity - 1.0),
            max_drawdown=float(episode.max_portfolio_drawdown),
            turnover=float(total_turnover),
            invested_fraction=float(invested / len(frame)),
        )


def execution_aligned_reward(
    *,
    exposure: float,
    previous_exposure: float,
    asset_return: float,
    benchmark_return: float,
    drawdown_increment: float,
    portfolio_risk: float,
    transaction_cost_bps: float,
    drawdown_penalty: float,
    portfolio_risk_penalty: float,
) -> RewardBreakdown:
    """Calculate the daily reward used by the shadow policy.

    ``portfolio_risk`` is a nonnegative daily risk estimate.  The penalty is
    proportional to current exposure.  Trading cost is charged only when the
    target exposure changes.
    """
    exposure_value = _fraction_value("exposure", exposure)
    previous_value = _fraction_value("previous_exposure", previous_exposure)
    asset_value = _finite_float(asset_return, "asset_return")
    benchmark_value = _finite_float(benchmark_return, "benchmark_return")
    drawdown_value = _nonnegative_value(
        "drawdown_increment",
        drawdown_increment,
    )
    risk_value = _nonnegative_value("portfolio_risk", portfolio_risk)
    cost_bps = _nonnegative_value("transaction_cost_bps", transaction_cost_bps)
    drawdown_coefficient = _nonnegative_value("drawdown_penalty", drawdown_penalty)
    risk_coefficient = _nonnegative_value(
        "portfolio_risk_penalty",
        portfolio_risk_penalty,
    )

    active_return = exposure_value * (asset_value - benchmark_value)
    turnover_cost = (
        cost_bps / 10_000.0 * abs(exposure_value - previous_value)
    )
    drawdown_cost = drawdown_coefficient * drawdown_value
    portfolio_risk_cost = risk_coefficient * exposure_value * risk_value
    reward = active_return - turnover_cost - drawdown_cost - portfolio_risk_cost
    return RewardBreakdown(
        reward=float(reward),
        active_return=float(active_return),
        turnover_cost=float(turnover_cost),
        drawdown_cost=float(drawdown_cost),
        portfolio_risk_cost=float(portfolio_risk_cost),
    )


@dataclass
class _EpisodeState:
    exposure: float = 0.0
    previous_action_fraction: float = 0.0
    entry_asset_index: float | None = None
    entry_drawdown: float = 0.0
    holding_period: int = 0
    asset_index: float = 1.0
    equity: float = 1.0
    equity_peak: float = 1.0
    portfolio_drawdown: float = 0.0
    max_portfolio_drawdown: float = 0.0

    @property
    def public_position(self) -> PolicyPosition:
        return PolicyPosition(
            exposure=self.exposure,
            entry_drawdown=self.entry_drawdown,
            holding_period=self.holding_period,
            previous_action_fraction=self.previous_action_fraction,
        )


@dataclass(frozen=True)
class _Transition:
    episode: _EpisodeState
    reward: RewardBreakdown


def _transition(
    previous: _EpisodeState,
    row: Mapping[str, object],
    *,
    action_fraction: float,
    config: ShadowPolicyConfig,
) -> _Transition:
    if action_fraction not in ACTION_FRACTIONS:
        raise ValueError(f"unsupported action fraction {action_fraction}")
    target_exposure = action_fraction * config.risk_budget_fraction
    asset_return = _context_float(row, FORWARD_ASSET_RETURN)
    benchmark_return = _context_float(row, FORWARD_BENCHMARK_RETURN)
    if asset_return < -1.0 or benchmark_return < -1.0:
        raise ValueError("forward returns cannot be below -100%")

    turnover = abs(target_exposure - previous.exposure)
    transaction_cost = config.transaction_cost_bps / 10_000.0 * turnover
    net_portfolio_return = target_exposure * asset_return - transaction_cost
    next_equity = previous.equity * max(1.0 + net_portfolio_return, 0.0)
    next_peak = max(previous.equity_peak, next_equity)
    next_drawdown = 0.0 if next_peak <= 0.0 else 1.0 - next_equity / next_peak
    drawdown_increment = max(next_drawdown - previous.portfolio_drawdown, 0.0)

    risk_measure = _portfolio_risk_measure(row)
    reward = execution_aligned_reward(
        exposure=target_exposure,
        previous_exposure=previous.exposure,
        asset_return=asset_return,
        benchmark_return=benchmark_return,
        drawdown_increment=drawdown_increment,
        portfolio_risk=risk_measure,
        transaction_cost_bps=config.transaction_cost_bps,
        drawdown_penalty=config.drawdown_penalty,
        portfolio_risk_penalty=config.portfolio_risk_penalty,
    )

    next_asset_index = previous.asset_index * max(1.0 + asset_return, 0.0)
    if target_exposure <= _TOLERANCE:
        entry_index = None
        entry_drawdown = 0.0
        holding_period = 0
    else:
        opened = previous.exposure <= _TOLERANCE
        entry_index = previous.asset_index if opened else previous.entry_asset_index
        if entry_index is None or entry_index <= 0.0:
            entry_drawdown = -1.0
        else:
            entry_return = next_asset_index / entry_index - 1.0
            entry_drawdown = min(entry_return, 0.0)
        holding_period = 1 if opened else previous.holding_period + 1

    return _Transition(
        episode=_EpisodeState(
            exposure=target_exposure,
            previous_action_fraction=action_fraction,
            entry_asset_index=entry_index,
            entry_drawdown=float(entry_drawdown),
            holding_period=holding_period,
            asset_index=float(next_asset_index),
            equity=float(next_equity),
            equity_peak=float(next_peak),
            portfolio_drawdown=float(next_drawdown),
            max_portfolio_drawdown=float(
                max(previous.max_portfolio_drawdown, next_drawdown)
            ),
        ),
        reward=reward,
    )


def _portfolio_risk_measure(row: Mapping[str, object]) -> float:
    explicit = row.get(OPTIONAL_PORTFOLIO_RISK)
    if explicit is not None and not pd.isna(explicit):
        return _nonnegative_value(OPTIONAL_PORTFOLIO_RISK, explicit)

    annualized_volatility = _nonnegative_value(
        "asset_volatility",
        _context_float(row, "asset_volatility"),
    )
    correlation = row.get("portfolio_correlation")
    positive_correlation = 0.0
    if correlation is not None and not pd.isna(correlation):
        correlation_value = _finite_float(correlation, "portfolio_correlation")
        if correlation_value < -1.0 or correlation_value > 1.0:
            raise ValueError("portfolio_correlation must be between -1 and 1")
        positive_correlation = max(correlation_value, 0.0)
    return float(
        annualized_volatility / np.sqrt(252.0) * (1.0 + positive_correlation)
    )


def _validate_policy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("policy data must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("policy data cannot be empty")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("policy data index must be unique and chronological")

    required = set(CONTEXT_FEATURES) | {
        FORWARD_ASSET_RETURN,
        FORWARD_BENCHMARK_RETURN,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"policy data is missing required columns: {missing}")

    numeric_columns = sorted(required)
    for column in numeric_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if invalid.any():
            bad_indices = list(frame.index[invalid][:5])
            raise ValueError(
                f"{column} must be finite for matured rows; invalid at {bad_indices}"
            )

    if (pd.to_numeric(frame["asset_volatility"]) < 0.0).any():
        raise ValueError("asset_volatility cannot be negative")
    if (pd.to_numeric(frame["vix_level"]) < 0.0).any():
        raise ValueError("vix_level cannot be negative")
    event_values = set(pd.to_numeric(frame["event_flag"]).astype(float).unique())
    if not event_values.issubset({0.0, 1.0}):
        raise ValueError("event_flag must contain only 0 or 1")
    event_score = pd.to_numeric(frame["event_score"])
    if ((event_score < -1.0) | (event_score > 1.0)).any():
        raise ValueError("event_score must be between -1 and 1")
    event_confidence = pd.to_numeric(frame["event_confidence"])
    if ((event_confidence < 0.0) | (event_confidence > 1.0)).any():
        raise ValueError("event_confidence must be between 0 and 1")
    breadth = pd.to_numeric(frame["sector_breadth"])
    if ((breadth < 0.0) | (breadth > 1.0)).any():
        raise ValueError("sector_breadth must be between 0 and 1")
    probability_up = pd.to_numeric(frame["forecast_probability_up"])
    if ((probability_up < 0.0) | (probability_up > 1.0)).any():
        raise ValueError("forecast_probability_up must be between 0 and 1")
    agreement = pd.to_numeric(frame["forecast_model_agreement"])
    if ((agreement < 0.0) | (agreement > 1.0)).any():
        raise ValueError("forecast_model_agreement must be between 0 and 1")
    uncertainty = pd.to_numeric(frame["forecast_uncertainty"])
    if (uncertainty < 0.0).any():
        raise ValueError("forecast_uncertainty cannot be negative")
    lower_bound = pd.to_numeric(frame["forecast_lower_bound"])
    forecast_return = pd.to_numeric(frame["forecast_return"])
    if (lower_bound > forecast_return + _TOLERANCE).any():
        raise ValueError("forecast_lower_bound cannot exceed forecast_return")
    for column in (FORWARD_ASSET_RETURN, FORWARD_BENCHMARK_RETURN):
        if (pd.to_numeric(frame[column]) < -1.0).any():
            raise ValueError(f"{column} cannot be below -100%")

    if "portfolio_correlation" in frame.columns:
        correlation = pd.to_numeric(frame["portfolio_correlation"], errors="coerce")
        finite = correlation.dropna()
        if ((finite < -1.0) | (finite > 1.0)).any():
            raise ValueError("portfolio_correlation must be between -1 and 1")
        original_missing = frame["portfolio_correlation"].isna()
        coerced_missing = correlation.isna()
        if (coerced_missing & ~original_missing).any():
            raise ValueError("portfolio_correlation must be numeric when supplied")
    if OPTIONAL_PORTFOLIO_RISK in frame.columns:
        risk = pd.to_numeric(frame[OPTIONAL_PORTFOLIO_RISK], errors="coerce")
        if (~np.isfinite(risk.to_numpy(dtype=float))).any() or (risk < 0.0).any():
            raise ValueError(f"{OPTIONAL_PORTFOLIO_RISK} must be finite and nonnegative")
    return frame


def _validate_position(position: PolicyPosition, risk_budget_fraction: float) -> None:
    exposure = _finite_float(position.exposure, "position.exposure")
    if exposure < 0.0 or exposure > risk_budget_fraction + _TOLERANCE:
        raise ValueError("position.exposure must be within the configured risk budget")
    drawdown = _finite_float(position.entry_drawdown, "position.entry_drawdown")
    if drawdown < -1.0 or drawdown > 0.0:
        raise ValueError("position.entry_drawdown must be between -1 and 0")
    if (
        not isinstance(position.holding_period, (int, np.integer))
        or position.holding_period < 0
    ):
        raise ValueError("position.holding_period must be a nonnegative integer")
    if position.previous_action_fraction not in ACTION_FRACTIONS:
        raise ValueError(
            f"position.previous_action_fraction must be one of {ACTION_FRACTIONS}"
        )


def _unique_index_position(
    index: pd.Index,
    label: Hashable,
    argument_name: str,
) -> int:
    try:
        location = index.get_loc(label)
    except KeyError as exc:
        raise ValueError(f"{argument_name} {label!r} is not in the data index") from exc
    if not isinstance(location, (int, np.integer)):
        raise ValueError(f"{argument_name} must identify exactly one row")
    return int(location)


def _context_float(context: Mapping[str, object], name: str) -> float:
    if name not in context:
        raise ValueError(f"missing required context feature {name!r}")
    return _finite_float(context[name], name)


def _fraction_value(name: str, value: object) -> float:
    number = _finite_float(value, name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _nonnegative_value(name: str, value: object) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return number


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_fraction(name: str, value: object, allow_zero: bool = True) -> None:
    number = _finite_float(value, name)
    minimum_valid = number >= 0.0 if allow_zero else number > 0.0
    if not minimum_valid or number > 1.0:
        bracket = "[" if allow_zero else "("
        raise ValueError(f"{name} must be in {bracket}0, 1]")


def _require_nonnegative(name: str, value: object) -> None:
    _nonnegative_value(name, value)
