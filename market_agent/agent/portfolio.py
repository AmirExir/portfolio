"""Portfolio-level allocation constraints for long-only policy targets.

The forecasting and symbol-level policy layers may propose independent target
weights, but orders should not be derived from those weights until they have
been constrained as one portfolio.  This module performs that final,
deterministic risk-allocation step.  It deliberately does not fetch market
data, infer classifications, or submit orders.

All weights, drawdowns, and volatility values use decimal fractions:
``0.05`` means 5%.  Covariance inputs must be annualized covariance of decimal
returns, with symbols on both axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PortfolioConstraints:
    """Hard limits applied to a set of long-only target weights.

    The defaults retain 15% cash, cap a name at 5%, cap a sector at 20%, and
    cap a supplied correlated cluster at 15%.  ``max_turnover`` is the sum of
    absolute risky-asset weight changes for one rebalance.  For example,
    selling 5% of one stock and buying 5% of another consumes 10% turnover.

    ``max_annual_volatility`` is only enforceable when an annualized covariance
    matrix is passed to :func:`allocate_target_weights`.
    """

    gross_limit: float = 1.0
    cash_reserve: float = 0.15
    max_name_weight: float = 0.05
    max_sector_weight: float = 0.20
    max_cluster_weight: float = 0.15
    max_annual_volatility: float | None = 0.15
    max_turnover: float | None = 0.20
    drawdown_circuit_breaker: float = 0.10

    def __post_init__(self) -> None:
        """Validate limits at construction so allocation failures are explicit."""
        _require_range("gross_limit", self.gross_limit, lower=0.0, upper=1.0)
        _require_range("cash_reserve", self.cash_reserve, lower=0.10, upper=0.20)
        _require_range(
            "max_name_weight",
            self.max_name_weight,
            lower=0.03,
            upper=0.05,
        )
        _require_range(
            "max_sector_weight",
            self.max_sector_weight,
            lower=0.0,
            upper=0.20,
            lower_inclusive=False,
        )
        _require_range(
            "max_cluster_weight",
            self.max_cluster_weight,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        _require_range(
            "drawdown_circuit_breaker",
            self.drawdown_circuit_breaker,
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        if self.max_annual_volatility is not None:
            _require_range(
                "max_annual_volatility",
                self.max_annual_volatility,
                lower=0.0,
                upper=np.inf,
                lower_inclusive=False,
            )
        if self.max_turnover is not None:
            _require_range(
                "max_turnover",
                self.max_turnover,
                lower=0.0,
                upper=np.inf,
            )

    @property
    def investable_limit(self) -> float:
        """Maximum risky-asset exposure after retaining the cash reserve."""
        return min(self.gross_limit, 1.0 - self.cash_reserve)


@dataclass(frozen=True)
class PortfolioAllocation:
    """Auditable output from portfolio-level target allocation."""

    target_weights: dict[str, float]
    cash_weight: float
    gross_exposure: float
    sector_exposures: dict[str, float]
    cluster_exposures: dict[str, float]
    annualized_volatility: float | None
    turnover: float
    circuit_breaker_triggered: bool
    turnover_cap_overridden: bool
    binding_constraints: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def allocate_target_weights(
    proposed_weights: Mapping[str, float],
    *,
    sectors: Mapping[str, str] | None = None,
    correlation_clusters: Mapping[str, str] | None = None,
    current_weights: Mapping[str, float] | None = None,
    annual_covariance: pd.DataFrame | None = None,
    current_drawdown: float = 0.0,
    constraints: PortfolioConstraints | None = None,
) -> PortfolioAllocation:
    """Constrain independent long-only targets as one portfolio.

    Parameters
    ----------
    proposed_weights:
        Desired symbol weights as fractions of equity.  Omitted current
        holdings are interpreted as a request to exit.
    sectors:
        Optional symbol-to-sector mapping.  The sector cap is enforced for
        mapped symbols.  Missing symbols are isolated rather than guessed.
    correlation_clusters:
        Optional symbol-to-cluster mapping created by an upstream, documented
        clustering process.  The cluster cap is enforced for mapped symbols.
    current_weights:
        Current long-only risky-asset weights.  These are required for a
        meaningful turnover calculation.
    annual_covariance:
        Annualized covariance matrix of decimal returns.  Symbols must appear
        on both axes.  If omitted, the volatility cap cannot be evaluated and
        a warning is returned.
    current_drawdown:
        Current portfolio drawdown as a signed return.  For example, ``-0.12``
        is a 12% drawdown.  At or below the configured threshold, all risky
        targets are set to zero.  The safety exit overrides turnover limits.
    constraints:
        Constraint configuration.  Defaults to :class:`PortfolioConstraints`.

    Returns
    -------
    PortfolioAllocation
        Final target weights and diagnostics.  The allocator never scales a
        proposed target upward to consume unused risk capacity.
    """
    config = constraints or PortfolioConstraints()
    proposed = _normalize_weights(proposed_weights, "proposed_weights")
    current = _normalize_weights(current_weights or {}, "current_weights")
    drawdown = _finite_float(current_drawdown, "current_drawdown")
    if drawdown < -1.0 or drawdown > 0.0:
        raise ValueError("current_drawdown must be a signed fraction between -1 and 0")

    symbols = tuple(sorted(set(proposed) | set(current)))
    requested = {symbol: proposed.get(symbol, 0.0) for symbol in symbols}
    current_full = {symbol: current.get(symbol, 0.0) for symbol in symbols}
    warnings: list[str] = []
    binding: list[str] = []

    sector_groups = _normalize_groups(
        sectors,
        symbols,
        label="sector",
        warnings=warnings,
    )
    cluster_groups = _normalize_groups(
        correlation_clusters,
        symbols,
        label="correlation cluster",
        warnings=warnings,
    )
    covariance = _validate_covariance(annual_covariance, symbols)

    circuit_breaker_triggered = (
        drawdown <= -abs(config.drawdown_circuit_breaker)
    )
    turnover_cap_overridden = False

    if circuit_breaker_triggered:
        targets = {symbol: 0.0 for symbol in symbols}
        binding.append("drawdown_circuit_breaker")
        turnover = _turnover(targets, current_full)
        if config.max_turnover is not None and turnover > config.max_turnover + _TOLERANCE:
            turnover_cap_overridden = True
            warnings.append(
                "Drawdown circuit breaker overrides max_turnover so the "
                "portfolio can move to cash."
            )
    else:
        targets = _apply_hard_caps(
            requested,
            sector_groups=sector_groups,
            cluster_groups=cluster_groups,
            covariance=covariance,
            constraints=config,
            binding=binding,
        )
        requested_turnover = _turnover(targets, current_full)
        if (
            config.max_turnover is not None
            and requested_turnover > config.max_turnover + _TOLERANCE
        ):
            scale = config.max_turnover / requested_turnover
            targets = {
                symbol: current_full[symbol]
                + scale * (targets[symbol] - current_full[symbol])
                for symbol in symbols
            }
            binding.append("turnover")

            # Interpolation preserves convex limits only when the current
            # portfolio is already compliant.  Reapply hard caps so unsafe
            # existing exposure cannot leak through a turnover constraint.
            targets = _apply_hard_caps(
                targets,
                sector_groups=sector_groups,
                cluster_groups=cluster_groups,
                covariance=covariance,
                constraints=config,
                binding=binding,
            )

        turnover = _turnover(targets, current_full)
        if config.max_turnover is not None and turnover > config.max_turnover + _TOLERANCE:
            turnover_cap_overridden = True
            warnings.append(
                "Hard exposure or volatility limits override max_turnover "
                "because the current portfolio is outside a configured limit."
            )

    targets = {
        symbol: 0.0 if abs(weight) <= _TOLERANCE else float(weight)
        for symbol, weight in targets.items()
    }
    gross = float(sum(targets.values()))
    annualized_volatility = _portfolio_volatility(targets, covariance)
    if config.max_annual_volatility is not None and covariance is None and symbols:
        warnings.append(
            "Annual volatility cap was not evaluated because no annualized "
            "covariance matrix was supplied."
        )

    return PortfolioAllocation(
        target_weights=targets,
        cash_weight=float(max(1.0 - gross, 0.0)),
        gross_exposure=gross,
        sector_exposures=_group_exposures(targets, sector_groups),
        cluster_exposures=_group_exposures(targets, cluster_groups),
        annualized_volatility=annualized_volatility,
        turnover=float(turnover),
        circuit_breaker_triggered=circuit_breaker_triggered,
        turnover_cap_overridden=turnover_cap_overridden,
        binding_constraints=tuple(dict.fromkeys(binding)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _apply_hard_caps(
    weights: Mapping[str, float],
    *,
    sector_groups: Mapping[str, str] | None,
    cluster_groups: Mapping[str, str] | None,
    covariance: pd.DataFrame | None,
    constraints: PortfolioConstraints,
    binding: list[str],
) -> dict[str, float]:
    """Apply only reductions, preserving the relative weights within a group."""
    constrained = {
        symbol: min(float(weight), constraints.max_name_weight)
        for symbol, weight in weights.items()
    }
    if any(
        constrained[symbol] < float(weights[symbol]) - _TOLERANCE
        for symbol in constrained
    ):
        binding.append("name")

    _scale_groups(
        constrained,
        sector_groups,
        limit=constraints.max_sector_weight,
        label="sector",
        binding=binding,
    )
    _scale_groups(
        constrained,
        cluster_groups,
        limit=constraints.max_cluster_weight,
        label="cluster",
        binding=binding,
    )

    gross = float(sum(constrained.values()))
    if gross > constraints.investable_limit + _TOLERANCE:
        _scale_all(constrained, constraints.investable_limit / gross)
        binding.append("gross_and_cash_reserve")

    volatility = _portfolio_volatility(constrained, covariance)
    if (
        volatility is not None
        and constraints.max_annual_volatility is not None
        and volatility > constraints.max_annual_volatility + _TOLERANCE
    ):
        _scale_all(constrained, constraints.max_annual_volatility / volatility)
        binding.append("annual_volatility")

    return constrained


def _scale_groups(
    weights: dict[str, float],
    groups: Mapping[str, str] | None,
    *,
    limit: float,
    label: str,
    binding: list[str],
) -> None:
    if groups is None:
        return

    members: dict[str, list[str]] = {}
    for symbol, group in groups.items():
        members.setdefault(group, []).append(symbol)

    for group in sorted(members):
        group_symbols = members[group]
        exposure = float(sum(weights[symbol] for symbol in group_symbols))
        if exposure <= limit + _TOLERANCE:
            continue
        scale = limit / exposure
        for symbol in group_symbols:
            weights[symbol] *= scale
        binding.append(f"{label}:{group}")


def _scale_all(weights: dict[str, float], scale: float) -> None:
    for symbol in weights:
        weights[symbol] *= scale


def _normalize_weights(
    weights: Mapping[str, float],
    argument_name: str,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_symbol, raw_weight in weights.items():
        symbol = _normalize_symbol(raw_symbol)
        if symbol in normalized:
            raise ValueError(
                f"{argument_name} contains duplicate symbol {symbol!r} "
                "after normalization"
            )
        weight = _finite_float(raw_weight, f"{argument_name}[{symbol!r}]")
        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                f"{argument_name}[{symbol!r}] must be between 0 and 1"
            )
        normalized[symbol] = weight
    return normalized


def _normalize_groups(
    groups: Mapping[str, str] | None,
    symbols: tuple[str, ...],
    *,
    label: str,
    warnings: list[str],
) -> dict[str, str] | None:
    if groups is None:
        if symbols:
            warnings.append(
                f"No {label} mapping supplied; the {label} limit was not evaluated."
            )
        return None

    normalized_input: dict[str, str] = {}
    for raw_symbol, raw_group in groups.items():
        symbol = _normalize_symbol(raw_symbol)
        if symbol in normalized_input:
            raise ValueError(
                f"{label} mapping contains duplicate symbol {symbol!r} "
                "after normalization"
            )
        group = str(raw_group).strip()
        if group:
            normalized_input[symbol] = group

    normalized: dict[str, str] = {}
    missing: list[str] = []
    for symbol in symbols:
        group = normalized_input.get(symbol)
        if group is None:
            # Do not guess that unrelated unclassified names share a sector or
            # correlation cluster.  Isolate each and surface the data gap.
            group = f"Unclassified:{symbol}"
            missing.append(symbol)
        normalized[symbol] = group

    if missing:
        warnings.append(
            f"Missing {label} classification for: {', '.join(missing)}. "
            "Each missing symbol was isolated; aggregate exposure for those "
            "symbols cannot be verified."
        )
    return normalized


def _normalize_symbol(raw_symbol: object) -> str:
    symbol = str(raw_symbol).strip().upper()
    if not symbol:
        raise ValueError("portfolio symbols must be non-empty")
    return symbol


def _validate_covariance(
    covariance: pd.DataFrame | None,
    symbols: tuple[str, ...],
) -> pd.DataFrame | None:
    if covariance is None:
        return None
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("annual_covariance must be a pandas DataFrame")
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise ValueError("annual_covariance axes must not contain duplicates")

    normalized = covariance.copy()
    normalized.index = [_normalize_symbol(symbol) for symbol in normalized.index]
    normalized.columns = [_normalize_symbol(symbol) for symbol in normalized.columns]
    if normalized.index.has_duplicates or normalized.columns.has_duplicates:
        raise ValueError(
            "annual_covariance axes contain duplicates after symbol normalization"
        )

    missing_rows = sorted(set(symbols) - set(normalized.index))
    missing_columns = sorted(set(symbols) - set(normalized.columns))
    if missing_rows or missing_columns:
        raise ValueError(
            "annual_covariance is missing symbols; "
            f"rows={missing_rows}, columns={missing_columns}"
        )

    normalized = normalized.reindex(index=symbols, columns=symbols)
    try:
        values = normalized.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("annual_covariance must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("annual_covariance must contain only finite values")
    if not np.allclose(values, values.T, rtol=1e-9, atol=1e-12):
        raise ValueError("annual_covariance must be symmetric")
    if values.size == 0:
        return pd.DataFrame(values, index=symbols, columns=symbols)
    diagonal = np.diag(values)
    if np.any(diagonal < -_TOLERANCE):
        raise ValueError("annual_covariance diagonal cannot be negative")
    eigenvalues = np.linalg.eigvalsh(values)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("annual_covariance must be positive semidefinite")
    return pd.DataFrame(values, index=symbols, columns=symbols)


def _portfolio_volatility(
    weights: Mapping[str, float],
    covariance: pd.DataFrame | None,
) -> float | None:
    if covariance is None:
        return None
    if not weights:
        return 0.0
    symbols = list(covariance.index)
    vector = np.asarray([weights.get(symbol, 0.0) for symbol in symbols], dtype=float)
    variance = float(vector @ covariance.to_numpy(dtype=float) @ vector)
    return float(np.sqrt(max(variance, 0.0)))


def _group_exposures(
    weights: Mapping[str, float],
    groups: Mapping[str, str] | None,
) -> dict[str, float]:
    if groups is None:
        return {}
    exposures: dict[str, float] = {}
    for symbol, group in groups.items():
        exposures[group] = exposures.get(group, 0.0) + float(weights[symbol])
    return {
        group: float(exposure)
        for group, exposure in sorted(exposures.items())
        if exposure > _TOLERANCE
    }


def _turnover(
    target_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
) -> float:
    symbols = set(target_weights) | set(current_weights)
    return float(
        sum(
            abs(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
            for symbol in symbols
        )
    )


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_range(
    name: str,
    value: float,
    *,
    lower: float,
    upper: float,
    lower_inclusive: bool = True,
) -> None:
    number = _finite_float(value, name)
    lower_valid = number >= lower if lower_inclusive else number > lower
    if not lower_valid or number > upper:
        lower_bracket = "[" if lower_inclusive else "("
        raise ValueError(
            f"{name} must be in {lower_bracket}{lower}, {upper}], got {number}"
        )
