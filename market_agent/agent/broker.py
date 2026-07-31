"""Fail-closed Alpaca broker boundary for live paper-trading execution.

All broker payloads are treated as untrusted external input.  Public mutation
functions either return a validated, broker-accepted result or raise
``BrokerError``; callers must not infer success from an HTTP status alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from typing import Any, Mapping

import requests
import streamlit as st


DEFAULT_TIMEOUT_SECONDS = 12
DEFAULT_QUOTE_MAX_AGE_SECONDS = 300
DEFAULT_SLIPPAGE_BPS = 50.0
OPEN_ORDER_STATUSES = frozenset(
    {
        "accepted",
        "accepted_for_bidding",
        "calculated",
        "held",
        "new",
        "partially_filled",
        "pending_cancel",
        "pending_new",
        "pending_replace",
        "stopped",
    }
)
ACCEPTED_ORDER_STATUSES = frozenset(
    {
        "accepted",
        "accepted_for_bidding",
        "calculated",
        "filled",
        "held",
        "new",
        "partially_filled",
        "pending_new",
        "pending_replace",
    }
)
TERMINAL_REJECTION_STATUSES = frozenset(
    {
        "canceled",
        "done_for_day",
        "expired",
        "rejected",
        "replaced",
        "suspended",
    }
)


class BrokerError(RuntimeError):
    """Base class for broker communication or validation failures."""


class BrokerResponseError(BrokerError):
    """Raised when Alpaca returns an invalid or error-shaped payload."""


class OrderRejectedError(BrokerError):
    """Raised when an order was not accepted by the broker."""


class CancellationError(BrokerError):
    """Raised when open-order cancellation cannot be confirmed."""


class LiquidationError(BrokerError):
    """Raised after every position was attempted but one or more closes failed."""

    def __init__(
        self,
        message: str,
        *,
        positions: Mapping[str, "BrokerPosition"],
        accepted_orders: tuple[dict[str, Any], ...],
        failures: Mapping[str, str],
        cancellations: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.positions = dict(positions)
        self.accepted_orders = accepted_orders
        self.failures = dict(failures)
        self.cancellations = dict(cancellations)


@dataclass(frozen=True)
class BrokerPosition:
    """Validated broker position without integer-quantity truncation."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float | None = None


@dataclass(frozen=True)
class BrokerQuote:
    """Validated current National Best Bid and Offer snapshot."""

    symbol: str
    bid_price: float
    ask_price: float
    timestamp: datetime

    @property
    def midpoint(self) -> float:
        """Return the quote midpoint."""
        return (self.bid_price + self.ask_price) / 2.0


@dataclass(frozen=True)
class BrokerPortfolioSnapshot:
    """Positions plus reserved risk from all currently open buy orders."""

    equity: float
    positions: dict[str, BrokerPosition]
    position_weights: dict[str, float]
    pending_buy_weights: dict[str, float]
    risk_weights: dict[str, float]
    open_orders: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BuyNotionalSizing:
    """Buy notional bounded by a verified ask and portfolio target capacity."""

    notional: float
    estimated_qty: float
    conservative_ask: float
    available_target_notional: float


@dataclass(frozen=True)
class LiquidationResult:
    """Accepted close-position orders for a portfolio circuit breaker."""

    position_symbols: tuple[str, ...]
    positions: Mapping[str, BrokerPosition]
    accepted_orders: tuple[dict[str, Any], ...]
    cancellations: Mapping[str, Any]


def _secret(name: str, default: Any = None) -> Any:
    """Read Streamlit secrets when configured, otherwise use env/default."""
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def _base_url() -> str:
    """Return an Alpaca trading API URL with exactly one ``/v2`` suffix."""
    endpoint = str(
        _secret("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
    ).rstrip("/")
    return endpoint if endpoint.endswith("/v2") else f"{endpoint}/v2"


def _data_base_url() -> str:
    """Return the Alpaca market-data API v2 root."""
    endpoint = str(
        _secret("ALPACA_DATA_ENDPOINT", "https://data.alpaca.markets")
    ).rstrip("/")
    return endpoint if endpoint.endswith("/v2") else f"{endpoint}/v2"


def _headers() -> dict[str, str]:
    key = _secret("ALPACA_KEY")
    secret = _secret("ALPACA_SECRET")
    if not key or not secret:
        raise BrokerError("Alpaca API keys are not configured")
    return {
        "APCA-API-KEY-ID": str(key),
        "APCA-API-SECRET-KEY": str(secret),
        "Content-Type": "application/json",
    }


def get_account() -> dict[str, Any]:
    """Fetch and validate the Alpaca account payload."""
    response = requests.get(
        f"{_base_url()}/account",
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    payload = _response_mapping(response, expected_statuses={200}, operation="get account")
    for field_name in ("equity", "cash", "buying_power"):
        _positive_or_zero(payload.get(field_name), f"account.{field_name}")
    return dict(payload)


def get_positions() -> dict[str, BrokerPosition]:
    """Return every broker position with fractional quantity and entry price."""
    response = requests.get(
        f"{_base_url()}/positions",
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    payload = _response_list(response, expected_statuses={200}, operation="get positions")
    positions: dict[str, BrokerPosition] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise BrokerResponseError(f"position {index} is not an object")
        symbol = _symbol(item.get("symbol"))
        qty = _finite(item.get("qty"), f"{symbol}.qty")
        market_value = _finite(item.get("market_value"), f"{symbol}.market_value")
        avg_entry_price = _positive(
            item.get("avg_entry_price"),
            f"{symbol}.avg_entry_price",
        )
        current_price_raw = item.get("current_price")
        current_price = (
            _positive(current_price_raw, f"{symbol}.current_price")
            if current_price_raw not in (None, "")
            else None
        )
        if abs(qty) <= 1e-12:
            continue
        positions[symbol] = BrokerPosition(
            symbol=symbol,
            qty=qty,
            market_value=market_value,
            avg_entry_price=avg_entry_price,
            current_price=current_price,
        )
    return positions


def get_open_orders(symbol: str | None = None) -> tuple[dict[str, Any], ...]:
    """Return validated open orders, optionally restricted to one symbol."""
    normalized_symbol = _symbol(symbol) if symbol is not None else None
    response = requests.get(
        f"{_base_url()}/orders",
        headers=_headers(),
        params={
            "status": "open",
            "limit": 500,
            "direction": "asc",
            "nested": "false",
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    payload = _response_list(response, expected_statuses={200}, operation="get open orders")
    orders: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise BrokerResponseError(f"open order {index} is not an object")
        order = dict(item)
        order_symbol = _symbol(order.get("symbol"))
        order_id = str(order.get("id", "")).strip()
        status = str(order.get("status", "")).strip().lower()
        side = str(order.get("side", "")).strip().lower()
        if not order_id:
            raise BrokerResponseError(f"open order {index} has no id")
        if side not in {"buy", "sell"}:
            raise BrokerResponseError(f"open order {order_id} has invalid side")
        if status not in OPEN_ORDER_STATUSES:
            raise BrokerResponseError(
                f"open order {order_id} has unexpected status {status!r}"
            )
        order["symbol"] = order_symbol
        order["id"] = order_id
        order["side"] = side
        order["status"] = status
        if normalized_symbol is None or order_symbol == normalized_symbol:
            orders.append(order)
    return tuple(orders)


def get_latest_quote(
    symbol: str,
    *,
    max_age_seconds: int = DEFAULT_QUOTE_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> BrokerQuote:
    """Fetch a positive, timestamped, sufficiently fresh Alpaca quote."""
    normalized_symbol = _symbol(symbol)
    if isinstance(max_age_seconds, bool) or int(max_age_seconds) <= 0:
        raise ValueError("max_age_seconds must be a positive integer")
    response = requests.get(
        f"{_data_base_url()}/stocks/{normalized_symbol}/quotes/latest",
        headers=_headers(),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    payload = _response_mapping(
        response,
        expected_statuses={200},
        operation=f"get quote for {normalized_symbol}",
    )
    quote_payload = payload.get("quote")
    if not isinstance(quote_payload, Mapping):
        raise BrokerResponseError("latest quote payload has no quote object")
    bid = _positive(
        quote_payload.get("bp", quote_payload.get("bid_price")),
        f"{normalized_symbol}.bid_price",
    )
    ask = _positive(
        quote_payload.get("ap", quote_payload.get("ask_price")),
        f"{normalized_symbol}.ask_price",
    )
    timestamp = _timestamp(
        quote_payload.get("t", quote_payload.get("timestamp")),
        f"{normalized_symbol}.quote_timestamp",
    )
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age_seconds = (current_time.astimezone(timezone.utc) - timestamp).total_seconds()
    if age_seconds < -5.0:
        raise BrokerResponseError("latest quote timestamp is in the future")
    if age_seconds > int(max_age_seconds):
        raise BrokerResponseError(
            f"latest quote is stale ({age_seconds:.0f}s old)"
        )
    if ask < bid:
        raise BrokerResponseError("latest quote ask is below bid")
    return BrokerQuote(
        symbol=normalized_symbol,
        bid_price=bid,
        ask_price=ask,
        timestamp=timestamp,
    )


def get_portfolio_snapshot(
    equity: float,
    *,
    pending_order_slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BrokerPortfolioSnapshot:
    """Return gross risk weights including all unfilled pending buy exposure."""
    validated_equity = _positive(equity, "equity")
    slippage_bps = _nonnegative(
        pending_order_slippage_bps,
        "pending_order_slippage_bps",
    )
    positions = get_positions()
    orders = get_open_orders()
    position_weights = {
        symbol: abs(position.market_value) / validated_equity
        for symbol, position in positions.items()
    }
    pending_buy_notional: dict[str, float] = {}
    quote_cache: dict[str, BrokerQuote] = {}
    for order in orders:
        if order["side"] != "buy":
            continue
        symbol = order["symbol"]
        reserve = _pending_buy_reserve(
            order,
            symbol=symbol,
            slippage_bps=slippage_bps,
            quote_cache=quote_cache,
        )
        pending_buy_notional[symbol] = (
            pending_buy_notional.get(symbol, 0.0) + reserve
        )
    pending_buy_weights = {
        symbol: notional / validated_equity
        for symbol, notional in pending_buy_notional.items()
        if notional > 0.0
    }
    risk_weights = dict(position_weights)
    for symbol, pending_weight in pending_buy_weights.items():
        risk_weights[symbol] = risk_weights.get(symbol, 0.0) + pending_weight
    return BrokerPortfolioSnapshot(
        equity=validated_equity,
        positions=positions,
        position_weights=position_weights,
        pending_buy_weights=pending_buy_weights,
        risk_weights=risk_weights,
        open_orders=orders,
    )


def size_capped_buy_notional(
    *,
    equity: float,
    allowed_target_weight: float,
    reserved_symbol_weight: float,
    quote: BrokerQuote,
    requested_qty: float | None = None,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> BuyNotionalSizing:
    """Size a market-notional buy without exceeding its authorized target.

    Alpaca receives a fixed notional rather than a share quantity.  A gap can
    therefore reduce shares received but cannot increase dollars spent beyond
    the authorized capacity.
    """
    validated_equity = _positive(equity, "equity")
    allowed = _fraction(allowed_target_weight, "allowed_target_weight")
    reserved = _nonnegative(reserved_symbol_weight, "reserved_symbol_weight")
    slippage = _nonnegative(slippage_bps, "slippage_bps") / 10_000.0
    conservative_ask = quote.ask_price * (1.0 + slippage)
    capacity = max((allowed - reserved) * validated_equity, 0.0)
    requested_notional = capacity
    if requested_qty is not None:
        qty = _positive(requested_qty, "requested_qty")
        requested_notional = qty * conservative_ask
    # Round down to cents; never round a risk authorization upward.
    notional = math.floor(min(capacity, requested_notional) * 100.0) / 100.0
    estimated_qty = notional / conservative_ask if notional > 0.0 else 0.0
    return BuyNotionalSizing(
        notional=notional,
        estimated_qty=estimated_qty,
        conservative_ask=conservative_ask,
        available_target_notional=capacity,
    )


def submit_order(
    symbol: str,
    qty: float | None = None,
    side: str = "",
    type: str = "market",
    tif: str = "day",
    stop_price: float | None = None,
    *,
    notional: float | None = None,
    cancel_existing: bool = True,
) -> dict[str, Any]:
    """Submit an order and return only a broker-accepted response.

    ``qty`` and ``notional`` are mutually exclusive.  Notional market buys are
    preferred for capped live purchases because price gaps cannot increase the
    dollars committed.
    """
    normalized_symbol = _symbol(symbol)
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    order_type = str(type).strip().lower()
    time_in_force = str(tif).strip().lower()
    if (qty is None) == (notional is None):
        raise ValueError("provide exactly one of qty or notional")
    if notional is not None and normalized_side != "buy":
        raise ValueError("notional orders are supported only for buys")
    if notional is not None and order_type != "market":
        raise ValueError("notional orders must use market type")
    if notional is not None and time_in_force != "day":
        raise ValueError("notional orders must use day time-in-force")

    if cancel_existing:
        cancel_open_orders(normalized_symbol)

    payload: dict[str, Any] = {
        "symbol": normalized_symbol,
        "side": normalized_side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if qty is not None:
        payload["qty"] = _decimal_string(_positive(qty, "qty"))
    else:
        payload["notional"] = _decimal_string(
            _positive(notional, "notional"),
            cents=True,
        )
    if stop_price is not None:
        payload["stop_price"] = _decimal_string(
            _positive(stop_price, "stop_price")
        )

    response = requests.post(
        f"{_base_url()}/orders",
        headers=_headers(),
        json=payload,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    order = _response_mapping(
        response,
        expected_statuses={200, 201},
        operation=f"submit {normalized_side} order for {normalized_symbol}",
    )
    return validate_accepted_order(
        order,
        expected_symbol=normalized_symbol,
        expected_side=normalized_side,
    )


def validate_accepted_order(
    payload: Mapping[str, Any],
    *,
    expected_symbol: str | None = None,
    expected_side: str | None = None,
) -> dict[str, Any]:
    """Reject error-shaped, terminal, or unidentifiable order payloads."""
    if not isinstance(payload, Mapping):
        raise BrokerResponseError("order response is not an object")
    if payload.get("error") or (
        payload.get("code") is not None and payload.get("message")
    ):
        raise OrderRejectedError("broker returned an error payload")
    order_id = str(payload.get("id", "")).strip()
    symbol = _symbol(payload.get("symbol"))
    side = str(payload.get("side", "")).strip().lower()
    status = str(payload.get("status", "")).strip().lower()
    if not order_id:
        raise BrokerResponseError("order response has no id")
    if expected_symbol is not None and symbol != _symbol(expected_symbol):
        raise BrokerResponseError("order response symbol does not match request")
    if expected_side is not None and side != str(expected_side).strip().lower():
        raise BrokerResponseError("order response side does not match request")
    if status in TERMINAL_REJECTION_STATUSES:
        raise OrderRejectedError(f"broker order status is {status}")
    if status not in ACCEPTED_ORDER_STATUSES:
        raise BrokerResponseError(f"unexpected broker order status {status!r}")
    result = dict(payload)
    result.update(
        {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "status": status,
            "broker_accepted": True,
        }
    )
    return result


def cancel_open_orders(symbol: str | None = None) -> dict[str, Any]:
    """Cancel matching open orders and confirm none remain.

    A deletion response alone is not considered confirmation.  The function
    re-queries open orders and raises ``CancellationError`` if any targeted
    order remains or if any cancellation request failed.
    """
    normalized_symbol = _symbol(symbol) if symbol is not None else None
    orders = get_open_orders(normalized_symbol)
    cancelled_ids: list[str] = []
    for order in orders:
        response = requests.delete(
            f"{_base_url()}/orders/{order['id']}",
            headers=_headers(),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        if response.status_code not in {200, 204}:
            raise CancellationError(
                f"broker did not accept cancellation for order {order['id']}"
            )
        if response.status_code == 200:
            payload = _optional_json(response)
            if isinstance(payload, Mapping) and (
                payload.get("error")
                or (payload.get("code") is not None and payload.get("message"))
            ):
                raise CancellationError(
                    f"broker returned an error cancelling order {order['id']}"
                )
        cancelled_ids.append(order["id"])

    remaining = get_open_orders(normalized_symbol)
    if remaining:
        remaining_ids = ", ".join(order["id"] for order in remaining)
        raise CancellationError(
            f"open-order cancellation was not confirmed: {remaining_ids}"
        )
    return {
        "confirmed": True,
        "symbol": normalized_symbol,
        "cancelled_order_ids": cancelled_ids,
        "remaining_open_order_ids": [],
    }


def close_all_positions() -> LiquidationResult:
    """Cancel all orders and attempt an exact close for every broker position.

    A failure for one symbol does not prevent close attempts for the remaining
    symbols.  If any attempt fails, ``LiquidationError`` exposes only the
    individually validated accepted orders and every failed symbol so callers
    can report a partial liquidation without claiming portfolio-wide success.
    """
    cancellations = cancel_open_orders()
    positions = get_positions()
    accepted: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for symbol in sorted(positions):
        try:
            response = requests.delete(
                f"{_base_url()}/positions/{symbol}",
                headers=_headers(),
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            order = _response_mapping(
                response,
                expected_statuses={200},
                operation=f"close position {symbol}",
            )
            accepted.append(
                validate_accepted_order(
                    order,
                    expected_symbol=symbol,
                    expected_side=(
                        "sell" if positions[symbol].qty > 0.0 else "buy"
                    ),
                )
            )
        except Exception as exc:
            failures[symbol] = f"{type(exc).__name__}: {exc}"
    if failures:
        raise LiquidationError(
            (
                f"portfolio liquidation accepted {len(accepted)} of "
                f"{len(positions)} close orders"
            ),
            positions=positions,
            accepted_orders=tuple(accepted),
            failures=failures,
            cancellations=cancellations,
        )
    return LiquidationResult(
        position_symbols=tuple(sorted(positions)),
        positions=dict(positions),
        accepted_orders=tuple(accepted),
        cancellations=cancellations,
    )


def _pending_buy_reserve(
    order: Mapping[str, Any],
    *,
    symbol: str,
    slippage_bps: float,
    quote_cache: dict[str, BrokerQuote],
) -> float:
    notional_raw = order.get("notional")
    if notional_raw in (None, ""):
        notional = None
    else:
        notional = _positive(notional_raw, f"{symbol}.order_notional")
    if notional is not None:
        # Reserving the full notional is conservative when partially filled.
        return notional

    qty = _positive(order.get("qty"), f"{symbol}.order_qty")
    filled_qty = _nonnegative(
        order.get("filled_qty", 0.0) or 0.0,
        f"{symbol}.filled_qty",
    )
    remaining_qty = max(qty - filled_qty, 0.0)
    if remaining_qty <= 0.0:
        return 0.0
    limit_price_raw = order.get("limit_price")
    if limit_price_raw not in (None, ""):
        reserve_price = _positive(
            limit_price_raw,
            f"{symbol}.limit_price",
        )
    else:
        quote = quote_cache.get(symbol)
        if quote is None:
            quote = get_latest_quote(symbol)
            quote_cache[symbol] = quote
        stop_price_raw = order.get("stop_price")
        stop_price = (
            _positive(stop_price_raw, f"{symbol}.stop_price")
            if stop_price_raw not in (None, "")
            else 0.0
        )
        reserve_price = max(quote.ask_price, stop_price) * (
            1.0 + slippage_bps / 10_000.0
        )
    return remaining_qty * reserve_price


def _response_mapping(
    response: Any,
    *,
    expected_statuses: set[int],
    operation: str,
) -> Mapping[str, Any]:
    if getattr(response, "status_code", None) not in expected_statuses:
        raise BrokerResponseError(
            f"{operation} failed with HTTP {getattr(response, 'status_code', 'unknown')}"
        )
    payload = _optional_json(response)
    if not isinstance(payload, Mapping):
        raise BrokerResponseError(f"{operation} returned a non-object payload")
    if payload.get("error") or (
        payload.get("code") is not None and payload.get("message")
    ):
        raise BrokerResponseError(f"{operation} returned an error payload")
    return payload


def _response_list(
    response: Any,
    *,
    expected_statuses: set[int],
    operation: str,
) -> list[Any]:
    if getattr(response, "status_code", None) not in expected_statuses:
        raise BrokerResponseError(
            f"{operation} failed with HTTP {getattr(response, 'status_code', 'unknown')}"
        )
    payload = _optional_json(response)
    if isinstance(payload, Mapping) and (
        payload.get("error")
        or (payload.get("code") is not None and payload.get("message"))
    ):
        raise BrokerResponseError(f"{operation} returned an error payload")
    if not isinstance(payload, list):
        raise BrokerResponseError(f"{operation} returned a non-list payload")
    return payload


def _optional_json(response: Any) -> Any:
    try:
        return response.json()
    except (TypeError, ValueError) as exc:
        if getattr(response, "status_code", None) == 204:
            return None
        raise BrokerResponseError("broker response is not valid JSON") from exc


def _symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol or any(character.isspace() for character in symbol):
        raise BrokerResponseError("broker symbol is missing or invalid")
    return symbol


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise BrokerResponseError(f"{field_name} must be numeric")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise BrokerResponseError(f"{field_name} must be numeric") from exc
    if not math.isfinite(converted):
        raise BrokerResponseError(f"{field_name} must be finite")
    return converted


def _positive(value: Any, field_name: str) -> float:
    converted = _finite(value, field_name)
    if converted <= 0.0:
        raise BrokerResponseError(f"{field_name} must be positive")
    return converted


def _positive_or_zero(value: Any, field_name: str) -> float:
    converted = _finite(value, field_name)
    if converted < 0.0:
        raise BrokerResponseError(f"{field_name} cannot be negative")
    return converted


def _nonnegative(value: Any, field_name: str) -> float:
    converted = _finite(value, field_name)
    if converted < 0.0:
        raise ValueError(f"{field_name} cannot be negative")
    return converted


def _fraction(value: Any, field_name: str) -> float:
    converted = _finite(value, field_name)
    if converted < 0.0 or converted > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return converted


def _timestamp(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise BrokerResponseError(
                f"{field_name} must be an ISO timestamp"
            ) from exc
    else:
        raise BrokerResponseError(f"{field_name} is required")
    if parsed.tzinfo is None:
        raise BrokerResponseError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _decimal_string(value: float, *, cents: bool = False) -> str:
    if cents:
        return f"{value:.2f}"
    rendered = f"{value:.9f}".rstrip("0").rstrip(".")
    return rendered or "0"
