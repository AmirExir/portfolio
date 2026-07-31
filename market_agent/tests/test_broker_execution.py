from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from market_agent.agent.broker import (
    BrokerQuote,
    BrokerResponseError,
    CancellationError,
    LiquidationError,
    OrderRejectedError,
    cancel_open_orders,
    close_all_positions,
    get_latest_quote,
    get_portfolio_snapshot,
    size_capped_buy_notional,
    submit_order,
)


UTC = timezone.utc


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _position(
    symbol: str,
    *,
    qty: str,
    market_value: str,
    avg_entry_price: str,
    current_price: str = "100",
) -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "market_value": market_value,
        "avg_entry_price": avg_entry_price,
        "current_price": current_price,
    }


def _open_order(
    order_id: str,
    symbol: str,
    *,
    side: str = "buy",
    qty: str = "1",
    filled_qty: str = "0",
    limit_price: str | None = "100",
    notional: str | None = None,
    stop_price: str | None = None,
) -> dict:
    return {
        "id": order_id,
        "symbol": symbol,
        "side": side,
        "status": "new",
        "qty": qty,
        "filled_qty": filled_qty,
        "limit_price": limit_price,
        "notional": notional,
        "stop_price": stop_price,
    }


def _accepted_order(
    order_id: str,
    symbol: str,
    side: str,
    status: str = "accepted",
) -> dict:
    return {
        "id": order_id,
        "symbol": symbol,
        "side": side,
        "status": status,
    }


class BrokerPortfolioSnapshotTests(unittest.TestCase):
    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.get")
    def test_preserves_fractional_qty_entry_and_reserves_pending_buy(
        self,
        mock_get,
        _mock_base,
        _mock_headers,
    ) -> None:
        def response_for(url: str, **kwargs):
            if url.endswith("/positions"):
                return FakeResponse(
                    200,
                    [
                        _position(
                            "MU",
                            qty="1.375",
                            market_value="137.50",
                            avg_entry_price="94.25",
                        )
                    ],
                )
            if url.endswith("/orders"):
                return FakeResponse(
                    200,
                    [
                        _open_order(
                            "pending-1",
                            "MU",
                            qty="2.5",
                            filled_qty="0.5",
                            limit_price="110",
                        )
                    ],
                )
            raise AssertionError(url)

        mock_get.side_effect = response_for
        snapshot = get_portfolio_snapshot(10_000.0)

        self.assertEqual(snapshot.positions["MU"].qty, 1.375)
        self.assertEqual(snapshot.positions["MU"].avg_entry_price, 94.25)
        self.assertAlmostEqual(snapshot.position_weights["MU"], 0.01375)
        self.assertAlmostEqual(snapshot.pending_buy_weights["MU"], 0.022)
        self.assertAlmostEqual(snapshot.risk_weights["MU"], 0.03575)

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._data_base_url", return_value="https://data/v2")
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.get")
    def test_market_pending_buy_uses_verified_ask_and_slippage(
        self,
        mock_get,
        _mock_trade_base,
        _mock_data_base,
        _mock_headers,
    ) -> None:
        now = datetime.now(UTC)

        def response_for(url: str, **kwargs):
            if url.endswith("/positions"):
                return FakeResponse(200, [])
            if url.endswith("/orders"):
                return FakeResponse(
                    200,
                    [
                        _open_order(
                            "market-1",
                            "SNDK",
                            qty="4",
                            limit_price=None,
                        )
                    ],
                )
            if url.endswith("/stocks/SNDK/quotes/latest"):
                return FakeResponse(
                    200,
                    {
                        "quote": {
                            "bp": 49.90,
                            "ap": 50.00,
                            "t": now.isoformat(),
                        }
                    },
                )
            raise AssertionError(url)

        mock_get.side_effect = response_for
        snapshot = get_portfolio_snapshot(
            10_000.0,
            pending_order_slippage_bps=50.0,
        )

        self.assertAlmostEqual(
            snapshot.pending_buy_weights["SNDK"],
            4.0 * 50.0 * 1.005 / 10_000.0,
        )

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._data_base_url", return_value="https://data/v2")
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.get")
    def test_pending_stop_buy_reserves_above_current_ask(
        self,
        mock_get,
        _mock_trade_base,
        _mock_data_base,
        _mock_headers,
    ) -> None:
        now = datetime.now(UTC)

        def response_for(url: str, **kwargs):
            if url.endswith("/positions"):
                return FakeResponse(200, [])
            if url.endswith("/orders"):
                return FakeResponse(
                    200,
                    [
                        _open_order(
                            "stop-1",
                            "MU",
                            qty="2",
                            limit_price=None,
                            stop_price="110",
                        )
                    ],
                )
            if url.endswith("/stocks/MU/quotes/latest"):
                return FakeResponse(
                    200,
                    {
                        "quote": {
                            "bp": 99.9,
                            "ap": 100.0,
                            "t": now.isoformat(),
                        }
                    },
                )
            raise AssertionError(url)

        mock_get.side_effect = response_for
        snapshot = get_portfolio_snapshot(
            10_000.0,
            pending_order_slippage_bps=50.0,
        )

        self.assertAlmostEqual(
            snapshot.pending_buy_weights["MU"],
            2.0 * 110.0 * 1.005 / 10_000.0,
        )

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._data_base_url", return_value="https://data/v2")
    @patch("market_agent.agent.broker.requests.get")
    def test_stale_quote_fails_closed(
        self,
        mock_get,
        _mock_data_base,
        _mock_headers,
    ) -> None:
        now = datetime(2026, 7, 30, 15, 0, tzinfo=UTC)
        mock_get.return_value = FakeResponse(
            200,
            {
                "quote": {
                    "bp": 99.0,
                    "ap": 100.0,
                    "t": (now - timedelta(minutes=10)).isoformat(),
                }
            },
        )

        with self.assertRaisesRegex(BrokerResponseError, "stale"):
            get_latest_quote("MU", max_age_seconds=300, now=now)

    def test_notional_sizing_cannot_exceed_remaining_target(self) -> None:
        quote = BrokerQuote(
            symbol="MU",
            bid_price=99.5,
            ask_price=100.0,
            timestamp=datetime.now(UTC),
        )
        sizing = size_capped_buy_notional(
            equity=100_000.0,
            allowed_target_weight=0.05,
            reserved_symbol_weight=0.0475,
            quote=quote,
            requested_qty=100.0,
            slippage_bps=100.0,
        )

        self.assertEqual(sizing.notional, 250.0)
        self.assertEqual(sizing.conservative_ask, 101.0)
        self.assertLessEqual(
            sizing.notional + 0.0475 * 100_000.0,
            0.05 * 100_000.0,
        )
        self.assertAlmostEqual(sizing.estimated_qty, 250.0 / 101.0)


class BrokerMutationTests(unittest.TestCase):
    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.post")
    @patch("market_agent.agent.broker.requests.delete")
    @patch("market_agent.agent.broker.requests.get")
    def test_submit_requires_confirmed_cancellation_and_accepted_order(
        self,
        mock_get,
        mock_delete,
        mock_post,
        _mock_base,
        _mock_headers,
    ) -> None:
        mock_get.side_effect = [
            FakeResponse(200, [_open_order("old-1", "MU")]),
            FakeResponse(200, []),
        ]
        mock_delete.return_value = FakeResponse(204)
        mock_post.return_value = FakeResponse(
            201,
            _accepted_order("new-1", "MU", "buy"),
        )

        result = submit_order("mu", side="buy", notional=250.0)

        self.assertTrue(result["broker_accepted"])
        self.assertEqual(result["id"], "new-1")
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["notional"], "250.00")
        self.assertNotIn("qty", payload)

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.post")
    @patch("market_agent.agent.broker.requests.delete")
    @patch("market_agent.agent.broker.requests.get")
    def test_unconfirmed_cancellation_prevents_submission(
        self,
        mock_get,
        mock_delete,
        mock_post,
        _mock_base,
        _mock_headers,
    ) -> None:
        order = _open_order("old-1", "MU")
        mock_get.side_effect = [
            FakeResponse(200, [order]),
            FakeResponse(200, [order]),
        ]
        mock_delete.return_value = FakeResponse(204)

        with self.assertRaises(CancellationError):
            submit_order("MU", qty=1.25, side="sell")
        mock_post.assert_not_called()

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.post")
    @patch("market_agent.agent.broker.requests.get")
    def test_http_success_with_error_payload_is_rejected(
        self,
        mock_get,
        mock_post,
        _mock_base,
        _mock_headers,
    ) -> None:
        mock_get.side_effect = [FakeResponse(200, []), FakeResponse(200, [])]
        mock_post.return_value = FakeResponse(
            201,
            {"code": 40310000, "message": "insufficient buying power"},
        )

        with self.assertRaises(BrokerResponseError):
            submit_order("MU", side="buy", notional=100.0)

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.post")
    @patch("market_agent.agent.broker.requests.get")
    def test_terminal_order_status_is_not_treated_as_executed(
        self,
        mock_get,
        mock_post,
        _mock_base,
        _mock_headers,
    ) -> None:
        mock_get.side_effect = [FakeResponse(200, []), FakeResponse(200, [])]
        mock_post.return_value = FakeResponse(
            201,
            _accepted_order("rejected-1", "MU", "buy", status="rejected"),
        )

        with self.assertRaises(OrderRejectedError):
            submit_order("MU", side="buy", notional=100.0)

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.delete")
    @patch("market_agent.agent.broker.requests.get")
    def test_cancellation_delete_failure_is_fail_closed(
        self,
        mock_get,
        mock_delete,
        _mock_base,
        _mock_headers,
    ) -> None:
        mock_get.return_value = FakeResponse(
            200,
            [_open_order("old-1", "MU")],
        )
        mock_delete.return_value = FakeResponse(
            422,
            {"code": 422, "message": "cannot cancel"},
        )

        with self.assertRaises(CancellationError):
            cancel_open_orders("MU")

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.delete")
    @patch("market_agent.agent.broker.requests.get")
    def test_drawdown_liquidation_closes_every_position(
        self,
        mock_get,
        mock_delete,
        _mock_base,
        _mock_headers,
    ) -> None:
        positions = [
            _position(
                "MU",
                qty="1.375",
                market_value="137.50",
                avg_entry_price="94.25",
            ),
            _position(
                "SNDK",
                qty="2.125",
                market_value="106.25",
                avg_entry_price="47.00",
                current_price="50",
            ),
        ]
        mock_get.side_effect = [
            FakeResponse(200, []),  # initial open orders
            FakeResponse(200, []),  # cancellation confirmation
            FakeResponse(200, positions),
        ]

        def delete_for(url: str, **kwargs):
            symbol = url.rsplit("/", 1)[-1]
            return FakeResponse(
                200,
                _accepted_order(f"close-{symbol}", symbol, "sell"),
            )

        mock_delete.side_effect = delete_for
        result = close_all_positions()

        self.assertEqual(result.position_symbols, ("MU", "SNDK"))
        self.assertEqual(len(result.accepted_orders), 2)
        self.assertTrue(
            all(order["broker_accepted"] for order in result.accepted_orders)
        )
        self.assertEqual(mock_delete.call_count, 2)

    @patch("market_agent.agent.broker._headers", return_value={})
    @patch("market_agent.agent.broker._base_url", return_value="https://trade/v2")
    @patch("market_agent.agent.broker.requests.delete")
    @patch("market_agent.agent.broker.requests.get")
    def test_drawdown_liquidation_attempts_all_and_reports_partial_failure(
        self,
        mock_get,
        mock_delete,
        _mock_base,
        _mock_headers,
    ) -> None:
        positions = [
            _position(
                "MU",
                qty="1.5",
                market_value="150",
                avg_entry_price="90",
            ),
            _position(
                "SNDK",
                qty="2.5",
                market_value="125",
                avg_entry_price="45",
                current_price="50",
            ),
            _position(
                "WDC",
                qty="3.25",
                market_value="195",
                avg_entry_price="55",
                current_price="60",
            ),
        ]
        mock_get.side_effect = [
            FakeResponse(200, []),
            FakeResponse(200, []),
            FakeResponse(200, positions),
        ]
        mock_delete.side_effect = [
            FakeResponse(200, _accepted_order("close-MU", "MU", "sell")),
            FakeResponse(
                422,
                {"code": 422, "message": "position unavailable"},
            ),
            FakeResponse(200, _accepted_order("close-WDC", "WDC", "sell")),
        ]

        with self.assertRaises(LiquidationError) as raised:
            close_all_positions()
        self.assertEqual(mock_delete.call_count, 3)
        self.assertEqual(set(raised.exception.failures), {"SNDK"})
        self.assertEqual(
            [order["symbol"] for order in raised.exception.accepted_orders],
            ["MU", "WDC"],
        )


if __name__ == "__main__":
    unittest.main()
