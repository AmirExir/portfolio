"""Regression tests for bounded ERCOT Public Reports API requests."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

import requests

from ERCOTAPI.ercotapi import ERCOT_REQUEST_TIMEOUT, ErcotAPI


class ErcotAPIClientTests(unittest.TestCase):
    @patch("ERCOTAPI.ercotapi.time.sleep")
    @patch("ERCOTAPI.ercotapi.requests.get")
    def test_public_request_retries_timeout_with_bounded_wait(
        self,
        mock_get: Mock,
        mock_sleep: Mock,
    ) -> None:
        response = Mock()
        response.json.return_value = {"data": [{"total": 1.0}]}
        mock_get.side_effect = [requests.Timeout("upstream stalled"), response]
        api = ErcotAPI(bearer_token="token", subscription_key="key")

        payload = api.get_public("report", params={"page": 1})

        self.assertEqual(payload, {"data": [{"total": 1.0}]})
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(
            mock_get.call_args_list,
            [
                call(
                    "https://api.ercot.com/api/public-reports/report",
                    headers={
                        "Authorization": "Bearer token",
                        "Ocp-Apim-Subscription-Key": "key",
                    },
                    params={"page": 1},
                    timeout=ERCOT_REQUEST_TIMEOUT,
                ),
                call(
                    "https://api.ercot.com/api/public-reports/report",
                    headers={
                        "Authorization": "Bearer token",
                        "Ocp-Apim-Subscription-Key": "key",
                    },
                    params={"page": 1},
                    timeout=ERCOT_REQUEST_TIMEOUT,
                ),
            ],
        )
        mock_sleep.assert_called_once_with(1)

    @patch("ERCOTAPI.ercotapi.requests.post")
    def test_authentication_timeout_is_actionable(self, mock_post: Mock) -> None:
        mock_post.side_effect = requests.Timeout("upstream stalled")

        with self.assertRaisesRegex(ValueError, "Authentication timed out"):
            ErcotAPI(username="user", password="password", client_id="client")

        mock_post.assert_called_once_with(
            unittest.mock.ANY,
            data=unittest.mock.ANY,
            timeout=ERCOT_REQUEST_TIMEOUT,
        )


if __name__ == "__main__":
    unittest.main()
