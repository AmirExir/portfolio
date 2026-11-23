"""An inaccessible service is not evidence that offline database repair is safe."""

import errno
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.repair_n8n_market_optimization import _n8n_is_running, main


class N8nRepairPreflightTests(unittest.TestCase):
    def test_cli_selects_news_only_on_request(self) -> None:
        arguments = ["repair", "--database", "fixture.sqlite", "--backup", "backup.sqlite", "--workflow-id", "fixture"]
        for extra, expected in (([], False), (["--news-only"], True)):
            with self.subTest(news_only=expected), patch("sys.argv", arguments + extra), \
                    patch("scripts.repair_n8n_market_optimization.repair_database", return_value=()) as repair, \
                    redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(repair.call_args.kwargs["news_only"], expected)

    def test_cli_owner_environment_keeps_public_forecasts_off(self) -> None:
        arguments = ["repair", "--database", "fixture.sqlite", "--backup", "backup.sqlite", "--workflow-id", "fixture"]
        with patch.dict("os.environ", {"ERCOT_FORECAST_OWNER_CHAT_ID": "123"}), \
                patch("sys.argv", arguments), \
                patch("scripts.repair_n8n_market_optimization.repair_database", return_value=()) as repair, \
                redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(repair.call_args.kwargs["ercot_forecast_owner_chat_id"], "123")
        self.assertFalse(repair.call_args.kwargs["ercot_forecast_publish_scheduled"])

    def test_public_forecasts_require_explicit_cli_opt_in(self) -> None:
        arguments = ["repair", "--database", "fixture.sqlite", "--backup", "backup.sqlite", "--workflow-id", "fixture",
                     "--ercot-forecast-owner-chat-id", "456", "--publish-scheduled-ercot-forecasts"]
        with patch("sys.argv", arguments), \
                patch("scripts.repair_n8n_market_optimization.repair_database", return_value=()) as repair, \
                redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(repair.call_args.kwargs["ercot_forecast_owner_chat_id"], "456")
        self.assertTrue(repair.call_args.kwargs["ercot_forecast_publish_scheduled"])

    def test_only_connection_refused_establishes_stopped_service(self) -> None:
        with patch("scripts.repair_n8n_market_optimization.socket.socket") as socket:
            client = socket.return_value.__enter__.return_value
            client.connect_ex.return_value = 0
            self.assertTrue(_n8n_is_running())
            client.connect_ex.return_value = errno.ECONNREFUSED
            self.assertFalse(_n8n_is_running())
            for status in (errno.EPERM, errno.EACCES, errno.ETIMEDOUT):
                client.connect_ex.return_value = status
                with self.subTest(status=status), self.assertRaisesRegex(
                    RuntimeError, "Cannot verify whether n8n is stopped",
                ):
                    _n8n_is_running()


if __name__ == "__main__":
    unittest.main()
