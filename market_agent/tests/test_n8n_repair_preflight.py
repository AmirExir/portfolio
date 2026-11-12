"""An inaccessible service is not evidence that offline database repair is safe."""

import errno
import unittest
from unittest.mock import patch

from scripts.repair_n8n_market_optimization import _n8n_is_running


class N8nRepairPreflightTests(unittest.TestCase):
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
