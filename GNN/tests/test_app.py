"""Exercise the real Streamlit entry points and result invalidation on mode change."""
from pathlib import Path
import unittest

import torch
from streamlit.testing.v1 import AppTest


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_launchers_start_without_pickled_artifacts(self):
        directory = Path(__file__).resolve().parents[1]
        for launcher in ("app_gnn_streamlit.py", "app_streamlit_gnn_powergrid.py"):
            with self.subTest(launcher=launcher):
                app = AppTest.from_file(str(directory / launcher), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(app.sidebar.selectbox[-1].label, "Scenario ID")
                self.assertEqual(len(app.sidebar.selectbox[-1].options), 200)

    def test_training_and_mode_switch_do_not_reuse_stale_results(self):
        path = Path(__file__).resolve().parents[1] / "app_gnn_streamlit.py"
        app = AppTest.from_file(str(path), default_timeout=60).run()
        app.sidebar.slider[0].set_value(10).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        report = app.session_state["hist_df"].attrs["evaluation"]
        self.assertEqual({key: len(value) for key, value in report["splits"].items()},
                         {"train": 120, "validation": 40, "test": 40})
        app.sidebar.selectbox[0].set_value("thermal").run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("hist_df", app.session_state)
        self.assertTrue(any("dispatch and load are absent" in warning.value for warning in app.warning))


if __name__ == "__main__":
    unittest.main()
