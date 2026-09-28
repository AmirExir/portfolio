"""Exercise the real Streamlit entry points and result invalidation on mode change."""
from pathlib import Path
import unittest
from unittest.mock import patch

import streamlit as st
import torch
from streamlit.testing.v1 import AppTest


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_launchers_offer_the_same_title_modes_and_scenarios(self):
        directory = Path(__file__).resolve().parents[1]
        for launcher in ("app_gnn_streamlit.py", "app_streamlit_gnn_powergrid.py",
                         "app_streamlit_gnn_powergrid_GCN_N_2.py"):
            with self.subTest(launcher=launcher):
                with patch("streamlit.set_page_config", wraps=st.set_page_config) as configure:
                    app = AppTest.from_file(str(directory / launcher), default_timeout=30).run()
                self.assertEqual(configure.call_args.kwargs["page_title"], "Power Grid GNN")
                self.assertEqual([title.value for title in app.title], ["Power Grid GNN"])
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(len(app.warning), 0)
                self.assertEqual(app.sidebar.selectbox[0].options, ["voltage", "thermal"])
                self.assertEqual(app.sidebar.selectbox[-1].label, "Scenario ID")
                self.assertEqual(len(app.sidebar.selectbox[-1].options), 200)
                self.assertEqual(next(m.value for m in app.metric if m.label == "Classes"), "5")
                app.sidebar.selectbox[0].set_value("thermal").run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertEqual(next(m.value for m in app.metric if m.label == "Classes"), "4")
                self.assertFalse(any("alarm_flag" in w.value for w in app.warning))

    def test_missing_target_stops_the_app_without_substituting_an_alarm(self):
        path = Path(__file__).resolve().parents[1] / "app_streamlit_gnn_powergrid_GCN_N_2.py"
        app = AppTest.from_file(str(path), default_timeout=30).run()
        st.cache_data.clear()
        try:
            with patch("data_pipeline.load_csv_graphs", side_effect=ValueError(
                "bus data is missing required columns: voltage_class"
            )):
                app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 1)
            self.assertIn("voltage_class", app.error[0].value)
            self.assertEqual(len(app.button), 0)
            self.assertEqual(len(app.warning), 0)
        finally:
            st.cache_data.clear()

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
        expected_accuracy = next(m.value for m in app.metric if m.label == "Accuracy")
        app.sidebar.selectbox[-1].set_value("1").run()
        self.assertEqual(next(m.value for m in app.metric if m.label == "Accuracy"), expected_accuracy)
        app.sidebar.selectbox[0].set_value("thermal").run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("hist_df", app.session_state)
        self.assertTrue(any("dispatch and load are absent" in warning.value for warning in app.warning))

    def test_failed_retry_clears_the_previous_successful_run(self):
        path = Path(__file__).resolve().parents[1] / "app_gnn_streamlit.py"
        app = AppTest.from_file(str(path), default_timeout=60).run()
        app.sidebar.slider[0].set_value(10).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn("hist_df", app.session_state)
        with patch("gnn_clean.train_gnn_multi_graph", side_effect=RuntimeError("training interrupted")):
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertIn("training interrupted", app.error[0].value)
        for key in ("model", "hist_df", "trained", "training_configuration"):
            self.assertNotIn(key, app.session_state)
        self.assertFalse(any(metric.label == "Accuracy" for metric in app.metric))
        self.assertEqual(len(app.tabs[2].get("download_button")), 0)
        self.assertTrue(any("Start a run" in info.value for info in app.tabs[2].info))


if __name__ == "__main__":
    unittest.main()
