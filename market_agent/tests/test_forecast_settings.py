"""Dashboard settings and provenance for automatic forecast tuning."""

from types import SimpleNamespace
import unittest

from streamlit.testing.v1 import AppTest

from market_agent.forecast_settings import tuning_results_table


class ForecastSettingsTests(unittest.TestCase):
    def test_automatic_defaults_and_manual_choices_survive_mode_changes(self) -> None:
        app = AppTest.from_string(
            "import streamlit as st\n"
            "from market_agent.forecast_settings import render_forecast_settings\n"
            "st.session_state['current_settings'] = render_forecast_settings()"
        ).run()
        automatic = app.session_state["current_settings"]
        self.assertTrue(automatic.automatic)
        self.assertTrue(automatic.optimize_forecast_model)
        self.assertEqual(automatic.primary_model_choice, "Best Validation")
        self.assertEqual((automatic.sequence_model_choice, automatic.one_day_sequence_model_choice), ("off", "off"))
        self.assertFalse(automatic.include_rl_policy)
        self.assertFalse(app.checkbox)

        mode = "forecast_settings_mode"
        app.radio(key=mode).set_value("Manual").run()
        app.slider(key="forecast_manual_lag").set_value(45)
        app.slider(key="dashboard_forecast_horizon").set_value(10)
        app.selectbox(key="forecast_manual_model").set_value("Ridge")
        app.selectbox(key="forecast_manual_sequence").set_value("LSTM")
        app.selectbox(key="forecast_manual_one_day_sequence").set_value("Same as main forecast")
        app.checkbox(key="forecast_manual_optimize").uncheck().run()
        manual = app.session_state["current_settings"]
        self.assertFalse(manual.optimize_forecast_model)
        self.assertEqual(manual.primary_model_choice, "Ridge")
        self.assertEqual((manual.forecast_horizon, manual.forecast_lookback), (10, 45))
        self.assertEqual((manual.sequence_model_choice, manual.one_day_sequence_model_choice), ("lstm", "lstm"))

        app.radio(key=mode).set_value("Automatic").run()
        self.assertEqual(app.session_state["current_settings"].forecast_horizon, 10)
        self.assertEqual(app.session_state["current_settings"].forecast_lookback, 20)
        app.run()
        app.radio(key=mode).set_value("Manual").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["current_settings"], manual)

    def test_display_only_recorded_parameters_and_independent_errors(self) -> None:
        models = {
            "Ensemble": {"metrics": {"component_weights": {"Ridge": 0.5, "XGBoost": 0.5, "Excluded": 0}}},
            "Ridge": SimpleNamespace(metrics={
                "selected_lookback_window": 10, "selected_ridge_alpha": 25, "selection_validation_mae_pct": 3.0,
                "holdout_mae_pct": 4.1, "validation_is_oos": True,
            }),
            "XGBoost": {"metrics": {"selected_lookback_window": 45, "holdout_mae_pct": 9, "validation_is_oos": False}},
            "Excluded": {"metrics": {"selected_lookback_window": 60}},
        }
        table = tuning_results_table(models, "Ensemble").set_index("Model")
        self.assertEqual(table.index.tolist(), ["Ridge", "XGBoost"])
        self.assertEqual(table.loc["Ridge", "Lag (sessions)"], 10)
        self.assertEqual(table.loc["Ridge", "Regularization"], 25)
        self.assertEqual(table.loc["Ridge", "Out-of-sample MAE (%)"], 4.1)
        self.assertTrue(table.isna().loc["XGBoost", ["Regularization", "Out-of-sample MAE (%)"]].all())
        self.assertEqual(tuning_results_table(models, "Ridge")["Model"].tolist(), ["Ridge"])
        self.assertTrue(tuning_results_table({"Ensemble": {"metrics": {}}}, "Ensemble").empty)


if __name__ == "__main__":
    unittest.main()
