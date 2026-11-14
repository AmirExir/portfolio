"""Stable, plain-language names for the forecasting methods actually recorded."""

from __future__ import annotations


def model_display_name(model_type: str) -> str:
    """Name a recorded estimator without guessing an unknown model's method.

    The label must come from the saved run's ``model_type``, not from whichever
    optional libraries happen to be installed when a report is displayed.
    Unknown identifiers are deliberately not echoed into chat or chart text.
    """
    if not isinstance(model_type, str):
        return "Experimental load model"
    return {
        "XGBRegressor": "XGBoost",
        "RandomForestRegressor": "Random Forest",
    }.get(model_type, "Experimental load model")
