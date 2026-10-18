"""Validate declared raw study data without inventing missing engineering metadata."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

RAW_BUS_FEATURES = ("load_MW", "p_inj_mw", "load_Mvar", "q_inj_mvar")
RAW_NOTE = (
    "Raw-data hashes, declared units, criteria, solver status and label consistency were checked. "
    "Metadata is supplied by the dataset owner; source cases and physical correctness still require engineering review."
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Study manifest requires nonempty {name}.")
    return value


def _values(data: pd.DataFrame, names: list[str]) -> np.ndarray:
    if not set(names) <= set(data.columns):
        raise ValueError(f"Raw study data requires columns: {', '.join(names)}")
    values = data[names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Raw study quantities must be finite; missing values cannot be inferred.")
    return values


def validate_study_manifest(path: Path, buses: pd.DataFrame, edges: pd.DataFrame,
                            hashes: dict[str, str], mode: str) -> dict[str, Any]:
    """Require hashes, study identity and exact source-label agreement for raw data."""
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("data_kind") != "raw_physical":
        raise ValueError("Study manifest requires schema_version 1 and data_kind raw_physical.")
    for field in ("dataset_id", "source", "preprocessing"):
        _text(manifest.get(field), field)
    availability = manifest.get("feature_availability")
    if not isinstance(availability, dict) or availability.get("role") != "pre_study_inputs":
        raise ValueError("feature_availability must declare role pre_study_inputs; solved/post-outcome predictors are not allowed.")
    _text(availability.get("description"), "feature_availability.description")
    if manifest["preprocessing"] != "none":
        raise ValueError("Raw study path requires unscaled physical data; upstream preprocessing is not inferred.")
    if manifest.get("source_hashes") != hashes:
        raise ValueError("Study manifest source hashes do not match both CSV files.")
    expected_units = {"load_MW": "MW", "p_inj_mw": "MW", "load_Mvar": "Mvar", "q_inj_mvar": "Mvar",
                      "voltage_pu": "pu", "x_pu": "pu", "length_km": "km", "flow_mva": "MVA",
                      "normal_rating_mva": "MVA", "emergency_rating_mva": "MVA"}
    if manifest.get("units") != expected_units:
        raise ValueError("Declare all raw-column units exactly as documented; no unit conversion is applied.")
    for frame, name in ((buses, "bus"), (edges, "edge")):
        if "scenario" not in frame or frame.scenario.isna().any():
            raise ValueError(f"Raw {name} table requires scenario identifiers.")
    scenarios = {str(value) for value in edges.scenario}
    if {str(value) for value in buses.scenario} != scenarios:
        raise ValueError("Raw bus and edge scenario IDs must match.")
    definitions = manifest.get("scenarios")
    if not isinstance(definitions, dict) or set(definitions) != scenarios:
        raise ValueError("Study manifest must describe every source scenario exactly once.")
    _values(buses, list(RAW_BUS_FEATURES))
    _values(edges, ["x_pu", "length_km", "normal_rating_mva", "emergency_rating_mva"])
    if (edges[["normal_rating_mva", "emergency_rating_mva"]] <= 0).any().any() or (edges.length_km < 0).any():
        raise ValueError("Line ratings must be positive and lengths nonnegative.")
    if "branch_id" not in edges or edges.branch_id.isna().any() or edges.branch_id.astype(str).str.strip().eq("").any():
        raise ValueError("Raw branches require nonempty branch_id identifiers.")
    if edges.duplicated(["scenario", "branch_id"]).any():
        raise ValueError("Branch IDs must be unique within each scenario, including parallel branches.")
    if "bus" not in buses or buses.bus.isna().any() or buses.duplicated(["scenario", "bus"]).any():
        raise ValueError("Raw bus IDs must be unique within each scenario.")
    criteria = manifest.get("class_thresholds")
    if not isinstance(criteria, dict):
        raise ValueError("Declare class_thresholds as an object keyed by classification mode.")
    thresholds = criteria.get(mode)
    needed = 4 if mode == "voltage" else 3
    if not isinstance(thresholds, list) or len(thresholds) != needed:
        raise ValueError(f"Declare {needed} increasing {mode} class thresholds.")
    limits = np.asarray(thresholds, dtype=float)
    if not np.isfinite(limits).all() or (np.diff(limits) <= 0).any() or (limits <= 0).any():
        raise ValueError("Class thresholds must be finite, positive and strictly increasing.")
    _text(manifest.get("criteria_source"), "criteria_source")
    for scenario, definition in definitions.items():
        if not isinstance(definition, dict):
            raise ValueError("Every scenario requires an object with study provenance.")
        for field in ("source_case", "split_group_id", "topology_assumptions", "load_dispatch_assumptions", "solver_version"):
            _text(definition.get(field), field)
        digest = definition.get("source_case_sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Each scenario requires a source case SHA-256.")
        timestamp = datetime.fromisoformat(_text(definition.get("executed_at"), "executed_at").replace("Z", "+00:00"))
        if timestamp.utcoffset() is None:
            raise ValueError("Study execution timestamps must identify their timezone.")
        if definition.get("solver_converged") is not True:
            raise ValueError("Unconverged or unverified scenarios cannot enter the raw training path.")
        condition = definition.get("condition")
        if condition not in ("pre_contingency", "post_contingency"):
            raise ValueError("Specify pre_contingency or post_contingency explicitly.")
        _text(definition.get("contingency_definition"), "contingency_definition (use none for base cases)")
        if definition.get("rating_basis") not in ("normal", "emergency"):
            raise ValueError("Specify normal or emergency rating basis for each scenario.")
        scenario_buses = buses.loc[buses.scenario.astype(str) == scenario]
        scenario_edges = edges.loc[edges.scenario.astype(str) == scenario]
        if not {"from_bus", "to_bus"} <= set(scenario_edges.columns) or not set(scenario_edges.from_bus).union(scenario_edges.to_bus) <= set(scenario_buses.bus):
            raise ValueError("Every raw branch endpoint, including open branches, must exist in its bus scenario.")
        if mode == "voltage":
            values = _values(scenario_buses, ["voltage_pu"]).ravel()
            if (values <= 0).any():
                raise ValueError("Physical voltage must be positive pu.")
            labels = _values(scenario_buses, ["voltage_class"]).ravel()
        else:
            values = _values(scenario_edges, ["flow_mva"]).ravel()
            if (values < 0).any():
                raise ValueError("flow_mva must be nonnegative max terminal apparent power.")
            rating = scenario_edges[f"{definition['rating_basis']}_rating_mva"].to_numpy(dtype=float)
            values = values / rating * 100
            labels = _values(scenario_edges, ["thermal_class"]).ravel()
        if not np.array_equal(labels, np.digitize(values, limits, right=False)):
            raise ValueError(f"Scenario {scenario}: source labels disagree with declared {mode} criteria.")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    return manifest
