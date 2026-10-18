"""Explicit licensed Windows smoke test of a copied .sav case; never run generated code.

Run from the Python environment supplied/configured for your PSS/E installation.
The wrapper checks API returns and convergence, but does not evaluate engineering
criteria or validate an assistant's generated script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence


class PSSEError(RuntimeError):
    """A simulator call failed or did not satisfy its documented return contract."""


def _checked(api: Any, name: str, *args: Any) -> None:
    result = getattr(api, name)(*args)
    if type(result) is not int or result != 0:
        raise PSSEError(f"PSS/E call {name} failed or returned an unexpected status: {result!r}")


def validate_case(api: Any, source: Path, output: Path, *, version: str,
                  buses: int, solve_options: Sequence[int], assumptions: str) -> dict[str, Any]:
    """Load and solve a copy, save separately, restore the copy, and record results.

    The injected API permits deterministic tests without a license. Output must be
    a new directory. Solve options are supplied by the operator, never inferred.
    """
    source, output = source.resolve(), output.resolve()
    if source.suffix.lower() != ".sav" or not source.is_file():
        raise ValueError("Source must be an existing .sav case.")
    if buses < 1 or len(solve_options) != 8 or any(type(value) is not int for value in solve_options):
        raise ValueError("Supply a positive bus capacity and eight explicit integer FNSL options.")
    if not version.strip() or not assumptions.strip():
        raise ValueError("Version and study assumptions are required.")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    copy = output / "input_copy.sav"
    saved = output / "solved_copy.sav"
    shutil.copyfile(source, copy)
    report: dict[str, Any] = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_case": str(source), "source_sha256": source_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "declared_psse_version": version, "bus_capacity": buses,
        "fnsl_options": list(solve_options), "assumptions": assumptions,
        "contingency_set": "none; base case only", "criteria": "API success and solver convergence only",
        "engineering_validation": "not performed", "status": "failed", "calls_completed": [],
    }
    cwd = Path.cwd()
    loaded = False
    try:
        os.chdir(output)  # PSS/E ancillary outputs stay beside the derived case.
        _checked(api, "psseinit", buses)
        report["calls_completed"].append("psseinit")
        reported = api.psseversion()
        if not isinstance(reported, (list, tuple)) or len(reported) != 6:
            raise PSSEError("Unexpected psseversion result; validate this wrapper against the installed API.")
        installed = ".".join(str(value) for value in reported[1:4])
        report["installed_psse_version"] = installed
        if installed != version:
            raise PSSEError("Installed PSS/E version does not exactly match the requested version.")
        _checked(api, "case", str(copy))
        loaded = True
        report["calls_completed"].append("case")
        _checked(api, "fnsl", list(solve_options))
        report["calls_completed"].append("fnsl")
        convergence = api.solved()
        report["solver_status"] = convergence
        if type(convergence) is not int or convergence != 0:
            raise PSSEError(f"Solver did not reach tolerance: solved={convergence!r}")
        if saved.exists():
            raise FileExistsError("Derived output already exists; refusing overwrite.")
        _checked(api, "save", str(saved))
        if not saved.is_file():
            raise PSSEError("Save returned success but no derived case was created.")
        report["calls_completed"].append("save")
        report["derived_case"] = str(saved)
        report["derived_sha256"] = hashlib.sha256(saved.read_bytes()).hexdigest()
        report["status"] = "solver smoke test completed"
    except Exception as exc:
        # PSS/E installations expose version-specific exception types. Preserve
        # failure in the report and CLI status, with no silent retry or fallback.
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if loaded:
            try:
                _checked(api, "case", str(copy))
                report["restored_input_copy"] = True
            except Exception as exc:
                report["restore_error"] = f"{type(exc).__name__}: {exc}"
                report["status"] = "failed"
        os.chdir(cwd)
        try:
            report["source_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        except OSError as exc:
            report["source_unchanged"] = None
            report["source_integrity_error"] = f"Source integrity could not be checked ({type(exc).__name__}). Investigate before using any outputs."
        if report["source_unchanged"] is not True:
            report["status"] = "failed"
            report.setdefault("source_integrity_error", "Source hash changed during validation. Investigate before using any outputs.")
        with (output / "validation.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
    return report


def main() -> int:
    """Run only when explicitly invoked in a configured Windows PSS/E environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--psse-version", required=True, help="Exact major.minor.modification version")
    parser.add_argument("--bootstrap-module", help="Optional installed PSS/E environment module")
    parser.add_argument("--buses", type=int, required=True)
    parser.add_argument("--fnsl-options", nargs=8, type=int, required=True)
    parser.add_argument("--assumptions", required=True, help="Study scenario, controls, load/dispatch, and applicable criteria")
    args = parser.parse_args()
    if os.name != "nt":
        parser.exit(2, "Run in your licensed Windows PSS/E environment. No simulation was started.\n")
    try:
        if args.bootstrap_module:
            importlib.import_module(args.bootstrap_module)
        api = importlib.import_module("psspy")
        report = validate_case(api, args.case, args.output_dir, version=args.psse_version,
                               buses=args.buses, solve_options=args.fnsl_options, assumptions=args.assumptions)
    except (ImportError, OSError, ValueError) as exc:
        parser.exit(2, f"Validation could not start: {exc}\n")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] == "solver smoke test completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
