"""Static PSS/E code review without importing PSS/E or executing supplied code.

This is a conservative syntax/call-shape screen, not a safety sandbox, control-flow
proof, signature authority, or electrical validation. Unknowns remain visible.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import inspect
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Finding:
    """A review finding located in the supplied Python source."""
    severity: str
    code: str
    line: int
    message: str


def python_blocks(answer: str) -> list[str]:
    """Extract explicit Python fences; accept unfenced text as a single script."""
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", answer, re.DOTALL | re.IGNORECASE)
    return blocks or ([answer] if "```" not in answer else [])


def validate_catalog(catalog: Mapping[str, Any], version: str) -> dict[str, inspect.Signature]:
    """Validate an operator-supplied, version-specific signature catalog."""
    if not isinstance(catalog, Mapping) or catalog.get("schema_version") != 1:
        raise ValueError("Signature catalog requires schema_version 1.")
    if catalog.get("psse_version") != version:
        raise ValueError("Signature catalog PSS/E version must exactly match the selected version.")
    if not isinstance(catalog.get("source"), str) or not catalog["source"].strip():
        raise ValueError("Signature catalog must identify its source.")
    if not isinstance(catalog.get("functions"), dict):
        raise ValueError("Signature catalog requires a functions object.")
    signatures = {}
    kinds = {kind.name: kind for kind in (inspect.Parameter.POSITIONAL_ONLY,
             inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.VAR_POSITIONAL,
             inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)}
    for name, parameters in catalog["functions"].items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(parameters, list):
            raise ValueError("Invalid function signature entry.")
        parsed = []
        for parameter in parameters:
            if not isinstance(parameter, dict) or parameter.get("kind") not in kinds or not isinstance(parameter.get("required"), bool):
                raise ValueError(f"Invalid parameter declaration for {name}.")
            kind = kinds[parameter["kind"]]
            default = inspect.Parameter.empty if parameter["required"] or kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) else None
            parsed.append(inspect.Parameter(parameter["name"], kind, default=default))
        signatures[name] = inspect.Signature(parsed)
    return signatures


def _path_key(value: str) -> str:
    # PSS/E runs on Windows: compare drive, separators and case using its semantics.
    return str(PureWindowsPath(value)).casefold()


def review_code(code: str, *, psse_version: str, catalog: Mapping[str, Any] | None = None,
                source_cases: Sequence[str] = ()) -> dict[str, Any]:
    """Report syntax, initialization, return-code, signature and save-path risks."""
    if not isinstance(psse_version, str) or not psse_version.strip():
        raise ValueError("Specify the installed PSS/E version; it is never inferred.")
    if len(code) > 200_000:
        raise ValueError("Code exceeds the 200,000 character review limit.")
    signatures = validate_catalog(catalog, psse_version) if catalog is not None else {}
    findings: list[Finding] = []
    calls_report = []
    def add(severity: str, category: str, line: int, message: str) -> None:
        findings.append(Finding(severity, category, line, message))
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        add("error", "syntax", exc.lineno or 0, exc.msg)
        tree = ast.Module(body=[], type_ignores=[])
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    aliases = {"psspy"}
    direct = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(item.asname or item.name for item in node.names if item.name == "psspy")
        if isinstance(node, ast.ImportFrom) and node.module == "psspy":
            direct.update({item.asname or item.name: item.name for item in node.names})
            if any(item.name == "*" for item in node.names):
                add("warning", "dynamic_calls", node.lineno, "Wildcard import prevents complete API identification.")
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (function.attr if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                and function.value.id in aliases else direct.get(function.id) if isinstance(function, ast.Name) else None)
        if name:
            calls.append((node, name))
    calls.sort(key=lambda item: item[0].lineno)
    init_lines = [node.lineno for node, name in calls if name == "psseinit"]
    loaded_paths = {_path_key(path) for path in source_cases}
    for node, name in calls:
        if name in {"case", "read"}:
            loaded_paths.update(_path_key(arg.value) for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
            loaded_paths.update(_path_key(kw.value.value) for kw in node.keywords if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str))
    if not calls:
        add("warning", "no_calls", 0, "No statically identifiable psspy calls; dynamic calls are not analyzed.")
    if catalog is None:
        add("warning", "signature_catalog", 0, "No version-specific signature catalog supplied. API names and argument signatures remain unverified.")
    for node, name in calls:
        status = "unverified"
        if name in signatures:
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                add("warning", "dynamic_arguments", node.lineno, f"{name}: expanded arguments cannot be bound statically.")
            else:
                try:
                    signatures[name].bind(*([None] * len(node.args)), **{kw.arg: None for kw in node.keywords})
                    status = "call shape matches supplied catalog"
                except TypeError as exc:
                    status = "mismatch"
                    add("error", "signature", node.lineno, f"{name}: {exc}")
        else:
            add("warning", "unknown_api", node.lineno, f"{name}: absent from the supplied version-specific catalog.")
        calls_report.append({"function": name, "line": node.lineno, "signature_status": status})
        if name != "psseinit" and not any(line < node.lineno for line in init_lines):
            add("warning", "initialization", node.lineno, f"{name}: no earlier psseinit call detected; inspect execution order and initialization result.")
        parent = parents.get(node)
        assigned = isinstance(parent, (ast.Assign, ast.AnnAssign))
        add("review" if assigned else "warning", "return_code", node.lineno,
            f"{name}: {'assigned result requires review of ierr handling on every path' if assigned else 'result is not directly assigned; verify the documented return contract and handle failures'}.")
        if name in {"save", "rawd", "rawd_2", "snap"}:
            paths = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
            paths += [kw.value.value for kw in node.keywords if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)]
            if not paths:
                add("warning", "save_path_unknown", node.lineno, f"{name}: output path is dynamic; establish a new path and refuse overwrite at runtime.")
            for path in paths:
                if _path_key(path) in loaded_paths:
                    add("error", "source_overwrite", node.lineno, f"{name}: output path matches an identified input case.")
                else:
                    add("review", "save_path", node.lineno, f"{name}: literal output path still needs existence, alias and permission checks at runtime.")
    return {"schema_version": 1, "psse_version": psse_version,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "catalog_source": catalog.get("source") if catalog else None,
            "syntax_valid": not any(item.code == "syntax" for item in findings),
            "has_errors": any(item.severity == "error" for item in findings),
            "status": "Static review only; engineering and runtime behavior unverified",
            "calls": calls_report, "findings": [asdict(item) for item in findings],
            "limitations": "No code executed. Control flow, value ranges, dynamic calls, filesystem aliases, solver convergence and electrical criteria require separate validation."}


def main() -> int:
    """Review a local script and print a JSON report without executing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("--psse-version", required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--source-case", action="append", default=[])
    args = parser.parse_args()
    try:
        report = review_code(args.script.read_text(), psse_version=args.psse_version,
                             catalog=json.loads(args.catalog.read_text()) if args.catalog else None,
                             source_cases=args.source_case)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Preflight failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return int(report["has_errors"])


if __name__ == "__main__":
    raise SystemExit(main())
