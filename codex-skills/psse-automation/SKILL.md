---
name: psse-automation
description: Build, debug, and review PSS/E automation for power flow, contingency, scaling, interconnection, and reporting workflows.
---

# PSS/E Automation

## Use this skill when

Use this skill for tasks involving PSS/E APIs, `.sav` cases, subsystem definitions, power-flow automation, contingency analysis, load or generation scaling, violation extraction, dynamic-study setup, or automated engineering reports.

## Objectives

- Preserve model integrity.
- Make study assumptions explicit.
- Check all PSS/E return codes.
- Keep PSS/E API calls isolated from engineering logic.
- Produce reproducible outputs.

## Required workflow

1. Identify the supported PSS/E version and Python environment.
2. Inspect existing wrappers and utilities before adding new API calls.
3. Load a copy of the source case; never overwrite the original.
4. Record case name, scenario, subsystem, dispatch assumptions, and criteria.
5. Apply changes incrementally and solve after meaningful modifications.
6. Check every `ierr` and raise a descriptive exception on failure.
7. Validate convergence and physical reasonableness.
8. Save outputs under new names with scenario metadata.
9. Add deterministic tests around logic that does not require PSS/E.
10. Summarize changes, validation, and unresolved risks.

## Engineering rules

- Preserve bus numbers, generator IDs, load IDs, and branch identifiers unless explicitly instructed.
- Do not silently change topology, status, ratings, areas, zones, or owners.
- Distinguish MW, Mvar, MVA, kV, per unit, degrees, and percentages.
- Distinguish normal and emergency ratings.
- Distinguish pre-contingency and post-contingency results.
- Confirm that subsystem and contingency definitions match the requested study.
- Do not treat successful API execution as proof of a valid engineering result.

## Preferred architecture

```text
src/
├── psse/
│   ├── environment.py
│   ├── api.py
│   ├── case.py
│   ├── solve.py
│   └── errors.py
├── studies/
│   ├── contingency.py
│   ├── scaling.py
│   └── interconnection.py
└── reporting/
    └── violations.py
```

Keep direct `psspy` calls in adapter or wrapper modules. Keep engineering calculations testable without PSS/E where practical.

## Error handling

```python
ierr = psspy.some_api_call(...)
if ierr != 0:
    raise PSSEError(
        f"PSS/E some_api_call failed with ierr={ierr}; case={case_name}"
    )
```

Never ignore nonzero return codes.

## Validation checklist

- Case loaded successfully.
- Base case converges.
- Modified case converges.
- Swing bus and islanding behavior are understood.
- Total load and generation changes match the request.
- Voltage and thermal criteria use correct ratings and units.
- Output counts and major violations are plausible.
- Original case remains unchanged.

## Completion format

Report:

- What changed
- Study assumptions
- Files modified
- Validation performed
- Engineering limitations
