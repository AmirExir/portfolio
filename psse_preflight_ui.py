"""Shared, explicit static-review panel for both PSS/E assistant entry points."""
import json

import streamlit as st

from psse_preflight import python_blocks, review_code


def render_preflight(latest_answer: str = "") -> None:
    """Keep a review report available across Streamlit reruns; never execute code."""
    with st.expander("Review generated PSS/E code", expanded=False):
        st.caption("Static checks only. Specify your installed version; PSS/E is not run by this panel.")
        with st.form("psse_preflight_form"):
            version = st.text_input("Installed PSS/E version", placeholder="Exact version from your installation")
            source = st.text_area("Source case paths (one per line)")
            code = st.text_area("Python code to review (leave empty to use the latest answer)", height=180)
            catalog = st.file_uploader("Version-specific signature catalog (optional JSON)", type=["json"])
            run = st.form_submit_button("Run static checks")
        if run:
            st.session_state.pop("psse_preflight_reports", None)
            try:
                if catalog is not None and catalog.size > 2_000_000:
                    raise ValueError("Signature catalog exceeds 2 MB.")
                specs = json.loads(catalog.getvalue()) if catalog else None
                blocks = python_blocks(code or latest_answer)
                if not blocks or not any(block.strip() for block in blocks):
                    raise ValueError("Paste Python code or first generate an answer with a Python code block.")
                st.session_state.psse_preflight_reports = [review_code(
                    block, psse_version=version, catalog=specs,
                    source_cases=[line.strip() for line in source.splitlines() if line.strip()]
                ) for block in blocks]
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Static review could not run: {exc}")
        reports = st.session_state.get("psse_preflight_reports")
        if reports:
            st.warning("Static review only; engineering and runtime behavior remain unverified.")
            for index, report in enumerate(reports, 1):
                st.write(f"Python block {index}")
                st.caption(f"Reviewed source SHA-256: {report['code_sha256']}")
                st.dataframe(report["findings"], hide_index=True)
            st.download_button("Download static review", json.dumps(reports, indent=2),
                               "psse_static_review.json", "application/json")
