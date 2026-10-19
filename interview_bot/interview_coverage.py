"""Explain partial links between posting language and saved STAR evidence."""

import re
from typing import Any

from interview_retrieval import JOB_FOCUSES, extract_tag_value


# Bounded related-language groups do not imply equivalent expertise. Software,
# agreement responsibilities, and each standards family remain independent.
TERM_FAMILIES = {
    "mentoring": ("mentor", "mentors", "mentored", "mentoring", "coached", "coaching"),
    "planning studies": (
        "planning study", "planning studies", "planning assessment", "planning assessments",
        "annual assessment", "annual transmission planning assessment", "power system studies",
        "transmission planning", "power system planning", "bulk electric system planning",
    ),
    "study assumptions": ("study assumptions", "study assumption", "assumptions"),
    "study scenarios": ("study scenarios", "study scenario", "scenarios", "scenario"),
    "contingencies": ("contingencies", "contingency", "contingency analysis", "contingency generation"),
    "sensitivities": ("sensitivities", "sensitivity analysis", "sensitivity development"),
    "transmission alternatives": ("mitigation alternatives", "transmission alternatives", "transmission solutions", "transmission upgrades"),
    "ERCOT requirements": (
        "ERCOT requirements", "ERCOT planning requirements", "ERCOT standards",
        "ERCOT procedures",
    ),
    "ERCOT protocols": ("ERCOT protocols", "Nodal Protocols"),
    "ERCOT planning guides": ("ERCOT planning guides", "ERCOT planning guide", "planning guides", "planning guide"),
    "NERC standards": ("NERC Reliability Standards", "NERC standards", "NERC requirements"),
    "TPL standard family": ("TPL",),
    "FAC standard family": ("FAC",),
    "MOD standard family": ("MOD",),
    "PSS/E": ("PSS/E", "PSSE", "PSS®E", "PSS®️E"),
    "TARA": ("PowerGEM TARA", "TARA"),
    "PowerWorld": ("PowerWorld",),
    "ASPEN OneLiner": ("ASPEN OneLiner", "OneLiner"),
    "PSCAD": ("PSCAD",),
    "TSAT": ("TSAT",),
    "GE PSLF": ("GE PSLF", "PSLF"),
    "UPLAN": ("UPLAN",),
    "PROMOD": ("PROMOD",),
    "GRIDVIEW": ("GRIDVIEW",),
    "consultant agreements": (
        "consultant and subcontractor agreements", "consultant agreements",
        "subcontractor agreements", "consultant contracts", "subcontractor contracts",
    ),
    "interconnection agreements": ("interconnection agreement", "interconnection agreements"),
    "interconnection studies": ("interconnection studies", "interconnection study", "interconnections", "interconnection"),
    "large load": ("large load", "large loads"),
    "steady-state analysis": ("steady state", "steady state analysis"),
    "dynamic analysis": ("dynamic analysis", "dynamic studies", "dynamic stability", "dynamic simulations"),
    "power-flow analysis": ("power flow", "load flow", "power system analysis"),
    "consultant coordination": ("consultant", "consultants"),
    "stakeholder coordination": ("stakeholder", "stakeholders", "stakeholder coordination"),
    "deadlines": ("deadline", "deadlines", "schedule", "schedules"),
    "parallel project delivery": (
        "multiple concurrent", "concurrent projects", "multiple projects", "multiple complex projects",
        "multiple deadlines", "several interconnection studies",
    ),
    "prioritization": ("prioritization", "priorities", "prioritized", "prioritizing"),
    "delegation": ("delegation", "delegated", "delegating", "assigned work", "assigned tasks"),
    "technical recommendations": ("technical recommendations", "recommendations", "recommended solutions"),
    "technical presentations": ("present findings", "presented findings", "presented the identified needs", "technical presentations"),
}


def _normalized(value: str) -> str:
    """Normalize labels without discarding the original source text."""
    value = re.sub(r"pss[\s/®\ufe0f]*e", "psse", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", re.sub(r"[-\u2010-\u2015]", " ", value.lower())).strip()


def _pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile explicit aliases with flexible hyphens and PSS/E typography."""
    expressions = []
    for term in sorted(set(terms), key=len, reverse=True):
        if _normalized(term) == "psse":
            expressions.append(r"pss[\s/®\ufe0f]*e")
        else:
            expressions.append(r"[\s\-\u2010-\u2015]+".join(re.escape(part) for part in _normalized(term).split()))
    return re.compile(r"(?<!\w)(?:" + "|".join(expressions) + r")(?!\w)", re.IGNORECASE)


def _families() -> dict[str, re.Pattern[str]]:
    """Retain existing specific role vocabulary alongside the explicit aliases."""
    families = dict(TERM_FAMILIES)
    grouped = {_normalized(term) for terms in families.values() for term in terms}
    for profile in JOB_FOCUSES.values():
        for term in profile["keywords"]:
            normalized = _normalized(term)
            if normalized not in grouped and normalized not in {"project", "projects", "contract", "contracts"}:
                families[normalized] = (term,)
    return {name: _pattern(terms) for name, terms in families.items()}


FAMILY_PATTERNS = _families()


def _row_kind(line: str) -> str:
    """Classify explicit headings and context without hiding unfamiliar requirements."""
    value = _normalized(line).strip(" •\t:")
    if value in {
        "position specific description", "key responsibilities", "responsibilities",
        "qualifications", "required qualifications", "preferred qualifications",
        "job description", "the successful candidate must possess",
    }:
        return "heading"
    if re.match(r"(?:travel\b|must be willing to travel\b|ideal candidate will sit\b|other locations\b|(?:work )?location:)", value):
        return "logistics"
    if not re.search(r"\b(?:must|required|experience|candidate|candidates)\b", value) and re.match(
        r"(?:requisition id\b|nextera energy transmission.*\bis the\b|"
        r"we develop, finance, construct, operate and maintain transmission assets\b|"
        r"we operate through our regional subsidiaries\b|do you want to be part\b|join our\b|"
        r"perform other job related duties as assigned\b)", value,
    ):
        return "context"
    return "requirement"


def _snippet(value: str, match: re.Match[str]) -> str:
    """Return the exact saved sentence containing a matched evidence term."""
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|\n+", value):
        if boundary.start() >= match.end():
            return value[start:boundary.start()].strip()
        if boundary.end() <= match.start():
            start = boundary.end()
    return value[start:].strip()


def _story_terms(item: dict) -> dict[str, dict[str, Any]]:
    """Find related terms only in saved STAR fields, including retained variants."""
    found: dict[str, dict[str, Any]] = {}
    variants = item.get("variants") or [item["candidate"]["text"]]
    sources = item.get("sources", [])
    for variant_index, text in enumerate(variants):
        for field in ("Situation", "Task", "Action", "Result"):
            value = extract_tag_value(text, field)
            for family, pattern in FAMILY_PATTERNS.items():
                if family in found:
                    continue
                match = pattern.search(value)
                if match:
                    found[family] = {
                        "family": family,
                        "evidence_term": match.group(),
                        "evidence_field": field,
                        "evidence_snippet": _snippet(value, match),
                        "variant_index": variant_index,
                        "source_text_sha256": sources[variant_index].get("text_sha256", "") if variant_index < len(sources) else "",
                    }
    return found


def coverage_map(description: str, library: list[dict]) -> list[dict[str, Any]]:
    """Link posting phrases to exact saved facts while retaining gaps and context.

    Matches are partial language evidence, not verification of accomplishments,
    standards compliance, hands-on skill, or a complete job requirement. Unknown
    requirements remain visible, and each software/contract topic is independent.
    """
    evidence = {item["id"]: _story_terms(item) for item in library}
    rows = []
    for original_line in description.splitlines():
        line = original_line.strip()
        if not line:
            continue
        kind = _row_kind(line)
        terms = {
            family: match.group() for family, pattern in FAMILY_PATTERNS.items()
            if (match := pattern.search(line))
        } if kind == "requirement" else {}
        matches = []
        supported = set()
        for item in library:
            shared = set(terms) & set(evidence[item["id"]])
            if shared:
                supported.update(shared)
                links = [dict(evidence[item["id"]][family], posting_term=terms[family]) for family in sorted(shared)]
                matches.append({
                    "story_id": item["id"],
                    "matched_phrases": sorted({terms[family] for family in shared}),
                    "evidence_links": links,
                })
        matches.sort(key=lambda match: len(match["evidence_links"]), reverse=True)
        if kind == "requirement":
            status = "Partial related evidence; review required" if matches else "No related phrase found; review manually"
            note = (
                "These links support only the displayed phrases. Review the full requirement and "
                "source; they do not establish complete coverage, ownership, proficiency, or compliance. "
                "Unmatched terms are recognized topics without a saved phrase match, not proof of missing experience."
            )
        else:
            status = {"heading": "Posting heading", "context": "Posting context", "logistics": "Logistics; confirm separately"}[kind]
            note = "Retained for context; no engineering-story match is inferred."
        rows.append({
            "posting_text": line,
            "status": status,
            "matches": matches,
            "row_kind": kind,
            "unmatched_terms": sorted({terms[family] for family in set(terms) - supported}),
            "review_note": note,
        })
    return rows
