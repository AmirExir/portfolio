"""Test explainable posting matches and visible gaps from the NextEra report."""

import unittest

from interview_evidence import build_story_library, coverage_map
from interview_retrieval import extract_tag_value


def saved_story(action: str, situation: str = "Our transmission planning team needed support.") -> dict[str, str]:
    """Create a bounded STAR fixture with no unrelated implied qualifications."""
    return {
        "response_type": "story",
        "text": (
            "Response Type: Story | Principle: Ownership | Question: Describe your work. | "
            f"Situation: {situation} | Task: I supported the team. | Action: {action} | "
            "Result: I documented the findings."
        ),
    }


class CoverageEvidenceTests(unittest.TestCase):
    def test_mentoring_aliases_explain_the_exact_saved_phrase(self) -> None:
        for phrase in ("mentored", "mentoring", "coached", "coaching"):
            with self.subTest(evidence=phrase):
                source = saved_story(f"I {phrase} engineers through practical study reviews.")
                library = build_story_library([source])
                row = coverage_map("Mentor and develop transmission planners.", library)[0]
                self.assertEqual(row["status"], "Partial related evidence; review required")
                link = row["matches"][0]["evidence_links"][0]
                self.assertEqual(link["posting_term"], "Mentor")
                self.assertEqual(link["evidence_term"], phrase)
                self.assertEqual(link["evidence_field"], "Action")
                self.assertIn(link["evidence_snippet"], source["text"])
                self.assertEqual(link["source_text_sha256"], library[0]["sources"][0]["text_sha256"])

    def test_study_preparation_terms_link_to_actual_saved_details(self) -> None:
        source = saved_story(
            "I checked study assumptions, developed scenarios and contingencies, performed "
            "sensitivity analysis, and compared transmission alternatives."
        )
        posting = "Develop study assumptions, scenarios, contingencies, sensitivities, mitigation alternatives, and technical recommendations."
        row = coverage_map(posting, build_story_library([source]))[0]
        links = row["matches"][0]["evidence_links"]
        self.assertEqual({link["family"] for link in links}, {
            "study assumptions", "study scenarios", "contingencies", "sensitivities", "transmission alternatives",
        })
        self.assertIn("technical recommendations", row["unmatched_terms"])
        for link in links:
            self.assertIn(link["posting_term"], posting)
            self.assertIn(link["evidence_term"], link["evidence_snippet"])
            self.assertIn(link["evidence_snippet"], extract_tag_value(source["text"], link["evidence_field"]))

    def test_planning_study_wording_finds_saved_assessment(self) -> None:
        source = saved_story("I completed the annual assessment and documented assumptions.")
        row = coverage_map(
            "Identify ERCOT system needs and transmission drivers, reproduce or validate relevant planning studies, and develop robust transmission solutions.",
            build_story_library([source]),
        )[0]
        self.assertTrue(row["matches"])
        self.assertIn("transmission solutions", row["unmatched_terms"])
        self.assertEqual(row["status"], "Partial related evidence; review required")

    def test_psse_typography_preserves_both_original_terms(self) -> None:
        for posting_term, evidence_term in (("PSS®E", "PSS/E"), ("PSS®️E", "PSSE"), ("PSS/E", "PSS®E")):
            with self.subTest(posting=posting_term, evidence=evidence_term):
                library = build_story_library([saved_story(f"I used {evidence_term} for analysis.")])
                row = coverage_map(f"Hands-on experience with {posting_term}.", library)[0]
                link = row["matches"][0]["evidence_links"][0]
                self.assertEqual((link["posting_term"], link["evidence_term"]), (posting_term, evidence_term))

    def test_specific_software_gaps_are_not_hidden_by_tara_match(self) -> None:
        source = saved_story("I used PSS/E, TARA, and ASPEN OneLiner for the assessment.")
        row = coverage_map(
            "Hands-on experience with relevant planning tools, such as PSS®E, PowerGEM TARA, PowerWorld, ASPEN OneLiner, PSCAD, TSAT, GE PSLF, UPLAN, PROMOD, or GRIDVIEW.",
            build_story_library([source]),
        )[0]
        self.assertEqual(set(row["unmatched_terms"]), {"PowerWorld", "PSCAD", "TSAT", "GE PSLF", "UPLAN", "PROMOD", "GRIDVIEW"})
        self.assertIn("Partial", row["status"])
        self.assertIn("proficiency", row["review_note"])

    def test_consultant_work_does_not_establish_contract_ownership(self) -> None:
        source = saved_story("I worked with consultants and reviewed their analysis.")
        row = coverage_map(
            "Structure, execute, and manage consultant and subcontractor agreements needed to support transmission planning analyses.",
            build_story_library([source]),
        )[0]
        self.assertTrue(row["matches"])
        self.assertIn("consultant and subcontractor agreements", row["unmatched_terms"])
        self.assertFalse(any(link["family"] == "consultant agreements" for match in row["matches"] for link in match["evidence_links"]))

    def test_standards_families_remain_separate(self) -> None:
        source = saved_story("I developed the assessment using NERC TPL-001 requirements.")
        row = coverage_map("Apply NERC Reliability Standards, including TPL, FAC, and MOD.", build_story_library([source]))[0]
        self.assertEqual({link["family"] for link in row["matches"][0]["evidence_links"]}, {"TPL standard family"})
        self.assertTrue({"FAC", "MOD"}.issubset(row["unmatched_terms"]))

    def test_ercot_mention_does_not_establish_protocol_or_guide_experience(self) -> None:
        source = saved_story("I coordinated with ERCOT about project status.")
        row = coverage_map(
            "Ensure work products comply with applicable ERCOT protocols, planning guides, NERC Reliability Standards, and company requirements.",
            build_story_library([source]),
        )[0]
        self.assertEqual(row["matches"], [])
        self.assertEqual(set(row["unmatched_terms"]), {"ERCOT protocols", "planning guides", "NERC Reliability Standards"})

    def test_context_logistics_headers_and_unknown_requirements_are_retained(self) -> None:
        posting = "\n".join((
            "Requisition ID: 97852", "Position Specific Description", "Key Responsibilities:",
            "We operate through our regional subsidiaries to integrate generation resources and enhance grid reliability.",
            "Travel approximately 10%–25%.", "Must speak Japanese.", "Must speak Japanese.",
            "Other locations, including remote, may be considered for qualified candidates.",
        ))
        rows = coverage_map(posting, [])
        self.assertEqual([row["posting_text"] for row in rows], posting.splitlines())
        self.assertEqual([row["row_kind"] for row in rows], ["context", "heading", "heading", "context", "logistics", "requirement", "requirement", "logistics"])
        for row in rows[5:7]:
            self.assertEqual(row["matches"], [])
            self.assertIn("review manually", row["status"])

    def test_context_prefix_does_not_hide_a_requirement_later_in_same_line(self) -> None:
        row = coverage_map("We operate in Texas. Candidates must speak Japanese.", [])[0]
        self.assertEqual(row["row_kind"], "requirement")
        self.assertIn("review manually", row["status"])

    def test_unfamiliar_we_develop_responsibility_remains_a_requirement(self) -> None:
        row = coverage_map("We develop planning studies using PSCAD.", [])[0]
        self.assertEqual(row["row_kind"], "requirement")
        self.assertEqual(set(row["unmatched_terms"]), {"planning studies", "PSCAD"})

    def test_only_saved_star_facts_establish_a_match(self) -> None:
        source = saved_story("I organized records.", "Records were incomplete.")
        source["text"] = source["text"].replace("Describe your work.", "Describe your PSCAD and XGBoost work.")
        source["text"] += " | Aliases: Experience with PSCAD | Answer: I would use PSCAD. | References: XGBoost"
        rows = coverage_map("Use PSCAD and XGBoost.\nLead model work.", build_story_library([source]))
        self.assertTrue(all(not row["matches"] for row in rows))

    def test_matching_variant_is_identified_instead_of_implying_first_variant(self) -> None:
        first = saved_story("I organized the records.")
        second = saved_story("I coached engineers through study reviews.")
        library = build_story_library([first, second])
        self.assertEqual(len(library), 1)
        link = coverage_map("Mentor engineers.", library)[0]["matches"][0]["evidence_links"][0]
        self.assertEqual(link["variant_index"], 1)
        self.assertIn(link["evidence_snippet"], second["text"])


if __name__ == "__main__":
    unittest.main()
