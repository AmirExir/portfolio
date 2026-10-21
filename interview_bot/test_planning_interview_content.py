"""Verify the imported planning preparation and retrieval without external APIs."""

import hashlib
import json
from pathlib import Path
import unittest

from embedding_utils import plan_embedding_reuse
from interview_retrieval import build_candidate, extract_tag_value, rank_candidates
from interview_services import answer_messages, practice_messages
from sync_manager_content import build_chunks, serialize_content


BASE_DIR = Path(__file__).resolve().parent
SOURCE_ID = "private:transmission-planning-interview-2026-10-19"
SOURCE_SHA256 = "4a453873e0c6b555248c4bd715738a5b789513e2e603b9efbd871305b2896068"
REVISION_ID = "private:transmission-planning-director-revision-2026-10-20"
REVISION_SHA256 = "f0c4325b6c752a553c777ff48f9c4c998de70ea939472d4699a15bd46a48ab1b"
REVISED_NUMBERS = set(range(1, 21))
Q14_REVISION_ID = "private:q14-weak-grid-investment-2026-10-20"
Q14_REVISION_SHA256 = "c637a1441eb077a52cd26068be93790cb5ded930fe30134c735a47a86edcef94"
SPECIAL_REVISIONS = {
    1: ("private:q1-batch-zero-batch1-ctp-2026-10-21", "8ee0ebd796062fcf0d28fd0e74ce5aa4ab46ccd4a81516209643566faae66cd4"),
    14: (Q14_REVISION_ID, Q14_REVISION_SHA256),
    15: ("private:transmission-planning-disagreement-2026-10-20", "115e460060f804e0de940e1331ecda70a43fbf3038501d2150a9c022adff3229"),
    20: ("private:transmission-planning-stakeholders-2026-10-20", "020fc4351e1090439ab59cc239f409def2111b4fe87169ce3a69418704a51350"),
}
BACKUP_ID = "planning-backup-15-ratings"
STORY_NUMBERS = {12, 13, 15, 16, 17, 18, 19}


def answer_digest(text: str) -> str:
    """Fingerprint original wording without publishing another source copy."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


class PlanningInterviewContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_hashes = json.loads(
            (BASE_DIR / "fixtures" / "planning_source_hashes.json").read_text(encoding="utf-8")
        )
        cls.original = {int(number): hashes for number, hashes in cls.source_hashes["questions"].items()}
        cls.content = json.loads((BASE_DIR / "manager_content.json").read_text(encoding="utf-8"))
        cls.active_chunks = json.loads((BASE_DIR / "chunks_cleaned.json").read_text(encoding="utf-8"))
        cls.target_chunks, _, _ = build_chunks(cls.active_chunks, cls.content)
        cls.imported = {
            item["source_question"]: item for item in cls.content
            if item["id"].startswith("planning-interview-")
        }
        cls.target_by_id = {chunk.get("source_id"): chunk for chunk in cls.target_chunks}

    def test_source_fingerprints_do_not_require_published_drafts(self) -> None:
        self.assertEqual(self.source_hashes["source_sha256"], SOURCE_SHA256)
        self.assertEqual(set(self.original), set(range(1, 21)))
        for path in BASE_DIR.rglob("transmission_planning_interview*.txt"):
            self.assertEqual(path.relative_to(BASE_DIR).parts[0], ".private")

    def test_original_questions_and_source_fingerprints_are_preserved(self) -> None:
        self.assertEqual(set(self.imported), set(self.original))
        for number, original in self.original.items():
            with self.subTest(question=number):
                item = self.imported[number]
                self.assertEqual(item["id"], f"planning-interview-{number:02d}")
                self.assertEqual(answer_digest(item["question"]), original["question_sha256"])
                self.assertTrue(item["preparation_note"].strip())
                self.assertEqual(item["source_sha256"], SOURCE_SHA256)
                self.assertEqual(item["source_document"], SOURCE_ID)
                expected_type = "story" if number in STORY_NUMBERS else "direct"
                self.assertEqual(item["response_type"], expected_type)
                if expected_type == "story":
                    for field in ("situation", "task", "action", "result"):
                        self.assertTrue(item[field].strip(), field)
                    self.assertIn("verify", item["source_status"].lower())
                else:
                    self.assertIn("I would", item["answer"])

    def test_revised_answers_trace_editorial_changes_to_each_draft(self) -> None:
        revised_numbers = {number for number, item in self.imported.items() if "answer_revision" in item}
        self.assertEqual(revised_numbers, REVISED_NUMBERS)
        for number in sorted(REVISED_NUMBERS):
            with self.subTest(question=number):
                item = self.imported[number]
                revision = item["answer_revision"]
                revision_id, revision_hash = SPECIAL_REVISIONS.get(number, (REVISION_ID, REVISION_SHA256))
                self.assertEqual(revision["source_document"], revision_id)
                self.assertEqual(revision["source_question"], number)
                self.assertEqual(revision["source_sha256"], revision_hash)
                self.assertTrue(revision["editorial_note"].strip())
                self.assertNotEqual(answer_digest(item["answer"]), self.original[number]["answer_sha256"])

    def test_story_answers_use_the_rewritten_star_narrative(self) -> None:
        for number in sorted(STORY_NUMBERS):
            with self.subTest(question=number):
                item = self.imported[number]
                narrative = "\n\n".join(
                    f"**{field.title()}**\n{item[field]}" for field in ("situation", "task", "action", "result")
                )
                self.assertEqual(item["answer"], narrative)

    def test_metadata_fields_do_not_leak_into_answers_or_star_sections(self) -> None:
        for number, item in self.imported.items():
            with self.subTest(question=number):
                text = self.target_by_id[item["id"]]["text"]
                for tag, field in (
                    ("Question", "question"), ("Answer", "answer"),
                    ("Source Document", "source_document"), ("Source Status", "source_status"),
                    ("Source SHA256", "source_sha256"),
                ):
                    self.assertEqual(extract_tag_value(text, tag), item[field].strip())
                self.assertEqual(extract_tag_value(text, "Source Question"), str(number))
                self.assertEqual(extract_tag_value(text, "References"), "\n".join(item.get("references", [])))
                self.assertEqual(extract_tag_value(text, "Preparation Note"), item.get("preparation_note", ""))
                verification = extract_tag_value(text, "Verification").lower()
                self.assertIn("user-provided", verification)
                self.assertIn("not been independently verified", verification)
                self.assertNotIn("answer revision", verification)
                revision_text = extract_tag_value(text, "Answer Revision")
                if "answer_revision" in item:
                    revision = item["answer_revision"]
                    self.assertIn(revision["source_document"], revision_text)
                    self.assertIn(f"question {number}", revision_text)
                    self.assertIn(revision["source_sha256"], revision_text)
                    self.assertIn(revision["editorial_note"], revision_text)
                else:
                    self.assertEqual(revision_text, "")
                if number in STORY_NUMBERS:
                    for field in ("situation", "task", "action", "result"):
                        self.assertEqual(extract_tag_value(text, field.title()), item[field].strip())
                if item.get("aliases"):
                    self.assertEqual(extract_tag_value(text, "Aliases"), "; ".join(item["aliases"]))

    def test_material_experience_qualifiers_remain_in_evidence(self) -> None:
        self.assertIn("potential", self.imported[13]["result"].lower())
        self.assertIn("$300,000", self.imported[13]["answer"])
        self.assertIn("before internal effort", self.imported[13]["result"])
        self.assertIn("no ERCOT endorsement", self.imported[13]["preparation_note"])
        self.assertIn("rather than a project that had to be canceled", self.imported[17]["answer"])
        self.assertIn("not a technically infeasible or canceled interconnection", self.imported[17]["preparation_note"])
        self.assertIn("Those were his achievements", self.imported[19]["result"])

    def test_process_answer_covers_case_building_tracking_and_review_paths(self) -> None:
        """The saved Q1 response must include the requested end-to-end process."""
        answer = extract_tag_value(self.target_by_id["planning-interview-01"]["text"], "Answer")
        for detail in (
            "Steady State Working Group (SSWG)", "Dynamics Working Group (DWG)",
            "Planning Working Group (PLWG)", "contingency database", "six years",
            "20 years", "TSPs also perform their own planning studies", "thermal overloads",
            "voltage violations", "GINR", "RIOO", "(MIS)", "(GIS)",
            "Transmission Project and Information Tracking (TPIT)", "Model On Demand (MOD)",
            "three times a year", "60 kV", "reporting exclusions", "lifecycle cost",
            "Certificate of Convenience and Necessity (CCN)", "PUCT", "commissioning",
        ):
            with self.subTest(detail=detail):
                self.assertIn(detail, answer)
        for tier in range(1, 5):
            self.assertIn(f"| **Tier {tier}** |", answer)
        self.assertIn("$200 million", answer)
        self.assertIn("$50 million", answer)
        self.assertIn("TAC review and comment", answer)
        self.assertIn("ERCOT Board endorsement", answer)
        self.assertIn("not every Tier 4 project is neutral", answer)
        self.assertIn("does not by itself constitute project endorsement", answer)

    def test_process_answer_keeps_effective_and_developing_processes_distinct(self) -> None:
        """Preserve dated process status and QSA's actual engineering function."""
        item = self.imported[1]
        answer = item["answer"]
        for detail in (
            "July 10, 2026", "Batch Zero", "October 21, 2026",
            "Comprehensive Transmission Planning (CTP)", "remain under stakeholder development",
            "Full Interconnection Study (FIS)", "Quarterly Stability Assessment (QSA)",
            "additional stability assessment", "generator initial synchronization",
            "large-load initial energization eligibility", "remaining commissioning",
        ):
            with self.subTest(detail=detail):
                self.assertIn(detail, answer)
        references = "\n".join(item["references"])
        for source in ("NPRR1274", "M-B062326-04", "M-D070826-01", "§5.3.5", "§6.4"):
            self.assertIn(source, references)
        self.assertIn("PGRR146", item["preparation_note"])
        self.assertIn("not treated as effective", item["preparation_note"])
        self.assertIn("do not establish its disposition conclusively", item["preparation_note"])

    def test_batch_transition_keeps_allocation_commitment_and_proposals_distinct(self) -> None:
        """Prevent a future process or capacity allocation becoming an approval claim."""
        item = self.imported[1]
        answer = item["answer"]
        for detail in (
            "Batch Zero — the implemented transition", "conditional classification does not establish final inclusion",
            "steady-state analysis and stability screening", "Load Commissioning Plan (LCP)",
            "refinement of the transmission upgrades for committed loads",
            "Batch 1+ — the recurring process being developed", "unmet load ramps",
            "Load Integration & Ongoing Operations (LIOO)", "planned portal capabilities",
            "standalone Batch → customer commitment → full reliability studies",
            "full NERC TPL and ERCOT reliability criteria", "annual, overlapping cycles",
            "Project Transparency and Alternatives (PTA)", "RPG 2.0",
            "today's RPG tier requirements have already been replaced",
            "entering a batch does not guarantee the requested MW or energization date",
        ):
            with self.subTest(detail=detail):
                self.assertIn(detail, answer)
        self.assertIn("These are proposed changes", answer)
        self.assertIn("Batch 1+ and CTP remain under stakeholder development", answer)
        self.assertNotIn("Preliminary Transmission Assessment", answer)
        references = "\n".join(item["references"])
        self.assertIn("20261015-ERCOT-CTP-Batch-workshop_3a_Batch_v02.pptx", references)
        self.assertIn("ERCOT-CTP-Workshop-3b-20261020.pptx", references)
        self.assertIn("proposed Batch 1+/CTP", item["quick_view"]["cue"])

    def test_assigned_episodes_are_not_reused_across_questions(self) -> None:
        """Check narrative content as well as unique assignment labels."""
        scenario_ids = [item["scenario_id"] for item in self.imported.values()]
        self.assertEqual(len(set(scenario_ids)), 20)
        anchors = {
            12: "AELAB", 13: "$300,000", 14: "minimax regret", 15: "Highway 29", 16: "backlog",
            17: "seven capacitor", 18: "DYRE", 19: "P.E. license", 20: "alignment matrix",
        }
        for number, phrase in anchors.items():
            with self.subTest(episode=number):
                matches = [n for n, item in self.imported.items() if phrase.casefold() in item["answer"].casefold()]
                self.assertEqual(matches, [number])
        for number, item in self.imported.items():
            with self.subTest(question=number):
                self.assertEqual(item["answer_set"], "ercot-planning-20")
                self.assertEqual(int(item["question_key"].split()[0]), number)
                self.assertGreater(len(item["answer"].split()), 450)

    def test_new_personal_facts_and_decision_boundaries_survive_in_saved_answers(self) -> None:
        """Keep user-confirmed outcomes distinct from proposed controls and approvals."""
        leadership = self.imported[13]["answer"]
        for detail in ("HRML", "South Texas GTC", "above the applicable allowed limit", "redispatched generation", "rerun was stable", "required stress cases"):
            self.assertIn(detail, leadership)

        disagreement = self.imported[15]["answer"]
        for detail in ("native-load growth", "neighboring utility", "one substation and two lines", "two substations and three lines", "ownership", "ERCOT 3", "autotransformer outage plus another outage", "Muscovy", "Another stakeholder questioned", "ERCOT also preferred", "RPG review"):
            self.assertIn(detail, disagreement)
        self.assertNotIn("tap bus", disagreement)
        self.assertNotIn("I also objected to eliminating", disagreement)

        error = self.imported[18]["answer"]
        for detail in ("greater than 10%", "initially missed", "DYRE", "proportional-integral (PI)", "phase-locked-loop (PLL)", "resolved the discrepancy on rerun"):
            self.assertIn(detail, error)
        self.assertNotIn("Lost Pines", error)

        stakeholders = self.imported[20]
        self.assertEqual(stakeholders["story_basis"], "illustrative_scenario")
        for detail in ("The Development Team", "The Major Customer", "ERCOT", "CapEx", "alignment matrix", "PLEXOS", "phased commitment", "Biweekly"):
            self.assertIn(detail, stakeholders["answer"])
        self.assertNotIn("ICCP", stakeholders["answer"])
        self.assertNotIn("operational transition", stakeholders["answer"])

    def test_rating_disagreement_is_only_an_explicit_q15_backup(self) -> None:
        backup = next(item for item in self.content if item["id"] == BACKUP_ID)
        self.assertEqual(backup["backup_for"], self.imported[15]["id"])
        self.assertNotIn("answer_set", backup)
        self.assertNotIn("question_key", backup)
        self.assertIn("tap bus", backup["answer"])
        for query in (backup["question"], *backup["aliases"]):
            with self.subTest(query=query):
                ranked = rank_candidates(query, self.target_chunks, {}, focus="Transmission planning")
                self.assertEqual([candidate["source_id"] for candidate in ranked], [BACKUP_ID])
                self.assertTrue(ranked[0]["assigned_match"])
                self.assertEqual(extract_tag_value(ranked[0]["text"], "Answer"), backup["answer"])

    def test_mentoring_answer_explains_diagnosis_and_accountable_delegation(self) -> None:
        """Keep the user's requested teaching method in the actual answer context."""
        item = self.imported[19]
        candidate = build_candidate(1, 0, self.target_by_id[item["id"]])
        context = answer_messages(item["question"], candidate, "")[1]["content"]
        action = extract_tag_value(candidate["text"], "Action")
        for concept in (
            "why a violation occurred", "redistributed power", "limiting facility rating",
            "reactive reserves", "PV/QV", "credible alternatives", "delegated",
            "acceptance criteria", "review points", "final decision",
        ):
            with self.subTest(concept=concept):
                self.assertIn(concept.casefold(), action.casefold())
                self.assertIn(concept.casefold(), context.casefold())
        self.assertIn("earned his P.E. license", item["result"])
        self.assertIn("promoted to Engineer", item["result"])
        self.assertNotIn("within a year", item["answer"])
        for concept in ("Train", "Trust", "Automate", "Gate", "strengths", "interests", "passionate", "negotiable", "non-negotiable"):
            self.assertIn(concept, item["answer"])
        for item in self.imported.values():
            self.assertNotIn("I still need to confirm", item["answer"])
            self.assertNotIn("My preparation record", item["answer"])

    def test_every_authored_question_and_alias_retrieves_its_primary_and_only_linked_backups(self) -> None:
        for number, item in self.imported.items():
            for query in (item["question"], *item.get("aliases", [])):
                with self.subTest(question=number, query=query):
                    ranked = rank_candidates(query, self.target_chunks, {}, focus="Transmission planning")
                    expected_ids = [item["id"], BACKUP_ID] if number == 15 else [item["id"]]
                    self.assertEqual([candidate["source_id"] for candidate in ranked], expected_ids)
                    self.assertTrue(ranked[0]["assigned_match"])
                    self.assertEqual(self.target_chunks[ranked[0]["index"]].get("source_id"), item["id"])
                    self.assertEqual(extract_tag_value(ranked[0]["text"], "Answer"), item["answer"])
                    if number in STORY_NUMBERS:
                        for field in ("situation", "task", "action", "result"):
                            self.assertEqual(extract_tag_value(ranked[0]["text"], field.title()), item[field])

    def test_representative_queries_find_the_new_material(self) -> None:
        queries = {
            1: "Walk us through the ERCOT transmission planning process",
            4: "1,000 MW solar-plus-storage interconnection technical studies",
            5: "750 MW hyperscale data center interconnection energization schedule",
            18: "Tell us about a time you identified a significant technical error or modeling inconsistency",
            20: "ERCOT customer development team conflicting priorities scope cost schedule",
        }
        for number, query in queries.items():
            with self.subTest(question=number):
                ranked = rank_candidates(query, self.target_chunks, {}, focus="Transmission planning")
                ids = [self.target_chunks[candidate["index"]].get("source_id") for candidate in ranked]
                self.assertIn(self.imported[number]["id"], ids)

    def test_exact_story_alias_precedes_a_more_general_planning_story(self) -> None:
        question = "Tell me about a time you resolved a technical planning disagreement."
        self.assertIn(question, self.imported[15]["aliases"])
        for scores in ({}, dict.fromkeys(range(len(self.target_chunks)), 0.5)):
            with self.subTest(semantic_scores=bool(scores)):
                ranked = rank_candidates(question, self.target_chunks, scores, focus="Transmission planning")
                self.assertEqual(self.target_chunks[ranked[0]["index"]].get("source_id"), self.imported[15]["id"])

    def test_investment_answer_preserves_user_wording_and_separate_review_notes(self) -> None:
        item = self.imported[14]
        chunk = self.target_by_id[item["id"]]
        for focus in ("All roles", "Transmission planning", "Project management"):
            with self.subTest(focus=focus):
                ranked = rank_candidates(item["question"], [chunk], {0: 0.99}, focus=focus)
                self.assertEqual([candidate["index"] for candidate in ranked], [0])
        self.assertEqual(item["response_type"], "direct")
        self.assertEqual(item["story_basis"], "illustrative_scenario")
        for field in ("situation", "task", "action", "result"):
            self.assertNotIn(field, item)
        # Fingerprint the authored Markdown without publishing a second copy.
        self.assertEqual(hashlib.sha256((item["answer"] + "\n").encode("utf-8")).hexdigest(), Q14_REVISION_SHA256)
        for detail in (
            "Minimax Regret", "TARA", "PSS®E", "PSCAD", "PLEXOS", "750 MW", "STATCOM", "HTLS",
        ):
            self.assertIn(detail, item["answer"])
        self.assertIn("preserved verbatim", item["source_status"])
        self.assertIn("overstatements", item["source_status"])
        for qualification in (
            "not universally conservative", "not missing dynamic behavior",
            "conditional estimates", "separate from planning compliance",
            "cannot shift all grid-reliability obligations",
        ):
            self.assertIn(qualification, item["preparation_note"])
        for unsupported in ("Waterloo", "$120 Million", "mandatory 10%"):
            self.assertNotIn(unsupported, item["answer"])
        self.assertNotIn("Waterloo", " ".join(item["aliases"]))
        self.assertNotIn("How did you evaluate upgrading versus replacing the GE EMS?", item["aliases"])

    def test_incremental_plan_reuses_all_unchanged_evidence(self) -> None:
        # Simulate the previous published corpus even after this batch is indexed.
        introduced = {item["id"] for item in self.imported.values()}
        introduced.add("story-pec-annual-transmission-assessment")
        introduced.add(BACKUP_ID)
        previous = [chunk for chunk in self.active_chunks if chunk.get("source_id") not in introduced]
        indices, missing = plan_embedding_reuse(previous, self.target_chunks)
        expected_new = {chunk["text"].strip() for chunk in self.target_chunks if chunk.get("source_id") in introduced}
        self.assertEqual(len(expected_new), 22)
        self.assertEqual(set(missing), expected_new)
        for chunk, index in zip(self.target_chunks, indices):
            if chunk.get("source_id") not in introduced:
                self.assertLess(index, len(previous))
                self.assertEqual(previous[index]["text"].strip(), chunk["text"].strip())
        synchronized, _, _ = build_chunks(self.target_chunks, self.content)
        self.assertEqual(synchronized, self.target_chunks)
        _, missing_after_sync = plan_embedding_reuse(synchronized, self.target_chunks)
        self.assertEqual(missing_after_sync, [])

    def test_refresh_reuses_vectors_for_unchanged_published_content(self) -> None:
        previous = self.active_chunks
        previous_by_id = {chunk.get("source_id"): chunk for chunk in previous}
        indices, missing = plan_embedding_reuse(previous, self.target_chunks)
        # Text edits remain separate from the validated saved embedding corpus.
        planning_ids = {f"planning-interview-{number:02d}" for number in range(1, 21)}
        planning_ids.add(BACKUP_ID)
        expected_changed = {
            chunk["text"].strip() for chunk in self.target_chunks
            if chunk.get("source_id") in planning_ids
            and chunk["text"] != previous_by_id.get(chunk["source_id"], {}).get("text")
        }
        self.assertEqual(set(missing), expected_changed)
        for chunk, index in zip(self.target_chunks, indices):
            if chunk["text"].strip() not in expected_changed:
                self.assertLess(index, len(previous))
                self.assertEqual(previous[index]["text"].strip(), chunk["text"].strip())

    def test_invalid_source_metadata_is_rejected(self) -> None:
        item = self.imported[1]
        for field, value in (("source_question", True), ("source_question", 0), ("source_sha256", "not-a-hash"), ("references", "invented citation")):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    serialize_content({**item, field: value})

    def test_invalid_answer_revision_is_rejected(self) -> None:
        item = self.imported[1]
        revision = {
            "source_document": REVISION_ID,
            "source_question": 1,
            "source_sha256": REVISION_SHA256,
            "editorial_note": "Adapt the technical sequence while retaining applicable-criteria caveats.",
        }
        invalid_revisions = [
            None, [], "draft",
            {key: value for key, value in revision.items() if key != "editorial_note"},
            {**revision, "unknown": "field"},
        ]
        invalid_revisions.extend(
            {**revision, field: value} for field, value in (
                ("source_document", ""), ("source_document", 1),
                ("source_question", True), ("source_question", 0),
                ("source_sha256", "not-a-hash"), ("editorial_note", ""),
            )
        )
        for invalid in invalid_revisions:
            with self.subTest(revision=invalid):
                with self.assertRaises(ValueError):
                    serialize_content({**item, "answer_revision": invalid})
        for number in (1, 12):
            with self.subTest(response_type=self.imported[number]["response_type"]):
                record = self.imported[number]
                serialized = serialize_content({**record, "answer_revision": revision})["text"]
                for field in ("Answer", "Situation", "Task", "Action", "Result", "References", "Preparation Note"):
                    baseline = serialize_content({key: value for key, value in record.items() if key != "answer_revision"})["text"]
                    self.assertEqual(extract_tag_value(serialized, field), extract_tag_value(baseline, field))

    def test_answer_prompt_uses_structured_star_steps_and_preserves_technical_detail(self) -> None:
        direct = self.target_by_id[self.imported[1]["id"]]
        prompt = answer_messages(self.imported[1]["question"], build_candidate(1, 0, direct), "")[0]["content"]
        for instruction in ("four to six", "ordered steps", "study sequence", "methods", "rationale", "decision criteria", "validation"):
            self.assertIn(instruction, prompt)
        self.assertIn("I would", prompt)
        story = self.target_by_id[self.imported[12]["id"]]
        story_prompt = answer_messages(self.imported[12]["question"], build_candidate(1, 0, story), "")[0]["content"]
        for heading in ("**Situation**", "**Task**", "**Action**", "**Result**"):
            self.assertIn(heading, story_prompt)
        for instruction in (
            "numbered steps", "bold descriptive labels", "Preserve numbered steps",
            "phased recommendation", "decision triggers", "delivery constraints",
            "without an introduction explaining STAR",
        ):
            self.assertIn(instruction, story_prompt)
        self.assertNotIn("Use four spoken paragraphs", story_prompt)
        self.assertNotIn("four to six", story_prompt)
        for instruction in (
            "fuller Answer field", "technical reasoning", "stakes", "personal responsibility",
            "most detail", "diagnosed", "engineering choices", "validated",
            "supported outcomes", "explicitly hypothetical",
            "never turn those proposals into completed STAR actions or results",
            "Do not append a generic future-process framework",
            "cause-and-effect explanations", "delegation boundaries",
            "review checkpoints", "Do not replace these details with generic claims",
        ):
            self.assertIn(instruction, story_prompt)

    def test_preparation_notes_constrain_answers_without_becoming_spoken_instructions(self) -> None:
        item = self.imported[12]
        chunk = self.target_by_id[item["id"]]
        answer_prompt = answer_messages(item["question"], build_candidate(1, 0, chunk), "")
        instructions = answer_prompt[0]["content"]
        self.assertIn("they are not a spoken script", instructions)
        self.assertIn("Do not recite preparation instructions", instructions)
        self.assertIn("Keep material uncertainty", instructions)
        self.assertIn("potential versus realized outcomes", instructions)
        self.assertIn("omit that detail while retaining the supported outcome", instructions)
        self.assertIn("Do not resurrect omitted details", instructions)
        self.assertIn(item["preparation_note"], answer_prompt[1]["content"])
        coaching_prompt = practice_messages(item["question"], "My practice answer", chunk["text"], "")
        self.assertIn("preparation notes", coaching_prompt[0]["content"])
        self.assertIn(item["preparation_note"], json.loads(coaching_prompt[1]["content"])["selected_context"])

    def test_answer_and_practice_prompts_preserve_source_limits(self) -> None:
        item = self.imported[4]
        chunk = self.target_by_id[item["id"]]
        candidate = build_candidate(1, 0, chunk)
        answer_prompt = answer_messages(item["question"], candidate, "Desired role")
        practice_prompt = practice_messages(item["question"], "I completed the project.", chunk["text"], "Desired role")
        for messages in (answer_prompt, practice_prompt):
            with self.subTest(prompt=messages[0]["content"][:40]):
                instructions = messages[0]["content"].lower()
                self.assertIn("i would", instructions)
                self.assertIn("not independently verified", instructions)
                self.assertIn("potential fee avoidance", instructions)
                self.assertIn("tool named only in the question", instructions)
                self.assertIn("effective", instructions)
                self.assertIn("preparation notes", instructions)
                self.assertNotIn(chunk["text"], messages[0]["content"])
        self.assertIn(chunk["text"], answer_prompt[1]["content"])
        self.assertEqual(json.loads(practice_prompt[1]["content"])["selected_context"], chunk["text"])


if __name__ == "__main__":
    unittest.main()
