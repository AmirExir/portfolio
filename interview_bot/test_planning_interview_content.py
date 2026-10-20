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
REVISION_ID = "private:transmission-planning-interview-2026-10-20-draft"
REVISION_SHA256 = "39e6a326290ce43c17d148525af85719ba85322d62031425a8ec7d5019bc61dd"
EXPANDED_REVISION_ID = "private:transmission-planning-interview-2026-10-20-expanded-draft"
EXPANDED_REVISION_SHA256 = "ac81de485b80bbd488a8a5bcc7bc6b795236990f3a83df0bbe735edde6391e99"
COACHING_REVISION_ID = "private:transmission-planning-interview-2026-10-20-coaching-draft"
COACHING_REVISION_SHA256 = "f0c4325b6c752a553c777ff48f9c4c998de70ea939472d4699a15bd46a48ab1b"
Q14_REVISION_ID = "private:q14-transmission-planning-example-2026-10-20"
Q14_REVISION_SHA256 = "2334af04d1047f7bb88c70faa0346ddebfaa282b67948e062077d0f16bba0787"
FIRST_REVISION_NUMBERS = set(range(1, 11))
EXPANDED_REVISION_NUMBERS = set(range(12, 21))
COACHING_REVISION_NUMBERS = {8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
REVISED_NUMBERS = FIRST_REVISION_NUMBERS | EXPANDED_REVISION_NUMBERS | COACHING_REVISION_NUMBERS
STORY_NUMBERS = {12, 13, 14, 15, 17, 18, 19}


def answer_digest(text: str) -> str:
    """Fingerprint original wording without publishing another source copy."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def metadata_digest(item: dict) -> str:
    """Compare original questions and caveats using a hash-only fixture."""
    metadata = {
        "question": item["question"],
        "source_status": item["source_status"],
        "references": item.get("references", []),
        "preparation_note": item.get("preparation_note", ""),
    }
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
                if number not in COACHING_REVISION_NUMBERS:
                    self.assertEqual(metadata_digest(item), original["metadata_sha256"])
                else:
                    self.assertTrue(item["preparation_note"].strip())
                if number not in REVISED_NUMBERS:
                    self.assertEqual(answer_digest(item["answer"]), original["answer_sha256"])
                    self.assertNotIn("answer_revision", item)
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
                if number == 14:
                    identifier, digest = Q14_REVISION_ID, Q14_REVISION_SHA256
                elif number in COACHING_REVISION_NUMBERS:
                    identifier, digest = COACHING_REVISION_ID, COACHING_REVISION_SHA256
                elif number in FIRST_REVISION_NUMBERS:
                    identifier, digest = REVISION_ID, REVISION_SHA256
                else:
                    identifier, digest = EXPANDED_REVISION_ID, EXPANDED_REVISION_SHA256
                self.assertEqual(revision["source_document"], identifier)
                self.assertEqual(revision["source_question"], number)
                self.assertEqual(revision["source_sha256"], digest)
                self.assertTrue(revision["editorial_note"].strip())
                self.assertNotEqual(answer_digest(item["answer"]), self.original[number]["answer_sha256"])

    def test_story_answers_use_the_rewritten_star_narrative(self) -> None:
        for number in sorted(STORY_NUMBERS):
            with self.subTest(question=number):
                item = self.imported[number]
                narrative = "\n\n".join(item[field] for field in ("situation", "task", "action", "result"))
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
        for number in (12, 13):
            with self.subTest(question=number):
                self.assertIn("potential", self.imported[number]["result"].lower())
                self.assertIn("$300,000", self.imported[number]["answer"])
        self.assertIn("realized net savings", self.imported[12]["preparation_note"])
        self.assertIn("do not imply an ERCOT project endorsement", self.imported[13]["preparation_note"])
        self.assertIn("project remained feasible", self.imported[17]["answer"].lower())
        self.assertIn("colleague’s own achievement", self.imported[19]["preparation_note"])

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
                self.assertIn(concept, action)
                self.assertIn(concept, context)
        self.assertIn("earned his P.E. license", item["result"])
        self.assertIn("promoted to Engineer", item["result"])
        self.assertNotIn("within a year", item["answer"])
        for item in self.imported.values():
            self.assertNotIn("I still need to confirm", item["answer"])
            self.assertNotIn("My preparation record", item["answer"])

    def test_every_exact_question_retrieves_its_imported_answer(self) -> None:
        for number, item in self.imported.items():
            with self.subTest(question=number):
                focus = "Transmission planning"
                ranked = rank_candidates(item["question"], self.target_chunks, {}, focus=focus)
                self.assertTrue(ranked)
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

    def test_investment_answer_uses_planning_scenario_and_phased_decision(self) -> None:
        item = self.imported[14]
        chunk = self.target_by_id[item["id"]]
        for focus in ("All roles", "Transmission planning", "Project management"):
            with self.subTest(focus=focus):
                ranked = rank_candidates(item["question"], [chunk], {0: 0.99}, focus=focus)
                self.assertEqual([candidate["index"] for candidate in ranked], [0])
        for heading in (
            "Root-Cause Model Reconciliation", "Multi-Scenario Sensitivity Analysis",
            "Least-Regrets Phased Investment", "Stakeholder Alignment",
        ):
            self.assertIn(f"**{heading}**", item["action"])
        for detail in (
            "SSWG", "PSS®E", "PV/QV", "charging", "345-kV", "reactive",
            "growth milestones", "in-service dates", "permitting lead times",
        ):
            self.assertIn(detail, item["answer"])
        self.assertIn("recommendation", item["result"])
        for unsupported in ("GE", "SCADA", "Site Acceptance", "zero formal objections", "millions"):
            self.assertNotIn(unsupported, item["answer"])
        self.assertNotIn("How did you evaluate upgrading versus replacing the GE EMS?", item["aliases"])

    def test_incremental_plan_reuses_all_unchanged_evidence(self) -> None:
        # Simulate the previous published corpus even after this batch is indexed.
        introduced = {item["id"] for item in self.imported.values()}
        introduced.add("story-pec-annual-transmission-assessment")
        previous = [chunk for chunk in self.active_chunks if chunk.get("source_id") not in introduced]
        indices, missing = plan_embedding_reuse(previous, self.target_chunks)
        expected_new = {chunk["text"].strip() for chunk in self.target_chunks if chunk.get("source_id") in introduced}
        self.assertEqual(len(expected_new), 21)
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
        expected_changed = {
            chunk["text"].strip() for chunk in self.target_chunks
            if chunk.get("source_id") in planning_ids
            and chunk["text"] != previous_by_id[chunk["source_id"]]["text"]
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
