# Offline evidence evaluation

This harness checks the retrieval/answer pipeline without making embedding or
answer-model requests. It does not establish regulatory correctness.

`repository_cases.json` contains three draft source-location/date cases anchored
to checked-in Planning Guide sections. Their expected evidence lists are
non-exhaustive and require independent engineer review. `synthetic_cases.json`
and `synthetic_predictions.json` are invented, explicitly labeled fixtures that
exercise quantity mismatch detection and canonical abstention. They are not
observed model answers or engineering results.

Run from the repository root:

```sh
# Verify that the draft section/title anchors still exist; records source hashes.
python -m ERCOTAPI.rag_ingestion.evaluation \
  --cases ERCOTAPI/evaluation/repository_cases.json --check-sources

# Exercise local source extraction and lexical section retrieval.
python -m ERCOTAPI.rag_ingestion.evaluation \
  --cases ERCOTAPI/evaluation/repository_cases.json --lexical-smoke

# Exercise the evaluator's support and abstention measurements with canned fixtures.
python -m ERCOTAPI.rag_ingestion.evaluation \
  --cases ERCOTAPI/evaluation/synthetic_cases.json \
  --predictions ERCOTAPI/evaluation/synthetic_predictions.json
```

Smoke mode ranks entire numbered sections by lexical cosine similarity and
emits an extracted section title. It does **not** use the deployed hybrid
retriever, query embeddings, answer model, or an independently inferred as-of
date. Its answer and date-echo scores are harness checks, not production
performance. The initial three-case smoke run retrieved one of three expected
sections in its top five; this deliberately untuned result does not characterize
the deployed system.

For actual system evaluation, collect a separate JSON list of recorded outputs:

```json
[
  {
    "id": "generator-initiation-section",
    "as_of": "2026-07-22",
    "answer": "The recorded model answer with [E1] citations.",
    "retrieved": [
      {
        "evidence_id": "E1",
        "source_path": "ERCOTAPI/sources/official/planning_guides/05-071126.docx",
        "section_number": "5.2.2",
        "effective_state": "effective",
        "text": "The actual text supplied to the model, without substitutions."
      }
    ]
  }
]
```

Pass that file through `--predictions`. Preserve the corpus generation, model,
prompt, retrieval settings, and run date alongside each recorded evaluation;
do not replace actual retrieved text with expected evidence. No recorded
production outputs are included here. Missing predictions count as failures,
and duplicate/unknown IDs are rejected. Source anchors that no longer match
fail visibly rather than silently continuing with outdated labels.

Metrics distinguish expected-evidence recall, requested-as-of match rate,
effective-state agreement, overall abstention accuracy, abstention recall,
non-abstention rate on answerable cases,
citation-format pass rate, and expected support-review signal detection. An
empty class is unscored (`null`). `semantic_support_accuracy` is always `null`:
word overlap, matching numbers, or a citation ID cannot establish entailment,
completeness, or engineering applicability. A canonical abstention mixed with
additional factual claims is not scored as a clean abstention.
Matching the requested date is not proof of date applicability, and answering
an answerable case is not proof the answer is correct; these deliberately narrow
metrics measure only their named behavior.

Before using scores for model selection, independently review and expand the
cases to include conflicting revisions, applicability exceptions, tables and
footnotes, absent evidence, and historical dates. Separate development cases
from a held-out evaluation set and retain review provenance. A dataset labeled
`engineer_reviewed` must name its reviewer and review date; this declaration is
input provenance, not an automated certification.

Validation:

```sh
python -m unittest ERCOTAPI.tests.test_evidence_evaluation \
  ERCOTAPI.tests.test_answer_support ERCOTAPI.tests.test_loaders -v
```
