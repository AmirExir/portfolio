# Verification before displaying ERCOT answers

The All-in-One, Planning Guide, Nodal Protocols, Resource Integration, and
DWG/SSWG assistants now withhold generated drafts until a shared claim review
finishes. The retrieval-only API does not generate answers and does not perform
this review on behalf of downstream callers.

## Processing

1. `retrieval.prepare_evidence` copies the passages actually supplied to the
   answer model, assigns stable evidence IDs, and marks any budget truncation.
   Retries use their own exact evidence subset.
2. `claim_verification.verify_answer` extracts self-contained factual claims
   with their original cited IDs and exact draft spans using structured output.
3. A separate model call judges each claim against its cited passages and
   metadata as `supported`, `contradicted`, or `insufficient_evidence`. It also
   checks extraction fidelity and citation attachment. The review covers
   negation, thresholds, units, scope, exceptions, applicability, and dates.
4. Supported claims require exact supporting quotations from every cited ID.
   Local checks reject invented IDs, unavailable, stale, generated, truncated,
   or extraction-flagged evidence. Current binding requirements require
   governing/effective metadata for the requested date. Recognized quantities
   and quoted language in accepted statements must occur in their cited text.
5. Proposed corrections receive another independent judgment against the same
   original sources before they may appear. Corrections cannot add sources.
6. The application assembles the answer only from accepted claims and their
   citations. It does not display an unchecked rewrite, rejected draft text, or
   source-list fallback. Every cited ID retains a footer entry, including
   different passages from the same document.

Failures, refusals, incomplete or malformed responses, missing verdicts, and
invalid quotation evidence withhold the draft. If no claims qualify, the app
returns an abstention. Existing unmarked assistant history is excluded before
display or reuse; user messages remain available. Planning's deterministic
document-visibility and raw-source lookup responses do not use a model.

## Configuration and operating limits

The existing `OPENAI_API_KEY` is used. `ERCOT_RAG_VERIFIER_MODEL` optionally selects
a model supporting the Responses API and strict structured output; the default
is `gpt-6-astra`, matching the existing answer model. There is no option to bypass
verification. No additional dependency is needed.

Normal answers add two model calls (claim extraction and judgment). Corrections
add one more. Each review call has a 45-second timeout, SDK retries disabled,
and response storage disabled. There are explicit draft, source, response, and
request size bounds, with at most 32 extracted claims. Exceeding a bound
withholds the answer instead of silently checking only a prefix.

The adapter uses the Responses API `text.format` JSON-schema interface described
in the [official OpenAI documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
It additionally validates returned fields, IDs, verdict coverage, and source
quotations locally. Structured output constrains format; it does not prove that
a model's semantic judgment is correct.

## Validation and limits

Run the focused regression checks from the repository root:

```sh
python -m unittest ERCOTAPI.tests.test_claim_verification ERCOTAPI.tests.test_prepared_evidence ERCOTAPI.tests.test_verified_answer_apps ERCOTAPI.tests.test_response_handling
```

The verifier tests inject controlled model responses to exercise rejection,
correction, re-verification, malformed output, and fail-closed behavior. The
app tests execute the real submission branches with substituted UI/model
services and check that raw drafts never reach rendering or conversation
history. These tests validate orchestration; they do not measure a live model's
ability to recognize contradictions or produce complete atomic claims.

Model judgments and extraction remain fallible. The upgrade checks support in
the supplied passages, not truth outside those passages, completeness of the
answer, correctness of all engineering reasoning, or freshness of the source
corpus. Existing effective-date/currentness metadata heuristics remain in use.
Missing extraction warnings do not prove complete table or footnote coverage.
Conservative quantity membership can withhold valid calculations, conversions,
or paraphrases; these are omitted rather than silently treated as proven.

Before claiming measured accuracy, benchmark the deployed model on
engineer-reviewed ERCOT questions, especially negation, inequality boundaries,
applicability, conflicting versions, and missing exceptions. Failed verification
is visible to users and never falls back to the original generated draft.
