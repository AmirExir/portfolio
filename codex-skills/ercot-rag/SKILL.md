---
name: ercot-rag
description: Build and evaluate trustworthy ERCOT regulatory retrieval systems with document status, effective-date resolution, persistent embeddings, and section-level citations.
---

# ERCOT RAG

## Use this skill when

Use this skill for ERCOT document ingestion, Protocols, Planning Guides, Operating Guides, procedures, xRRs, meeting materials, market notices, retrieval, embeddings, citations, version tracking, or regulatory question answering.

## Core standard

The system must identify which document governs, determine whether it is effective, combine related requirements, explain engineering impact, track what changed, and prove the answer with section-level evidence.

## Required workflow

1. Discover and classify source documents.
2. Capture title, identifier, document type, revision, approval date, effective date, status, and source URL.
3. Distinguish effective, approved-not-effective, proposed, withdrawn, superseded, and historical material.
4. Parse headings, sections, tables, appendices, and page references.
5. Chunk by logical section rather than arbitrary token windows where possible.
6. Hash source content and metadata.
7. Reuse existing embeddings for unchanged content.
8. Persist embeddings and indexes across runs.
9. Retrieve using lexical and semantic methods.
10. Rerank evidence before answer generation.
11. Verify citations against retrieved source text.
12. Evaluate retrieval and answer faithfulness.

## Evidence hierarchy

Prefer, in order:

1. Effective governing document
2. Approved implementation document
3. Official procedure or methodology
4. Approved revision request and implementation record
5. Official meeting or ballot material
6. Supporting presentation or market notice

Do not treat a proposed revision as an effective requirement.

## Answer structure

Separate:

- Governing requirement
- Related requirements
- Effective status and date
- Engineering impact
- Change history
- Uncertainty or missing evidence
- Citations

## Retrieval requirements

- Preserve section titles and page numbers.
- Preserve document status in chunk metadata.
- Filter or rerank by question date.
- Avoid mixing requirements from incompatible effective periods.
- Detect duplicate and near-duplicate editions.
- Prefer authoritative source files over derived summaries.
- Never fabricate a section number or citation.

## Persistence design

Use a content hash based on normalized text plus material metadata. Store:

- source hash
- parser version
- chunking version
- embedding model
- embedding dimension
- chunk IDs
- source document ID
- index generation ID

Only reprocess documents whose source or relevant pipeline version changed.

## Evaluation

Create a test set covering:

- governing-document identification
- effective-date questions
- section retrieval
- cross-document synthesis
- superseded requirements
- engineering-impact explanations
- unanswerable questions

Track at minimum:

- Recall@k
- MRR or nDCG
- citation correctness
- answer faithfulness
- abstention quality

## Failure behavior

When evidence is insufficient, say so explicitly. Do not fill gaps from model memory.

## Completion format

Report:

- Sources added or changed
- Index and embedding behavior
- Retrieval changes
- Evaluation results
- Remaining evidence gaps
