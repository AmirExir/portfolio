# ERCOT document ingestion

The ERCOT retrieval indexes now use one centralized ingestion pipeline. It keeps
official ERCOT source material separate from generated summaries, embeds only
new official content, and publishes complete index generations atomically.

## Data flow

```text
ERCOT source page
  -> ercot_link_monitor.py
  -> hash-addressed local archive + provenance sidecar
  -> classification and chatbot routing
  -> safe text extraction
  -> deterministic chunks
  -> reuse existing vectors or embed unseen content
  -> validated index generation
  -> atomic CURRENT switch
  -> collection-filtered chatbot retrieval with citations
```

The twice-daily n8n job already invokes `ercot_link_monitor.py`. After each
successful scan, the monitor runs incremental ingestion over the complete
durable official archive. Content hashes still prevent unchanged documents from
being re-embedded, while the complete reconciliation lets repaired parse
errors, changed sidecars, deletions, and renamed files converge even during a
continuous download backlog. Set `ERCOT_RAG_AUTO_INGEST=false` to disable this
hook.

The tracked `scripts/run_ercot_link_monitor.sh` launcher resolves the repository
from its own location and uses `ERCOTAPI/.venv/bin/python` by default. Relocated
or externally orchestrated deployments can set `ERCOT_REPO_ROOT`,
`ERCOT_MONITOR_DIR`, or `ERCOT_MONITOR_PYTHON`. The local ignored n8n workflow
uses `ERCOT_REPO_ROOT` with a `$HOME/Documents/GitHub/portfolio` compatibility
fallback. If `OPENAI_API_KEY` is absent on macOS, the launcher checks the
`openai-api-key` Keychain service without printing the credential; override the
service/account with `ERCOT_OPENAI_KEYCHAIN_SERVICE` and
`ERCOT_OPENAI_KEYCHAIN_ACCOUNT`. Workflow credentials remain outside version
control.

Generated news summaries do not feed the shared RAG store. They can still be
published for other workflows, but they are intentionally excluded from bot
reindexing so they do not change retrieval results. A GitHub branch is not a
shared filesystem: the existing n8n flow that publishes `ercot_news_summary_*.txt`
to the remote `generated-output` branch does not make those files visible to a
locally running ingester, and the central RAG configuration no longer watches
that content at all. Likewise, hosted chatbots need the published generation
store on shared persistent storage (or a deployment artifact); another machine's
local `.rag_store` is not visible to them automatically.

The monitor archives official bytes beneath:

```text
ERCOTAPI/NEWS/official/<source>/<year>/<sha256>.<extension>
```

Every archived document has an adjacent `<filename>.metadata.json` sidecar with
its source page, original/final URL, title, download time, MIME type, size, and
SHA-256. Identical bytes discovered at multiple official URLs share one archive
object; the sidecar deterministically merges URL aliases and per-URL provenance
without oscillating between sources. Both the archive and sidecar are written
atomically. A URL is marked seen only after the durable archive succeeds. Known
URLs are still checked so a publisher replacing content at the same URL creates
a new content-addressed file. HTML issue pages are inspected one level deeper
for key attachments. To
avoid a first-run bulk download, unseen top-level links are limited to five per
source per run by default and the remaining backlog drains on later runs;
configure `ERCOT_LINK_MAX_ITEMS_PER_SOURCE` to change that limit. All-revision
report pages use a numeric newest-first window of 100 candidates
(`ERCOT_LINK_REPORT_WINDOW`), and at most 10 already-known candidates per source
are rechecked on each run (`ERCOT_LINK_MAX_KNOWN_RECHECKS_PER_SOURCE`). Increase
those bounds when older lifecycle changes must be revisited more aggressively.
The monitor attempts up to 20 unseen candidates to obtain the default five
successful archives, preventing a few broken top links from starving the
backlog (`ERCOT_LINK_MAX_UNSEEN_ATTEMPTS_PER_SOURCE`). Archive responses are
streamed and capped at 50 MiB before persistence; configure
`ERCOT_LINK_MAX_RESPONSE_BYTES` when a known official document requires a
different bound. Detail-page attachments use a separate round-robin budget of
20 per source and run by default; set `ERCOT_LINK_MAX_NESTED_ITEMS_PER_SOURCE`
to tune that bound without letting one large issue page starve other parents.

The default source roots are:

- `chatbot_ercot_all_in_one/ercot_sources/` — canonical checked-in manuals.
- `ERCOTAPI/NEWS/official/` — downloaded authoritative ERCOT material.
- `chatbot_ercot/` — planning guide source texts.
- `chatbot_ercot_nodalprotocols/` — nodal protocol source texts.
- `DWG_SSWG_Chatbot/` — DWG/SSWG manual source texts.
- Generated news and market-agent summaries are excluded from the shared RAG
  store.

The older per-chatbot split files and experimental FAISS/copy scripts remain
legacy artifacts; active chatbot entry points now read the central store (or
the checked-in all-in-one JSON/NumPy fallback). Update the canonical static
manuals directory instead of maintaining separate chatbot indexes.

## Store and manifest

The default persistent store is `ERCOTAPI/.rag_store/`. Override it with
`ERCOT_RAG_STORE` (or the compatibility name `ERCOT_RAG_INDEX_DIR`). The store
is local generated state and is ignored by Git.

```text
ERCOTAPI/.rag_store/
  CURRENT
  generations/
    <generation-id>/
      manifest.json
      chunks.json
      embeddings.npy
```

`CURRENT` contains the only reader-visible commit point. A writer builds and
validates all three generation files before atomically replacing that pointer.
Per-file parsing or embedding failures are isolated: healthy documents can
still publish, the failing path is recorded in the manifest, and previously
indexed chunks for that path remain available but are explicitly marked stale.
A first-time file that fails has no retrievable chunks until a later retry.
Generation-wide failures—such as incompatible dimensions/configuration, a
failed full rebuild, store validation, or publication failure—never switch
`CURRENT`, so readers keep the previous complete generation.

The manifest records each observed path's repository-relative path, byte size,
nanosecond mtime and readable modification timestamp, SHA-256, type, source
root/category, authority, generated flag, document kind/number/status/version,
URL and download metadata, target collections, deterministic document/chunk
IDs, ingestion timestamp/status, duplicate alias, and any error. Deleted paths
remain as tombstones while their chunks leave the active generation when no
live alias remains. Complete generations are retained for rollback according to
`ERCOT_RAG_KEEP_GENERATIONS`, which defaults to 10 and is clamped to a minimum
of 2. Retention always protects the active generation and its recorded previous
generation.

Exact byte duplicates are embedded once and represented as path aliases.
Renames therefore retain document/chunk IDs and reuse vectors. If identical
bytes appear through official and generated paths, the manifest preserves both
path records and retrieval metadata chooses the authoritative ERCOT alias.
Different official revisions have different hashes and remain independently
retrievable.

## Authority and routing

Classification precedence is:

1. Downloader sidecar metadata.
2. Known canonical static filenames.
3. Configured source-root trust defaults.
4. Centralized title/path patterns.

The physical vector matrix is shared, but every chunk carries logical
collections:

| Source kind | Collections |
| --- | --- |
| NPRR / current Protocol | `general`, `protocols`, `market` |
| PGRR / Planning Guide | `general`, `planning` |
| NOGRR / Operating Guide | `general`, `operations` |
| OBDRR | `general`, `protocols`, `operations` |
| SCR | `general`, `operations` |
| Market Notice / report | `general`, `market` |
| Resource Integration / interconnection | `general`, `resource_integration` |
| RIWG meeting material | `general`, `resource_integration` |
| DWG | `general`, `dwg_sswg` |
| SSWG | `general`, `dwg_sswg`, `planning` |
| RPG / Regional Transmission Plan (RTP) | `general`, `planning` |
| Generated news/market summary | `news`, `market` only |

Normal regulatory chatbots do not load generated summaries. Retrieval adds an
authority boost for official ERCOT material, demotes generated text, and emits
a citation containing authority, document kind/number, title, repository path,
chunk number, and original URL when known.

To add a source kind, update the rules in `rag_ingestion/classify.py`. To add a
chatbot, add a stable collection in `rag_ingestion/config.py`, route the source
kinds to it, and call `load_index("collection_name")` from the application.

## Supported documents

- PDF through `pypdf`
- TXT
- HTML / HTM (script, style, SVG, and other hidden content removed)
- DOCX through bounded ZIP/XML parsing
- CSV with bounded rows and columns
- XLSX with bounded ZIP/XML and worksheet text extraction

Malformed files record an error without stopping healthy documents. Legacy
DOC/XLS, PowerPoint, ZIP bundles, encrypted files, and other binaries are
archived when discovered but skipped by ingestion until a safe loader is added.

## Commands

Install the declared dependencies into the same environment used by the
monitor/chatbot process, then run commands from the repository root:

```bash
python -m pip install -r ERCOTAPI/requirements.txt
```

```bash
# Read-only discovery; performs no parsing, embedding, lock, or write.
python -m ERCOTAPI.rag_ingestion scan --dry-run

# Incremental update; unchanged files and existing vectors are reused.
python -m ERCOTAPI.rag_ingestion update
python -m ERCOTAPI.rag_ingestion update --changed-only

# Limit repair/update to a configured source path.
python -m ERCOTAPI.rag_ingestion update \
  --path ERCOTAPI/NEWS/official

# Index generated summaries after their producer writes or syncs them locally.
python -m ERCOTAPI.rag_ingestion update \
  --path ERCOTAPI/market_agent

# Inspect the active generation and per-file errors.
python -m ERCOTAPI.rag_ingestion status

# Stage and validate fresh vectors before switching generations.
python -m ERCOTAPI.rag_ingestion rebuild --force
```

The first update can bootstrap the four existing all-in-one manuals from the
checked-in 3,072-dimensional JSON/NumPy cache. Matching legacy chunks reuse
their vectors instead of making paid embedding requests. Bootstrap validates
that the normalized cached chunks continuously cover the current source through
its end; a stale or incomplete source/cache pair falls through to normal parsing
and embedding. OpenAI is imported and contacted only if genuinely unseen chunks
need embeddings. Configure the key in the environment; it is never stored in the
manifest or repository.

## Recovery

1. Run `status` and inspect its `errors` entries.
2. Repair the source file, optional parser dependency, or embedding credentials.
3. Run a path-scoped `update`; failed files are eligible for retry, and stale
   chunks are replaced only after the changed source embeds successfully.
4. If a newly published generation has an operational problem, atomically point
   `CURRENT` back to the previous complete generation ID recorded in its
   manifest, then run `status`.
5. Use `rebuild --force` only when changing embedding/chunking behavior or when
   a complete repair is required. The old `CURRENT` remains active until the
   replacement validates.

Chatbot startup only loads an active generation (or the legacy read-only
fallback). It never scans, parses, or embeds source documents.
