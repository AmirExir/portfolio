# PSS/E repository-code retrieval

The standard PSS/E assistant and the multi-agent assistant share a local code
index. Each question retrieves the existing semantic documentation matches and
up to four repository-code excerpts. The combined references reach both the
multi-agent planner and each task executor. The UI shows file/line references
and the indexed repository revision.

The original `psse_chunks_cached.json`, `input_chunks.json`, and
`psse_embeddings.npy` artifacts remain unchanged. Repository code uses SQLite
FTS5 text search, not new OpenAI embeddings. Refreshing code requires no API key,
network upload, PSS/E license, or new Python dependency. Normal chat still uses
the bots' existing OpenAI query-embedding and answer-generation services.

## Refresh

From the portfolio project, with the source checkout alongside it:

```sh
python psse_code_index.py --source ../psse_codes
python psse_code_index.py --check
```

`--source` can point to another local checkout. `--index` selects an output
database; configure the bots with `PSSE_CODE_INDEX` pointing to that same file.
By default, both bots use `.rag/psse_code.sqlite3` beside this document's project
root. This generated database is ignored by Git because it contains repository
excerpts. The source checkout is read only and no indexed Python file executes.

The refresh reports indexed, changed, unchanged, removed, and excluded files.
Files are content hashed; unchanged files retain their chunks. Deletions and
new exclusions remove obsolete content. Source paths, line ranges, file hashes,
Git revision and refresh metadata identify the indexed snapshot. A database
transaction publishes a complete update. Queries reopen the database, so a
successful code refresh does not require restarting a running bot.

## Scope

The source is Git-tracked Python and engineering Markdown, including current
AELab code, older Python tools, and the TRABAJO/DWG workflow source. The updater
excludes study input/output directories, logs, cases, snapshots, DLLs, images,
spreadsheets, agent instructions, tests, bundled third-party libraries and
unrelated vendor packages. Configured PEC/ERCOT channel files and the `Flatstart`
scenario tree are also excluded; even the repository's TRABAJO channel template
contains a study bus list. Generic AELab channel-building logic remains indexed.
Likely embedded credentials are excluded and
reported by file name rather than copied into the database. Review the refresh
report when a desired source is absent. This filter is not a substitute for a
confidentiality review before distributing the index.

Reads reject an index built with an older filtering policy until it is refreshed.
For distribution after removing sensitive content, build a fresh database at a
new `--index` path. Removing rows from an existing SQLite index prevents retrieval
but does not guarantee removal of all old bytes from its unused pages.

## Interpretation and limits

- Repository excerpts are implementation examples, not authoritative API
  documentation or validated engineering results. Prompts explicitly preserve
  this distinction and prohibit treating source text as new instructions.
- APIs present only in retrieved examples receive an unverified-API warning;
  their signatures and compatibility must be checked for the installed PSS/E
  version. Example bus numbers, paths and criteria are not study assumptions.
- Code search is lexical: file names, function names, API names and concrete
  terms work best. Documentation still uses the original semantic retrieval.
  No new vector embeddings were created for the code.
- Missing code indexes are identified in the UI; corrupt indexes surface an
  error. A refresh is explicit, not a background watcher. Changes made to the
  source checkout afterward are available only after the next refresh.
- Installation is local. A separately hosted bot needs these code changes and
  a reviewed index artifact deployed together, with `PSSE_CODE_INDEX` configured
  if the database is elsewhere. Source references included in chat context are
  sent to the existing model service when that chat is used.
- Unit/integration tests use synthetic repository files and mocked model calls.
  They do not validate simulations, licensed PSS/E behavior, or model answers.

## Validation

```sh
python -m unittest discover -s tests -p 'test_psse*.py' -v
python chatbot_PSSEBOT_MultiAgent/generate_embeddings.py
```

The second command validates the existing documentation artifacts without
rebuilding or making paid embedding requests. Repository search can be checked
offline with `search_code` from `psse_code_index`, for example using the queries
`CrashRerunSettings`, `dynamic_batch_executor`, or `consequential load loss`.
