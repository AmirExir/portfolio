# Market news and report delivery

The news brief is contextual research. It does not change a price forecast,
promote a failed validation result, authorize an allocation, or place an order.
The acknowledgement states this distinction even when an optimization request
does not request news.

## Evaluation and changes

The earlier news query requested only the 30 newest articles. In the supplied
response, several articles came from the same few minutes and included
aggregator summaries. That is a limited retrieval sample, not evidence that
the market has no significant events. The old prompt explicitly requested a
"thin" judgment and raw `publishedAt` dates. It did not require source links
or concrete event watch actions.

Telegram `/news`, `/briefing` and scheduled briefs read MarketWatch Top Stories,
BBC Business and NVIDIA Newsroom RSS on every run. They also request up to 100
relevant NewsAPI articles for context. The November 23, 2026 9:04 a.m. Central
run illustrates why both are needed. Its newest NewsAPI article was already
24 hours old, while NVIDIA's RSS item announcing its board decision was stamped
7:55 a.m. that morning. Publisher feeds and their URLs are configured in
`breaking_news_sources.py`.

The evidence filter keeps source links and publication times, excludes future,
undated, malformed and duplicate items, and displays times in `America/Chicago`.
It reserves room for the newest eligible headline and material publisher reports
before filling the brief across topics. Articles up to 36 hours old can supply
context. When every eligible article is at least 24 hours old, the model receives
an empty article list and instructions to report that a current update is unavailable.
Newer decisions take precedence over earlier speculation about the same event.

In the Streamlit news tab, **Refresh market news** queries the same publishers
and shows their linked headlines and publication times. The saved AI brief has
its own publication timestamp. Refreshing the dashboard does not publish a brief.

The summary leads with material developments and explains conditional market
implications. A reported investigation and its company response must be
distinguished from proof of an allegation. Each development includes a source,
readable date and direct URL. The event watchlist identifies what to monitor
and what would change the assessment. It cannot invent probabilities, price
moves or buy/sell instructions. The channel brief contains short news items,
conditional watch actions, source links and dates. It omits retrieval counts,
provider limits, settings and workflow commentary. Each distinct claim in a
combined item needs its own supporting source. Material uncertainty stays
with the affected story; the brief must not claim exhaustive market coverage.

Retrieval diagnostics remain in the evidence node's `news_coverage` output for
operator inspection in n8n execution data. Only its readable generation time
is passed to the news model, along with the selected articles; the counts are
not supplied for publication. The news text archive uses the same audience
brief. Operators can inspect each source's status and article count, the newest
eligible and selected timestamps, and the delayed-only flag. Available sources
continue through partial failures. If every source fails, the workflow reports
an error.

The forecast Telegram summary likewise shows only forecasts passing the
existing qualification rules, with forecast dates, probabilities and error
estimates. Failed candidates, policy-only signals, holdings/allocation checks,
run settings and raw errors remain in the detailed report and JSON. A short
availability note appears if some forecasts could not be updated. This is a
presentation change; forecast calculations and qualification gates are unchanged.

The normalization node runs before the news model in both the exported
workflow and the repair helper. The existing news GitHub publication branch
is connected again. The optimization branch publishes validated Python text
without asking a language model to rewrite its ratings. Complete news and
optimization reports are split for Telegram after formatting; long reports
no longer lose the sell section to an earlier 3,900-character truncation.
All optimization profiles request five calendar years of history. This makes
the existing independent validation-window requirement attainable for more
30-session forecasts; it does not relax any validation threshold or guarantee
that a forecast qualifies.

## Configuration and installation

Local source and the JSON export do not by themselves update a running n8n
installation. The local workflow export intentionally contains no NewsAPI key;
it is gitignored, while the repair modules carry the versioned changes.

Before importing/activating the export:

1. In **News API** and **ERCOT News API**, either select an n8n HTTP Header Auth
   credential whose header is `X-Api-Key` and remove the inline header
   expression, or set `NEWS_API_KEY` in the n8n service environment if that
   deployment permits environment access from nodes. Never paste the key into
   a tracked workflow. The repair preserves an existing credential binding.
2. In **Save Optimization To GitHub** and **Save News To GitHub**, replace the
   `CONFIGURE_GITHUB_HEADER_AUTH` placeholder with a real n8n HTTP Header Auth
   credential for the existing repository. Keep secrets in n8n credentials.
3. Select existing OpenAI, Telegram and Calendar credentials as appropriate.
   Confirm intended publication destinations before activating the workflow.

For an existing installation, let active jobs finish and stop n8n. Install
only the market news update using the existing workflow ID and a new private
backup path. This updates the draft and each referenced published version
independently. Restart n8n after the command finishes.

```bash
.venv/bin/python scripts/repair_n8n_market_optimization.py --news-only \
  --database /path/to/n8n/database.sqlite \
  --backup /path/to/backups/before-breaking-news.sqlite \
  --workflow-id YOUR_EXISTING_WORKFLOW_ID
```

For a combined optimizer and news installation, supply the GitHub credential source.

```bash
.venv/bin/python scripts/repair_n8n_market_optimization.py \
  --database /path/to/n8n/database.sqlite \
  --backup /path/to/backups/market-workflow-before-repair.sqlite \
  --workflow-id YOUR_EXISTING_WORKFLOW_ID \
  --credential-source-node 'Save ERCOT To GitHub'
```

The default credential source above is appropriate only if that node already
has a working `httpHeaderAuth` binding. An alternative is a configured **Save
Optimization To GitHub** node. The export uses that alternative for offline
validation because its placeholder is deliberately unresolved. Configure
NewsAPI authentication before repairing an installation with a literal key:
repairing an inline key replaces it with the environment expression and does
not create or populate the environment variable.

For targeted offline news-export validation (no calls to external services):

```python
import json
from pathlib import Path
from market_agent.news_workflow import repair_market_news_workflow

workflow = json.loads(Path("n8n_market_telegram_calendar_combined.workflow.json").read_text())
repaired, changes = repair_market_news_workflow(workflow)
assert repaired == workflow and not changes
```

## Validation and limitations

```bash
.venv/bin/python -m unittest \
  market_agent.tests.test_n8n_workflow \
  market_agent.tests.test_news_workflow \
  market_agent.tests.test_breaking_news_workflow \
  market_agent.tests.test_news_only_database_repair \
  market_agent.tests.test_publisher_headlines \
  market_agent.tests.test_daily_report_output \
  market_agent.tests.test_report_presentation -v
```

The tests execute the actual JavaScript using Node.js VM isolation (without
browser URL globals, matching the n8n Code-node environment) and synthetic fixtures.
They cover date freshness, Chicago date rollover/DST, duplicate removal,
provider errors, preservation of an older governance event amid newer feed
items, exclusion of operational metadata from model inputs while preserving
source evidence, graph/repair idempotence, and complete long-message delivery
through the last sell row. They do not call NewsAPI, the language model, GitHub or
Telegram. Prompt assertions verify the written contract, not model obedience.

Coverage depends on when each publisher posts or updates its feed. It is
polled on request and on the existing schedule. NewsAPI contributes one page
per run, and the internal `sample_limited` flag records when its result count
exceeds that page. Each brief has a maximum of 24 selected articles.
The workflow cannot independently confirm allegations or establish complete
market coverage. Live model prose must still be checked against its links.
No historical news panel was added to forecast training.

Provider parameter reference: [NewsAPI Everything documentation](https://newsapi.org/docs/endpoints/everything).
