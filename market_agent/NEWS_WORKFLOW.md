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

`news_workflow.py` now requests up to 100 relevant results within 36 hours,
then rejects stale, future, missing-time, removed, malformed-URL and duplicate
records. It retains original timestamps and URLs, formats publication times
in `America/Chicago` with daylight-saving changes, and reserves coverage for
governance/legal, earnings, macro, semiconductors, energy and crypto topics.
The topic rules are editorial coverage heuristics, not investment scores.
Publisher diversity is preferred; repeated snippets are not corroboration.
Both publisher and aggregator snippets remain explicitly unverified.

The summary leads with material developments and explains conditional market
implications. A reported investigation and its company response must be
distinguished from proof of an allegation. Each development includes a source,
readable date and direct URL. The event watchlist identifies what to monitor
and what would change the assessment. It cannot invent probabilities, price
moves or buy/sell instructions. A brief ends with actual retrieval counts and
any sample limitation instead of a blanket claim that news is quiet.

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

For an existing installation, the repository's offline repair tool updates
the stored workflow. It requires n8n to be stopped, the correct workflow ID,
an unused backup path, and a configured GitHub credential source node:

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

For offline export validation (no calls to external services):

```python
import json
from pathlib import Path
from market_agent.n8n_workflow import repair_market_optimization_workflow

workflow = json.loads(Path("n8n_market_telegram_calendar_combined.workflow.json").read_text())
repaired, changes = repair_market_optimization_workflow(
    workflow, credential_source_node="Save Optimization To GitHub"
)
assert repaired == workflow and not changes
```

## Validation and limitations

```bash
.venv/bin/python -m unittest \
  market_agent.tests.test_n8n_workflow \
  market_agent.tests.test_news_workflow -v
```

The tests execute the actual JavaScript using Node.js VM isolation (without
browser URL globals, matching the n8n Code-node environment) and synthetic fixtures.
They cover date freshness, Chicago date rollover/DST, duplicate removal,
provider errors, preservation of an older governance event amid newer feed
items, graph/repair idempotence, and complete long-message delivery through
the last sell row. They do not call NewsAPI, the language model, GitHub or
Telegram. Prompt assertions verify the written contract, not model obedience.

Retrieval still uses one provider page, so matching stories can be missed.
The brief explicitly reports this when the provider's `totalResults` exceeds
the fetched count. NewsAPI supplies snippets, not verified full articles.
The workflow cannot independently confirm allegations or establish complete
market coverage. Live model prose must still be checked against its links.
No historical news panel was added to forecast training.

Provider parameter reference: [NewsAPI Everything documentation](https://newsapi.org/docs/endpoints/everything).
