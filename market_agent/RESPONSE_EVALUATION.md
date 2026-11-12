# Market-agent response evaluation — November 11, 2026

## Findings

The supplied responses show a real input/configuration defect and several
presentation defects. A small number of qualified signals is also sometimes
the correct result: a large forecast or an important headline does not, by
itself, demonstrate a reliable return prediction.

| Finding | Evidence | Change |
| --- | --- | --- |
| Quick runs could not meet their own validation sample minimum | The November 5, 12:46 PM CST saved quick report requested 913 calendar days. MSFT and NVDA had five non-overlapping 30-session validation windows; eight were required. Their sole qualification failure was the window count. | Default to 1,825 calendar days in the CLI and generated workflow commands; preserve explicit overrides and all validation gates. |
| A limited news sample was described as a quiet market | The old request took 30 newest articles, while its model prompt asked for a judgment about whether news was thin. Repeated aggregator snippets could crowd out consequential coverage. | Request up to 100 relevant items, filter publication times, deduplicate, diversify topics, and report actual coverage counts. Prioritize significance separately from verification quality. |
| Material company developments were dismissed as “noise” | The supplied brief mentioned leadership scrutiny but minimized it in the conclusion. A November 8 Reuters report described an independent company investigation and included the board's response. | Require attributed event summaries, source links, possible business effects, and conditional watch actions. Allegations remain allegations; source snippets are not independent verification. |
| Raw timestamps were hard to scan | The supplied news used ISO timestamps and the forecast used numeric target dates. | Use dates such as `Dec 18, 2026`; generated report times use Chicago time by default, with DST and an explicit UTC offset. Preserve machine timestamps and market-session dates. |
| Some candidates disappeared or were cut off | Candidate lists appeared only when that side had no qualified signals. n8n payload builders truncated reports at 3,900 characters before delivery. | Keep qualified signals and research candidates visible together, show displayed/total counts, and split complete messages without deleting later sell sections. |
| News and model outputs appeared more integrated than they are | The detailed report already explained the separation, but the short response omitted it. | State that general news does not enter forecast scores. News watch actions do not promote failed quantitative signals. |

The independent news check used [Reuters' November 8 report](https://www.investing.com/news/stock-market-news/nvidia-launches-probe-after-report-ceo-huang-was-target-of-foreign-intelligence-5010307).
It supports prominent coverage of the investigation; it does not establish
every allegation or a particular future stock return. Later news must not be
used to judge what an earlier report should have known.

The saved forecast evidence is
`reports/ml_forecast_rankings_2026-11-05T18-46-02Z.json`. No historical report
or prediction-ledger entry was rewritten. The July examples' high-confidence
RL/large-return calls should not be treated as a target for recommendation
volume under the current validation contract.

## Resulting response behavior

The forecast summary leads with qualified buy and sell/avoid counts, then
lists both research directions even if a qualified signal exists. An
unqualified historical example now reads:

```text
MSFT: forecast +9.65% by Dec 17, 2026
  Not qualified: too few non-overlapping historical tests (5/8 required).
```

This is a formatting example from the saved November 5 run, not a new forecast.
Qualified entries include estimated error; rejected entries do not advertise
unreliable probabilities. Missing holdings block position sizing separately
from forecast qualification.

News briefs lead with consequential developments, cite their source and
publication date, and explain what evidence would strengthen or weaken a
conditional watch action. Coverage limitations are specific to the retrieved
sample. The workflow does not fetch full articles or independently verify them.

## Verification and limits

Run the relevant local suites from the repository root:

```bash
.venv/bin/python -m unittest discover -s market_agent/tests -p 'test_*.py'
.venv/bin/python -m unittest tests.test_market_agent_evaluation tests.test_market_agent_ledger tests.test_market_agent_ledger_promotion tests.test_market_agent_shadow_policy
```

Regression cases cover the actual holdout-window count on deterministic
exchange-session data, unchanged qualification requirements, coexistence of
qualified and candidate lists, date round trips across DST, stale/future and
duplicate news, topic coverage, workflow repair idempotence, and retention of
sell sections in long messages. Provider and notification calls in these tests
are mocked; JavaScript contract tests execute locally in Node.

Longer history costs more download/fit time and includes older regimes. New
listings, missing data, and longer horizons may still lack enough history.
Models that fail baseline skill or calibration must still abstain. These
changes do not demonstrate improved investment returns, and no fresh full
universe backtest or live trading evaluation is claimed.

Set `MARKET_REPORT_TIMEZONE` to change Python report time presentation; the
news workflow currently uses `America/Chicago`. The local latest-text
freshness parser understands both historical ISO headers and the new readable
headers, including their explicit offsets.

For activation and credentials, see
[the n8n setup guide](N8N_ML_FORECAST_RANKINGS_SETUP.md). Editing a workflow
export does not itself update the active n8n workflow. Offline repair requires
a stopped service and creates a database backup before modifying the draft and
published versions; an inaccessible port is not treated as a stopped service.

## Local activation — November 12, 2026

After the existing optimization completed successfully, the local n8n service
was stopped, backed up using SQLite's backup API, updated, and restarted.
The active workflow and its published history match the repaired configuration;
a repeat repair reports no remaining changes. The existing market-news API key
was migrated to an encrypted n8n HTTP-header credential owned by the same
project, and the duplicate inline authentication header was removed.

Both service health endpoints returned HTTP 200. A read-only NewsAPI check at
12:11 AM CST returned 96 articles; the evidence filter found 87 eligible unique
items and selected 24 from 16 sources, covering all seven topic groups. The
provider reported 991 matching results, so this remains a sample. Credential
decryption and isolated execution of the production evidence JavaScript passed.
No model-generation, publishing, notification, or trade nodes were executed for
this verification, and the completed optimizer was not rerun.

The protected pre-installation backup and installation record are under the
local n8n directory at
`backups/market-response-fixes-20261112T061002Z/`.
