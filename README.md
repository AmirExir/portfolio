# Engineering and AI portfolio

Power-system analysis, applied machine learning, ERCOT document retrieval, and
engineering automation by Amir Exir. The repository contains a static portfolio
site and independently launched Python applications.

## Explore the projects

| Project | What to inspect | Run and validation guide |
| --- | --- | --- |
| Hourly load forecasting | Causal load features, chronological evaluation, seasonal baselines, recursive forecasts | [Forecasting guide](energy_forcast/README.md) |
| Power Fault Classifier | Validated current/voltage inputs, stratified model selection, held-out per-class metrics | [Classifier guide](fault_classifier/README.md) |
| Power Grid GNN | Scenario graphs, supplied violation labels, train-only preprocessing, held-out scenario evaluation | [GNN guide](GNN/README.md) |
| ERCOT Grid Intelligence Dashboard | Grid views and hourly load forecasting with explicit evaluation scope | [Dashboard guide](ERCOTAPI/README.md) |
| ERCOT knowledge assistants | Persistent document ingestion, retrieval, source evidence, and revision tracking | [RAG architecture](ERCOTAPI/RAG_INGESTION.md) |
| Grid Atlas | Regional infrastructure map, source metadata, and dataset boundaries | [Atlas guide](ERCOTAPI/GRID_ATLAS.md) |
| Market intelligence agent | Experimental forecasting, prospective evaluation, and paper-trading workflows | [Evaluation contract](market_agent/POLICY_EVALUATION.md) |
| Portfolio website | Project narratives, screenshots, credentials, and application links | [Site maintenance](docs/portfolio.md) |

These applications have separate dependencies. Follow each project's guide in
an isolated environment; there is no repository-wide application build. Live
services may require credentials. Local engineering and ML tests do not require
calling those services.

## Preview the website

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`. This serves the static site; it does not start the
Streamlit applications. Published screenshots and live links may reflect an
earlier deployment than the checked-out source.

## Review the engineering evidence

See the [project review](docs/project-review.md) for the selected improvement
priorities, resolved defects, remaining limitations, and reproducible checks.
Source datasets and previously saved models are retained. Historical plots and
artifacts are not evidence of the revised pipelines' performance.

ML results in this repository are research evaluations. The GNN CSVs lack
complete preprocessing/physical-study provenance, the fault dataset lacks
event grouping, and short-window load forecasts lack operational validation.
Each project guide describes those boundaries and how its results can be
independently checked.
