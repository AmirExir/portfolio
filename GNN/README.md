# Power Grid GNN

This project explores bus voltage classes and line thermal classes with GCN, GAT,
GIN, and graph Transformer models. It provides inspectable CSV graph construction,
scenario-separated evaluation, and a Streamlit interface for training and topology
inspection. It is an exploratory machine-learning example, not a validated
power-flow solver, contingency screen, or compliance assessment.

## Run

Use Python 3.13 with the existing dependencies in `requirements.txt`. From the
repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r GNN/requirements.txt
.venv/bin/python -m streamlit run GNN/app_streamlit_gnn_powergrid.py
```

All three existing entry points now run the same **Power Grid GNN** application:
`app_gnn_streamlit.py`, `app_streamlit_gnn_powergrid.py`, and
`app_streamlit_gnn_powergrid_GCN_N_2.py`. Source files resolve relative to the
project directory, so the working directory does not determine the data loaded.

The last entry point previously served a separate node-alarm experiment with
different CSVs. Selecting voltage or thermal could substitute `alarm_flag` when
the requested class column was missing, and its noisy graph copies were not
independent scenarios. It now delegates to the shared scenario application;
missing class columns stop loading with an explicit error. Combined alarms and
noise augmentation are no longer exposed by any app launcher. Original source
data are retained, and the historical implementation remains in Git history.
Existing deployments can keep their entry-point path, but must receive these
changes and restart to serve the shared app. Multiple hosted URLs are deployment
aliases, not separate GNN products.

Use **Train & compare** for model training and validation history, **Explore
topology** for source graph inspection, and **Evaluation** for held-out metrics.
The research/provenance note stays visible; detailed limitations and class counts
are in **Data and research assumptions**.

For a short execution check (one epoch does not establish model quality):

```sh
.venv/bin/python GNN/gnn_clean.py --mode voltage --epochs 1 --use_relu
.venv/bin/python GNN/gnn_clean.py --mode thermal --epochs 1 --use_relu
```

A longer experiment can save a report at a **new** path:

```sh
.venv/bin/python GNN/gnn_clean.py --mode voltage --model gcn \
  --epochs 100 --use_relu --seed 42 --report /tmp/gnn_voltage_evaluation.json
```

The report includes source CSV SHA-256 hashes, feature names, source limitations,
random seed, hyperparameters, package versions, execution time, scenario IDs in
each split, normalization parameters, selected epoch, held-out metrics, per-class
support/precision/recall/F1, confusion matrix, and a training-majority baseline.
The UI offers the same report for download. Model weights are not automatically
exported; rerunning with the recorded configuration is required to recreate them.
Torch seeds improve repeatability but do not guarantee bitwise agreement across
hardware or package versions.

## Architecture and data contract

- `data_pipeline.py`: CSV validation and one graph per scenario.
- `gnn_clean.py`: architectures, scenario grouping, training, checkpoint selection,
  evaluation, and command-line entry point.
- `app_gnn_streamlit.py`: training controls, progress, source topology and feature
  inspection, validation history, and held-out/baseline comparisons.
- `create_graph_dataset*.py`: optional exports of validated graph objects.

The voltage input requires `bus_scenarios.csv` columns `scenario`, `bus`,
`load_MW`, `p_inj_mw`, and `voltage_class`, plus `edge_scenarios.csv` columns
`scenario`, `from_bus`, `to_bus`, and `in_service`. Bus IDs must be unique within
a scenario; branch endpoints and scenario IDs must match the bus table. Node
predictors are source load, source active-power injection, and active incident
branch count. Supplied `voltage_class` integers 0–4 are preserved exactly.

Thermal input requires `edge_scenarios.csv` columns `scenario`, `from_bus`,
`to_bus`, `in_service`, `x_pu`, `length_km`, and `thermal_class`. Line predictors
are source reactance and source length. Supplied `thermal_class` integers 0–3
are preserved exactly. Every scenario needs at least one active line.

Missing/non-finite required values, duplicate bus IDs, unknown branch endpoints,
unknown equipment status, and fractional/out-of-range class IDs cause explicit
errors. Values are not silently replaced with zero. Status must be `True`/`False`
or `1`/`0` (case-insensitive text is accepted).

### Topology policy

Only rows explicitly marked `in_service=True` belong to the active graph. The
bundled edge CSV contains 776 open-line rows across its 200 scenarios; these rows
are excluded from message-passing topology, and from thermal classification
samples. The UI reports the excluded count. Source CSVs remain unchanged.

Each active branch has message edges in both directions. Parallel branches remain
parallel in the bus graph and each contributes to the incident branch count.
Thermal graphs use one node per active line, linking lines that share a bus in
both directions; duplicate adjacency links are removed. Isolated nodes remain in
the graph. Artificial self-loops are not added to source topology (individual
neural-network layers may apply their own self-connection convention). The
NetworkX topology preview collapses parallel visual edges, while model input
retains the bus graph's branch multiplicity.

### Evaluation policy

At least five distinct scenarios are required. A seeded split assigns complete
scenario groups to approximately 60% training, 20% validation, and 20% testing;
repeated copies carrying the same scenario ID stay together. For the 200 bundled
scenarios this is 120/40/40. This is a random scenario split, **not** a temporal
split or a holdout of entire grid topologies. Related scenarios need a shared
group ID if they must stay together.

The new pipeline fits normalization and inverse-frequency class weights using
training nodes only. Normalization is stored in model buffers, so direct model
inference takes the same source feature values as training. No source graph is
modified in place. Validation loss selects the checkpoint; that checkpoint is
restored before a single final test evaluation. Loss aggregation accounts for
node/class weights across batches. Missing training classes are reported. Macro
F1 and confusion matrices include all declared classes, including unsupported
ones; inspect class support before interpreting a score.

A majority classifier is fitted from training labels and evaluated on the same
test labels. Metrics shown as test results are no longer taken from the last
validation epoch. Changing UI mode, training settings, or source-file revision
clears the previous experiment's results. Repeated architecture/parameter
selection against the test report uses up its independence: use separate external
data for a final generalization claim. Starting another training attempt also
clears the prior results; if that attempt fails, no previous metrics or report
are shown as its output.

## Important limitations of the bundled data

`bus_scenarios.csv` contains negative standardized values in its `voltage` column;
`edge_scenarios.csv` also contains preprocessed quantities. The generating scaler,
physical inverse transform, and authoritative label threshold definitions are not
available with these files. The app therefore shows **source values and source
class IDs**, without assigning physical pu/MW/km/% units or regulatory meanings
to them. The historical `voltage_to_class` helper is retained for callers with
raw positive finite voltage in pu, but is not used to relabel these CSVs. Its
five demonstration bins are not verified engineering criteria.

Direct targets (`voltage`, `loading_percent`) and voltage-derived features are
excluded from predictors. However, train-only normalization here cannot undo or
establish the independence of unknown upstream preprocessing. Thus scores remain
exploratory, even with disjoint scenario IDs. Feature availability before a
contingency is also not established by the bundled provenance.

Thermal predictors contain line properties and topology only, with no scenario
load or dispatch inputs. Identical inputs can correspond to different loading
classes. This limits predictive information and makes this model an exploratory
baseline. Meaningful engineering validation needs traceable raw cases, ratings,
contingency definitions, label criteria, load/dispatch features, and independent
studies with known solver convergence. No accuracy ranges are claimed here.

The original `.pt` artifacts are preserved but are not automatically loaded by
training or the app. Pickled graph artifacts can execute code when loaded; use
only artifacts whose origin you trust. Optional export commands create new v2
files and refuse to overwrite existing paths:

```sh
.venv/bin/python GNN/create_graph_dataset.py --output /tmp/voltage_graphs_v2.pt
.venv/bin/python GNN/create_graph_dataset_thermal.py --output /tmp/thermal_graphs_v2.pt
```

The old notebooks and `generate_dataset.py` remain historical experiments outside this validated entry
path. In particular, `generate_dataset.py` emits a different schema and is not a
compatible way to regenerate these labeled CSVs. Do not overwrite the bundled
sources with its output. Its simulation dependencies are not part of the app's
supported execution path.

## Verification

No new test dependency is needed beyond the application requirements:

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s GNN/tests -v
.venv/bin/python -m compileall -q GNN
```

Deterministic tests cover preserved source labels, exclusion of target features,
active/undirected topology, parallel lines, isolated buses, malformed data,
disjoint scenario groups, training-only scaling, source immutability, restoration
of the selected checkpoint, batch-independent evaluation loss, all four model
architectures, all three Streamlit launchers in both modes, rejection of missing
class columns even when `alarm_flag` exists, actual training, retained metrics
after topology selection, and clearing stale results on a mode switch.
They also cover a successful run followed by a failed retry with unchanged settings.
