# Application names and deployment mapping

| Public name | Portfolio launch | Source entry point |
| --- | --- | --- |
| Power Grid GNN | [Open app](https://ai-in-power-system-electrical-engineering-hw6ktvbujtbw5zygqxxj.streamlit.app/) | `GNN/app_gnn_streamlit.py` or the equivalent `GNN/app_streamlit_gnn_powergrid.py` |
| Power Fault Classifier | [Open app](https://portfolio-xmc8hpyiryrkj8acggxe6k.streamlit.app/) | `fault_classifier/fault_classifier_app.py` |
| ERCOT Grid Intelligence Dashboard | [Open app](https://portfolio-w5bmjqktrj969skqfxavxm.streamlit.app/) | `ERCOTAPI/ercotapi.py` |

The application addresses above are the existing portfolio links. Confirm the
repository, branch, and main-file settings in Streamlit Cloud when updating a
deployment; local source inspection does not establish those remote settings.

## Why there were two GNN apps

The former **Open Analyzer** link pointed to
`portfolio-kucwkosbrixdtcgkae7yvm.streamlit.app`. The screenshots of that app
match `GNN/app_streamlit_gnn_powergrid_GCN_N_2.py`, an older implementation that
substituted a combined `alarm_flag` when a voltage/thermal target was absent.
That changes the classification task and cannot support the selected label.

The portfolio now offers one GNN launch link. The older script path is a
compatibility launcher for the shared **Power Grid GNN** app, as is
`GNN/app_streamlit_gnn_powergrid.py`. Existing deployments using either path
can continue to launch after their source updates. The unified app preserves
supplied voltage/thermal class labels and reports missing schema fields as
errors. The old synthetic combined-alarm workflow is retired; its source remains
in Git history.

## Why the fault warning can remain online

Editing or testing local files does not update a hosted Streamlit process.
The repeated **Model compatibility warning** text identifies the older path
that loads the committed scikit-learn 1.6.1 pickle files. The corrected default
builds an evaluated model from the bundled CSV in the serving runtime and caches
it; it does not deserialize those legacy files.

Publish the reviewed source changes to the repository/branch used by the app,
then allow its deployment to update or reboot it in Streamlit Cloud. If the app
uses `FAULT_CLASSIFIER_ARTIFACT_DIR`, that selected bundle must be trained in a
matching environment; remove that setting to use the runtime-trained default.
No unrelated application settings or credentials need to be changed.

## Name consistency

**ERCOT Grid Intelligence Dashboard** is the name of the complete application.
Its load, forecasting, renewables, prices, outages, news, and document views
remain features within it. The former portfolio heading **ERCOT Load Forecast
Dashboard** referred to that same app. The AEP/PJM hourly forecasting example is
a separate project and retains its own launch link.

## Verify a published update

- GNN: browser and page title read **Power Grid GNN**; voltage uses bus classes
  and thermal uses line classes, with no fallback to a combined alarm label.
- Fault classifier: page title reads **Power Fault Classifier**; analysis is the
  first view and model details are separate. Default startup shows no legacy
  model-version warnings.
- ERCOT: page, browser, and portfolio headings all read **ERCOT Grid Intelligence
  Dashboard**.

These checks verify app identity and startup behavior. They do not validate
model performance or the effectiveness of any engineering/regulatory dataset.
