---
title: Object-centric explanation explorer
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.67.0"
app_file: app.py
pinned: false
---

# Explanation explorer for object-centric predictive process monitoring

Interactive explorer of the explanations produced for the paper *Explaining Object-Centric Processes Predictions*
(leave-one-out explanations and counterfactuals of remaining-time predictions on OCEL 2.0 logs).

Everything shown was **precomputed** from the trained heterogeneous graph transformer and the event logs
(`scripts/export_dashboard.py` of the main repository): the explorer needs neither the model nor the logs, only the
files in `data/`. It covers **every test prefix** of the nine log / KPI pairs of the paper:

| view | what it shows |
|---|---|
| **Leave-one-out** | for the selected execution and prefix: the top-k nodes and arcs by leave-one-out shift (red: the element pushes the prediction later, blue: earlier), whiskers = ±1 std of the shift of the same role over the test prefixes, the explanation subgraph highlighted on the prefix graph, fidelity⁺ / fidelity⁻ / characterization, and the prediction along the execution |
| **Counterfactual** | the closest training prefix (dissimilarity D) whose prediction is at least τ hours lower (or higher), drawn as the union graph of the two prefixes, with the three best candidates, the components of D, the viewpoint attributes and the activities that differ. Two directions and three thresholds (τ/2, τ, 2τ) were precomputed |
| **Aggregate** | oracle MAE and mean fidelity over the test set, the most influential roles (activities, object types / identities, arc types) with their mean ± std shift, and the coverage of the counterfactual search |

The button *Load the paper example* selects the prefix used in the tables and figures of the paper.

## Run

Python 3.10+ is required.

```
python3 run_dashboard.py          # creates .venv, installs requirements.txt, starts http://localhost:8501
```

or, in an environment of your own:

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — the Streamlit application; `data.py` loads the precomputed files; `graphs.py` slices prefixes out of the
  compact executions and draws prefix / union graphs; `charts.py` draws the bar charts.
- `data/<log>/executions.json.gz` — compact copy of the process executions (events, objects, relations, viewpoint
  attributes) used to draw graphs; `data/<log>__<kpi>/` — `meta.json`, `prefixes.csv.gz` (prediction and fidelity of
  every test prefix), `loo_top.csv.gz` (top-20 nodes and arcs of every prefix), `roles.csv` (shift statistics per role),
  `cf.csv.gz` (counterfactual grid); `data/paper_examples.json` — the prefixes shown in the paper.

Anonymised artifact of a submission under double-blind review.
