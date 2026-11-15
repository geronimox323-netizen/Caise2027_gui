# Explanation explorer for object-centric predictive process monitoring

Interactive explorer of the explanations produced for the paper *Explaining Object-Centric Processes Predictions*
(leave-one-out explanations and counterfactuals of remaining-time predictions on OCEL 2.0 logs).

Everything shown was **precomputed** from the trained heterogeneous graph transformer and the event logs
(`scripts/export_dashboard.py` of the main repository): the explorer needs neither the model nor the logs, only the
files in `data/`. It covers **every test prefix** of the nine log / KPI pairs of the paper:

| view | what it shows |
|---|---|
| **Leave-one-out** | for the selected execution and prefix: bar chart of the top-k nodes and arcs by leave-one-out shift (red: the element pushes the prediction later, blue: earlier; whiskers = ±1 std of the shift of the same role over the test prefixes), the explanation subgraph (top-k nodes and top-k arcs) highlighted on the prefix graph, and the prediction along the execution |
| **Counterfactual** | for the selected execution and prefix, direction and threshold: the union graph of the original prefix and of its counterfactual (grey: in both, red: original prefix only, green: counterfactual only) and the decomposition of the dissimilarity D |
| **Aggregated Leave-One-Out** | bar chart of the k most influential elements of any kind (events, objects, arcs), by mean leave-one-out shift over the test prefixes in which they occur, with ±1 std whiskers |

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
