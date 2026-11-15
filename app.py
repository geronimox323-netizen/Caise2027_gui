"""Explanation explorer for object-centric predictive process monitoring (static version).

Every explanation shown here was precomputed by scripts/export_dashboard.py from the trained heterogeneous graph
transformer and the event logs; the dashboard itself needs neither. It covers every test prefix of every KPI of the
paper: leave-one-out explanations with fidelity metrics, and counterfactuals searched among the training prefixes for
two directions and three thresholds.

Run (from this folder):  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

import charts
import data as D
import graphs as G

st.set_page_config(page_title="Object-centric explanation explorer", layout="wide")
MODES = {"loo": "Leave-one-out", "cf": "Counterfactual", "agg": "Aggregated Leave-One-Out"}


# ------------------------------------------------------------------------------------------------------- helpers
def kpi_label(slug, meta):
    return f"{meta['dataset_label']} — {meta['kpi_label']}"


def pick_prefix(slug, meta, prefixes, valid=None):
    """Execution + prefix pickers (state scoped per case study). Returns the row of prefixes.csv of the selected prefix.
    valid: optional set of prefix_ids that have something to show (counterfactual view); the pickers then default to an
    execution and to the deepest prefix in that set, and remember the choice of the visitor afterwards."""
    ids = list(prefixes["vp_id"].drop_duplicates())
    okey, dkey, vkey = f"exec_{slug}", f"depth_{slug}", f"valid_{slug}"
    if valid is not None and st.session_state.get(vkey) != valid:
        # new threshold / direction (or first visit of the view): move to a prefix that has a counterfactual
        st.session_state[vkey] = valid
        ok = prefixes[prefixes["prefix_id"].isin(valid)]
        cur = st.session_state.get(okey)
        if cur not in set(ok["vp_id"]) and len(ok):
            cur = ok["vp_id"].iloc[0]; st.session_state[okey] = cur
        d = ok[ok["vp_id"] == cur]["depth"]
        if len(d):
            st.session_state[dkey] = int(d.max())
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.session_state.get(okey) not in ids:
            st.session_state[okey] = ids[0]
        vp_id = st.selectbox("Execution", ids, key=okey, label_visibility="collapsed")
    sub = prefixes[prefixes["vp_id"] == vp_id].sort_values("depth")
    depths = [int(d) for d in sub["depth"]]
    with c2:
        if st.session_state.get(dkey) not in depths:
            st.session_state[dkey] = depths[-1]
        depth = st.select_slider("Prefix (events observed)", options=depths, key=dkey,
                                 format_func=str)
    return sub[sub["depth"] == depth].iloc[0], sub


def fmt_h(v):
    return f"{v:.1f} h"


# ------------------------------------------------------------------------------------------------------- views
def view_loo(slug, meta, row, sub, top_k, relative):
    loo = D.load_loo(slug); roles = D.load_roles(slug)
    schema, execs = D.load_executions(meta["dataset"])
    el = loo[loo["prefix_id"] == row["prefix_id"]].copy()
    el["abs"] = el["shift_h"].abs(); el = el.sort_values("abs", ascending=False)
    top = el.head(top_k).reset_index(drop=True)
    stds, _, _ = D.whiskers(top, roles, relative=relative)

    vals = (100 * top["rel_shift"]) if relative else top["shift_h"]
    labels = [charts.short(l) for l in top["label"]]; kinds = [charts.kind_of(k, key) for k, key in zip(top["kind"], top["key"])]
    st.pyplot(charts.loo_bars(labels, kinds, vals.to_numpy(), stds, relative=relative), width='content')

    with st.expander("Explanation subgraph on the prefix graph", expanded=False):
        p = G.slice_prefix(execs[row["vp_id"]], schema, int(row["depth"]))
        sg = el[el["rank"] <= top_k]
        hn = [(r.nt, int(r.idx)) for r in sg[sg["kind"] == "node"].itertuples()]
        hu = [int(r.uid) for r in sg[sg["kind"] == "edge"].itertuples()]
        st.pyplot(G.draw_prefix(p, schema, hn, hu, title=f"{meta['viewpoint_label']} {p.vp_id} after {p.depth} events: explanation subgraph (top {top_k} nodes, top {top_k} arcs)"),
                  width='stretch')
    with st.expander("Prediction along the execution"):
        st.pyplot(charts.prediction_timeline(sub, int(row["depth"])), width='content')


def view_cf(slug, meta, row, sub, direction, threshold, rank):
    cf = D.load_cf(slug)
    if cf is None:
        st.warning("No counterfactual data exported for this KPI."); return
    schema, execs = D.load_executions(meta["dataset"])
    rows = cf[(cf["prefix_id"] == row["prefix_id"]) & (cf["direction"] == direction) & (np.isclose(cf["threshold_h"], threshold))]
    rows = rows[rows["rank"] > 0].sort_values("rank")
    if not len(rows):
        st.info(f"No counterfactual for this prefix: no training prefix at a similar stage is predicted at least {threshold:g} h "
                f"{'earlier' if direction == 'lower' else 'later'}. Choose another prefix, lower the threshold or flip the direction.")
        return
    best = rows[rows["rank"] == rank].iloc[0] if rank in set(rows["rank"]) else rows.iloc[0]
    q = G.slice_prefix(execs[row["vp_id"]], schema, int(row["depth"]))
    c = G.slice_prefix(execs[best["cf_vp_id"]], schema, int(best["cf_depth"]))
    spec = G.build_union(q, c, schema, float(row["pred_h"]), float(best["cf_pred_h"]), f"{meta['dataset_label']}, {meta['kpi_label']}")
    st.pyplot(G.draw_union(spec), width='stretch')
    st.pyplot(charts.dissimilarity_bars(best), width='content')


def view_aggregate(slug, meta, top_k, relative):
    roles = D.load_roles(slug)
    # one ranking over every kind of element: events (by activity), objects (by identity when seen in >= 3 executions, by type
    # otherwise) and arcs (by type)
    ident = roles[(roles["level"] == "identity") & (roles["n_traces"] >= 3)]
    covered = set(ident[~ident["key"].str.startswith("Events=")]["key"].str.split("=").str[0])
    typ = roles[(roles["level"] == "type") & ~roles["key"].isin(covered)]
    arcs = roles[roles["level"] == "role"]
    r = pd.concat([ident, typ, arcs]); r = r[r["n_traces"] >= 3].copy()
    col, scol = ("mean_rel", "std_rel") if relative else ("mean_shift_h", "std_shift_h")
    r["abs"] = r[col].abs(); r = r.sort_values("abs", ascending=False).head(top_k)
    if len(r):
        f = 100.0 if relative else 1.0
        st.pyplot(charts.role_bars(r["key"].tolist(), (f * r[col]).to_numpy(), (f * r[scol]).to_numpy(), r["n_traces"].tolist(), relative=relative),
                  width='content')
    else:
        st.info("No element occurs in at least 3 executions.")



# ------------------------------------------------------------------------------------------------------- layout
def main():
    kpis = D.list_kpis()
    if not kpis:
        st.error("No data found in dashboard/data: run scripts/export_dashboard.py first."); return
    with st.sidebar:
        st.title("Explanation explorer")
        slug = st.selectbox("Log and viewpoint", [s for s, _ in kpis], format_func=lambda s: kpi_label(s, dict(kpis)[s]))
        meta = D.load_meta(slug); prefixes = D.load_prefixes(slug)
        mode = st.radio("View", list(MODES), format_func=MODES.get)
        top_k, relative, direction, threshold, rank = meta["k"], False, "lower", meta["tau_h"], 1
        if mode in ("loo", "agg"):
            top_k = st.select_slider("Top k", options=D.ks_available(prefixes, meta["k"]), value=meta["k"],
                                     help="elements listed and size of the explanation subgraph (top-k nodes and top-k arcs) on which fidelity and characterization are computed")
            relative = st.checkbox("Shifts relative to the execution duration", value=False)
        else:
            direction = st.radio("Search direction", ["Earlier than predicted", "Later than predicted"])
            direction = "lower" if direction.startswith("Earlier") else "higher"
            threshold = st.select_slider("Minimum change of the prediction (h)", options=meta["thresholds_h"], value=meta["tau_h"],
                                         format_func=lambda v: v if isinstance(v, str) else f"{v:g} h" + (" (τ of the paper)" if v == meta["tau_h"] else ""))
            rank = st.radio("Candidate shown", [1, 2, 3], horizontal=True)

    st.title(kpi_label(slug, meta))
    prefixes = D.load_prefixes(slug)
    if mode == "agg":
        view_aggregate(slug, meta, top_k, relative); return
    valid = None
    if mode == "cf":
        cf = D.load_cf(slug)
        if cf is not None:
            ok = cf[(cf["direction"] == direction) & np.isclose(cf["threshold_h"], threshold) & (cf["rank"] > 0)]
            valid = frozenset(ok["prefix_id"])
    row, sub = pick_prefix(slug, meta, prefixes, valid)
    if mode == "loo":
        row = D.metrics_for_k(row.to_frame().T, top_k).iloc[0]; sub = D.metrics_for_k(sub, top_k)
        view_loo(slug, meta, row, sub, top_k, relative)
    else:
        view_cf(slug, meta, row, sub, direction, threshold, rank)


if __name__ == "__main__":
    main()
