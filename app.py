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
MODES = {"loo": "Leave-one-out", "cf": "Counterfactual", "agg": "Aggregate (test set)"}


# ------------------------------------------------------------------------------------------------------- helpers
def kpi_label(slug, meta):
    return f"{meta['dataset_label']} — {meta['kpi_label']}"


def pick_prefix(slug, meta, prefixes):
    """Execution + prefix pickers (state scoped per case study). Returns the row of prefixes.csv of the selected prefix."""
    ids = list(prefixes["vp_id"].drop_duplicates())
    okey, dkey = f"exec_{slug}", f"depth_{slug}"
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.session_state.get(okey) not in ids:
            st.session_state[okey] = ids[0]
        vp_id = st.selectbox(f"{meta['viewpoint_label']} ({len(ids)} test executions" + (f", a {100 * meta['sample_fraction']:.0f}% sample" if "sample_fraction" in meta else "") + ")", ids, key=okey)
    sub = prefixes[prefixes["vp_id"] == vp_id].sort_values("depth")
    depths = [int(d) for d in sub["depth"]]
    with c2:
        if st.session_state.get(dkey) not in depths:
            st.session_state[dkey] = depths[-1]
        depth = st.select_slider("Prefix (events observed)", options=depths, key=dkey,
                                 format_func=lambda d: f"{d} (last before the KPI)" if d == depths[-1] else str(d))
    return sub[sub["depth"] == depth].iloc[0], sub


def fmt_h(v):
    return f"{v:.1f} h"


# ------------------------------------------------------------------------------------------------------- views
def view_loo(slug, meta, row, sub, top_k, relative, scope):
    loo = D.load_loo(slug); roles = D.load_roles(slug)
    schema, execs = D.load_executions(meta["dataset"])
    el = loo[loo["prefix_id"] == row["prefix_id"]].copy()
    el["abs"] = el["shift_h"].abs(); el = el.sort_values("abs", ascending=False)
    top = el.head(top_k).reset_index(drop=True)
    stds, ns, lv = D.whiskers(top, roles, scope=scope, relative=relative)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Predicted remaining time", fmt_h(row["pred_h"]))
    m2.metric("Actual remaining time", fmt_h(row["true_h"]))
    m3.metric("Events observed", int(row["depth"]))
    m4.metric("Elements in the prefix", f"{int(row['n_nodes'])} nodes, {int(row['n_edges'])} arcs")
    m5.metric("Characterization", f"{row['characterization']:.2f}" if np.isfinite(row["characterization"]) else "n/a",
              help="Fid⁺ / (Fid⁺ + Fid⁻) of the explanation subgraph (top-k nodes and top-k arcs)")

    vals = (100 * top["rel_shift"]) if relative else top["shift_h"]
    labels = [charts.short(l) for l in top["label"]]; kinds = [charts.kind_of(k, key) for k, key in zip(top["kind"], top["key"])]
    st.subheader(f"Top {top_k} elements by leave-one-out shift")
    st.pyplot(charts.loo_bars(labels, kinds, vals.to_numpy(), stds, relative=relative), width='content')
    scope_txt = "last-event test prefixes" if scope == "last" else "test prefixes"
    st.caption(f"Red: the presence of the element pushes the prediction later; blue: earlier. Whiskers: ±1 std of the shift of the same "
               f"role (activity, object identity or type, arc type) over all the {scope_txt} in which it occurs"
               + (f"; shifts in % of the {row['duration_h']:.0f} h the execution lasted until the KPI." if relative else "."))

    tbl = top[["kind", "label", "shift_h", "rel_shift"]].copy()
    tbl["rel_shift"] = 100 * tbl["rel_shift"]; tbl["std"] = stds; tbl["n_executions"] = ns; tbl["aggregated_by"] = lv
    tbl["kind"] = kinds
    tbl = tbl.rename(columns={"label": "element", "shift_h": "shift (h)", "rel_shift": "shift (% of duration)",
                              "std": "std of the role (%)" if relative else "std of the role (h)"})
    st.dataframe(tbl.round(2), width="stretch", hide_index=True)

    with st.expander("Explanation subgraph on the prefix graph", expanded=False):
        p = G.slice_prefix(execs[row["vp_id"]], schema, int(row["depth"]))
        sg = el[el["in_subgraph"]]
        hn = [(r.nt, int(r.idx)) for r in sg[sg["kind"] == "node"].itertuples()]
        hu = [int(r.uid) for r in sg[sg["kind"] == "edge"].itertuples()]
        st.pyplot(G.draw_prefix(p, schema, hn, hu, title=f"{meta['viewpoint_label']} {p.vp_id} after {p.depth} events: explanation subgraph ({meta['k']} nodes, {meta['k']} arcs)"),
                  width='stretch')
    with st.expander("Fidelity of the explanation subgraph"):
        c = st.columns(4)
        c[0].metric("Fidelity⁺ (h)", f"{row['fidelity_plus_h']:.1f}", help="|ŷ − Φ(prefix without the subgraph)|: higher is better")
        c[1].metric("Fidelity⁻ (h)", f"{row['fidelity_minus_h']:.1f}", help="|ŷ − Φ(subgraph only)|: lower is better")
        c[2].metric("Node sparsity", f"{100 * row['node_sparsity']:.0f}%")
        c[3].metric("Arc sparsity", f"{100 * row['edge_sparsity']:.0f}%" if np.isfinite(row["edge_sparsity"]) else "n/a")
        st.caption(f"Prediction without the subgraph: {row['pred_without_h']:.1f} h; with the subgraph only: {row['pred_only_h']:.1f} h. "
                   f"MAE of the oracle on this KPI: {meta['mae_h']:.1f} h.")
    with st.expander("Prediction along the execution"):
        st.pyplot(charts.prediction_timeline(sub, int(row["depth"])), width='content')


def view_cf(slug, meta, row, sub, direction, threshold, rank):
    cf = D.load_cf(slug)
    if cf is None:
        st.warning("No counterfactual data exported for this KPI."); return
    schema, execs = D.load_executions(meta["dataset"])
    rows = cf[(cf["prefix_id"] == row["prefix_id"]) & (cf["direction"] == direction) & (np.isclose(cf["threshold_h"], threshold))]
    rows = rows[rows["rank"] > 0].sort_values("rank")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted remaining time (query)", fmt_h(row["pred_h"]))
    m2.metric("Events observed", int(row["depth"]))
    m3.metric("Candidates in the training set", int(rows["n_candidates"].iloc[0]) if len(rows) else 0,
              help=f"training prefixes of other executions within ±{meta['depth_window']} events whose prediction is at least {threshold:g} h "
                   f"{'lower' if direction == 'lower' else 'higher'}")
    if not len(rows):
        st.warning(f"No training prefix at a similar stage is predicted at least {threshold:g} h "
                   f"{'earlier' if direction == 'lower' else 'later'} than this one: the oracle sees no realistic way to "
                   f"{'complete the execution earlier' if direction == 'lower' else 'delay it'}. Lower the threshold or flip the direction.")
        return
    best = rows[rows["rank"] == rank].iloc[0] if rank in set(rows["rank"]) else rows.iloc[0]
    st.subheader(f"Counterfactual #{int(best['rank'])}: {meta['viewpoint_label']} {best['cf_vp_id']} after {int(best['cf_depth'])} events, "
                 f"predicted {best['cf_pred_h']:.1f} h ({best['cf_pred_h'] - row['pred_h']:+.1f} h)")
    q = G.slice_prefix(execs[row["vp_id"]], schema, int(row["depth"]))
    c = G.slice_prefix(execs[best["cf_vp_id"]], schema, int(best["cf_depth"]))
    spec = G.build_union(q, c, schema, float(row["pred_h"]), float(best["cf_pred_h"]), f"{meta['dataset_label']}, {meta['kpi_label']}")
    st.pyplot(G.draw_union(spec), width='stretch')
    st.caption("Union graph of the query prefix and of its counterfactual: grey elements belong to both (labelled \"query / counterfactual\" "
               "when two objects play the same role), red ones to the query only, green ones to the counterfactual only. "
               "The ringed node is the viewpoint (orange ring: its attributes differ).")

    st.subheader(f"Top {len(rows)} candidates (lowest dissimilarity first)")
    t = rows[["rank", "cf_vp_id", "cf_depth", "cf_pred_h", "cf_true_h", "feat", "type", "edge", "struct", "total"]].copy()
    t.insert(4, "shift (h)", t["cf_pred_h"] - row["pred_h"])
    t = t.rename(columns={"cf_vp_id": meta["viewpoint_label"], "cf_depth": "events", "cf_pred_h": "predicted (h)", "cf_true_h": "actual (h)",
                          "feat": "d_feat", "type": "d_type", "edge": "d_edge", "struct": "d_struct", "total": "D"})
    st.dataframe(t.round(3), width="stretch", hide_index=True)

    with st.expander("Differences between the query and the counterfactual", expanded=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.pyplot(charts.dissimilarity_bars(best), width='content')
        with c2:
            names = schema["vp_features"]
            fd = pd.DataFrame({"feature": names, "query": q.vp_x, "counterfactual": c.vp_x})
            fd["delta"] = fd["counterfactual"] - fd["query"]
            fd = fd[(fd["query"] != 0) | (fd["counterfactual"] != 0)]
            st.markdown(f"**{meta['viewpoint_label']} node features** (standardized; counts as log(1+n))")
            st.dataframe(fd.round(3), width="stretch", hide_index=True)
        aq, ac = G.activity_counts(q), G.activity_counts(c)
        ad = pd.DataFrame([{"activity": a, "query": aq.get(a, 0), "counterfactual": ac.get(a, 0), "delta": ac.get(a, 0) - aq.get(a, 0)}
                           for a in sorted(set(aq) | set(ac)) if aq.get(a, 0) != ac.get(a, 0)])
        if len(ad):
            st.markdown("**Events per activity that differ**")
            st.dataframe(ad, width="stretch", hide_index=True)
        else:
            st.caption("The two prefixes contain the same activities with the same multiplicities.")


def view_aggregate(slug, meta, top_k, relative, scope):
    roles = D.load_roles(slug); prefixes = D.load_prefixes(slug); cf = D.load_cf(slug)
    st.subheader("Oracle and explanation quality over the test set")
    last = prefixes[prefixes["is_last"]]
    c = st.columns(5)
    sampled = "n_sampled_prefixes" in meta
    c[0].metric("Prefixes shown" if sampled else "Test prefixes", f"{len(prefixes)}",
                help=(f"random sample of {100 * meta['sample_fraction']:.0f}% of the test executions ({meta['n_sampled_executions']} of "
                      f"{meta['n_test_executions']}, all their prefixes; {meta['n_test_prefixes_full']} test prefixes in total)") if sampled
                else f"{meta['n_test_executions']} executions")
    c[1].metric("MAE of the oracle", fmt_h(meta["mae_h"]))
    c[2].metric("Mean characterization", f"{prefixes['characterization'].mean():.2f}", help="over all test prefixes")
    c[3].metric("Mean fidelity⁺", fmt_h(prefixes["fidelity_plus_h"].mean()))
    c[4].metric("Mean fidelity⁻", fmt_h(prefixes["fidelity_minus_h"].mean()))
    st.caption(("Whisker statistics (roles) are computed on the whole test set; the other figures of this page on the sample. " if sampled else "") + f"On the {len(last)} last-event prefixes: characterization {last['characterization'].mean():.2f}, "
               f"fidelity⁺ {last['fidelity_plus_h'].mean():.1f} h, fidelity⁻ {last['fidelity_minus_h'].mean():.1f} h.")

    st.subheader("Most influential roles (leave-one-out shift averaged over the prefixes in which the role occurs)")
    level = st.radio("Aggregate by", ["activity", "object type", "object identity", "arc type"], horizontal=True, key=f"lvl_{slug}")
    r = roles[roles["scope"] == scope]
    if level == "activity":
        r = r[(r["level"] == "identity") & r["key"].str.startswith("Events=")]
    elif level == "object type":
        r = r[r["level"] == "type"]
    elif level == "object identity":
        r = r[(r["level"] == "identity") & ~r["key"].str.startswith("Events=")]
    else:
        r = r[r["level"] == "role"]
    r = r[r["n_traces"] >= 3].copy()
    col, scol = ("mean_rel", "std_rel") if relative else ("mean_shift_h", "std_shift_h")
    r["abs"] = r[col].abs(); r = r.sort_values("abs", ascending=False).head(top_k)
    if len(r):
        f = 100.0 if relative else 1.0
        st.pyplot(charts.role_bars(r["key"].tolist(), (f * r[col]).to_numpy(), (f * r[scol]).to_numpy(), r["n_traces"].tolist(), relative=relative),
                  width='content')
        st.caption(f"n = executions in which the role occurs; whiskers = ±1 std over the {'last-event ' if scope == 'last' else ''}test prefixes. "
                   "Roles occurring in fewer than 3 executions are not shown.")
        st.dataframe(r[["key", col, scol, "n", "n_traces"]].rename(columns={"key": "role", col: "mean shift", scol: "std", "n": "prefixes", "n_traces": "executions"}).round(3),
                     width="stretch", hide_index=True)
    else:
        st.info("No role of this kind occurs in at least 3 executions.")

    if cf is not None:
        st.subheader("Counterfactual coverage")
        g = cf.groupby(["direction", "threshold_h"]).agg(prefixes=("prefix_id", "nunique"),
                                                         covered=("n_candidates", lambda s: int((s > 0).sum() / 3)),
                                                         median_candidates=("n_candidates", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else 0.0)).reset_index()
        best = cf[cf["rank"] == 1].groupby(["direction", "threshold_h"]).agg(median_shift_h=("cf_pred_h", lambda s: 0.0), median_D=("total", "median")).reset_index()
        sh = cf[cf["rank"] == 1].assign(shift=lambda x: x["cf_pred_h"] - x["query_pred_h"]).groupby(["direction", "threshold_h"])["shift"].median().reset_index()
        g = g.merge(sh, on=["direction", "threshold_h"], how="left").merge(best[["direction", "threshold_h", "median_D"]], on=["direction", "threshold_h"], how="left")
        g["coverage"] = g["covered"] / g["prefixes"]
        g["direction"] = g["direction"].map({"lower": "earlier (lower prediction)", "higher": "later (higher prediction)"})
        st.dataframe(g.rename(columns={"threshold_h": "threshold τ (h)", "covered": "prefixes with a counterfactual", "shift": "median shift of the best (h)",
                                       "median_D": "median D of the best"}).round(3), width="stretch", hide_index=True)
        st.caption("A prefix is covered when at least one training prefix of another execution, within the depth window, is predicted "
                   "at least τ hours earlier (or later). The search is exhaustive on the dissimilarity D, so every returned counterfactual is valid.")


# ------------------------------------------------------------------------------------------------------- layout
def main():
    kpis = D.list_kpis()
    if not kpis:
        st.error("No data found in dashboard/data: run scripts/export_dashboard.py first."); return
    with st.sidebar:
        st.title("Explanation explorer")
        slug = st.selectbox("Log and viewpoint", [s for s, _ in kpis], format_func=lambda s: kpi_label(s, dict(kpis)[s]))
        meta = D.load_meta(slug)
        st.caption(f"Remaining time until *{meta['target_activity'] if meta['target_activity'] != '*' else 'the end of the execution'}* "
                   f"of each {meta['viewpoint_label']}. τ = {meta['tau_h']:g} h.")
        mode = st.radio("View", list(MODES), format_func=MODES.get)
        top_k, relative, scope, direction, threshold, rank = 8, False, "last", "lower", meta["tau_h"], 1
        if mode in ("loo", "agg"):
            top_k = st.select_slider("Top k", options=[5, 8, 10, 15, 20], value=8)
            relative = st.checkbox("Shifts relative to the execution duration", value=False)
            scope = st.radio("Whiskers over", ["last-event prefixes", "all test prefixes"], horizontal=True)
            scope = "last" if scope.startswith("last") else "all"
            st.caption("Every node (except the viewpoint) and every arc of the prefix is masked in turn; the shift of the prediction "
                       "measures how much the oracle relies on it. The k nodes and k arcs with the largest shift form the explanation subgraph.")
        else:
            direction = st.radio("Search direction", ["Earlier than predicted", "Later than predicted"])
            direction = "lower" if direction.startswith("Earlier") else "higher"
            threshold = st.select_slider("Minimum change of the prediction (h)", options=meta["thresholds_h"], value=meta["tau_h"],
                                         format_func=lambda v: v if isinstance(v, str) else f"{v:g} h" + (" (τ of the paper)" if v == meta["tau_h"] else ""))
            rank = st.radio("Candidate shown", [1, 2, 3], horizontal=True)
            st.caption("The counterfactual is a real prefix of the training set, at a similar stage, whose prediction differs by at least "
                       "the threshold; the least dissimilar one is shown as a union graph.")
        st.divider()
        st.caption("All explanations were precomputed from the trained oracle; this explorer needs neither the model nor the event logs.")

    st.title(kpi_label(slug, meta))
    prefixes = D.load_prefixes(slug)
    if mode == "agg":
        view_aggregate(slug, meta, top_k, relative, scope); return
    row, sub = pick_prefix(slug, meta, prefixes)
    if mode == "loo":
        view_loo(slug, meta, row, sub, top_k, relative, scope)
    else:
        view_cf(slug, meta, row, sub, direction, threshold, rank)


if __name__ == "__main__":
    main()
