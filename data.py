"""Loading of the precomputed dashboard data (scripts/export_dashboard.py). No model, no event log: only CSV / JSON."""
import gzip, json, os
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ORDER = ["logistics", "order_management", "recruiting", "github_pm4py", "transfer_order", "logistics_official"]


@st.cache_data(show_spinner=False)
def list_kpis():
    """[(slug, meta)] of every exported KPI, in the order of the paper."""
    out = []
    for name in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, name, "meta.json")
        if os.path.exists(p) and os.path.exists(os.path.join(DATA, name, "prefixes.csv.gz")):
            out.append((name, json.load(open(p))))
    key = lambda s: (ORDER.index(s[1]["dataset"]) if s[1]["dataset"] in ORDER else 99, s[1]["kpi_label"])
    return sorted(out, key=key)


@st.cache_data(show_spinner=False)
def load_meta(slug):
    return json.load(open(os.path.join(DATA, slug, "meta.json")))


@st.cache_data(show_spinner="Loading explanations…")
def load_prefixes(slug):
    df = pd.read_csv(os.path.join(DATA, slug, "prefixes.csv.gz"))
    return df.sort_values(["vp_id", "depth"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading explanations…")
def load_loo(slug):
    return pd.read_csv(os.path.join(DATA, slug, "loo_top.csv.gz"))


@st.cache_data(show_spinner=False)
def load_roles(slug):
    r = pd.read_csv(os.path.join(DATA, slug, "roles.csv"))
    return r[r["scope"] == "all"].drop(columns="scope") if "scope" in r.columns else r


@st.cache_data(show_spinner="Loading counterfactuals…")
def load_cf(slug):
    p = os.path.join(DATA, slug, "cf.csv.gz")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_resource(show_spinner="Loading executions…")
def load_executions(dataset):
    """Schema + executions of a dataset, indexed by viewpoint id."""
    with gzip.open(os.path.join(DATA, dataset, "executions.json.gz"), "rt") as fh:
        d = json.load(fh)
    by_id = {e["vp_id"]: e for e in d["executions"]}
    schema = {k: v for k, v in d.items() if k != "executions"}
    schema["edge_types"] = [tuple(et) for et in schema["edge_types"]]
    return schema, by_id


@st.cache_data(show_spinner=False)
def whiskers(df_top, roles, relative=False, min_n=3):
    """±1 std of the role of each explained element over the test prefixes (identity when it occurs in >= min_n
    executions, object type otherwise; arcs by role). Returns (std, n_traces, level) arrays aligned with df_top."""
    r = roles
    ident = r[r["level"] == "identity"].set_index("key"); typ = r[r["level"] == "type"].set_index("key"); role = r[r["level"] == "role"].set_index("key")
    col = "std_rel" if relative else "std_shift_h"
    stds, ns, lv = [], [], []
    for row in df_top.itertuples():
        if row.kind == "edge":
            src, level = (role.loc[row.key] if row.key in role.index else None), "role"
        else:
            src, level = (ident.loc[row.key] if row.key in ident.index else None), "identity"
            if (src is None or src["n_traces"] < min_n) and not str(row.key).startswith("Events=") and row.type_key in typ.index:
                src, level = typ.loc[row.type_key], "type"
        stds.append(float(src[col]) * (100.0 if relative else 1.0) if src is not None else 0.0)
        ns.append(int(src["n_traces"]) if src is not None else 0); lv.append(level)
    return np.array(stds), np.array(ns), lv
