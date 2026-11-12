"""Prefix slicing of the compact executions and the union graph of a prefix and its counterfactual.

Port of xocel.explain.union_graph without torch: the drawing spec and the rendering are the same as the paper figures."""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import linear_sum_assignment

EVENT = "event"
COLORS = {"both": "#8c8c8c", "query": "#e04b4b", "cf": "#1fae7a", "vp": "#f0a30a"}
COL_GAP, ROW_Y, VP_EXTRA = 2.4, (2.8, 4.4), 1.4


# ------------------------------------------------------------------------------------------------------- prefixes
@dataclass
class Prefix:
    vp_id: str
    depth: int
    t: float
    acts: List[str]
    times: List[float]
    node_ids: Dict[str, List[str]]                      # object type -> identifiers of the objects present
    edges: List[Tuple[str, int, str, int, str, int]]    # (src type, src idx, dst type, dst idx, qualifier, uid), local indices
    vp_x: np.ndarray

    @property
    def n_events(self):
        return self.depth

    def n_nodes(self):
        return self.depth + sum(len(v) for v in self.node_ids.values())


def _rel(st, dt):
    if st == EVENT and dt == EVENT:
        return "e2e"
    return "o2e" if dt == EVENT else "o2o"


def slice_prefix(e: dict, schema: dict, depth: int) -> Prefix:
    """The prefix of execution `e` with its first `depth` events (the same elements and edge uids as the model input)."""
    t = float(e["event_times"][depth - 1])
    present, remap = {}, {}
    vp = schema["viewpoint"]
    for nt, fts in e["first_time"].items():
        # None encodes an infinite first time: -inf for the viewpoint object (always present), +inf for objects that take
        # part in no event of the execution (never present)
        m = np.array([(ft <= t) if ft is not None else (nt == vp and i == 0) for i, ft in enumerate(fts)], dtype=bool)
        present[nt] = m
        idx = np.full(len(m), -1); idx[m] = np.arange(int(m.sum())); remap[nt] = idx
    node_ids = {nt: [i for i, ok in zip(e["ids"][nt], present[nt]) if ok] for nt in e["ids"]}
    groups = defaultdict(list)
    for st, si, dt, di, q in e["edges"]:
        ok_s = si < depth if st == EVENT else bool(present[st][si])
        ok_d = di < depth if dt == EVENT else bool(present[dt][di])
        if ok_s and ok_d:
            s2 = si if st == EVENT else int(remap[st][si]); d2 = di if dt == EVENT else int(remap[dt][di])
            groups[(st, _rel(st, dt), dt)].append((st, s2, dt, d2, q))
    edges, uid = [], 0
    for et in schema["edge_types"]:
        for st, s2, dt, d2, q in groups.get(tuple(et), []):
            edges.append((st, s2, dt, d2, q, uid)); uid += 1
    counts = []
    for nt in schema.get("vp_dynamic_types", []):
        counts.append(np.log1p(float(depth if nt == EVENT else present[nt].sum())) if nt == EVENT or nt in present else 0.0)
    vp_x = np.array(list(e["x_vp"]) + counts, dtype=np.float32)
    return Prefix(e["vp_id"], depth, t, [str(a) for a in e["activities"][:depth]], [float(x) for x in e["event_times"][:depth]], node_ids, edges, vp_x)


def activity_counts(p: Prefix):
    c = defaultdict(int)
    for a in p.acts:
        c[a] += 1
    return dict(c)


# ------------------------------------------------------------------------------------------------------- matching
def _lcs_pairs(a: Sequence[str], b: Sequence[str]) -> List[Tuple[int, int]]:
    n, m = len(a), len(b)
    L = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i, j] = L[i + 1, j + 1] + 1 if a[i] == b[j] else max(L[i + 1, j], L[i, j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif L[i + 1, j] >= L[i, j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _merge_events(n_q: int, n_c: int, pairs: List[Tuple[int, int]]) -> List[dict]:
    merged, i, j = [], 0, 0
    for pi, pj in pairs + [(n_q, n_c)]:
        merged += [{"status": "query", "q": k, "c": None} for k in range(i, pi)]
        merged += [{"status": "cf", "q": None, "c": k} for k in range(j, pj)]
        if pi < n_q and pj < n_c:
            merged.append({"status": "both", "q": pi, "c": pj})
        i, j = pi + 1, pj + 1
    return merged


def type_name(schema: dict, nt: str) -> str:
    shown = schema["display"].get(nt, nt)
    if shown != nt:
        return shown
    if nt.endswith("ies"):
        nt = nt[:-3] + "y"
    elif nt.endswith("s") and not nt.endswith("ss"):
        nt = nt[:-1]
    return nt[:1].upper() + nt[1:]


def _short(s: str, maxlen: int = 20) -> str:
    m = re.match(r"^[A-Za-z_ ]+\[(.+?)\](.*)$", s)
    if m:
        s = (m.group(1) + m.group(2)).strip()
    return s if len(s) <= maxlen + 2 else s[:maxlen] + "…"


def _abbrev(s: str) -> str:
    words = s.split()
    if len(words) >= 2 and all(w[:1].isalpha() and w.isalpha() for w in words):
        return " ".join(w[0] + "." for w in words[:-1]) + " " + words[-1]
    return s


def _pair_label(a: Optional[str], b: Optional[str]) -> str:
    if a is not None and b is not None:
        if a == b:
            return _short(a)
        sa, sb = _abbrev(_short(a)), _abbrev(_short(b))
        return f"{sa} / {sb}" if len(sa) + len(sb) <= 22 else f"{sa} /\n{sb}"
    return _short(a if a is not None else b)


def build_union(q: Prefix, c: Prefix, schema: dict, q_pred_h=None, c_pred_h=None, kpi_title: str = "") -> dict:
    vp = schema["viewpoint"]
    n_q, n_c = q.n_events, c.n_events
    merged = _merge_events(n_q, n_c, _lcs_pairs(q.acts, c.acts))
    key_q, key_c, nodes, pos = {}, {}, {}, {}
    for k, ev in enumerate(merged):
        key = f"E{k}"
        if ev["q"] is not None:
            key_q[(EVENT, ev["q"])] = key
        if ev["c"] is not None:
            key_c[(EVENT, ev["c"])] = key
        act = q.acts[ev["q"]] if ev["q"] is not None else c.acts[ev["c"]]
        if ev["status"] == "both":
            tline = f"{q.times[ev['q']] - q.times[0]:.0f} h / {c.times[ev['c']] - c.times[0]:.0f} h"
        elif ev["status"] == "query":
            tline = f"{q.times[ev['q']] - q.times[0]:.0f} h"
        else:
            tline = f"{c.times[ev['c']] - c.times[0]:.0f} h"
        nodes[key] = (f"Event\n{act}\n{tline}", ev["status"])
        pos[key] = (COL_GAP * k, 0.0)
    ev_x = {k: pos[k][0] for k in nodes}

    def linked_positions(p: Prefix, nt: str, key_map: dict) -> Dict[int, set]:
        out: Dict[int, set] = defaultdict(set)
        for st, si, dt, di, _, _ in p.edges:
            if EVENT not in (st, dt) or nt not in (st, dt) or st == dt:
                continue
            o_idx, e_idx = (si, di) if st == nt else (di, si)
            ekey = key_map.get((EVENT, e_idx))
            if ekey is not None:
                out[o_idx].add(ev_x[ekey])
        return out

    vp_same = np.allclose(q.vp_x, c.vp_x, atol=1e-6)
    nodes["VP"] = (f"{type_name(schema, vp)}\n{_pair_label(q.node_ids[vp][0], c.node_ids[vp][0])}", "vp0" if vp_same else "vp")
    key_q[(vp, 0)], key_c[(vp, 0)] = "VP", "VP"

    objects: Dict[str, List[Tuple[str, set]]] = {}
    for nt in schema["node_types"]:
        if nt in (EVENT, vp):
            continue
        ids_q, ids_c = q.node_ids.get(nt, []), c.node_ids.get(nt, [])
        if not ids_q and not ids_c:
            continue
        lq, lc = linked_positions(q, nt, key_q), linked_positions(c, nt, key_c)
        pairs: List[Tuple[Optional[int], Optional[int]]] = []
        free_q, free_c = list(range(len(ids_q))), list(range(len(ids_c)))
        if free_q and free_c:
            sim = np.zeros((len(free_q), len(free_c)))
            for a, i in enumerate(free_q):
                for b, j in enumerate(free_c):
                    u, v = lq.get(i, set()), lc.get(j, set())
                    sim[a, b] = (1.0 if not u and not v else len(u & v) / max(1, len(u | v))) + (0.1 if ids_q[i] == ids_c[j] else 0.0)
            for a, b in zip(*linear_sum_assignment(-sim)):
                if sim[a, b] > 0:
                    pairs.append((free_q[a], free_c[b]))
            used_q, used_c = {p[0] for p in pairs}, {p[1] for p in pairs}
            free_q, free_c = [i for i in free_q if i not in used_q], [j for j in free_c if j not in used_c]
        pairs += [(i, None) for i in free_q] + [(None, j) for j in free_c]
        objects[nt] = []
        for n, (i, j) in enumerate(pairs):
            key = f"{nt}#{n}"
            status = "both" if i is not None and j is not None else ("query" if i is not None else "cf")
            nodes[key] = (f"{type_name(schema, nt)}\n{_pair_label(ids_q[i] if i is not None else None, ids_c[j] if j is not None else None)}", status)
            if i is not None:
                key_q[(nt, i)] = key
            if j is not None:
                key_c[(nt, j)] = key
            linked = (lq.get(i, set()) if i is not None else set()) | (lc.get(j, set()) if j is not None else set())
            objects[nt].append((key, linked))

    def edge_set(p: Prefix, key_map: dict) -> set:
        es = set()
        for st, si, dt, di, _, _ in p.edges:
            u, v = key_map.get((st, si)), key_map.get((dt, di))
            if u is not None and v is not None and u != v:
                es.add(tuple(sorted((u, v))))
        return es
    eq, ec = edge_set(q, key_q), edge_set(c, key_c)
    edges = [(u, v, "both") for u, v in sorted(eq & ec)] + [(u, v, "query") for u, v in sorted(eq - ec)] + [(u, v, "cf") for u, v in sorted(ec - eq)]
    ev_keys = [f"E{k}" for k in range(len(merged))]
    curved = {}
    for u, v, _ in edges:
        if u in ev_keys and v in ev_keys and abs(ev_keys.index(u) - ev_keys.index(v)) > 1:
            curved[(u, v)] = 0.22

    n_cols = max(1, len(merged)); capacity = n_cols + 1
    x_centre = COL_GAP * (n_cols - 1) / 2
    sides: Dict[str, List[str]] = {"above": [], "below": []}
    for nt in sorted(objects, key=lambda t: -len(objects[t])):
        side = "above" if len(sides["above"]) <= len(sides["below"]) else "below"
        sides[side].append(nt)
    rows_per_side = {}
    for side, types in sides.items():
        rows: List[list] = [[]]
        for nt in types:
            if rows[-1] and len(rows[-1]) + len(objects[nt]) > capacity and len(rows) < 2:
                rows.append([])
            rows[-1].extend(objects[nt])
        rows_per_side[side] = [r for r in rows if r]
    above: set = set()
    for side, rows in rows_per_side.items():
        sign = 1 if side == "above" else -1
        for r, row in enumerate(rows):
            y = sign * ROW_Y[r]
            bary = {k: (float(np.mean(sorted(l))) if l else x_centre) for k, l in row}
            order = sorted(bary, key=lambda k: bary[k])
            xs = [bary[k] for k in order]
            for i in range(1, len(xs)):
                xs[i] = max(xs[i], xs[i - 1] + COL_GAP)
            if len(xs) > 1:
                shift = float(np.mean([bary[k] for k in order]) - np.mean(xs))
                xs = [x + shift for x in xs]
            for n, (k, x) in enumerate(zip(order, xs)):
                pos[k] = (x, y)
                outward = n % 2 == 0
                if (side == "above") == outward:
                    above.add(k)
    top = max([ROW_Y[len(r) - 1] for s, r in rows_per_side.items() if s == "above" and r] + [ROW_Y[0]])
    pos["VP"] = (x_centre, top + VP_EXTRA); above.add("VP")
    for k in ev_keys:
        above.discard(k)
    title = f"{kpi_title + ': ' if kpi_title else ''}{type_name(schema, vp)} {_short(q.node_ids[vp][0])} ({n_q} events"
    title += f", {q_pred_h:.1f} h)" if q_pred_h is not None else ")"
    title += f" vs. {_short(c.node_ids[vp][0])} ({n_c} events" + (f", {c_pred_h:.1f} h)" if c_pred_h is not None else ")")
    return dict(nodes=nodes, edges=edges, pos=pos, above=above, curved=curved, title=title, n_cols=n_cols, key_q=key_q, key_c=key_c)


# ------------------------------------------------------------------------------------------------------- drawing
def legend_handles(spec: dict) -> List[Line2D]:
    statuses = {s for _, s in spec["nodes"].values()}; est = {s for _, _, s in spec["edges"]}
    h = []
    if "query" in statuses:
        h.append(Line2D([], [], marker="o", ls="", ms=12, color=COLORS["query"], label="only in the original prefix"))
    if "cf" in statuses:
        h.append(Line2D([], [], marker="o", ls="", ms=12, color=COLORS["cf"], label="only in counterfactual"))
    h.append(Line2D([], [], marker="o", ls="", ms=12, color=COLORS["both"], label="in both (\"original / counterfactual\")"))
    differ = "vp" in statuses
    h.append(Line2D([], [], marker="o", ls="", ms=12, color=COLORS["both"], markeredgecolor=COLORS["vp"] if differ else "black", markeredgewidth=3,
                    label="viewpoint, attributes differ" if differ else "viewpoint, same attributes"))
    h.append(Line2D([], [], color=COLORS["both"], lw=1.6, label="relation in both"))
    if "query" in est:
        h.append(Line2D([], [], color=COLORS["query"], lw=2.6, ls=(0, (5, 3)), label="relation only in the original prefix"))
    if "cf" in est:
        h.append(Line2D([], [], color=COLORS["cf"], lw=3.0, label="relation only in counterfactual"))
    return h


def draw_union(spec: dict, title: bool = True):
    """Matplotlib figure of a union-graph spec (same rendering as the paper figures)."""
    pos, nodes, edges = spec["pos"], spec["nodes"], spec["edges"]
    n_cols = spec.get("n_cols") or max(1, sum(1 for k in nodes if k.startswith("E")))
    fs = 11 if n_cols <= 10 else (9 if n_cols <= 20 else 7)
    ys = [y for _, y in pos.values()]
    figsize = (min(60.0, max(11.0, 0.95 * n_cols + 3.0)), min(16.0, max(7.2, 1.3 * (max(ys) - min(ys)) + 1.5)))
    st_ = {"both": dict(color=COLORS["both"], lw=1.6, ls="-"), "query": dict(color=COLORS["query"], lw=2.6, ls=(0, (5, 3))),
           "cf": dict(color=COLORS["cf"], lw=3.0, ls="-")}
    curved = spec.get("curved", {})
    fig, ax = plt.subplots(figsize=figsize)
    for u, v, s in edges:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        rad = curved.get((u, v), curved.get((v, u)))
        if rad is not None:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                                         color=st_[s]["color"], lw=st_[s]["lw"], ls=st_[s]["ls"], zorder=1))
        else:
            ax.plot([x1, x2], [y1, y2], zorder=1, solid_capstyle="round", **st_[s])
    for n, (lab, s) in nodes.items():
        x, y = pos[n]
        if s in ("vp", "vp0"):
            ax.scatter([x], [y], s=800, c=COLORS["both"], edgecolors=COLORS["vp"] if s == "vp" else "black", linewidths=4, zorder=3)
        else:
            ax.scatter([x], [y], s=600 if s == "both" else 800, c=COLORS[s], edgecolors="white", linewidths=1.5, zorder=3)
        up = n in spec.get("above", ())
        ax.annotate(lab, (x, y), xytext=(0, 15 if up else -15), textcoords="offset points", ha="center", va="bottom" if up else "top",
                    fontsize=fs, zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_axis_off(); ax.margins(0.08); ax.set_aspect("equal")
    if title:
        ax.set_title(spec["title"], fontsize=fs + 1, pad=24)
    handles = legend_handles(spec)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4 if len(handles) > 4 else len(handles), fontsize=fs - 1,
              frameon=False, handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout()
    return fig


def draw_prefix(p: Prefix, schema: dict, highlight_nodes=(), highlight_uids=(), title: str = ""):
    """The prefix alone (events on a row, objects above / below) with the explanation subgraph highlighted:
    nodes and relations of the subgraph in colour, the rest in grey."""
    spec = build_union(p, p, schema)
    key_q = spec["key_q"]
    hn = {key_q[(nt, i)] for nt, i in highlight_nodes if (nt, i) in key_q}
    hu = set()
    for st, si, dt, di, _, uid in p.edges:
        if uid in highlight_uids and (st, si) in key_q and (dt, di) in key_q and key_q[(st, si)] != key_q[(dt, di)]:
            hu.add(tuple(sorted((key_q[(st, si)], key_q[(dt, di)]))))
    pos, nodes = spec["pos"], spec["nodes"]
    n_cols = spec["n_cols"]; fs = 11 if n_cols <= 10 else (9 if n_cols <= 20 else 7)
    ys = [y for _, y in pos.values()]
    fig, ax = plt.subplots(figsize=(min(60.0, max(11.0, 0.95 * n_cols + 3.0)), min(16.0, max(7.2, 1.3 * (max(ys) - min(ys)) + 1.5))))
    HL = "#c8553d"
    for u, v, _ in spec["edges"]:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        on = tuple(sorted((u, v))) in hu
        rad = spec["curved"].get((u, v), spec["curved"].get((v, u)))
        kw = dict(color=HL if on else COLORS["both"], lw=3.0 if on else 1.4, zorder=2 if on else 1)
        if rad is not None:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-", connectionstyle=f"arc3,rad={rad}", **kw))
        else:
            ax.plot([x1, x2], [y1, y2], solid_capstyle="round", **kw)
    for n, (lab, s) in nodes.items():
        x, y = pos[n]
        lab = lab.replace(" / ", "/")
        if s in ("vp", "vp0"):
            ax.scatter([x], [y], s=800, c=COLORS["both"], edgecolors="black", linewidths=4, zorder=3)
        else:
            ax.scatter([x], [y], s=800 if n in hn else 600, c=HL if n in hn else COLORS["both"], edgecolors="white", linewidths=1.5, zorder=3)
        up = n in spec.get("above", ())
        ax.annotate(lab, (x, y), xytext=(0, 15 if up else -15), textcoords="offset points", ha="center", va="bottom" if up else "top",
                    fontsize=fs, zorder=4, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_axis_off(); ax.margins(0.08); ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=fs + 1, pad=24)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=12, color=HL, label="explanation subgraph (top-k nodes and arcs)"),
                       Line2D([], [], marker="o", ls="", ms=12, color=COLORS["both"], label="other elements of the prefix"),
                       Line2D([], [], marker="o", ls="", ms=12, color=COLORS["both"], markeredgecolor="black", markeredgewidth=3, label="viewpoint")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=fs - 1, frameon=False)
    fig.tight_layout()
    return fig
