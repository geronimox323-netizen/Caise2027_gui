"""Bar charts of the dashboard, in the style of the paper figures (scripts/fig_loo_single.py)."""
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

POS, NEG, INK, MUTED, GRID = "#c8553d", "#2f6f9f", "#222222", "#666666", "#dddddd"
GLYPH = {"event": "o", "object": "s", "arc": ">"}


def _cut(t, n):
    return t if len(t) <= n else t[: n - 1] + "…"


def short(l: str) -> str:
    l = re.sub(r"https://api\.github\.com/users/([^/\s]+)", r"user \1", l)
    l = re.sub(r"https://api\.github\.com/repos/[^/]+/[^/]+/issues/(\d+)/timeline", r"issue \1", l)
    l = re.sub(r"Application\[(\d+)\]", r"application \1", l).replace("Events: ", "")
    m = re.match(r"^(.*?) -\((.*?)\)-> (.*)$", l)
    if m:
        return f"{_cut(m.group(1), 24)} → {_cut(m.group(3), 24)}" if m.group(2) == "follows" else f"{_cut(m.group(1), 18)} ({m.group(2)}) → {_cut(m.group(3), 18)}"
    m = re.match(r"^(.*?) -> (.*)$", l)
    if m:
        return f"{_cut(m.group(1), 24)} → {_cut(m.group(2), 26)}"
    return _cut(l, 44)


def kind_of(kind: str, key: str) -> str:
    return "arc" if kind == "edge" else ("event" if str(key).startswith("Events=") else "object")


def loo_bars(labels, kinds, vals, errs=None, relative=False, legend=True, std_label=r"$\pm$1 std"):
    """Horizontal bars of the LOO shifts (largest |shift| on top), with glyphs for the kind of element and whiskers."""
    vals = np.asarray(vals, dtype=float)[::-1]; labels = list(labels)[::-1]; kinds = list(kinds)[::-1]
    ext = np.asarray(errs, dtype=float)[::-1] if errs is not None else np.zeros(len(vals))
    n = len(vals); y = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * n + 1.5))
    ax.barh(y, vals, color=[POS if v > 0 else NEG for v in vals], height=0.6, zorder=3)
    if errs is not None:
        ax.errorbar(vals, y, xerr=ext, fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4, alpha=0.9)
    lo = min(float((vals - ext).min()), 0.0) if n else -1.0; hi = max(float((vals + ext).max()), 0.0) if n else 1.0; span = (hi - lo) or 1.0
    for yi, (v, e) in enumerate(zip(vals, ext)):
        off = e + 0.015 * span
        txt = (f"{v:+.1f}%" if relative else f"{v:+.1f}") if abs(v) >= 0.05 else ("0.0%" if relative else "0.0")
        ax.text(v + off if v > 0 else v - off, yi, txt, va="center", ha="left" if v > 0 else "right", fontsize=8, color=INK, zorder=5)
    ax.axvline(0, color=INK, lw=0.9, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    tr = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, k in zip(y, kinds):
        ax.plot(-0.012, yi, marker=GLYPH[k], ms=4.2, color=MUTED, transform=tr, clip_on=False, ls="none")
    ax.tick_params(axis="y", length=0, pad=14); ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=3)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Leave-one-out shift of the predicted remaining time, % of the execution duration" if relative else
                  "Leave-one-out shift of the predicted remaining time (h)", fontsize=8.5, color=INK)
    ax.set_xlim(lo - (0.38 if lo < 0 else 0.1) * span, hi + (0.2 if hi > 0 else 0.08) * span); ax.set_ylim(-0.6, n - 0.4)
    if legend:
        glyphs = {k: Line2D([], [], marker=GLYPH[k], ls="none", color=MUTED, ms=4.5, label=k) for k in ("event", "object", "arc")}
        std_h = Line2D([], [], color="#333333", lw=0.9, marker="|", ms=5, label=std_label) if errs is not None else Patch(color="none", label="")
        handles = [glyphs["event"], Patch(color=POS, label="presence increases the prediction"), glyphs["object"],
                   Patch(color=NEG, label="presence decreases it"), glyphs["arc"], std_h]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17 - 0.9 / max(n, 3)), ncol=3, fontsize=7, frameon=False,
                  handlelength=1.4, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout()
    return fig


def role_bars(labels, means, stds, ns, relative=False):
    """Aggregate LOO shift of the most important roles (mean ± std over the test prefixes in which the role occurs)."""
    labels = [f"{short(l)}  (n={n})" for l, n in zip(labels, ns)]
    kinds = ["arc" if "->" in str(l) else ("event" if str(l).startswith("Events=") else "object") for l in labels]
    return loo_bars(labels, kinds, means, stds, relative=relative, legend=True, std_label=r"$\pm$1 std over the prefixes")


def dissimilarity_bars(comp: dict):
    keys = ["feat", "type", "edge", "struct"]
    names = ["node attributes", "object types", "relation types", "number of arcs"]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    vals = [float(comp[k]) for k in keys]
    ax.barh(range(4), vals, color="#2f6f9f", height=0.6)
    ax.set_yticks(range(4)); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(1.0, max(vals) * 1.25)); ax.set_xlabel(f"dissimilarity component (total D = {float(comp['total']):.3f})", fontsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def prediction_timeline(df_exec, highlight_depth=None):
    """Predicted vs actual remaining time along the prefixes of one execution."""
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.plot(df_exec["depth"], df_exec["pred_h"], marker="o", ms=3.5, lw=1.2, color="#2f6f9f", label="predicted remaining time")
    ax.plot(df_exec["depth"], df_exec["true_h"], marker="s", ms=3, lw=1.0, color="#999999", ls="--", label="actual remaining time")
    if highlight_depth is not None:
        ax.axvline(highlight_depth, color="#c8553d", lw=1.0, ls=":")
    ax.set_xlabel("events observed", fontsize=8.5); ax.set_ylabel("hours", fontsize=8.5)
    ax.tick_params(labelsize=8); ax.legend(fontsize=7.5, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig
