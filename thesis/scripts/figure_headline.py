#!/usr/bin/env python3
"""Generate figures/canon/headline.pdf for the introduction.

Two panels, default lazy mmap (gray) vs the new backend (red).

  - GPT-OSS-20B: grouped bars at 7, 8, and 9 GiB RAM caps. Dashed line marks
    the 14.32 tok/s ceiling reached at 22 GiB, when the whole model fits in
    RAM and the default loader is no longer paging.
  - GPT-OSS-120B: grouped bars at 12 and 28 GiB. The model is 61 GiB and
    never fits in the 30 GiB of RAM on the test machine, so there is no
    in-RAM ceiling for this model.

Numbers come from _meta/final_canon.json. The "new backend" series picks the
best canonical configuration per cap (LFU-aging at 7 GiB on 20B because the
cache is below the per-token working set, LRU at every other cap).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ORACLE_PATH = Path("/home/keri/BSC/thesis/_meta/final_canon.json")
OUT = Path("/home/keri/BSC/thesis/figures/canon/headline.pdf")

with ORACLE_PATH.open() as f:
    oracle = json.load(f)


def tps(key: str) -> float:
    return oracle["configs"][key]["tok_per_s_mean"]


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


COL_DEFAULT = "#7f7f7f"
COL_OURS    = "#d62728"
COL_CEIL    = "#2ca02c"


def annotate(ax, x, y, s, dy):
    ax.text(x, y + dy, s, ha="center", va="bottom", fontsize=8.5)


fig, (ax20, ax120) = plt.subplots(1, 2, figsize=(11.0, 4.8))

# --- 20B panel ---
caps_20 = [7, 8, 9]
default_20 = [
    tps("lazy_20b_7g"),
    tps("lazy_20b_8g"),
    tps("lazy_20b_9g"),
]
ours_20 = [
    tps("async_projection_overlap_lfua_20b_7g"),
    tps("async_projection_overlap_lru_20b_8g"),
    tps("async_projection_overlap_lru_20b_9g"),
]
ceiling_20 = tps("lazy_20b_22g")

x = np.arange(len(caps_20))
w = 0.36

ax20.bar(x - w/2, default_20, w, color=COL_DEFAULT)
ax20.bar(x + w/2, ours_20, w, color=COL_OURS)
for xi, v in zip(x - w/2, default_20):
    annotate(ax20, xi, v, f"{v:.2f}", dy=0.20)
for xi, v in zip(x + w/2, ours_20):
    annotate(ax20, xi, v, f"{v:.2f}", dy=0.20)

ax20.axhline(ceiling_20, linestyle="--", color=COL_CEIL, linewidth=1.4, zorder=2)

ax20.set_xticks(x)
ax20.set_xticklabels([f"{c} GiB" for c in caps_20])
ax20.set_xlabel("RAM available")
ax20.set_ylabel("tokens / second")
ax20.set_title("GPT-OSS-20B (12.85 GiB on disk)")
ax20.set_ylim(0, ceiling_20 + 3.2)
ax20.grid(axis="y", linestyle=":", alpha=0.5)

legend20 = [
    Patch(facecolor=COL_DEFAULT, label="default llama.cpp loader"),
    Patch(facecolor=COL_OURS,    label="our backend"),
    Line2D([0], [0], color=COL_CEIL, linestyle="--", linewidth=1.4,
           label=f"22 GiB ceiling: {ceiling_20:.2f} tok/s (model fits in RAM)"),
]
ax20.legend(handles=legend20, loc="upper left", frameon=True,
            edgecolor="lightgray", facecolor="white", framealpha=0.95)

# --- 120B panel ---
caps_120 = [12, 28]
default_120 = [tps("lazy_120b_12g"), tps("lazy_120b_28g")]
ours_120 = [
    tps("async_projection_overlap_lru_120b_12g"),
    tps("async_projection_overlap_lru_120b_28g"),
]

x = np.arange(len(caps_120))

ax120.bar(x - w/2, default_120, w, color=COL_DEFAULT)
ax120.bar(x + w/2, ours_120, w, color=COL_OURS)
for xi, v in zip(x - w/2, default_120):
    annotate(ax120, xi, v, f"{v:.2f}", dy=0.15)
for xi, v in zip(x + w/2, ours_120):
    annotate(ax120, xi, v, f"{v:.2f}", dy=0.15)

ax120.set_xticks(x)
ax120.set_xticklabels([f"{c} GiB" for c in caps_120])
ax120.set_xlabel("RAM available")
ax120.set_ylabel("tokens / second")
ax120.set_title("GPT-OSS-120B (61 GiB on disk, never fits in 30 GiB of RAM)")
ax120.set_ylim(0, max(ours_120) + 1.8)
ax120.grid(axis="y", linestyle=":", alpha=0.5)

legend120 = [
    Patch(facecolor=COL_DEFAULT, label="default llama.cpp loader"),
    Patch(facecolor=COL_OURS,    label="our backend"),
]
ax120.legend(handles=legend120, loc="upper left", frameon=True,
             edgecolor="lightgray", facecolor="white", framealpha=0.95)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print(f"wrote {OUT}")
