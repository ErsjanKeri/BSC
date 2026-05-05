#!/usr/bin/env python3
"""
Generate chapter 4 (Tensor Tracing) figures from the canonical 20B 2000-token cache dump.

Outputs into thesis/figures/canon/:
  - expert_selection_heatmap.pdf  (24 layers x 32 experts, sorted by frequency within layer)
  - reuse_distance_cdf.pdf        (CDF of (layer, expert) reuse distances)

Source: BSC/tensor-tracing/traces/20b-2000tok-cache-dump/cache_dump.csv
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THESIS_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = THESIS_DIR / "figures" / "canon"
TRACE_PATH = Path("/home/keri/BSC/tensor-tracing/traces/20b-2000tok-cache-dump/cache_dump.csv")

N_LAYERS = 24
N_EXPERTS = 32


def load_dump():
    """Returns (le_freq, reuse_dists)."""
    le_freq = Counter()
    reuse_dists = []
    last_seen_token = {}

    with TRACE_PATH.open() as f:
        r = csv.DictReader(f)
        for row in r:
            layer = int(row["layer"])
            n_real = int(row["n_real"])
            p_start = int(row["p_start"])
            load_id = int(row["load_id"])
            token = load_id // 192
            if p_start != 1:
                continue
            for k in range(n_real):
                eid = int(row[f"eid{k}"])
                if eid < 0:
                    continue
                le_freq[(layer, eid)] += 1
                key = (layer, eid)
                if key in last_seen_token:
                    reuse_dists.append(token - last_seen_token[key])
                last_seen_token[key] = token

    return le_freq, reuse_dists


def plot_heatmap(le_freq):
    H = np.zeros((N_LAYERS, N_EXPERTS), dtype=int)
    for L in range(N_LAYERS):
        counts = sorted([le_freq.get((L, e), 0) for e in range(N_EXPERTS)], reverse=True)
        H[L, :] = counts

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    im = ax.imshow(H, aspect="auto", cmap="magma", origin="upper", interpolation="nearest")
    ax.set_xlabel("Expert rank within layer")
    ax.set_ylabel("Layer")
    ax.set_xticks([0, 7, 15, 23, 31])
    ax.set_xticklabels(["1", "8", "16", "24", "32"])
    ax.set_yticks(range(0, N_LAYERS, 4))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Times selected (out of 1999 tokens)")
    plt.tight_layout()
    out = FIG_DIR / "expert_selection_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def plot_reuse_cdf(reuse_dists):
    n = len(reuse_dists)
    counts = Counter(reuse_dists)
    xs = sorted(counts)
    cum = 0
    ys = []
    for d in xs:
        cum += counts[d]
        ys.append(cum / n)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogx(xs, ys, color="#c0392b", lw=2)
    ax.set_xlabel("Reuse distance (tokens)")
    ax.set_ylabel("Cumulative fraction of reuses")
    ax.set_xlim(1, max(xs))
    ax.set_ylim(0, 1.02)
    ax.grid(True, which="both", alpha=0.3)

    annotations = [(1, 0.453), (5, 0.761), (10, 0.871), (100, 0.994)]
    for x, y in annotations:
        ax.plot(x, y, "ko", markersize=4)
        ax.annotate(
            f"{int(round(y * 100))}%",
            (x, y),
            xytext=(8, -10),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    out = FIG_DIR / "reuse_distance_cdf.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    le_freq, reuse_dists = load_dump()
    print(f"loaded: {sum(le_freq.values())} selections, {len(reuse_dists)} reuses")
    plot_heatmap(le_freq)
    plot_reuse_cdf(reuse_dists)


if __name__ == "__main__":
    main()
