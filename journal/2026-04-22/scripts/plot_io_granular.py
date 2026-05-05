#!/usr/bin/env python3
"""First-10s granular NVMe read-bandwidth plots, stacked-panel style.

Mirrors the style of /home/keri/BSC/time-tracking/io_plots/april06_uring_pack/
(fill_between + thin line overlay + dashed mean + red SSD-ceiling reference).

One PNG + PDF per (model, budget): 3 stacked panels (overlap / v2 / v3), first
10 s of run-1 decode, raw 25 ms samples.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE      = Path(__file__).resolve().parent
PLOTS_DIR = HERE.parent / "plots" / "io_zoom"
DATA_DIR  = Path("/home/keri/BSC/time-tracking/results/cgroup_20260421_170500")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# SSD theoretical ceiling (same spec as april06 baseline plot).
SSD_CAPACITY_MIB = 7000
ZOOM_S = 10.0  # first 10 seconds

plt.rcParams.update({
    "figure.dpi"      : 100,
    "savefig.dpi"     : 150,
    "savefig.bbox"    : "tight",
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 9,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
})

COLOR = {
    "overlap": "#DD8452",  # orange
    "v2"     : "#8172B3",  # purple
    "v3"     : "#2CA02C",  # green
}
LABEL = {
    "overlap": "io_uring + overlap",
    "v2"     : "io_uring + pipeline v2",
    "v3"     : "io_uring + pipeline v3",
}
ORDER = ["overlap", "v2", "v3"]

# (model, budget) → { method → cfg filename prefix }
CONFIGS = {
    ("120B", 20): {
        "overlap": "overlap_120b_20g_c2700_lru_mt",
        "v2"     : "pipeline_v2_120b_20g_c2700_lru_mt",
        "v3"     : "pipeline_v3_120b_20g_c2700_lru_mt",
    },
    ("120B", 25): {
        "overlap": "overlap_120b_25g_c4000_lru_mt",
        "v2"     : "pipeline_v2_120b_25g_c4000_lru_mt",
        "v3"     : "pipeline_v3_120b_25g_c4000_lru_mt",
    },
    ("20B", 7): {
        "overlap": "overlap_20b_7g_c250_lfua3_mt",
        "v2"     : "pipeline_v2_20b_7g_c250_lfua3_mt",
        "v3"     : "pipeline_v3_20b_7g_c250_lfua3_mt",
    },
    ("20B", 8): {
        "overlap": "overlap_20b_8g_c498_lfua3_mt",
        "v2"     : "pipeline_v2_20b_8g_c498_lfua3_mt",
        "v3"     : "pipeline_v3_20b_8g_c498_lfua3_mt",
    },
    ("20B", 9): {
        "overlap": "overlap_20b_9g_c740_lfua3_mt",
        "v2"     : "pipeline_v2_20b_9g_c740_lfua3_mt",
        "v3"     : "pipeline_v3_20b_9g_c740_lfua3_mt",
    },
}
CONFIG_ORDER = [("120B", 20), ("120B", 25), ("20B", 7), ("20B", 8), ("20B", 9)]


def load_io_trace(cfg_name: str, iter_num: int = 1):
    path = DATA_DIR / f"{cfg_name}_io_run{iter_num}.csv"
    t, r = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["timestamp_s"]))
            r.append(float(row["read_mib_s"]))
    return np.array(t), np.array(r)


def stats_in_window(t, r, t_start, t_end, idle_threshold=100.0):
    """Mean of active samples in [t_start, t_end]; total read bytes in GiB."""
    mask = (t >= t_start) & (t <= t_end)
    t_w = t[mask]; r_w = r[mask]
    active = r_w[r_w > idle_threshold]
    mean_active = float(active.mean()) if len(active) else 0.0
    # Integrate bytes: rate * dt
    if len(t_w) >= 2:
        dt = np.diff(t_w, prepend=t_w[0])
        total_mib = float((r_w * dt).sum())
    else:
        total_mib = 0.0
    return mean_active, total_mib / 1024.0


def plot_zoom_one_config(key, zoom_s, iter_num=1):
    """One PNG/PDF: 3 stacked panels for one (model, budget)."""
    model, budget = key
    methods = CONFIGS[key]

    fig, axes = plt.subplots(len(ORDER), 1, figsize=(16, 9), sharex=True)
    y_max = 0.0
    # First pass: determine y-limit from data (so all 3 panels share a scale).
    traces = {}
    for m in ORDER:
        t, r = load_io_trace(methods[m], iter_num=iter_num)
        mask = t <= zoom_s
        traces[m] = (t[mask], r[mask])
        if len(r[mask]):
            y_max = max(y_max, float(r[mask].max()))
    y_lim = max(SSD_CAPACITY_MIB * 1.02, y_max * 1.1)

    for ax, m in zip(axes, ORDER):
        t, r = traces[m]
        color = COLOR[m]

        # Filled area + thin line overlay.
        ax.fill_between(t, r, alpha=0.35, color=color)
        ax.plot(t, r, color=color, alpha=0.85, linewidth=0.45)

        # Mean of active samples in the zoom window.
        mean_active, total_gib = stats_in_window(t, r, 0, zoom_s)

        # Dashed mean line + dashed SSD ceiling.
        ax.axhline(mean_active, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label=f"active mean {mean_active:.0f} MiB/s")
        ax.axhline(SSD_CAPACITY_MIB, color="red", linestyle="--", linewidth=1.0, alpha=0.35,
                   label=f"SSD ceiling {SSD_CAPACITY_MIB} MiB/s")

        title = (f"{LABEL[m]}    ·    first {zoom_s:.0f}s, run {iter_num}    ·    "
                 f"active mean {mean_active:.0f} MiB/s    ·    "
                 f"read in window {total_gib:.1f} GiB")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Read MiB/s")
        ax.set_ylim(0, y_lim)
        ax.grid(alpha=0.35)
        ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Wall time (s)")
    axes[-1].set_xlim(0, zoom_s)

    fig.suptitle(f"NVMe read bandwidth — first {zoom_s:.0f}s of decode · "
                 f"GPT-OSS-{model} · {budget} GiB cgroup",
                 fontsize=14, y=1.003)
    fig.tight_layout()

    stem = f"io_first{int(zoom_s)}s_{model.lower()}_{budget}g"
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}")
    print(f"  wrote {stem}.png / .pdf")
    plt.close(fig)


def plot_comparison_single_panel(zoom_s, iter_num=1):
    """Bonus: 5-row (one per config) × 1-col, all 3 methods overlaid per panel.
    Useful for a single-glance cross-budget comparison."""
    fig, axes = plt.subplots(len(CONFIG_ORDER), 1,
                              figsize=(16, 2.8 * len(CONFIG_ORDER)),
                              sharex=True)
    for ax, key in zip(axes, CONFIG_ORDER):
        model, budget = key
        for m in ORDER:
            t, r = load_io_trace(CONFIGS[key][m], iter_num=iter_num)
            mask = t <= zoom_s
            ax.fill_between(t[mask], r[mask], alpha=0.18, color=COLOR[m])
            ax.plot(t[mask], r[mask], color=COLOR[m], alpha=0.85, linewidth=0.5,
                    label=LABEL[m])
        ax.axhline(SSD_CAPACITY_MIB, color="red", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.set_title(f"GPT-OSS-{model} · {budget} GiB", fontsize=11, loc="left")
        ax.set_ylabel("Read MiB/s")
        ax.grid(alpha=0.35)
        if key == CONFIG_ORDER[0]:
            ax.legend(loc="upper right", ncol=3, framealpha=0.9, fontsize=9)
    axes[-1].set_xlabel("Wall time (s)")
    axes[-1].set_xlim(0, zoom_s)
    fig.suptitle(f"NVMe read bandwidth — first {zoom_s:.0f}s, all configs overlaid",
                 fontsize=13, y=1.003)
    fig.tight_layout()
    stem = f"io_first{int(zoom_s)}s_all_overlaid"
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}")
    print(f"  wrote {stem}.png / .pdf")
    plt.close(fig)


def main():
    print(f"Writing io-zoom plots to {PLOTS_DIR}")
    for key in CONFIG_ORDER:
        plot_zoom_one_config(key, zoom_s=ZOOM_S)
    plot_comparison_single_panel(zoom_s=ZOOM_S)
    # Also: 5 s zoom for even tighter structure view (matches april06 startup_5s style).
    for key in CONFIG_ORDER:
        plot_zoom_one_config(key, zoom_s=5.0)
    print("\nDone.")

if __name__ == "__main__":
    main()
