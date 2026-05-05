#!/usr/bin/env python3
"""Generate all plots for the April 21 v3 writeup.

Pulls data from three result directories (cross-session comparison validated
within 0.2% cross-day for overlap_20g and 0.1% for overlap_25g):
  - Phase 3 evening (cgroup_20260420_200322): lazy_25g, mmap_pin_25g
  - Phase 4 morning  (cgroup_20260421_063034): lazy_20g, mmap_pin_20g
  - V3 night         (cgroup_20260421_093420): overlap_* and pipeline_v3_* at
                                                both budgets, + io_monitor traces

Pipeline_v2 is intentionally excluded from plots per the advisor-meeting scope
(one-paragraph mention in summary.md only).

Outputs: PNG + PDF into ../plots/, reproducible from this single script.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE        = Path(__file__).resolve().parent
PLOTS_DIR   = HERE.parent / "plots"
RESULTS_ROOT = Path("/home/keri/BSC/time-tracking/results")

DIR_PHASE3_25G   = RESULTS_ROOT / "cgroup_20260420_200322"  # lazy/pin 25G
DIR_PHASE4_20G   = RESULTS_ROOT / "cgroup_20260421_063034"  # lazy/pin 20G
DIR_V3_NIGHT     = RESULTS_ROOT / "cgroup_20260421_093420"  # overlap/v3 both budgets + io

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi"      : 100,
    "savefig.dpi"     : 150,
    "savefig.bbox"    : "tight",
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 10,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
})

# Consistent method colors across plots.
COLOR = {
    "lazy"    : "#999999",    # grey — "no optimization"
    "pin"     : "#4C72B0",    # blue — "keep attn in RAM"
    "overlap" : "#DD8452",    # orange — "hide I/O behind compute"
    "v3"      : "#2CA02C",    # green — "first-ready split-tag dispatch"
}
LABEL = {
    "lazy"    : "lazy mmap (baseline)",
    "pin"     : "mmap + pin",
    "overlap" : "io_uring + overlap",
    "v3"      : "io_uring + pipeline v3",
}
ORDER = ["lazy", "pin", "overlap", "v3"]

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def read_metrics_csv(path: Path) -> list[dict]:
    """Read harness main-metrics CSV. Returns list of dict rows."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def summarize_config(path: Path) -> tuple[float, float, float, float]:
    """Return (eval_mean_s, eval_std_s, faults_mean, tok_per_s_mean)."""
    rows = read_metrics_csv(path)
    eval_ms = np.array([float(r["eval_time_ms"]) for r in rows])
    faults  = np.array([int(r["major_page_faults"]) for r in rows])
    eval_s  = eval_ms / 1000.0
    # tok/s = eval_tokens / eval_time
    tok_s = np.array([float(r["eval_tokens"]) / (float(r["eval_time_ms"]) / 1000.0)
                      for r in rows])
    return (float(eval_s.mean()),
            float(eval_s.std(ddof=0)),
            float(faults.mean()),
            float(tok_s.mean()))


# Config mapping: (budget, method) -> (dir, csv_name)
DATA_SOURCES = {
    (25, "lazy")   : (DIR_PHASE3_25G,  "lazy_25g.csv"),
    (25, "pin")    : (DIR_PHASE3_25G,  "mmap_pin_25g.csv"),
    (25, "overlap"): (DIR_V3_NIGHT,    "overlap_25g_c4000_lru.csv"),
    (25, "v3")     : (DIR_V3_NIGHT,    "pipeline_v3_25g_c4000_lru.csv"),

    (20, "lazy")   : (DIR_PHASE4_20G,  "lazy_20g.csv"),
    (20, "pin")    : (DIR_PHASE4_20G,  "mmap_pin_20g.csv"),
    (20, "overlap"): (DIR_V3_NIGHT,    "overlap_20g_c2700_lru.csv"),
    (20, "v3")     : (DIR_V3_NIGHT,    "pipeline_v3_20g_c2700_lru.csv"),
}


def load_all() -> dict:
    """Returns {(budget, method): {eval_mean, eval_std, faults_mean, tok_per_s_mean}}"""
    out = {}
    for k, (d, fn) in DATA_SOURCES.items():
        path = d / fn
        assert path.exists(), f"missing {path}"
        em, es, fm, ts = summarize_config(path)
        out[k] = dict(eval_mean=em, eval_std=es, faults_mean=fm, tok_per_s=ts)
    return out


def load_io_trace(cfg_dir: Path, cfg_name: str, run: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamp_s, read_mib_s) arrays for a given config's io_monitor run."""
    path = cfg_dir / f"{cfg_name}_io_run{run}.csv"
    t, r = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["timestamp_s"]))
            r.append(float(row["read_mib_s"]))
    return np.array(t), np.array(r)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def savefig(fig, stem: str) -> None:
    """Write PNG + PDF with the same stem."""
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}")
    print(f"  wrote  {stem}.png  {stem}.pdf")


def grouped_bar(ax, budgets, values_by_method, errs_by_method=None,
                ylabel="", title="", value_fmt="{:.2f}",
                show_values=True):
    """Draw a grouped bar chart: x = budgets, bars per method (4 bars/group)."""
    x = np.arange(len(budgets))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(ORDER)) * width
    for i, m in enumerate(ORDER):
        vals = np.array([values_by_method[b][m] for b in budgets])
        errs = np.array([errs_by_method[b][m]   for b in budgets]) if errs_by_method else None
        bars = ax.bar(x + offsets[i], vals, width,
                       yerr=errs, capsize=3,
                       color=COLOR[m], label=LABEL[m], edgecolor="black", linewidth=0.5)
        if show_values:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        value_fmt.format(v),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} GiB budget" for b in budgets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4,
              frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Plot 1 — tok/s by config
# ---------------------------------------------------------------------------

def plot_tok_per_sec(data):
    budgets = [25, 20]
    vals = {b: {m: data[(b, m)]["tok_per_s"] for m in ORDER} for b in budgets}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_bar(ax, budgets, vals, ylabel="Tokens per second (↑ better)",
                title="GPT-OSS-120B decode throughput by configuration and budget",
                value_fmt="{:.2f}")
    ax.set_ylim(0, max(max(v.values()) for v in vals.values()) * 1.18)
    savefig(fig, "01_tok_per_sec")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — eval time
# ---------------------------------------------------------------------------

def plot_eval_time(data):
    budgets = [25, 20]
    vals = {b: {m: data[(b, m)]["eval_mean"] for m in ORDER} for b in budgets}
    errs = {b: {m: data[(b, m)]["eval_std"]  for m in ORDER} for b in budgets}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_bar(ax, budgets, vals, errs,
                ylabel="Eval time for 2000 tokens (seconds, ↓ better)",
                title="Eval time by configuration and budget (error bars = σ across iters)",
                value_fmt="{:.1f}")
    ax.set_ylim(0, max(max(v.values()) for v in vals.values()) * 1.15)
    savefig(fig, "02_eval_time")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — speedup vs lazy
# ---------------------------------------------------------------------------

def plot_speedup_vs_lazy(data):
    budgets = [25, 20]
    vals = {}
    for b in budgets:
        lazy_t = data[(b, "lazy")]["eval_mean"]
        vals[b] = {m: lazy_t / data[(b, m)]["eval_mean"] for m in ORDER}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    grouped_bar(ax, budgets, vals,
                ylabel="Speedup over lazy mmap (×, higher is better)",
                title="Speedup relative to lazy-mmap baseline (lazy = 1.00×)",
                value_fmt="{:.3f}×")
    ax.set_ylim(0.95, max(max(v.values()) for v in vals.values()) * 1.08)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    savefig(fig, "03_speedup_vs_lazy")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — NVMe bandwidth time-series
# ---------------------------------------------------------------------------

def plot_bandwidth_timeseries():
    """Four panels (2 budgets × 2 configs with io_monitor data).

    Raw 25ms samples, no smoothing — shows the true bursty nature of NVMe
    activity and exposes the per-layer read pattern clearly."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=True)
    panels = [
        (0, 0, 25, "overlap", "overlap_25g_c4000_lru"),
        (0, 1, 25, "v3",      "pipeline_v3_25g_c4000_lru"),
        (1, 0, 20, "overlap", "overlap_20g_c2700_lru"),
        (1, 1, 20, "v3",      "pipeline_v3_20g_c2700_lru"),
    ]
    for r, c, b, m, cfg_name in panels:
        ax = axes[r, c]
        t, rd = load_io_trace(DIR_V3_NIGHT, cfg_name, run=1)
        # Raw 25ms samples, thin line.
        ax.plot(t, rd, color=COLOR[m], linewidth=0.35, alpha=0.85)
        ax.set_title(f"{b} GiB — {LABEL[m]}  (raw 25 ms samples, {len(t)} points)")
        ax.set_xlabel("Wall time (s)")
        if c == 0:
            ax.set_ylabel("NVMe read MiB/s")
        ax.grid(alpha=0.3)
        # Active-reading average (excludes zero samples).
        active_mean = float(rd[rd > 0].mean()) if (rd > 0).any() else 0.0
        # Total bytes read across the whole trace (cumulative MiB).
        total_mib = float((rd * np.diff(np.concatenate([[0], t]))).sum())
        ax.axhline(active_mean, color="black", linestyle=":", linewidth=0.9, alpha=0.7,
                   label=f"active-read avg = {active_mean:.0f} MiB/s")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.text(0.02, 0.95, f"total read ≈ {total_mib/1024:.1f} GiB",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85))
    fig.suptitle("NVMe read bandwidth during 2000-token decode (raw 25 ms samples)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "04_nvme_bandwidth_timeseries")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5 — avg bandwidth bar
# ---------------------------------------------------------------------------

def plot_bandwidth_avg():
    """Mean read MiB/s during steady-state decode (skipping first 2 s)."""
    budgets = [25, 20]
    methods_with_io = ["overlap", "v3"]   # only these have io_monitor
    vals = {b: {} for b in budgets}
    for b in budgets:
        for m in methods_with_io:
            cfg_name = {
                (25, "overlap"): "overlap_25g_c4000_lru",
                (25, "v3")     : "pipeline_v3_25g_c4000_lru",
                (20, "overlap"): "overlap_20g_c2700_lru",
                (20, "v3")     : "pipeline_v3_20g_c2700_lru",
            }[(b, m)]
            # average of runs 1 and 2
            avgs = []
            for run in (1, 2):
                t, rd = load_io_trace(DIR_V3_NIGHT, cfg_name, run=run)
                mask = (t > 2.0) & (rd > 0)
                if mask.any():
                    avgs.append(rd[mask].mean())
            vals[b][m] = float(np.mean(avgs)) if avgs else 0.0
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(budgets))
    width = 0.35
    offsets = [-width/2, width/2]
    for i, m in enumerate(methods_with_io):
        vv = np.array([vals[b][m] for b in budgets])
        bars = ax.bar(x + offsets[i], vv, width, color=COLOR[m], label=LABEL[m],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vv):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.0f} MiB/s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} GiB budget" for b in budgets])
    ax.set_ylabel("Mean NVMe read bandwidth during decode (MiB/s)")
    ax.set_title("Sustained NVMe read bandwidth — v3 pulls more bytes per unit time")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(v.values()) for v in vals.values()) * 1.15)
    savefig(fig, "05_nvme_bandwidth_avg")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6 — total bytes read from NVMe (replaces page faults)
# ---------------------------------------------------------------------------

def total_bytes_read_gib(cfg_name: str) -> float:
    """Integrate io_monitor read_bytes over the run. Returns GiB."""
    path = DIR_V3_NIGHT / f"{cfg_name}_io_run1.csv"
    total = 0.0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += float(row["read_bytes"])
    return total / (1024 ** 3)


def plot_total_bytes_read(data):
    """Total bytes read from NVMe during the run.

    lazy/pin have no io_monitor trace — estimate from major_page_faults × 4 KiB
    (one fault = one 4 KiB page read; ignores readahead so is a LOWER BOUND).
    overlap/v3 use direct io_monitor integrals.
    """
    budgets = [25, 20]
    vals = {b: {} for b in budgets}
    for b in budgets:
        for m in ORDER:
            if m in ("lazy", "pin"):
                # Estimate from page-fault count × 4 KiB (lower bound)
                faults = data[(b, m)]["faults_mean"]
                vals[b][m] = faults * 4096 / (1024 ** 3)
            else:
                cfg_name = {
                    (25, "overlap"): "overlap_25g_c4000_lru",
                    (25, "v3")     : "pipeline_v3_25g_c4000_lru",
                    (20, "overlap"): "overlap_20g_c2700_lru",
                    (20, "v3")     : "pipeline_v3_20g_c2700_lru",
                }[(b, m)]
                vals[b][m] = total_bytes_read_gib(cfg_name)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(budgets))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(ORDER)) * width
    for i, m in enumerate(ORDER):
        vv = np.array([vals[b][m] for b in budgets])
        bars = ax.bar(x + offsets[i], vv, width, color=COLOR[m], label=LABEL[m],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vv):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.1f} GiB", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} GiB budget" for b in budgets])
    ax.set_ylabel("Total bytes read from NVMe per 2000-token run (GiB)")
    ax.set_title(
        "Disk I/O per run — lazy/pin via mmap faults (estimate: faults × 4 KiB); overlap/v3 from io_monitor")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4,
              frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(v.values()) for v in vals.values()) * 1.25)
    savefig(fig, "06_bytes_read_from_nvme")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data from:")
    print(f"  {DIR_PHASE3_25G}")
    print(f"  {DIR_PHASE4_20G}")
    print(f"  {DIR_V3_NIGHT}")
    print()
    data = load_all()

    print("Summary table:")
    print(f"  {'budget':>6} {'method':>8} {'eval_s':>10} {'σ_s':>6} {'tok/s':>7} {'faults':>10}")
    for b in (25, 20):
        for m in ORDER:
            d = data[(b, m)]
            print(f"  {b:>6} {m:>8} {d['eval_mean']:>10.2f} {d['eval_std']:>6.2f} "
                  f"{d['tok_per_s']:>7.3f} {d['faults_mean']:>10.0f}")
    print()

    print(f"Generating plots into {PLOTS_DIR}:")
    plot_tok_per_sec(data)
    plot_eval_time(data)
    plot_speedup_vs_lazy(data)
    plot_bandwidth_timeseries()
    plot_bandwidth_avg()
    plot_total_bytes_read(data)
    print(f"\nDone. {len(list(PLOTS_DIR.glob('*.png')))} PNGs + "
          f"{len(list(PLOTS_DIR.glob('*.pdf')))} PDFs in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
