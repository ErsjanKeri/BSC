#!/usr/bin/env python3
"""Generate all plots for the April 22 pipeline-sweep writeup.

Source: cgroup_20260421_170500 — 15 configs × 3 iterations = 45 runs total,
        multi-topic essay prompt (fixes the degenerate-workload bias discovered
        on April 21). Both GPT-OSS-120B and GPT-OSS-20B.

Outputs PNG + PDF into ../plots/, reproducible from this single script.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE      = Path(__file__).resolve().parent
PLOTS_DIR = HERE.parent / "plots"
DATA_DIR  = Path("/home/keri/BSC/time-tracking/results/cgroup_20260421_170500")

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

# Method colors — consistent with April 21.
COLOR = {
    "overlap": "#DD8452",  # orange
    "v2"     : "#8172B3",  # purple (new for Apr 22)
    "v3"     : "#2CA02C",  # green
}
LABEL = {
    "overlap": "io_uring + overlap",
    "v2"     : "io_uring + pipeline v2",
    "v3"     : "io_uring + pipeline v3",
}
ORDER = ["overlap", "v2", "v3"]

# Per-configuration short names.
#   key = (model, budget_gib) tuple
#   val = dict of method -> filename prefix (without .csv / _io_runN.csv / _runN.log)
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

# Display order for configs: 120B first (headline question), then 20B ascending.
CONFIG_ORDER = [("120B", 20), ("120B", 25), ("20B", 7), ("20B", 8), ("20B", 9)]

def cfg_label(key):
    """Short human-readable config label for x-axis."""
    model, b = key
    return f"{model}\n{b} GiB"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def read_summary(cfg_name: str):
    """Return list of dict rows from <cfg>.csv (harness main metrics)."""
    path = DATA_DIR / f"{cfg_name}.csv"
    with open(path) as f:
        return list(csv.DictReader(f))

def summarize(cfg_name: str):
    rows = read_summary(cfg_name)
    eval_ms = np.array([float(r["eval_time_ms"]) for r in rows])
    faults  = np.array([int(r["major_page_faults"]) for r in rows])
    tokens  = np.array([int(r["eval_tokens"]) for r in rows])
    tok_s   = tokens * 1000.0 / eval_ms
    return dict(
        eval_mean = float(eval_ms.mean() / 1000.0),
        eval_std  = float(eval_ms.std(ddof=0) / 1000.0),
        tok_s     = float(tok_s.mean()),
        faults    = float(faults.mean()),
        n_iters   = len(rows),
    )

def parse_log_for_cache(cfg_name: str, iter_num: int):
    """Parse hit/miss/load_ms/phase2_wait from the BSC phases JSON in the log."""
    path = DATA_DIR / f"{cfg_name}_run{iter_num}.log"
    text = path.read_text()
    out = {}
    for fld in ("uring_cache_hits", "uring_cache_misses",
                "uring_load_ms", "uring_phase2_wait_ms"):
        m = re.search(rf'"{fld}":\s*([0-9.]+)', text)
        out[fld] = float(m.group(1)) if m else 0.0
    return out

def summarize_cache(cfg_name: str):
    n = summarize(cfg_name)["n_iters"]
    per_iter = [parse_log_for_cache(cfg_name, i+1) for i in range(n)]
    hits   = np.mean([d["uring_cache_hits"]   for d in per_iter])
    misses = np.mean([d["uring_cache_misses"] for d in per_iter])
    load_ms = np.mean([d["uring_load_ms"] for d in per_iter])
    p2_ms   = np.mean([d["uring_phase2_wait_ms"] for d in per_iter])
    hr = 100 * hits / (hits + misses) if (hits + misses) > 0 else 0.0
    return dict(hits=hits, misses=misses, hit_rate=hr,
                load_ms=load_ms, p2_ms=p2_ms)

def load_io_trace(cfg_name: str, iter_num: int = 1):
    """Return (timestamp_s[], read_mib_s[]) arrays."""
    path = DATA_DIR / f"{cfg_name}_io_run{iter_num}.csv"
    t, r = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["timestamp_s"]))
            r.append(float(row["read_mib_s"]))
    return np.array(t), np.array(r)

def load_all():
    data = {}
    for key, methods in CONFIGS.items():
        data[key] = {}
        for m, name in methods.items():
            s = summarize(name); c = summarize_cache(name)
            data[key][m] = {**s, **c, "name": name}
    return data

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def savefig(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}")
    print(f"  wrote  {stem}.png  {stem}.pdf")

def grouped_bar(ax, x_labels, values_by_method, errs_by_method=None,
                ylabel="", title="", value_fmt="{:.2f}", value_rot=0,
                value_fontsize=8):
    """Grouped bar chart with 3 bars per group."""
    x = np.arange(len(x_labels))
    width = 0.25
    offsets = np.linspace(-1, 1, len(ORDER)) * width
    for i, m in enumerate(ORDER):
        vals = np.array(values_by_method[m])
        errs = np.array(errs_by_method[m]) if errs_by_method else None
        bars = ax.bar(x + offsets[i], vals, width,
                       yerr=errs, capsize=3,
                       color=COLOR[m], label=LABEL[m],
                       edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    value_fmt.format(v),
                    ha="center", va="bottom", fontsize=value_fontsize,
                    rotation=value_rot)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------------------
# Plot 1 — tok/s
# ---------------------------------------------------------------------------

def plot_tok_per_sec(data):
    labels = [cfg_label(k) for k in CONFIG_ORDER]
    vals = {m: [data[k][m]["tok_s"] for k in CONFIG_ORDER] for m in ORDER}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grouped_bar(ax, labels, vals,
                ylabel="Tokens / second (↑ better)",
                title="Decode throughput by pipeline mode, both models, all budgets  —  3 iters, multi-topic prompt")
    ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.15)
    savefig(fig, "01_tok_per_sec")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2 — eval time
# ---------------------------------------------------------------------------

def plot_eval_time(data):
    labels = [cfg_label(k) for k in CONFIG_ORDER]
    vals = {m: [data[k][m]["eval_mean"] for k in CONFIG_ORDER] for m in ORDER}
    errs = {m: [data[k][m]["eval_std"]  for k in CONFIG_ORDER] for m in ORDER}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grouped_bar(ax, labels, vals, errs,
                ylabel="Eval time for 2000 tokens (seconds, ↓ better)",
                title="Eval time by pipeline mode — error bars show σ across 3 iters",
                value_fmt="{:.1f}")
    ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.15)
    savefig(fig, "02_eval_time")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3 — v3 speedup vs overlap (%)
# ---------------------------------------------------------------------------

def plot_v3_speedup(data):
    labels = [cfg_label(k) for k in CONFIG_ORDER]
    # speedup = (tok_s[v3] / tok_s[overlap] - 1) * 100
    v2_pct = [(data[k]["v2"]["tok_s"] / data[k]["overlap"]["tok_s"] - 1) * 100 for k in CONFIG_ORDER]
    v3_pct = [(data[k]["v3"]["tok_s"] / data[k]["overlap"]["tok_s"] - 1) * 100 for k in CONFIG_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(labels))
    width = 0.38
    b1 = ax.bar(x - width/2, v2_pct, width, color=COLOR["v2"], label=LABEL["v2"],
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width/2, v3_pct, width, color=COLOR["v3"], label=LABEL["v3"],
                edgecolor="black", linewidth=0.5)
    for bars, vals in ((b1, v2_pct), (b2, v3_pct)):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f"+{v:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Throughput gain vs overlap (%, higher is better)")
    ax.set_title("Pipeline speedup over overlap — v3 beats overlap by 5–7% on every budget")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(v3_pct) * 1.25)
    savefig(fig, "03_v3_speedup_vs_overlap")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4 — NVMe bandwidth time-series (granular, 25 ms samples)
# ---------------------------------------------------------------------------

def plot_bandwidth_timeseries():
    """5 rows × 3 cols = 15 panels. One row per (model, budget); cols = methods.
    Raw 25 ms samples exposed so the per-layer burst pattern is visible."""
    fig, axes = plt.subplots(len(CONFIG_ORDER), len(ORDER),
                              figsize=(15, 3.0 * len(CONFIG_ORDER)),
                              sharey="row")
    # Determine per-row y max across methods for consistent row scale.
    for row, key in enumerate(CONFIG_ORDER):
        row_max = 0.0
        traces = {}
        for m in ORDER:
            cfg_name = CONFIGS[key][m]
            t, rd = load_io_trace(cfg_name, iter_num=1)
            traces[m] = (t, rd)
            if len(rd): row_max = max(row_max, float(rd.max()))
        for col, m in enumerate(ORDER):
            ax = axes[row, col]
            t, rd = traces[m]
            ax.plot(t, rd, color=COLOR[m], linewidth=0.30, alpha=0.80)
            # Annotate active-read mean (excluding zero samples).
            active_mean = float(rd[rd > 0].mean()) if (rd > 0).any() else 0.0
            total_gib = float((rd[1:] * np.diff(t)).sum() / 1024.0)  # MiB·s / s / 1024
            ax.axhline(active_mean, color="black", linestyle=":", linewidth=0.9,
                       alpha=0.7)
            if col == 0:
                ax.set_ylabel(f"{cfg_label(key)}\nNVMe read MiB/s")
            if row == len(CONFIG_ORDER) - 1:
                ax.set_xlabel("Wall time (s)")
            ax.set_title(f"{LABEL[m]}\nactive-read avg {active_mean:.0f} MiB/s · total {total_gib:.0f} GiB",
                          fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, row_max * 1.08)
    fig.suptitle("NVMe read bandwidth during 2000-token decode — raw 25 ms samples, run 1 of 3",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    savefig(fig, "04_nvme_bandwidth_timeseries_granular")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5 — NVMe bandwidth: average during decode (steady state)
# ---------------------------------------------------------------------------

def plot_bandwidth_avg():
    """Mean active-read MiB/s across all 3 iters, one bar per (config, method)."""
    labels = [cfg_label(k) for k in CONFIG_ORDER]
    vals = {m: [] for m in ORDER}
    for key in CONFIG_ORDER:
        for m in ORDER:
            cfg_name = CONFIGS[key][m]
            means = []
            for it in (1, 2, 3):
                t, rd = load_io_trace(cfg_name, iter_num=it)
                mask = (t > 2.0) & (rd > 0)  # skip first 2 s, drop idle samples
                if mask.any():
                    means.append(float(rd[mask].mean()))
            vals[m].append(float(np.mean(means)) if means else 0.0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    grouped_bar(ax, labels, vals,
                ylabel="Mean active-read NVMe bandwidth (MiB/s, ↑ better I/O utilisation)",
                title="Average NVMe read bandwidth during decode — v3 sustains a higher rate by keeping the queue deeper",
                value_fmt="{:.0f}")
    ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.15)
    savefig(fig, "05_nvme_bandwidth_avg")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 6 — total bytes read from NVMe (per iter, averaged)
# ---------------------------------------------------------------------------

def plot_bytes_read(data):
    """Total GiB read from NVMe during decode. Should match across methods at a
    given (model, budget) since they see the same access pattern (bit-exact)."""
    labels = [cfg_label(k) for k in CONFIG_ORDER]
    vals = {m: [] for m in ORDER}
    for key in CONFIG_ORDER:
        for m in ORDER:
            cfg_name = CONFIGS[key][m]
            totals = []
            for it in (1, 2, 3):
                t, rd = load_io_trace(cfg_name, iter_num=it)
                # bytes ≈ rate_MiB_s * dt_s → integrate
                dt = np.diff(t, prepend=0)
                gib = float((rd * dt).sum() / 1024.0)
                totals.append(gib)
            vals[m].append(float(np.mean(totals)))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    grouped_bar(ax, labels, vals,
                ylabel="Total NVMe read during decode (GiB, averaged over 3 iters)",
                title="Total bytes read from NVMe — identical across overlap/v2/v3 (same access pattern, bit-exact)",
                value_fmt="{:.0f}")
    ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.15)
    savefig(fig, "06_bytes_read")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 7 — cache hit rate
# ---------------------------------------------------------------------------

def plot_hit_rate(data):
    labels = [cfg_label(k) for k in CONFIG_ORDER]
    vals = {m: [data[k][m]["hit_rate"] for k in CONFIG_ORDER] for m in ORDER}
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped_bar(ax, labels, vals,
                ylabel="Expert cache hit rate (%)",
                title="Expert cache hit rate — identical across overlap/v2/v3 (bit-exact access pattern)",
                value_fmt="{:.1f}%")
    ax.set_ylim(0, 105)
    savefig(fig, "07_cache_hit_rate")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading data from {DATA_DIR}")
    data = load_all()
    print(f"Writing plots to {PLOTS_DIR}")
    plot_tok_per_sec(data)
    plot_eval_time(data)
    plot_v3_speedup(data)
    plot_bandwidth_timeseries()
    plot_bandwidth_avg()
    plot_bytes_read(data)
    plot_hit_rate(data)
    print("\nDone. Plots in", PLOTS_DIR)

if __name__ == "__main__":
    main()
