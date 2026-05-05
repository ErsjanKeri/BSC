#!/usr/bin/env python3
"""
Generate canonical thesis figures from final_canon.json (May 1/2 sweep).

Outputs into thesis/figures/canon/:
  - tok_s_by_budget_20b.pdf   (4-budget headline for 20B)
  - tok_s_by_budget_120b.pdf  (4-budget headline for 120B)
  - speedup_vs_lazy.pdf       (cross-model speedup ratios)
  - lfu_lru_crossover.pdf     (May 1 7G policy sweep — LFU-aging > LFU > LRU at c=250 < WS=288)
  - cv_distribution.pdf       (CV across all 168 runs)

The chapter prose is responsible for narrative; this script only writes
faithful, deterministic visualizations of final_canon.json.
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The cache-policy figure uses the bit-exact simulator on the canonical
# tensor-trace dump; import its policy classes directly so the figure stays
# faithful to the same code path that simulate_from_dump.py executes.
sys.path.insert(0, str(Path("/home/keri/BSC/tensor-tracing/tools")))
from simulate_from_dump import (
    load_dump,
    simulate,
    LRUCache,
    LFUSlotCache,
    LFUAgingSlotCache,
    ARCCache,
    WTinyLFUCache,
    BeladyCache,
)

ORACLE_PATH = Path("/home/keri/BSC/thesis/_meta/final_canon.json")
CACHE_DUMP_PATH = Path("/home/keri/BSC/tensor-tracing/traces/20b-2000tok-cache-dump/cache_dump.csv")
OUT_DIR = Path("/home/keri/BSC/thesis/figures/canon")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "lazy": "#7f7f7f",
    "mmap_pin": "#1f77b4",
    "uring": "#9467bd",
    "projection_overlap": "#2ca02c",
    "async_projection_overlap": "#d62728",
    "async_experts": "#ff7f0e",
    "lfu": "#9467bd",
    "lfua": "#d62728",
    "lru": "#17becf",
}

PRETTY = {
    "lazy": "lazy mmap",
    "mmap_pin": "mmap + pin",
    "uring": "uring (sync)",
    "projection_overlap": "projection-group overlap",
    "async_projection_overlap": "async-projection-overlap",
    "async_experts": "async-experts",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_oracle() -> dict:
    with ORACLE_PATH.open() as f:
        return json.load(f)


def cfg(oracle: dict, key: str) -> dict:
    if key not in oracle["configs"]:
        raise KeyError(f"missing config: {key}")
    return oracle["configs"][key]


def tps(oracle: dict, key: str) -> float:
    return cfg(oracle, key)["tok_per_s_mean"]


def tps_stdev(oracle: dict, key: str) -> float:
    c = cfg(oracle, key)
    tokens = c["eval_tokens"]
    per_iter = [tokens / s for s in c["eval_seconds_per_iter"]]
    return statistics.stdev(per_iter) if len(per_iter) > 1 else 0.0


def annotate_bar(ax, bar, fmt="{:.2f}", offset_frac=0.02):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + offset_frac * h,
        fmt.format(h),
        ha="center",
        va="bottom",
        fontsize=8,
    )


def figure_tok_s_by_budget(oracle: dict, model: str, budgets: list[int], outname: str) -> None:
    """5-config × N-budget grouped bar chart. Best uring variant per cell uses LRU
    (the canonical cache-size-> WS regime for 8G/9G/12-28G). At 20B 7G we use LFU-aging
    since the cache is below the working set.
    """
    families = ["lazy", "mmap_pin", "projection_overlap", "async_projection_overlap", "async_experts"]

    def policy_for(model: str, budget: int) -> str:
        if model == "20b" and budget == 7:
            return "lfua"
        return "lru"

    means = {fam: [] for fam in families}
    stds = {fam: [] for fam in families}
    for b in budgets:
        for fam in families:
            if fam in ("lazy", "mmap_pin"):
                key = f"{fam}_{model}_{b}g"
            else:
                key = f"{fam}_{policy_for(model, b)}_{model}_{b}g"
            means[fam].append(tps(oracle, key))
            stds[fam].append(tps_stdev(oracle, key))

    x = np.arange(len(budgets))
    width = 0.16
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i, fam in enumerate(families):
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset, means[fam], width,
            yerr=stds[fam],
            label=PRETTY[fam],
            color=COLORS[fam],
            capsize=2,
            edgecolor="black",
            linewidth=0.4,
        )
        for b in bars:
            annotate_bar(ax, b)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} GiB" for g in budgets])
    ax.set_xlabel("cgroup memory.max")
    ax.set_ylabel("Throughput (tokens / second)")
    best = max(max(v) for v in means.values())
    ax.set_ylim(0, best * 1.18)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT_DIR / outname)
    plt.close(fig)
    print(f"  wrote {outname}")


def figure_speedup_vs_lazy(oracle: dict) -> None:
    """Speedup of best uring variant vs lazy at every (model, budget) cell."""
    cells = [
        ("20b", 7, "async_projection_overlap_lfua_20b_7g"),
        ("20b", 8, "async_projection_overlap_lru_20b_8g"),
        ("20b", 9, "async_projection_overlap_lru_20b_9g"),
        ("120b", 12, "async_projection_overlap_lru_120b_12g"),
        ("120b", 16, "async_projection_overlap_lru_120b_16g"),
        ("120b", 22, "async_projection_overlap_lru_120b_22g"),
        ("120b", 28, "async_projection_overlap_lru_120b_28g"),
    ]

    labels = []
    speedups = []
    for model, b, best_key in cells:
        labels.append(f"{model.upper()}\n{b} GiB")
        lazy_tps = tps(oracle, f"lazy_{model}_{b}g")
        best_tps = tps(oracle, best_key)
        speedups.append(best_tps / lazy_tps)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(
        x, speedups,
        color=COLORS["async_projection_overlap"],
        edgecolor="black",
        linewidth=0.4,
    )
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + 0.04,
            f"{b.get_height():.2f}x",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model and cgroup budget")
    ax.set_ylabel("Speedup vs lazy mmap (ratio)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(speedups) * 1.12)
    ax.set_title("async-projection-overlap speedup over lazy mmap baseline")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "speedup_vs_lazy.pdf")
    plt.close(fig)
    print("  wrote speedup_vs_lazy.pdf")


def figure_lfu_lru_crossover() -> None:
    """Hit-rate curves for five eviction policies plus Belady (offline-optimal)
    on the canonical 20B access trace, replayed by the bit-exact simulator
    across cache sizes that span the per-token working-set boundary of 288.

    The wall-clock at c=250 (LFU-aging 5.36 vs LRU 3.98 tok/s on the
    canonical sweep) corroborates the simulator-predicted gap at undersized
    cache; the figure extends the comparison across cache sizes the
    wall-clock sweep does not measure.
    """
    if not CACHE_DUMP_PATH.exists():
        print(f"  SKIP lfu_lru_crossover.pdf — missing {CACHE_DUMP_PATH}")
        return

    loads = load_dump(str(CACHE_DUMP_PATH))
    all_batches = [batch for batch, _, _ in loads]
    n_accesses = sum(len(b) for b in all_batches)

    cache_sizes = [150, 200, 250, 288, 350, 500, 750, 1000]
    policies = [
        ("LRU", LRUCache, "lru", "o", "-"),
        ("LFU", LFUSlotCache, "lfu", "s", "-"),
        ("LFU-aging $m=3$", LFUAgingSlotCache, "lfua", "^", "-"),
        ("ARC", ARCCache, "arc", "D", "--"),
        ("W-TinyLFU", WTinyLFUCache, "wtlfu", "v", "--"),
    ]
    POLICY_COLORS = {
        "lru": "#17becf",
        "lfu": "#9467bd",
        "lfua": "#d62728",
        "arc": "#2ca02c",
        "wtlfu": "#ff7f0e",
        "belady": "#7f7f7f",
    }

    hit_rates = {key: [] for _, _, key, _, _ in policies}
    for cs in cache_sizes:
        for name, cls, key, _, _ in policies:
            h, _ = simulate(loads, cs, cls, all_batches=all_batches)
            hit_rates[key].append(h / n_accesses * 100)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, _, key, marker, ls in policies:
        ax.plot(
            cache_sizes, hit_rates[key],
            marker=marker, linestyle=ls, linewidth=1.3,
            color=POLICY_COLORS[key], label=name, markersize=5,
        )

    ax.axvline(288, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(290, 5, "per-token WS = 288", fontsize=8, color="black", alpha=0.7)

    ax.set_xlabel("Cache size (slots)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_xticks(cache_sizes)
    ax.set_xticklabels([str(c) for c in cache_sizes])
    ax.set_ylim(-3, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.set_title(
        "Hit rate by policy on the canonical 20B access trace (1999 tokens, 575,724 expert accesses)"
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "lfu_lru_crossover.pdf")
    plt.close(fig)
    print("  wrote lfu_lru_crossover.pdf (simulator hit-rate curves)")


def figure_cv_distribution(oracle: dict) -> None:
    """Histogram of CV across all 168 runs."""
    cvs = [c["eval_seconds_cv_pct"] for c in oracle["configs"].values()]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(cvs, bins=20, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.axvline(statistics.mean(cvs), color="red", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"mean CV = {statistics.mean(cvs):.3f}%")
    ax.set_xlabel("Coefficient of variation across iterations (%)")
    ax.set_ylabel("Number of configurations")
    ax.set_title(f"CV distribution across {oracle['n_configs']} configurations × 3 iters = {oracle['n_runs']} runs")
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "cv_distribution.pdf")
    plt.close(fig)
    print("  wrote cv_distribution.pdf")


def main() -> None:
    oracle = load_oracle()
    print(f"Loaded oracle: {oracle['n_configs']} configs, {oracle['n_runs']} runs")
    print(f"Writing figures to {OUT_DIR}/")
    figure_tok_s_by_budget(oracle, "20b", [7, 8, 9, 22], "tok_s_by_budget_20b.pdf")
    figure_tok_s_by_budget(oracle, "120b", [12, 16, 22, 28], "tok_s_by_budget_120b.pdf")
    figure_speedup_vs_lazy(oracle)
    figure_lfu_lru_crossover()
    figure_cv_distribution(oracle)
    print("Done.")


if __name__ == "__main__":
    main()
