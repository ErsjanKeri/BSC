#!/usr/bin/env python3
"""
Build the canonical oracle JSON for thesis figures/tables from the May 1/2 sweep.

Reads per-config CSVs from:
  - cgroup_20260501_191023/  (20B canon, 96 runs)
  - cgroup_20260502_140144/  (120B retry canon, 72 runs)

Writes:
  - thesis/_meta/final_canon.json  (the source-of-truth for figures_canon.py + tables_canon.py)

Run: python3 thesis/scripts/build_final_canon.py
"""
import csv
import json
import statistics
from pathlib import Path

DIR_20B = Path("/home/keri/BSC/time-tracking/results/cgroup_20260501_191023")
DIR_120B = Path("/home/keri/BSC/time-tracking/results/cgroup_20260502_140144")
OUT = Path("/home/keri/BSC/thesis/_meta/final_canon.json")


def parse_run_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_cgroup_csv(path: Path) -> float:
    """Return the max cgroup memory.peak observed (MiB), or 0.0 if missing."""
    if not path.exists():
        return 0.0
    peaks_mib: list[float] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            peak_bytes = row.get("memory_peak_bytes")
            if peak_bytes:
                peaks_mib.append(float(peak_bytes) / (1024 * 1024))
    return max(peaks_mib) if peaks_mib else 0.0


def aggregate(name: str, rows: list[dict], peak_mib: float) -> dict:
    eval_seconds = [float(r["eval_time_ms"]) / 1000.0 for r in rows]
    tokens_per_iter = [int(r["eval_tokens"]) for r in rows]
    gen_faults = [int(r["phase_generation_faults"] or 0) for r in rows]
    n = len(eval_seconds)
    if n == 0:
        return {}
    mean = statistics.mean(eval_seconds)
    stdev = statistics.stdev(eval_seconds) if n > 1 else 0.0
    cv_pct = (stdev / mean) * 100.0 if mean > 0 else 0.0
    eval_tokens = tokens_per_iter[0]
    if any(t != eval_tokens for t in tokens_per_iter):
        # Take the most common value if mismatched, but flag it
        eval_tokens = max(set(tokens_per_iter), key=tokens_per_iter.count)
    tok_per_s_mean = eval_tokens / mean if mean > 0 else 0.0
    return {
        "n_runs": n,
        "eval_seconds_per_iter": eval_seconds,
        "eval_seconds_mean": mean,
        "eval_seconds_stdev": stdev,
        "eval_seconds_cv_pct": cv_pct,
        "tok_per_s_mean": tok_per_s_mean,
        "eval_tokens": eval_tokens,
        "gen_faults_mean": statistics.mean(gen_faults),
        "cgroup_peak_mib_max": peak_mib,
    }


def collect(dirpath: Path) -> dict:
    configs: dict = {}
    for csv_path in sorted(dirpath.glob("*.csv")):
        if csv_path.name.endswith("_cgroup.csv") or csv_path.name.endswith("_meminfo.csv"):
            continue
        name = csv_path.stem
        rows = parse_run_csv(csv_path)
        peak_mib = parse_cgroup_csv(dirpath / f"{name}_cgroup.csv")
        agg = aggregate(name, rows, peak_mib)
        if agg:
            configs[name] = agg
    return configs


def main() -> None:
    configs: dict = {}
    configs.update(collect(DIR_20B))
    configs.update(collect(DIR_120B))

    out = {
        "dataset": "May1/2_canonical",
        "description": (
            "BSC thesis canonical: GPT-OSS-20B from cgroup_20260501_191023 (96 runs, "
            "32 configs × 3 iters) + GPT-OSS-120B from cgroup_20260502_140144 (72 runs, "
            "24 configs × 3 iters). All multi-topic prompt, --eager-compute, seed=42, "
            "1999 eval_tokens on both models."
        ),
        "n_configs": len(configs),
        "n_runs": sum(c["n_runs"] for c in configs.values()),
        "configs": configs,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {OUT} — {out['n_configs']} configs, {out['n_runs']} runs")
    print()
    print("Configs:")
    for name in sorted(configs):
        c = configs[name]
        print(f"  {name:50s}  {c['n_runs']} iters, {c['tok_per_s_mean']:.3f} tok/s, CV {c['eval_seconds_cv_pct']:.3f}%")


if __name__ == "__main__":
    main()
