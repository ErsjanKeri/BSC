#!/usr/bin/env python3
"""
Generate LaTeX table snippets from final_canon.json (May 1/2 canonical sweep).

Reads:  thesis/_meta/final_canon.json
Writes: thesis/_meta/canon_tables.tex

Each table is a self-contained LaTeX `table` environment with caption and label.
Chapters can either copy the numbers manually or `\\input{_meta/canon_tables.tex}`
and reference labels.
"""
import json
import statistics
from pathlib import Path

ORACLE_PATH = Path("/home/keri/BSC/thesis/_meta/final_canon.json")
OUT_PATH = Path("/home/keri/BSC/thesis/_meta/canon_tables.tex")


def load_oracle() -> dict:
    with ORACLE_PATH.open() as f:
        return json.load(f)


def cfg(oracle: dict, key: str) -> dict:
    return oracle["configs"][key]


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def table_headline(oracle: dict, model: str, budgets: list[int], label: str, caption_extra: str) -> str:
    """5-config × N-budget headline table. Uses LFU-aging at 20B 7G, LRU elsewhere."""
    pretty = [
        ("lazy", "lazy mmap"),
        ("mmap_pin", "mmap $+$ pin"),
        ("projection_overlap", "projection-group overlap"),
        ("async_projection_overlap", "async-projection-overlap"),
        ("async_experts", "async-experts"),
    ]

    def policy(model: str, b: int) -> str:
        if model == "20b" and b == 7:
            return "lfua"
        return "lru"

    cap_for_budget = {
        "20b": {7: ("250 slots ($\\approx$ 1.03~GiB), $\\approx$ 0.10~GiB free", "below the per-token working set of 288"),
                8: ("498 slots ($\\approx$ 2.04~GiB), $\\approx$ 0.08~GiB free", None),
                9: ("740 slots ($\\approx$ 3.04~GiB), $\\approx$ 0.09~GiB free", None),
                22: (None, "model fits in RAM")},
        "120b": {12: ("693 slots ($\\approx$ 2.84~GiB), $\\approx$ 1.02~GiB free", None),
                 16: ("1668 slots ($\\approx$ 6.84~GiB), $\\approx$ 1.02~GiB free", None),
                 22: ("3130 slots ($\\approx$ 12.84~GiB), $\\approx$ 1.03~GiB free", None),
                 28: ("4592 slots ($\\approx$ 18.83~GiB), $\\approx$ 1.03~GiB free", None)},
    }

    rows: list[str] = []
    for b in budgets:
        suffix = ""
        if model == "20b" and b == 22:
            suffix = "; model fits in RAM"
        rows.append(f"  \\midrule\n  \\multicolumn{{4}}{{l}}{{\\textit{{cgroup memory.max = {b}~GiB{suffix}}}}} \\\\")
        # find best in this budget for bolding
        keys = []
        for fam, _ in pretty:
            if fam in ("lazy", "mmap_pin"):
                k = f"{fam}_{model}_{b}g"
            else:
                k = f"{fam}_{policy(model, b)}_{model}_{b}g"
            keys.append(k)
        best_key = max(keys, key=lambda k: cfg(oracle, k)["tok_per_s_mean"])

        # Insert "with buffer: ..." sub-row before the buffer-using configs.
        for (fam, name), key in zip(pretty, keys):
            if fam == "projection_overlap":
                buf_descr = cap_for_budget.get(model, {}).get(b, (None, None))[0]
                if buf_descr:
                    rows.append(f"  \\multicolumn{{4}}{{l}}{{\\textit{{with buffer: {buf_descr}}}}} \\\\")
            c = cfg(oracle, key)
            tps = c["tok_per_s_mean"]
            cv = c["eval_seconds_cv_pct"]
            lazy_key = f"lazy_{model}_{b}g"
            speedup = tps / cfg(oracle, lazy_key)["tok_per_s_mean"]
            tps_str = f"\\textbf{{{fmt(tps)}}}" if key == best_key else fmt(tps)
            rows.append(f"  {name:30s} & {tps_str} & {fmt(cv, 2)} & {fmt(speedup)}$\\times$ \\\\")

    body = "\n".join(rows)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption[GPT-OSS-{model.upper()} throughput at four cgroup budgets]{{GPT-OSS-{model.upper()} throughput at four cgroup limits, three iterations per cell. {caption_extra}}}
\\label{{{label}}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Configuration & Tok/s & Eval CV (\\%) & Speedup vs lazy \\\\
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def table_policy_at_7g(oracle: dict) -> str:
    """20B 7G policy sweep, 4 io_uring variants x 3 policies (LRU, LFU, LFU-aging)."""
    variants = [
        ("uring", "uring (sync)"),
        ("projection_overlap", "projection-group overlap"),
        ("async_projection_overlap", "async-projection-overlap"),
        ("async_experts", "async-experts"),
    ]
    policies = [("lru", "LRU"), ("lfu", "LFU"), ("lfua", "LFU-aging $m=3$")]

    rows = []
    for variant, vname in variants:
        cells = []
        for pol, _ in policies:
            key = f"{variant}_{pol}_20b_7g"
            cells.append(cfg(oracle, key)["tok_per_s_mean"])
        best_idx = cells.index(max(cells))
        formatted = []
        for i, v in enumerate(cells):
            if i == best_idx:
                formatted.append(f"\\textbf{{{fmt(v)}}}")
            else:
                formatted.append(fmt(v))
        rows.append(f"  {vname:30s} & " + " & ".join(formatted) + r" \\")

    body = "\n".join(rows)
    pol_header = " & ".join(p[1] for p in policies)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption[Cache eviction policy at 20B 7~GiB c=250]{{Cache eviction policy comparison at GPT-OSS-20B, cgroup memory.max~=~7~GiB, cache 250 slots (below the per-token working set of 288). Three iterations per cell, all CVs $\\leq$~1.7\\%. All values in tokens per second. LFU-aging dominates LFU dominates LRU at every io\\_uring code path: when the cache is below the working set, the LRU tail is exactly the next slice the access pattern asks for and the policy thrashes.}}
\\label{{tab:cache-policy-7g}}
\\begin{{tabular}}{{lrrr}}
\\toprule
io\\_uring code path & {pol_header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def table_reproducibility(oracle: dict) -> str:
    """Reproducibility summary: CV across all 168 runs."""
    cvs = [c["eval_seconds_cv_pct"] for c in oracle["configs"].values()]
    cv_mean = statistics.mean(cvs)
    cv_max = max(cvs)
    cv_p50 = statistics.median(cvs)
    n_below_05 = sum(1 for c in cvs if c < 0.5)
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption[Reproducibility of the May 1/2 canonical sweep]{{Reproducibility of the May 1/2 canonical sweep across all {oracle['n_configs']} configurations and {oracle['n_runs']} total iterations. Mean coefficient of variation in eval seconds is {fmt(cv_mean, 2)}\\%; median {fmt(cv_p50, 3)}\\%; max {fmt(cv_max, 2)}\\%. The kernel-enforced cgroup memory.max methodology is reproducible at the sub-percent level on this hardware.}}
\\label{{tab:reproducibility}}
\\begin{{tabular}}{{lr}}
\\toprule
Statistic & Value \\\\
\\midrule
Configurations    & {oracle['n_configs']} \\\\
Iterations / cell & 3 \\\\
Total runs        & {oracle['n_runs']} \\\\
Mean CV (\\%)      & {fmt(cv_mean, 2)} \\\\
Median CV (\\%)    & {fmt(cv_p50, 3)} \\\\
Max CV (\\%)       & {fmt(cv_max, 2)} \\\\
Configs with CV $<$ 0.5\\% & {n_below_05} of {oracle['n_configs']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def main() -> None:
    oracle = load_oracle()
    parts = [
        "% Auto-generated by thesis/scripts/tables_canon.py from _meta/final_canon.json.",
        "% Do not hand-edit. Re-run the script if the oracle changes.",
        "",
        table_headline(
            oracle, "20b", [7, 8, 9, 22],
            "tab:eval-20b-headline",
            "Each block is monotonic top to bottom when the model does not fit in RAM: pinning helps in proportion to what the kernel would otherwise evict, and our buffer-managed I/O stack adds roughly an additional $2\\times$ on top. At 22~GiB the working set fits and all configurations converge.",
        ),
        "",
        table_headline(
            oracle, "120b", [12, 16, 22, 28],
            "tab:eval-120b-headline",
            "GPT-OSS-120B (60.88~GiB on disk) exceeds the host's 30~GiB RAM by a factor of about 2, so every cell forces page-cache turnover regardless of the cgroup limit. The configuration ranking is the same as on the smaller model, and our buffer-managed I/O stack adds a factor of roughly $1.63$--$2.38\\times$ on top of pin.",
        ),
        "",
        table_policy_at_7g(oracle),
        "",
        table_reproducibility(oracle),
        "",
    ]
    OUT_PATH.write_text("\n".join(parts))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
