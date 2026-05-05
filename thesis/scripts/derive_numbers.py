#!/usr/bin/env python3
"""
Derive every quantitative claim used in the thesis from byte-level source data.

Outputs to stdout a single table of (claim_id, value_GiB_or_MiB, formatted_2dp,
source_file_or_log) so prose can cite the same authoritative number.

Inputs:
- /home/keri/BSC/tensor-tracing/desktopui/data/memory-map.json
    Per-tensor sizes for GPT-OSS-20B with offset_start, offset_end, size_bytes.
    Authoritative for every weight-breakdown number on 20B.

- /home/keri/BSC/time-tracking/results/cgroup_20260501_191023/<config>_run1.log
    A canonical run log. We pick async_projection_overlap_lfua_20b_7g (the
    headline 20B 7 GiB cell) and async_projection_overlap_lru_120b_28g (the
    headline 120B 28 GiB cell). The log lines we read:
      load_tensors:   CPU_Mapped model buffer size = N MiB     -> mmap region
      llama_kv_cache:        CPU KV buffer size =  N MiB        -> KV reservation
      llama_context:        CPU compute buffer size =   N MiB   -> compute buffer
      llama_uring_expert_buf_init: ... slot_bytes=N             -> exact slot size
      [BSC_MEM]   VmRSS:	  N kB                                -> RSS over time
      uring_cache_hits / uring_cache_misses                     -> hit rate

- /home/keri/BSC/thesis/_meta/final_canon.json
    Aggregated tok/s, CV, eval seconds for all 56 configs.

- /home/keri/BSC/tensor-tracing/traces/20b-2000tok-cache-dump/cache_dump.csv
    Authoritative cache-load-by-load record on the canonical 20B trace,
    used to derive per-(layer, expert) selection counts and the reuse-distance
    distribution.

Rules:
- All sizes in GiB / MiB / KiB only. Never GB / MB / KB.
- All numbers reported at 2 decimal places.
- Conversion uses 1024 powers (GiB = 2^30, MiB = 2^20, KiB = 2^10).
"""
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

GiB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10


def to_gib(b: int) -> float:
    return b / GiB


def to_mib(b: int) -> float:
    return b / MiB


def to_kib(b: int) -> float:
    return b / KiB


def fmt2(x: float) -> str:
    return f"{x:.2f}"


# -------------------------------------------------------------------------
# 1. Per-tensor weight breakdown (byte-exact).
# -------------------------------------------------------------------------
MEM_MAP_20B = Path("/home/keri/BSC/tensor-tracing/desktopui/data/memory-map.json")
GGUF_120B = Path("/home/keri/llama.cpp/models/gpt-oss-120b/gpt-oss-120b-F16.gguf")
GGUF_DUMP_BIN = Path("/home/keri/llama.cpp/build/bin/llama-gguf-dump")
GGUF_120B_CSV = Path("/tmp/gguf_120b_dump.csv")


def categorize(name: str) -> str:
    if "_exps." in name:
        return "experts"
    if "attn" in name:
        return "attention"
    if "token_embd" in name:
        return "token_embedding"
    if name.endswith("output.weight") or "output.weight" in name and "norm" not in name:
        return "output_projection"
    if "norm" in name:
        return "norms"
    if "ffn_gate_inp" in name:
        return "routers"
    return "other"


def load_20b_breakdown() -> dict:
    with MEM_MAP_20B.open() as f:
        d = json.load(f)
    file_size_bytes = d["total_size_bytes"]  # full file size including GGUF header
    tensors = d["tensors"]
    cat_bytes = defaultdict(int)
    expert_sizes = []
    for t in tensors:
        c = categorize(t["name"])
        cat_bytes[c] += t["size_bytes"]
        if c == "experts":
            expert_sizes.append(t["size_bytes"])
    tensor_total = sum(t["size_bytes"] for t in tensors)
    return {
        "model": "GPT-OSS-20B",
        "file_size_bytes": file_size_bytes,
        "tensor_data_bytes": tensor_total,
        "header_overhead_bytes": file_size_bytes - tensor_total,
        "categories": dict(cat_bytes),
        "expert_size_min": min(expert_sizes),
        "expert_size_max": max(expert_sizes),
        "expert_size_mode": Counter(expert_sizes).most_common(1)[0][0],
        "n_expert_tensors": len(expert_sizes),
        "n_tensors_total": len(tensors),
        "n_layers": d["metadata"]["n_layers"],
        "n_experts_per_layer": 32,
        "n_experts_used": 4,
    }


def load_120b_breakdown() -> dict:
    """Run llama-gguf-dump on the 120B file, parse output."""
    import subprocess
    if not GGUF_120B_CSV.exists():
        with GGUF_120B_CSV.open("w") as f:
            subprocess.run([str(GGUF_DUMP_BIN), str(GGUF_120B)],
                           stdout=f, stderr=subprocess.DEVNULL, check=True)
    file_size_bytes = GGUF_120B.stat().st_size
    cat_bytes = defaultdict(int)
    expert_sizes = []
    n_tensors = 0
    with GGUF_120B_CSV.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            n_tensors += 1
            name = row["tensor_name"]
            size = int(row["size_bytes"])
            c = categorize(name)
            cat_bytes[c] += size
            if c == "experts":
                expert_sizes.append(size)
    tensor_total = sum(cat_bytes.values())
    return {
        "model": "GPT-OSS-120B",
        "file_size_bytes": file_size_bytes,
        "tensor_data_bytes": tensor_total,
        "header_overhead_bytes": file_size_bytes - tensor_total,
        "categories": dict(cat_bytes),
        "expert_size_min": min(expert_sizes),
        "expert_size_max": max(expert_sizes),
        "expert_size_mode": Counter(expert_sizes).most_common(1)[0][0],
        "n_expert_tensors": len(expert_sizes),
        "n_tensors_total": n_tensors,
        "n_layers": 36,
        "n_experts_per_layer": 128,
        "n_experts_used": 4,
    }


# -------------------------------------------------------------------------
# 2. Per-token reads (dense + 4 experts × 3 projections × n_layers).
# -------------------------------------------------------------------------
def per_token_reads(b: dict) -> dict:
    cats = b["categories"]
    # "Read per word" interpretation: only one row of the token embedding is
    # actually read per generated word, and one row is small (n_embd * 2 B for
    # F16). Everything else in dense gets read fully on each word.
    dense_per_token_reads = (
        cats.get("attention", 0)
        + cats.get("output_projection", 0)
        + cats.get("norms", 0)
        + cats.get("routers", 0)
    )
    embed_row_bytes = 2880 * 2
    expert_slot = b["expert_size_mode"]
    n_layers = b["n_layers"]
    per_token_expert_bytes = 4 * 3 * n_layers * expert_slot
    total_reads_per_token = dense_per_token_reads + embed_row_bytes + per_token_expert_bytes
    # "RAM resident for any forward pass" interpretation: the full token
    # embedding must be in RAM because we don't know which row a word will
    # need. This is the larger, more lenient figure.
    dense_resident = dense_per_token_reads + cats.get("token_embedding", 0)
    total_resident_per_token = dense_resident + per_token_expert_bytes
    return {
        "dense_per_token_reads_bytes": dense_per_token_reads + embed_row_bytes,
        "per_token_expert_bytes": per_token_expert_bytes,
        "total_reads_per_token_bytes": total_reads_per_token,
        "total_resident_dense_per_token_bytes": total_resident_per_token,
        "frac_reads_of_file": total_reads_per_token / b["file_size_bytes"],
        "frac_resident_of_file": total_resident_per_token / b["file_size_bytes"],
    }


# Keep old function name for backwards compat
per_token_reads_20b = per_token_reads


# -------------------------------------------------------------------------
# 3. Run-log parsing for KV / compute / pin / RSS.
# -------------------------------------------------------------------------
LOG_20B = Path(
    "/home/keri/BSC/time-tracking/results/cgroup_20260501_191023/"
    "async_projection_overlap_lfua_20b_7g_run1.log"
)
LOG_120B = Path(
    "/home/keri/BSC/time-tracking/results/cgroup_20260502_140144/"
    "async_projection_overlap_lru_120b_28g_run1.log"
)


def parse_log(path: Path) -> dict:
    text = path.read_text()
    out = {}

    # CPU_Mapped model buffer size (mmap region of the GGUF)
    m = re.search(r"CPU_Mapped model buffer size = ([\d.]+) MiB", text)
    if m:
        out["cpu_mapped_mib"] = float(m.group(1))

    # KV cache buffer sizes (sum the two: non-SWA + SWA)
    kv_hits = [float(x) for x in re.findall(r"CPU KV buffer size =\s+([\d.]+) MiB", text)]
    if kv_hits:
        out["kv_buffer_mib_sum"] = sum(kv_hits)
        out["kv_buffer_mib_parts"] = kv_hits

    # Compute buffer
    m = re.search(r"CPU compute buffer size =\s+([\d.]+) MiB", text)
    if m:
        out["compute_buffer_mib"] = float(m.group(1))

    # Output buffer (small)
    m = re.search(r"CPU\s+output buffer size =\s+([\d.]+) MiB", text)
    if m:
        out["output_buffer_mib"] = float(m.group(1))

    # mlock'd amount
    mlock_hits = re.findall(r"mlock_mib[\":]?\s*[:=]\s*([\d.]+)", text)
    if not mlock_hits:
        mlock_hits = re.findall(r"\"mlock_mib\":\s*([\d.]+)", text)
    if mlock_hits:
        out["mlock_mib"] = float(mlock_hits[0])

    # Slot bytes from uring_expert_buf_init
    m = re.search(
        r"llama_uring_expert_buf_init:.*?expert_bytes=(\d+).*?slot_bytes=(\d+).*?cache_slots=(\d+)",
        text,
    )
    if m:
        out["expert_bytes"] = int(m.group(1))
        out["slot_bytes"] = int(m.group(2))
        out["cache_slots"] = int(m.group(3))

    # bsc_phases_json line: pull cache hits/misses for hit-rate
    m = re.search(r"\"uring_cache_hits\":\s*(\d+)", text)
    if m:
        out["uring_cache_hits"] = int(m.group(1))
    m = re.search(r"\"uring_cache_misses\":\s*(\d+)", text)
    if m:
        out["uring_cache_misses"] = int(m.group(1))
    m = re.search(r"\"uring_loads\":\s*(\d+)", text)
    if m:
        out["uring_loads"] = int(m.group(1))
    m = re.search(r"\"uring_reads\":\s*(\d+)", text)
    if m:
        out["uring_reads"] = int(m.group(1))

    # RSS deltas: capture all VmRSS values
    rss_kb = [int(x) for x in re.findall(r"VmRSS:\s*(\d+)\s*kB", text)]
    if rss_kb:
        out["vmrss_kb_sequence"] = rss_kb
        # The increase across "after KV cache creation" is the KV RSS delta.
        # We approximate by taking the max single-step increase that follows
        # "after KV cache creation".
        out["vmrss_max_kib"] = max(rss_kb)

    return out


# -------------------------------------------------------------------------
# 4. cache_dump.csv: per-(layer, expert) selection counts and reuse distance.
# -------------------------------------------------------------------------
CACHE_DUMP = Path(
    "/home/keri/BSC/tensor-tracing/traces/20b-2000tok-cache-dump/cache_dump.csv"
)


def selection_stats(path: Path) -> dict:
    """Per-(layer, expert) selection counts on the canonical 1999-token decode.

    Format observed: every (layer, token) layer-visit produces exactly 8 rows
    (4 experts x 2 rows per expert: one row covering projections [0,1) i.e.
    up, and one row covering [1,3) i.e. gate and down). The first 8 rows of
    the file are a single layer-23 group from prefill that we discard.
    Subsequent groups cycle layers 0..23 in order, repeated for each
    generated token.
    """
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            layer = int(row["layer"])
            n_real = int(row["n_real"])
            eids = []
            for k in range(n_real):
                e = int(row[f"eid{k}"])
                if e >= 0:
                    eids.append(e)
            rows.append((layer, eids))

    # Discard prefill: the first 8 rows are layer 23 (one group).
    assert all(r[0] == 23 for r in rows[:8]), "prefill assumption violated"
    rows = rows[8:]
    assert len(rows) % 8 == 0, f"row count {len(rows)} not divisible by 8"

    n_groups = len(rows) // 8
    n_layers = 24
    assert n_groups % n_layers == 0
    n_tokens = n_groups // n_layers

    counts = defaultdict(int)  # (layer, expert) -> selection count
    for g in range(n_groups):
        group_rows = rows[g * 8 : (g + 1) * 8]
        layer = group_rows[0][0]
        # All 8 rows in a group must share the same layer.
        assert all(r[0] == layer for r in group_rows)
        # The unique experts in this group are the four selected experts.
        experts = set()
        for _, eids in group_rows:
            experts.update(eids)
        assert len(experts) == 4, f"group {g} layer {layer}: {len(experts)} experts"
        for e in experts:
            counts[(layer, e)] += 1

    per_layer_stats = {}
    for layer in range(n_layers):
        layer_counts = [counts.get((layer, e), 0) for e in range(32)]
        per_layer_stats[layer] = {
            "n_tokens": n_tokens,
            "median": statistics.median(layer_counts),
            "max": max(layer_counts),
            "min": min(layer_counts),
        }

    all_pairs = [counts.get((layer, e), 0) for layer in range(n_layers) for e in range(32)]
    return {
        "per_layer": per_layer_stats,
        "overall_min": min(all_pairs),
        "overall_max": max(all_pairs),
        "n_tokens": n_tokens,
        "counts": counts,
    }


def reuse_distance_stats(path: Path) -> dict:
    """Reuse distance in tokens for each (layer, expert) selection pair.

    Group the cache_dump into 8-row blocks, discarding the 8-row prefill at
    the start. For each block, identify the four selected experts and the
    (layer, token) it belongs to. For each (layer, expert), every selection
    after the first contributes one reuse-distance value (current token minus
    last token at which the same expert was selected at the same layer).
    """
    rows = []
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            layer = int(row["layer"])
            n_real = int(row["n_real"])
            eids = []
            for k in range(n_real):
                e = int(row[f"eid{k}"])
                if e >= 0:
                    eids.append(e)
            rows.append((layer, eids))
    rows = rows[8:]
    assert len(rows) % 8 == 0
    n_groups = len(rows) // 8
    n_layers = 24
    n_tokens = n_groups // n_layers

    last_seen = {}  # (layer, expert) -> token
    distances = []
    for g in range(n_groups):
        group_rows = rows[g * 8 : (g + 1) * 8]
        layer = group_rows[0][0]
        token = g // n_layers
        experts = set()
        for _, eids in group_rows:
            experts.update(eids)
        for e in experts:
            key = (layer, e)
            if key in last_seen:
                distances.append(token - last_seen[key])
            last_seen[key] = token

    if not distances:
        return {}
    n = len(distances)
    return {
        "n": n,
        "frac_at_1": sum(1 for d in distances if d == 1) / n,
        "frac_le_5": sum(1 for d in distances if d <= 5) / n,
        "frac_le_10": sum(1 for d in distances if d <= 10) / n,
    }


# -------------------------------------------------------------------------
# 5. Final canon JSON: tok/s and CV stats.
# -------------------------------------------------------------------------
FINAL_CANON = Path("/home/keri/BSC/thesis/_meta/final_canon.json")


def canon_stats() -> dict:
    with FINAL_CANON.open() as f:
        d = json.load(f)
    cvs = [c["eval_seconds_cv_pct"] for c in d["configs"].values()]
    return {
        "n_configs": d["n_configs"],
        "n_runs": d["n_runs"],
        "cv_mean_pct": statistics.mean(cvs),
        "cv_median_pct": statistics.median(cvs),
        "cv_max_pct": max(cvs),
        "n_cv_below_0p5": sum(1 for c in cvs if c < 0.5),
        "tok_per_s": {k: v["tok_per_s_mean"] for k, v in d["configs"].items()},
        "configs": d["configs"],
    }


# -------------------------------------------------------------------------
# Main: print authoritative numbers in a single report.
# -------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("AUTHORITATIVE NUMBERS (units: GiB / MiB / KiB ; rounding: 2 decimals)")
    print("=" * 78)
    print()

    def print_breakdown(b: dict) -> None:
        print(f"[{b['model']}] weight breakdown (byte-exact)")
        print("-" * 78)
        print(f"  File size on disk           : {b['file_size_bytes']:>14d} bytes  ({fmt2(to_gib(b['file_size_bytes']))} GiB)")
        print(f"  Tensor data total           : {b['tensor_data_bytes']:>14d} bytes  ({fmt2(to_gib(b['tensor_data_bytes']))} GiB)")
        print(f"  GGUF header overhead        : {b['header_overhead_bytes']:>14d} bytes  ({fmt2(to_mib(b['header_overhead_bytes']))} MiB)")
        print(f"  N tensors                   : {b['n_tensors_total']}")
        print(f"  N layers / experts / used   : {b['n_layers']} / {b['n_experts_per_layer']} / {b['n_experts_used']}")
        print()
        print("  Categories (sum of size_bytes):")
        for cat, nb in sorted(b["categories"].items(), key=lambda x: -x[1]):
            print(f"    {cat:<22s}      : {nb:>14d} bytes  ({fmt2(to_gib(nb))} GiB / {fmt2(to_mib(nb))} MiB)")
        print()
        print(f"  Per-expert-tensor size      : min={b['expert_size_min']} max={b['expert_size_max']} mode={b['expert_size_mode']}")
        print(f"  One expert weight matrix    : {b['expert_size_mode']:>14d} bytes  ({fmt2(to_mib(b['expert_size_mode']))} MiB)")
        print(f"  N expert tensors            : {b['n_expert_tensors']} (vs {b['n_layers']*b['n_experts_per_layer']*3} weights; rest are biases)")
        print()
        pt = per_token_reads(b)
        n_layers = b["n_layers"]
        n_used = b["n_experts_used"]
        n_slices = n_used * 3 * n_layers
        print(f"  Per-token slices (count)    : {n_used} x 3 x {n_layers} = {n_slices}")
        print(f"  Per-token expert reads      : {pt['per_token_expert_bytes']:>14d} bytes  ({fmt2(to_gib(pt['per_token_expert_bytes']))} GiB)")
        print(f"  Per-token total reads       : {pt['total_reads_per_token_bytes']:>14d} bytes  ({fmt2(to_gib(pt['total_reads_per_token_bytes']))} GiB)")
        print(f"  Reads / file size           : {fmt2(pt['frac_reads_of_file']*100)} %")
        print(f"  Resident-dense + experts    : {pt['total_resident_dense_per_token_bytes']:>14d} bytes  ({fmt2(to_gib(pt['total_resident_dense_per_token_bytes']))} GiB)")
        print(f"    /file-size                : {fmt2(pt['frac_resident_of_file']*100)} %  (lenient interp)")
        print()

    print_breakdown(load_20b_breakdown())
    print_breakdown(load_120b_breakdown())

    print("[3] Run log: GPT-OSS-20B canonical 7 GiB cell (async_projection_overlap_lfua_20b_7g_run1.log)")
    print("-" * 78)
    log20 = parse_log(LOG_20B)
    if "cpu_mapped_mib" in log20:
        print(f"  CPU_Mapped model buffer     : {log20['cpu_mapped_mib']} MiB  ({fmt2(log20['cpu_mapped_mib']/1024)} GiB)")
    if "kv_buffer_mib_parts" in log20:
        print(f"  KV buffer parts             : {log20['kv_buffer_mib_parts']} (sum {log20['kv_buffer_mib_sum']} MiB = {fmt2(log20['kv_buffer_mib_sum']/1024)} GiB)")
    if "compute_buffer_mib" in log20:
        print(f"  Compute buffer              : {log20['compute_buffer_mib']} MiB  ({fmt2(log20['compute_buffer_mib']/1024)} GiB)")
    if "output_buffer_mib" in log20:
        print(f"  Output buffer               : {log20['output_buffer_mib']} MiB")
    if "mlock_mib" in log20:
        print(f"  mlock'd region              : {log20['mlock_mib']} MiB  ({fmt2(log20['mlock_mib']/1024)} GiB)")
    if "expert_bytes" in log20:
        print(f"  expert_bytes (one slice)    : {log20['expert_bytes']:>14d} bytes  ({fmt2(to_mib(log20['expert_bytes']))} MiB)")
        print(f"  slot_bytes (padded to 512)  : {log20['slot_bytes']:>14d} bytes  ({fmt2(to_mib(log20['slot_bytes']))} MiB)")
        print(f"  cache_slots                 : {log20['cache_slots']}")
        print(f"  cache total                 : {log20['cache_slots']*log20['slot_bytes']:>14d} bytes  ({fmt2(to_gib(log20['cache_slots']*log20['slot_bytes']))} GiB)")
    if "uring_cache_hits" in log20 and "uring_cache_misses" in log20:
        h = log20["uring_cache_hits"]
        m = log20["uring_cache_misses"]
        total = h + m
        print(f"  Cache hits / misses         : {h} / {m}  (total accesses {total})")
        print(f"  Cache hit rate              : {h/total*100:.4f} %  -> {fmt2(h/total*100)} %")
    if "vmrss_kb_sequence" in log20:
        seq = log20["vmrss_kb_sequence"]
        print(f"  VmRSS sequence (first 10)   : {seq[:10]}")
        print(f"  VmRSS max                   : {seq[-1]} kB  ({fmt2(seq[-1]/1024/1024)} GiB)")
    print()

    print("[4] Run log: GPT-OSS-120B canonical 28 GiB cell (async_projection_overlap_lru_120b_28g_run1.log)")
    print("-" * 78)
    log120 = parse_log(LOG_120B)
    if "cpu_mapped_mib" in log120:
        print(f"  CPU_Mapped model buffer     : {log120['cpu_mapped_mib']} MiB  ({fmt2(log120['cpu_mapped_mib']/1024)} GiB)")
    if "kv_buffer_mib_parts" in log120:
        print(f"  KV buffer parts             : {log120['kv_buffer_mib_parts']} (sum {log120['kv_buffer_mib_sum']} MiB = {fmt2(log120['kv_buffer_mib_sum']/1024)} GiB)")
    if "compute_buffer_mib" in log120:
        print(f"  Compute buffer              : {log120['compute_buffer_mib']} MiB  ({fmt2(log120['compute_buffer_mib']/1024)} GiB)")
    if "mlock_mib" in log120:
        print(f"  mlock'd region              : {log120['mlock_mib']} MiB  ({fmt2(log120['mlock_mib']/1024)} GiB)")
    if "expert_bytes" in log120:
        print(f"  expert_bytes (one slice)    : {log120['expert_bytes']:>14d} bytes  ({fmt2(to_mib(log120['expert_bytes']))} MiB)")
    if "vmrss_kb_sequence" in log120:
        seq = log120["vmrss_kb_sequence"]
        print(f"  VmRSS max                   : {seq[-1]} kB  ({fmt2(seq[-1]/1024/1024)} GiB)")
    print()

    print("[5] Per-(layer, expert) selection stats from cache_dump.csv")
    print("-" * 78)
    sel = selection_stats(CACHE_DUMP)
    print(f"  Tokens (decode)                     : {sel['n_tokens']}")
    print(f"  Overall min selections (a pair)     : {sel['overall_min']}")
    print(f"  Overall max selections (a pair)     : {sel['overall_max']}")
    print()
    print(f"  Per-layer stats (median, max, min over 32 experts):")
    for layer in sorted(sel["per_layer"]):
        s = sel["per_layer"][layer]
        print(f"    layer {layer:>2d}: n_tokens={s['n_tokens']:>4d}  median={s['median']:>5}  max={s['max']:>5}  min={s['min']:>4}")
    print()

    print("[6] Reuse-distance distribution (across tokens)")
    print("-" * 78)
    rd = reuse_distance_stats(CACHE_DUMP)
    if rd:
        print(f"  N reuse events    : {rd['n']}")
        print(f"  Distance == 1     : {rd['frac_at_1']*100:.4f} %  -> {fmt2(rd['frac_at_1']*100)} %")
        print(f"  Distance <= 5     : {rd['frac_le_5']*100:.4f} %  -> {fmt2(rd['frac_le_5']*100)} %")
        print(f"  Distance <= 10    : {rd['frac_le_10']*100:.4f} %  -> {fmt2(rd['frac_le_10']*100)} %")
    print()

    print("[7] CV statistics from final_canon.json")
    print("-" * 78)
    c = canon_stats()
    print(f"  N configs         : {c['n_configs']}")
    print(f"  N runs            : {c['n_runs']}")
    print(f"  Mean CV (%)       : {c['cv_mean_pct']:.4f}  -> {fmt2(c['cv_mean_pct'])}")
    print(f"  Median CV (%)     : {c['cv_median_pct']:.4f}  -> {fmt2(c['cv_median_pct'])}")
    print(f"  Max CV (%)        : {c['cv_max_pct']:.4f}  -> {fmt2(c['cv_max_pct'])}")
    print(f"  Configs CV<0.5%   : {c['n_cv_below_0p5']} of {c['n_configs']}")
    print()

    print("[8] Headline tok/s for every cell (rounded to 2 decimals)")
    print("-" * 78)
    for k in sorted(c["tok_per_s"]):
        print(f"  {k:50s}  {fmt2(c['tok_per_s'][k]):>6s} tok/s")
    print()


if __name__ == "__main__":
    main()
