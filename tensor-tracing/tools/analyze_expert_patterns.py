#!/usr/bin/env python3
"""
Analyze expert activation patterns from per-token trace JSON files.

Computes:
  1. Expert popularity distribution per layer (entropy, top-K concentration)
  2. Reuse distance distribution (how many tokens between re-accesses of same expert)
  3. Per-token expert overlap (fraction of experts shared with previous token)
  4. Cumulative unique (layer, expert) slots vs token index
  5. Belady's optimal miss count for a given cache size

Usage:
  python3 analyze_expert_patterns.py <trace_dir> [--cache-sizes 200,250,288,350]

  trace_dir: directory containing token-NNNNN.json files
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import math


def load_expert_selections(trace_dir: str) -> dict[int, dict[int, list[int]]]:
    """Load per-token, per-layer expert selections from trace JSON files.

    Returns: {token_id: {layer_id: [expert_ids]}}
    """
    trace_dir = Path(trace_dir)
    selections = {}

    for f in sorted(trace_dir.glob("token-*.json")):
        with open(f) as fh:
            data = json.load(fh)

        token_id = data.get("token_id", int(f.stem.split("-")[-1]))

        # Extract expert IDs from MUL_MAT_ID ops (skip ADD_ID — same experts)
        layer_experts = {}
        for entry in data["entries"]:
            if entry["operation_type"] == "MUL_MAT_ID" and entry["num_experts"] > 0:
                lid = entry["layer_id"]
                if lid not in layer_experts:
                    # First MUL_MAT_ID per layer = up projection. All 3 projections
                    # have the same expert IDs, so we only need the first one.
                    layer_experts[lid] = entry["expert_ids"][:entry["num_experts"]]

        if layer_experts:
            selections[token_id] = layer_experts

    return selections


def analyze_popularity(selections, n_expert=32):
    """Expert popularity distribution per layer."""
    print("\n" + "=" * 70)
    print("1. EXPERT POPULARITY PER LAYER")
    print("=" * 70)

    n_layers = max(max(layers.keys()) for layers in selections.values()) + 1
    n_tokens = len(selections)

    # Count per-layer expert frequencies
    layer_freq = defaultdict(lambda: defaultdict(int))
    for token_id, layers in selections.items():
        for lid, experts in layers.items():
            for eid in experts:
                layer_freq[lid][eid] += 1

    print(f"\n{'Layer':>5} {'Entropy':>8} {'Top1%':>7} {'Top4%':>7} {'Unique':>7} {'Top-4 experts':>30}")
    print("-" * 70)

    for lid in range(n_layers):
        freq = layer_freq[lid]
        total = sum(freq.values())
        if total == 0:
            continue

        # Entropy (bits)
        probs = [c / total for c in freq.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Top-1 and top-4 concentration
        sorted_experts = sorted(freq.items(), key=lambda x: -x[1])
        top1_pct = sorted_experts[0][1] / total * 100
        top4_pct = sum(c for _, c in sorted_experts[:4]) / total * 100

        # Unique experts seen
        n_unique = len(freq)

        # Top-4 expert IDs
        top4_str = ", ".join(f"e{eid}({cnt})" for eid, cnt in sorted_experts[:4])

        print(f"{lid:>5} {entropy:>8.2f} {top1_pct:>6.1f}% {top4_pct:>6.1f}% {n_unique:>7} {top4_str:>30}")

    # Summary statistics
    entropies = []
    for lid in range(n_layers):
        freq = layer_freq[lid]
        total = sum(freq.values())
        if total == 0:
            continue
        probs = [c / total for c in freq.values()]
        entropies.append(-sum(p * math.log2(p) for p in probs if p > 0))

    print(f"\nMax possible entropy (uniform over 32): {math.log2(32):.2f} bits")
    print(f"Mean layer entropy: {sum(entropies)/len(entropies):.2f} bits")
    print(f"Min/Max layer entropy: {min(entropies):.2f} / {max(entropies):.2f}")


def analyze_reuse_distance(selections):
    """Reuse distance: how many token-steps between consecutive accesses to same (layer, expert)."""
    print("\n" + "=" * 70)
    print("2. REUSE DISTANCE DISTRIBUTION")
    print("=" * 70)

    # Build access sequence: for each (layer, expert), record the token_ids it appears at
    access_tokens = defaultdict(list)
    for token_id in sorted(selections.keys()):
        for lid, experts in selections[token_id].items():
            for eid in experts:
                access_tokens[(lid, eid)].append(token_id)

    # Compute reuse distances
    distances = []
    for key, tokens in access_tokens.items():
        for i in range(1, len(tokens)):
            distances.append(tokens[i] - tokens[i - 1])

    if not distances:
        print("No reuse distances found (too few tokens?)")
        return

    distances.sort()
    n = len(distances)

    print(f"\nTotal reuse events: {n}")
    print(f"Distance percentiles:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * pct / 100), n - 1)
        print(f"  P{pct:>2}: {distances[idx]:>6} tokens")

    print(f"  Min: {distances[0]:>6}")
    print(f"  Max: {distances[-1]:>6}")
    print(f"  Mean: {sum(distances) / n:>6.1f}")

    # Histogram of short distances (most relevant for caching)
    print(f"\nShort-distance breakdown:")
    for d in [1, 2, 3, 4, 5, 10, 20, 50]:
        count = sum(1 for x in distances if x <= d)
        print(f"  distance <= {d:>3}: {count:>6} ({count/n*100:>5.1f}%)")


def analyze_token_overlap(selections):
    """Per-token expert overlap: fraction of experts shared with previous token."""
    print("\n" + "=" * 70)
    print("3. TOKEN-TO-TOKEN EXPERT OVERLAP")
    print("=" * 70)

    tokens = sorted(selections.keys())
    if len(tokens) < 2:
        print("Need at least 2 tokens")
        return

    overlaps_per_layer = defaultdict(list)
    total_overlaps = []

    for i in range(1, len(tokens)):
        prev_tok = tokens[i - 1]
        curr_tok = tokens[i]
        prev = selections[prev_tok]
        curr = selections[curr_tok]

        token_overlap_count = 0
        token_total_count = 0

        for lid in curr:
            if lid in prev:
                curr_set = set(curr[lid])
                prev_set = set(prev[lid])
                overlap = len(curr_set & prev_set)
                n_experts = len(curr[lid])
                overlaps_per_layer[lid].append(overlap / n_experts)
                token_overlap_count += overlap
                token_total_count += n_experts

        if token_total_count > 0:
            total_overlaps.append(token_overlap_count / token_total_count)

    print(f"\nGlobal token-to-token overlap: {sum(total_overlaps)/len(total_overlaps)*100:.1f}%")
    print(f"  (fraction of current token's experts that were also in previous token)")

    print(f"\nPer-layer overlap:")
    n_layers = max(overlaps_per_layer.keys()) + 1
    for lid in range(n_layers):
        vals = overlaps_per_layer.get(lid, [])
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  Layer {lid:>2}: {mean*100:>5.1f}%")


def analyze_cumulative_unique(selections):
    """Cumulative unique (layer, expert) slots touched vs token index."""
    print("\n" + "=" * 70)
    print("4. CUMULATIVE UNIQUE SLOTS vs TOKEN INDEX")
    print("=" * 70)

    tokens = sorted(selections.keys())
    seen = set()
    cumulative = []

    for tok in tokens:
        for lid, experts in selections[tok].items():
            for eid in experts:
                seen.add((lid, eid))
        cumulative.append((tok, len(seen)))

    # Print at key points
    print(f"\n{'Token':>6} {'Unique slots':>14} {'Δ since prev':>14}")
    print("-" * 40)
    checkpoints = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1999]
    prev = 0
    for tok, count in cumulative:
        if tok in checkpoints or tok == cumulative[-1][0]:
            print(f"{tok:>6} {count:>14} {count - prev:>+14}")
            prev = count

    print(f"\nTotal unique (layer, expert) pairs: {cumulative[-1][1]}")
    print(f"Max possible: {24 * 32} (24 layers × 32 experts)")
    print(f"Saturation: {cumulative[-1][1] / (24 * 32) * 100:.1f}%")


def analyze_belady(selections, cache_sizes):
    """Belady's optimal (MIN) algorithm: evict the slot whose next use is furthest.
    Gives the theoretical minimum cache misses for each cache size."""
    print("\n" + "=" * 70)
    print("5. BELADY'S OPTIMAL CACHE MISS RATE")
    print("=" * 70)

    # Build the full access sequence: list of (layer, proj, expert) tuples in order.
    # Since all 3 projections for the same expert are accessed together per layer,
    # we model the cache at (layer, expert) granularity and multiply by 3 for slot count.
    # Or more precisely: model at (layer, proj, expert) granularity.
    # For simplicity and accuracy: model at (layer, expert) and count 3 slots per entry.

    access_seq = []
    for tok in sorted(selections.keys()):
        for lid in range(24):
            if lid in selections[tok]:
                for eid in selections[tok][lid]:
                    # 3 projections per expert per layer
                    for proj in range(3):
                        access_seq.append((lid, proj, eid))

    n_accesses = len(access_seq)

    # Precompute next_use[i] = index of next occurrence of access_seq[i] after position i
    next_use_map = defaultdict(list)
    for i, key in enumerate(access_seq):
        next_use_map[key].append(i)

    # For each key, the positions are sorted. We can binary search.
    import bisect

    def get_next_use(key, after_pos):
        positions = next_use_map[key]
        idx = bisect.bisect_right(positions, after_pos)
        if idx < len(positions):
            return positions[idx]
        return float('inf')

    print(f"\nTotal accesses: {n_accesses}")
    print(f"Cache sizes to evaluate: {cache_sizes}")
    print()

    for cache_size in cache_sizes:
        # Simulate Belady's optimal
        cache = {}  # key -> True
        misses = 0
        hits = 0

        for i, key in enumerate(access_seq):
            if key in cache:
                hits += 1
            else:
                misses += 1
                if len(cache) >= cache_size:
                    # Evict the entry with the furthest next use
                    farthest_key = None
                    farthest_next = -1
                    for ck in cache:
                        nu = get_next_use(ck, i)
                        if nu > farthest_next:
                            farthest_next = nu
                            farthest_key = ck
                    del cache[farthest_key]
                cache[key] = True

        miss_rate = misses / n_accesses * 100
        print(f"  cache={cache_size:>4}: {misses:>7} misses, {hits:>7} hits, miss rate={miss_rate:>5.1f}%")

    # Also compute uncached (every access is a miss)
    print(f"  no cache:  {n_accesses:>7} misses, {0:>7} hits, miss rate=100.0%")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert activation patterns for cache policy design")
    parser.add_argument("trace_dir", help="Directory containing token-NNNNN.json trace files")
    parser.add_argument("--cache-sizes", default="100,150,200,250,288,350,500",
                        help="Comma-separated cache sizes for Belady analysis")
    args = parser.parse_args()

    cache_sizes = [int(x) for x in args.cache_sizes.split(",")]

    print(f"Loading traces from: {args.trace_dir}")
    selections = load_expert_selections(args.trace_dir)
    print(f"Loaded {len(selections)} tokens")

    if not selections:
        print("No traces found!", file=sys.stderr)
        sys.exit(1)

    analyze_popularity(selections)
    analyze_reuse_distance(selections)
    analyze_token_overlap(selections)
    analyze_cumulative_unique(selections)
    analyze_belady(selections, cache_sizes)


if __name__ == "__main__":
    main()
