#!/usr/bin/env python3
"""
analyze_blktrace.py - Analyze blkparse output for sequential vs random reads

Streaming implementation to handle very large traces without loading all rows.
"""

import sys
import json

SEQUENTIAL_THRESHOLD_SECTORS = 256  # 128KB gap tolerance

def stream_analyze(stream):
    """
    Stream through blkparse output and compute:
      - total reads / bytes
      - sequential vs random counts (consecutive reads)
      - average gap (sectors)
      - simple inter-arrival stats (mean/min/max)

    Assumes blkparse output is time-ordered (default behavior).
    """
    total_reads = 0
    total_bytes = 0
    sequential = 0
    random = 0
    gap_sum = 0
    gap_count = 0

    prev_ts = None
    prev_sector = None
    prev_size = None

    interval_count = 0
    interval_sum_ms = 0.0
    interval_min_ms = None
    interval_max_ms = None

    for line in stream:
            parts = line.split()
            # Expect: dev cpu seq time pid action rwbs sector + size ...
            # Example: 259,1 7 1 0.000000000 137 A W 1747855872 + 8 [...]
            if len(parts) < 10:
                continue

            ts_str = parts[3]
            rwbs = parts[6]
            sector_str = parts[7]
            size_str = parts[9]

            # Only reads
            if 'R' not in rwbs:
                continue

            try:
                ts = float(ts_str)
                sector = int(sector_str)
                size_sectors = int(size_str)
            except ValueError:
                continue

            size_bytes = size_sectors * 512
            total_reads += 1
            total_bytes += size_bytes

            if prev_sector is not None:
                expected_next = prev_sector + prev_size
                gap = abs(sector - expected_next)
                gap_sum += gap
                gap_count += 1
                if gap <= SEQUENTIAL_THRESHOLD_SECTORS:
                    sequential += 1
                else:
                    random += 1

                interval_ms = (ts - prev_ts) * 1000
                interval_count += 1
                interval_sum_ms += interval_ms
                interval_min_ms = interval_ms if interval_min_ms is None else min(interval_min_ms, interval_ms)
                interval_max_ms = interval_ms if interval_max_ms is None else max(interval_max_ms, interval_ms)

            prev_ts = ts
            prev_sector = sector
            prev_size = size_sectors

    if total_reads == 0:
        return None

    sequential_percent = (sequential / (sequential + random) * 100) if (sequential + random) > 0 else 0.0
    avg_gap_kb = (gap_sum / gap_count * 0.5) if gap_count else 0.0  # sectors→KB
    mean_interval_ms = (interval_sum_ms / interval_count) if interval_count else 0.0

    return {
        "total_reads": total_reads,
        "total_mb_read": total_bytes / (1024 * 1024),
        "sequential_reads": sequential,
        "random_reads": random,
        "sequential_percent": sequential_percent,
        "avg_gap_kb": avg_gap_kb,
        "mean_interval_ms": mean_interval_ms,
        "min_interval_ms": interval_min_ms if interval_min_ms is not None else 0.0,
        "max_interval_ms": interval_max_ms if interval_max_ms is not None else 0.0,
    }


def classify_pattern(sequential_percent):
    if sequential_percent > 80:
        return "HIGHLY SEQUENTIAL"
    elif sequential_percent > 50:
        return "MOSTLY SEQUENTIAL"
    elif sequential_percent > 20:
        return "MIXED (Sequential + Random)"
    else:
        return "HIGHLY RANDOM"


def main():
    if len(sys.argv) < 2:
        print("Usage: ./analyze_blktrace.py <blkparse_output.txt> [output.json]")
        sys.exit(1)

    blktrace_file = sys.argv[1]
    output_json = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Analyzing block trace: {blktrace_file}\n")

    if blktrace_file == "-":
        results = stream_analyze(sys.stdin)
    else:
        with open(blktrace_file, "r") as f:
            results = stream_analyze(f)
    if not results:
        print("No read operations found in trace file")
        sys.exit(1)

    pattern_type = classify_pattern(results["sequential_percent"])
    results["pattern_classification"] = pattern_type

    print("=" * 60)
    print("  Block I/O Access Pattern Analysis")
    print("=" * 60)
    print(f"Total read operations:    {results['total_reads']:,}")
    print(f"Total data read:          {results['total_mb_read']:.2f} MB")
    print("")
    print("Access Pattern:")
    print(f"  Sequential reads:       {results['sequential_reads']:,} ({results['sequential_percent']:.1f}%)")
    print(f"  Random reads:           {results['random_reads']:,} ({100 - results['sequential_percent']:.1f}%)")
    print(f"  Average gap:            {results['avg_gap_kb']:.1f} KB")
    print("")
    print("Inter-arrival (proxy for latency):")
    print(f"  Mean:                   {results['mean_interval_ms']:.3f} ms")
    print(f"  Min:                    {results['min_interval_ms']:.3f} ms")
    print(f"  Max:                    {results['max_interval_ms']:.3f} ms")
    print("")
    print(f"Pattern Classification:   {pattern_type}")
    print("=" * 60)

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
