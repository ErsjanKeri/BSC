"""
Analysis tools for blktrace data

Includes:
- blktrace to CSV conversion
- DuckDB-based analysis
- Gap distribution analysis
- Bandwidth calculations
- Dynamic GGUF sector range detection
"""

import json
import re
from pathlib import Path
from .setup_tools import log, run_cmd


def get_gguf_sector_range(model_path):
    """Get physical sector range for GGUF model file using filefrag

    Args:
        model_path: Path to .gguf model file

    Returns:
        tuple: (start_sector, end_sector, num_extents)

    Note:
        Sectors are 512 bytes each.
        filefrag shows blocks (4096 bytes = 8 sectors).
        We convert: block_number * 8 = sector_number

        Adjacent extents are merged (same as filefrag's summary count)
    """
    log(f"Detecting GGUF sector range for {model_path.name}...")

    # Run filefrag -v to get extent information
    result = run_cmd(f"filefrag -v {model_path}", capture=True)

    # Parse output to find physical extents
    # Format: "  0:   0.. 4095: 128862208.. 128866303:  4096:"
    extent_pattern = r'^\s*\d+:\s+\d+\.\.\s*\d+:\s+(\d+)\.\.\s*(\d+):\s+\d+:'

    extents = []
    for line in result.split('\n'):
        match = re.search(extent_pattern, line)
        if match:
            start_block = int(match.group(1))
            end_block = int(match.group(2))
            extents.append((start_block, end_block))

    if not extents:
        raise ValueError(f"Failed to parse filefrag output for {model_path}")

    # Sort extents by physical start block
    extents.sort(key=lambda x: x[0])

    # Merge adjacent extents (where end+1 == next_start)
    merged_extents = []
    current_start, current_end = extents[0]

    for start, end in extents[1:]:
        if start == current_end + 1:
            # Adjacent extent, merge it
            current_end = end
        else:
            # Gap found, save current extent and start new one
            merged_extents.append((current_start, current_end))
            current_start, current_end = start, end

    # Don't forget the last extent
    merged_extents.append((current_start, current_end))

    # Get overall range
    min_block = merged_extents[0][0]
    max_block = merged_extents[-1][1]
    num_extents = len(merged_extents)

    # Convert blocks (4096 bytes) to sectors (512 bytes)
    # 1 block = 8 sectors
    min_sector = min_block * 8
    max_sector = (max_block + 1) * 8  # +1 because filefrag end is inclusive

    log(f"  Found {num_extents} extent(s) (after merging adjacent)")
    log(f"  Physical blocks: {min_block:,} to {max_block:,}")
    log(f"  Physical sectors: {min_sector:,} to {max_sector:,}")

    return (min_sector, max_sector, num_extents)


def blktrace_to_csv(blktrace_dir, output_csv, result_dir):
    """Convert blktrace binary files to CSV with all columns

    blkparse output format:
    major,minor cpu seq timestamp pid action rwbs sector + size [process]

    CSV columns:
    device_major,device_minor,cpu,seq,timestamp,pid,action,rwbs,sector,size_sectors,size_bytes,process

    Args:
        blktrace_dir: Directory containing trace.blktrace.* files
        output_csv: Output CSV file path
        result_dir: Results directory for intermediate files

    Returns:
        Path: output CSV file path
    """
    log("Converting blktrace to CSV...")

    # Run blkparse
    blkparse_output = result_dir / "blkparse_raw.txt"

    run_cmd(
        f"blkparse -i {blktrace_dir}/trace -o {blkparse_output}",
        check=True
    )

    log(f"blkparse complete, parsing to CSV...")

    # Parse to CSV
    csv_lines = []
    csv_lines.append("# Columns: device_major,device_minor,cpu,seq,timestamp,pid,action,rwbs,sector,size_sectors,size_bytes,process")
    csv_lines.append("device_major,device_minor,cpu,seq,timestamp,pid,action,rwbs,sector,size_sectors,size_bytes,process")

    with open(blkparse_output, 'r') as f:
        for line in f:
            parts = line.split()

            # Skip header/summary lines
            if len(parts) < 10:
                continue

            # Skip non-I/O lines
            if parts[0] == "CPU" or parts[0] == "Total":
                continue

            try:
                # Device is "major,minor" - split it
                device_parts = parts[0].split(',')
                device_major = device_parts[0]
                device_minor = device_parts[1] if len(device_parts) > 1 else "0"

                cpu = parts[1]
                seq = parts[2]
                timestamp = parts[3]
                pid = parts[4]
                action = parts[5]
                rwbs = parts[6]
                sector = parts[7]
                # parts[8] is "+"
                size_sectors = parts[9]
                size_bytes = int(size_sectors) * 512

                # Process name is optional, in [brackets]
                process = ""
                if len(parts) > 10:
                    # Find [process]
                    for i in range(10, len(parts)):
                        if parts[i].startswith('[') and parts[i].endswith(']'):
                            process = parts[i][1:-1]  # Remove brackets
                            break

                csv_line = f"{device_major},{device_minor},{cpu},{seq},{timestamp},{pid},{action},{rwbs},{sector},{size_sectors},{size_bytes},{process}"
                csv_lines.append(csv_line)

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

    # Write CSV
    with open(output_csv, 'w') as f:
        f.write('\n'.join(csv_lines))

    log(f"CSV saved: {output_csv} ({len(csv_lines)-2} rows)")

    return output_csv


def analyze_with_duckdb(csv_path, result_dir, gap_small, gap_medium, gguf_start_sector, gguf_end_sector, num_extents):
    """Analyze blktrace CSV with DuckDB

    Performs comprehensive analysis including:
    - Total bytes read
    - Unique sectors accessed (corrected calculation)
    - Gap distribution (sequential vs random access)
    - Bandwidth over time

    Args:
        csv_path: Path to blktrace CSV file
        result_dir: Results directory for output files
        gap_small: Small gap threshold in sectors (e.g., 256 = 128KB)
        gap_medium: Medium gap threshold in sectors (e.g., 2048 = 1MB)
        gguf_start_sector: Start sector of .gguf file (from get_gguf_sector_range)
        gguf_end_sector: End sector of .gguf file (from get_gguf_sector_range)
        num_extents: Number of extents (for logging fragmentation level)
    """
    try:
        import duckdb
    except ImportError:
        log("ERROR: duckdb not installed. Install with: pip install duckdb")
        return

    # Read model size from config
    config_path = result_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_size_gb = config["model_size_gb"]

    log("Analyzing with DuckDB...")

    con = duckdb.connect()

    # Load CSV (skip comment line)
    con.execute(f"""
        CREATE TABLE trace AS
        SELECT * FROM read_csv(
            '{csv_path}',
            skip=1,
            header=true,
            delim=',',
            columns={{
                'device_major': 'INTEGER',
                'device_minor': 'INTEGER',
                'cpu': 'INTEGER',
                'seq': 'INTEGER',
                'timestamp': 'DOUBLE',
                'pid': 'INTEGER',
                'action': 'VARCHAR',
                'rwbs': 'VARCHAR',
                'sector': 'BIGINT',
                'size_sectors': 'INTEGER',
                'size_bytes': 'BIGINT',
                'process': 'VARCHAR'
            }},
            ignore_errors=true
        )
    """)

    # Read llama-cli PID from saved file
    pid_file = result_dir / "llama_pid.txt"
    if pid_file.exists():
        llama_pid = int(pid_file.read_text().strip())
        log(f"Filtering for llama-cli PID: {llama_pid}")

        # Use dynamic .gguf file sector range (from get_gguf_sector_range)
        log(f"Filtering for .gguf sectors: {gguf_start_sector:,} to {gguf_end_sector:,} ({num_extents} extents)")

        # Filter for reads from llama-cli AND within .gguf sector range
        con.execute(f"""
            CREATE TABLE reads AS
            SELECT * FROM trace
            WHERE action = 'D' AND rwbs LIKE '%R%'
            AND pid = {llama_pid}
            AND sector >= {gguf_start_sector}
            AND sector <= {gguf_end_sector}
            ORDER BY timestamp
        """)
    total_rows = con.execute("SELECT COUNT(*) FROM reads").fetchone()[0]
    log(f"Total read operations: {total_rows:,}")

    if total_rows == 0:
        log("WARNING: No read operations found in blktrace!")
        con.close()
        return

    # ========================================================================
    # Metric 1: Total bytes read (Amount swapped - Method A)
    # ========================================================================

    total_bytes = con.execute("SELECT SUM(size_bytes) FROM reads").fetchone()[0]
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)

    log(f"Total bytes read: {total_gb:.2f} GB ({total_mb:.2f} MB)")

    # ========================================================================
    # Metric 2: Unique sectors accessed (Amount swapped - Method B)
    # ========================================================================

    log("Calculating unique sectors accessed (expanding all ranges)...")

    # CRITICAL FIX: Properly count unique sectors by expanding each read's range
    # Each read covers [sector, sector+size_sectors-1]
    # We must expand all ranges and count DISTINCT sectors

    # Method: Use generate_series to expand each read into individual sectors
    unique_sectors_query = """
        WITH expanded_sectors AS (
            SELECT UNNEST(
                generate_series(sector, sector + size_sectors - 1)
            ) AS sector_num
            FROM reads
        )
        SELECT COUNT(DISTINCT sector_num) AS unique_count
        FROM expanded_sectors
    """

    unique_sectors = con.execute(unique_sectors_query).fetchone()[0]
    unique_bytes = unique_sectors * 512
    unique_mb = unique_bytes / (1024 * 1024)
    unique_gb = unique_bytes / (1024 * 1024 * 1024)

    log(f"Unique sectors accessed: {unique_sectors:,} (~{unique_mb:.2f} MB / {unique_gb:.2f} GB)")
    log(f"  → This represents the actual unique model data touched")
    log(f"  → Model size is {model_size_gb:.2f} GB, so {unique_gb/model_size_gb*100:.1f}% of model accessed")

    # ========================================================================
    # Metric 3: Gap distribution analysis
    # ========================================================================

    log("Calculating gap distribution...")

    # Create table with gaps
    con.execute(f"""
        CREATE TABLE gaps AS
        WITH ordered_reads AS (
            SELECT
                sector,
                size_sectors,
                LAG(sector) OVER (ORDER BY timestamp) as prev_sector,
                LAG(size_sectors) OVER (ORDER BY timestamp) as prev_size
            FROM reads
        )
        SELECT
            sector - (prev_sector + prev_size) as gap,
            sector,
            prev_sector,
            size_sectors
        FROM ordered_reads
        WHERE prev_sector IS NOT NULL
    """)

    # Categorize gaps
    gap_stats = con.execute(f"""
        SELECT
            SUM(CASE WHEN gap = 0 THEN 1 ELSE 0 END) as perfect_sequential,
            SUM(CASE WHEN gap > 0 AND gap < {gap_small} THEN 1 ELSE 0 END) as small_gap,
            SUM(CASE WHEN gap >= {gap_small} AND gap < {gap_medium} THEN 1 ELSE 0 END) as medium_gap,
            SUM(CASE WHEN gap >= {gap_medium} THEN 1 ELSE 0 END) as large_gap,
            SUM(CASE WHEN gap < 0 THEN 1 ELSE 0 END) as backward,
            COUNT(*) as total
        FROM gaps
    """).fetchone()

    perfect = gap_stats[0] or 0
    small = gap_stats[1] or 0
    medium = gap_stats[2] or 0
    large = gap_stats[3] or 0
    backward = gap_stats[4] or 0
    total_gaps = gap_stats[5] or 1

    log(f"\nGap Distribution:")
    log(f"  Perfect sequential (gap=0):     {perfect:6,} ({perfect/total_gaps*100:5.1f}%)")
    log(f"  Small gaps (<128KB):            {small:6,} ({small/total_gaps*100:5.1f}%)")
    log(f"  Medium gaps (128KB-1MB):        {medium:6,} ({medium/total_gaps*100:5.1f}%)")
    log(f"  Large gaps (>1MB):              {large:6,} ({large/total_gaps*100:5.1f}%)")
    log(f"  Backward seeks:                 {backward:6,} ({backward/total_gaps*100:5.1f}%)")

    # ========================================================================
    # Metric 4: Bandwidth over time (1 second windows)
    # ========================================================================

    log("\nCalculating bandwidth over time...")

    bandwidth = con.execute("""
        SELECT
            CAST(timestamp AS INTEGER) as time_bucket,
            COUNT(*) as operations,
            SUM(size_bytes) / 1024.0 / 1024.0 as mb_read
        FROM reads
        GROUP BY time_bucket
        ORDER BY time_bucket
    """).fetchall()

    if bandwidth:
        log(f"Bandwidth per second (first 10 seconds):")
        for i, (bucket, ops, mb) in enumerate(bandwidth[:10]):
            log(f"  {bucket}s: {mb:8.2f} MB/s ({ops:5,} ops)")

        avg_bandwidth = sum(row[2] for row in bandwidth) / len(bandwidth)
        max_bandwidth = max(row[2] for row in bandwidth)
        log(f"\nAverage bandwidth: {avg_bandwidth:.2f} MB/s")
        log(f"Peak bandwidth:    {max_bandwidth:.2f} MB/s")
    else:
        avg_bandwidth = 0
        max_bandwidth = 0

    # ========================================================================
    # Save analysis results
    # ========================================================================

    analysis = {
        "total_reads": int(total_rows),
        "total_bytes_read": int(total_bytes),
        "total_mb_read": float(total_mb),
        "total_gb_read": float(total_gb),
        "unique_sectors": int(unique_sectors),
        "unique_bytes": int(unique_bytes),
        "unique_mb": float(unique_mb),
        "unique_gb": float(unique_gb),
        "unique_coverage_pct": float(unique_gb/model_size_gb*100),
        "gap_distribution": {
            "perfect_sequential": int(perfect),
            "small_gaps": int(small),
            "medium_gaps": int(medium),
            "large_gaps": int(large),
            "backward_seeks": int(backward),
            "total": int(total_gaps),
            "percent_perfect": float(perfect/total_gaps*100),
            "percent_small": float(small/total_gaps*100),
            "percent_medium": float(medium/total_gaps*100),
            "percent_large": float(large/total_gaps*100)
        },
        "bandwidth": {
            "average_mb_per_sec": float(avg_bandwidth),
            "peak_mb_per_sec": float(max_bandwidth)
        }
    }

    with open(result_dir / "analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    log(f"\nAnalysis saved to: {result_dir / 'analysis.json'}")

    con.close()
