#!/usr/bin/env python3
"""
LLM Parameter Offloading Experiment - Main Runner

Measures block I/O access patterns during LLM inference under memory pressure.

Usage:
    sudo python3 run_experiment.py
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Import utilities
from utils import (
    log,
    run_cmd,
    check_root,
    drop_caches,
    compile_mem_locker,
    mount_dedicated_tmpfs,
    unmount_dedicated_tmpfs,
    blktrace_to_csv,
    analyze_with_duckdb
)


def load_settings():
    """Load experiment settings from JSON file

    Returns:
        dict: Settings dictionary
    """
    settings_path = Path(__file__).parent / "settings.json"

    if not settings_path.exists():
        print(f"ERROR: settings.json not found at {settings_path}")
        sys.exit(1)

    with open(settings_path, 'r') as f:
        settings = json.load(f)

    return settings


def resolve_paths(settings):
    """Resolve relative paths to absolute paths

    Args:
        settings: Settings dictionary

    Returns:
        dict: Paths dictionary with absolute paths
    """
    script_dir = Path(__file__).parent.absolute()
    bsc_dir = script_dir.parent

    paths = {
        'llama_cli': bsc_dir / settings['paths']['llama_cli'],
        'models_dir': bsc_dir / settings['paths']['models_dir'],
        'mlock_tool_cpp': bsc_dir / settings['paths']['mlock_tool_cpp'],
        'mlock_bin': bsc_dir / settings['paths']['mlock_bin'],
        'results_dir': script_dir / settings['paths']['results_dir']
    }

    return paths


def run_experiment(settings, paths):
    """Run single experiment with blktrace monitoring

    Args:
        settings: Settings dictionary
        paths: Paths dictionary

    Returns:
        Path: Results directory
    """
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = paths['results_dir'] / f"experiment_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    log(f"Results will be saved to: {result_dir}")

    # Get model file size
    model_path = paths['models_dir'] / settings['experiment']['model_file']
    model_size_bytes = model_path.stat().st_size
    model_size_gb = model_size_bytes / (1024 ** 3)

    log(f"Model: {settings['experiment']['model_file']} ({model_size_gb:.2f} GB)")

    # Save configuration
    config = {
        "timestamp": timestamp,
        "model_file": settings['experiment']['model_file'],
        "model_size_bytes": model_size_bytes,
        "model_size_gb": model_size_gb,
        "tokens": settings['experiment']['tokens_to_generate'],
        "prompt": settings['experiment']['prompt'],
        "mlock_gb": settings['memory']['mlock_size_gb'],
        "block_device": settings['storage']['block_device'],
        "gap_small_sectors": settings['analysis']['gap_small_sectors'],
        "gap_medium_sectors": settings['analysis']['gap_medium_sectors']
    }

    with open(result_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    log("Configuration saved")

    # Step 1: Drop caches
    drop_caches()

    # Step 2: Mount dedicated tmpfs (BEFORE mem_locker!)
    tmpfs_dir = mount_dedicated_tmpfs(
        settings['memory']['tmpfs_mount'],
        settings['memory']['tmpfs_size_gb']
    )
    log(f"Dedicated tmpfs ready: {tmpfs_dir} ({settings['memory']['tmpfs_size_gb']}GB isolated RAM)")

    # Step 3: Start blktrace
    log(f"Starting blktrace on {settings['storage']['block_device']}...")
    blktrace_proc = subprocess.Popen(
        ["blktrace", "-d", settings['storage']['block_device'], "-o", "trace"],
        cwd=str(tmpfs_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(2)
    log("blktrace running")

    # Step 4: Start mem_locker
    log(f"Starting mem_locker ({settings['memory']['mlock_size_gb']} GB)...")
    mlock_log = result_dir / "mlock.log"
    mlock_proc = subprocess.Popen(
        [str(paths['mlock_bin']), str(settings['memory']['mlock_size_gb'])],
        stdout=open(mlock_log, 'w'),
        stderr=subprocess.STDOUT
    )
    time.sleep(5)
    log("mem_locker running")

    # Step 5: Capture memory state BEFORE inference
    log("Capturing pre-inference memory state...")
    mem_before = run_cmd("free -b", capture=True)

    with open(result_dir / "memory_before.txt", 'w') as f:
        f.write("=== free -b ===\n")
        f.write(mem_before + "\n")

    # Step 6: Run llama-cli inference (capture PID)
    log(f"Running llama-cli inference ({settings['experiment']['tokens_to_generate']} tokens)...")
    inference_log = result_dir / "inference.log"

    start_time = time.time()

    # Start llama-cli as subprocess (non-blocking to capture PID)
    llama_proc = subprocess.Popen(
        [
            str(paths['llama_cli']),
            "-m", str(model_path),
            "-p", settings['experiment']['prompt'],
            "-n", str(settings['experiment']['tokens_to_generate']),
            "--log-disable"
        ],
        stdout=open(inference_log, 'w'),
        stderr=subprocess.STDOUT
    )

    # Capture PID immediately
    llama_pid = llama_proc.pid
    log(f"llama-cli PID: {llama_pid}")

    # Save PID for later analysis
    with open(result_dir / "llama_pid.txt", 'w') as f:
        f.write(str(llama_pid))

    # Wait for completion
    try:
        llama_proc.wait(timeout=3600)
        elapsed_time = time.time() - start_time

        if llama_proc.returncode == 0:
            tokens_per_sec = settings['experiment']['tokens_to_generate'] / elapsed_time
            log(f"Inference complete: {elapsed_time:.2f}s ({tokens_per_sec:.2f} tok/s)")
            inference_success = True
        else:
            log(f"ERROR: Inference failed with code {llama_proc.returncode}")
            elapsed_time = time.time() - start_time
            tokens_per_sec = 0
            inference_success = False

    except subprocess.TimeoutExpired:
        log("ERROR: Inference timed out!")
        llama_proc.kill()
        elapsed_time = time.time() - start_time
        tokens_per_sec = 0
        inference_success = False

    # Step 7: Capture memory state AFTER inference
    log("Capturing post-inference memory state...")
    mem_after = run_cmd("free -b", capture=True)

    with open(result_dir / "memory_after.txt", 'w') as f:
        f.write("=== free -b ===\n")
        f.write(mem_after + "\n")

    # Save performance metrics
    metrics = {
        "success": inference_success,
        "total_time_sec": elapsed_time,
        "tokens_generated": settings['experiment']['tokens_to_generate'] if inference_success else 0,
        "tokens_per_second": tokens_per_sec,
        "llama_pid": llama_pid
    }

    with open(result_dir / "performance.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Step 8: Stop blktrace
    log("Stopping blktrace...")
    run_cmd("killall blktrace", check=False)
    time.sleep(2)

    # Step 9: Stop mem_locker
    log("Stopping mem_locker...")
    mlock_proc.terminate()
    try:
        mlock_proc.wait(timeout=5)
    except:
        mlock_proc.kill()

    log("All processes stopped")

    # Step 10: Copy blktrace files from RAM to disk
    log("Copying blktrace data from RAM to disk...")
    blktrace_dest = result_dir / "blktrace"
    blktrace_dest.mkdir(exist_ok=True)

    trace_files = list(tmpfs_dir.glob("trace.blktrace.*"))

    if not trace_files:
        log("WARNING: No blktrace files found!")
    else:
        log(f"Found {len(trace_files)} blktrace files")
        for trace_file in trace_files:
            run_cmd(f"cp {trace_file} {blktrace_dest}/")
        log(f"Blktrace files saved to {blktrace_dest}")

    # Step 11: Unmount dedicated tmpfs (free 8GB RAM)
    unmount_dedicated_tmpfs(settings['memory']['tmpfs_mount'])

    log(f"\n{'='*70}")
    log(f"Experiment complete! Results in: {result_dir}")
    log(f"{'='*70}\n")

    return result_dir


def parse_memory_snapshot(filepath):
    """Extract memory stats from free output

    Args:
        filepath: Path to memory snapshot file

    Returns:
        dict: Memory statistics or None if parsing fails
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Parse "free -b" output
    for line in lines:
        if line.startswith('Mem:'):
            parts = line.split()
            total = int(parts[1])
            used = int(parts[2])
            free = int(parts[3])
            shared = int(parts[4])
            cache = int(parts[5])
            available = int(parts[6])

            return {
                'total_gb': total / 1024**3,
                'used_gb': used / 1024**3,
                'free_gb': free / 1024**3,
                'cache_gb': cache / 1024**3,
                'available_gb': available / 1024**3
            }

    return None


def main():
    """Main entry point"""

    print("\n" + "="*70)
    print("  LLM Parameter Offloading Experiment")
    print("="*70 + "\n")

    # Check root
    check_root()

    # Load settings
    log("Loading settings from settings.json...")
    settings = load_settings()

    # Resolve paths
    paths = resolve_paths(settings)

    # Compile mem_locker if needed
    compile_mem_locker(paths['mlock_bin'], paths['mlock_tool_cpp'])

    # Run experiment
    result_dir = run_experiment(settings, paths)

    # Convert blktrace to CSV
    blktrace_dir = result_dir / "blktrace"

    if not list(blktrace_dir.glob("trace.blktrace.*")):
        log("ERROR: No blktrace files to analyze!")
        return

    csv_path = result_dir / "blktrace.csv"
    blktrace_to_csv(blktrace_dir, csv_path, result_dir)

    # Analyze with DuckDB
    analyze_with_duckdb(
        csv_path,
        result_dir,
        settings['analysis']['gap_small_sectors'],
        settings['analysis']['gap_medium_sectors']
    )

    # Parse memory files for page cache metrics
    log("\nExtracting memory usage from snapshots...")

    mem_before = parse_memory_snapshot(result_dir / "memory_before.txt")
    mem_after = parse_memory_snapshot(result_dir / "memory_after.txt")

    if mem_before and mem_after:
        cache_delta = mem_after['cache_gb'] - mem_before['cache_gb']

        log(f"Page cache before: {mem_before['cache_gb']:.2f} GB")
        log(f"Page cache after:  {mem_after['cache_gb']:.2f} GB")
        log(f"Cache delta:       {cache_delta:.2f} GB")

        # Add to analysis if it exists
        analysis_file = result_dir / "analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)

            analysis["memory_metrics"] = {
                "page_cache_before_gb": mem_before['cache_gb'],
                "page_cache_after_gb": mem_after['cache_gb'],
                "page_cache_delta_gb": cache_delta,
                "method_a_total_read_gb": analysis.get("total_gb_read", 0),
                "method_b_unique_sectors_mb": analysis.get("unique_mb", 0)
            }

            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)

            log("Memory metrics added to analysis.json")

    print("\n" + "="*70)
    print(f"  ALL COMPLETE! Results in: {result_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
