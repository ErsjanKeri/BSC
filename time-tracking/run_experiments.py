#!/usr/bin/env python3
"""
MAP_POPULATE Impact Study - Systematic Experiment Runner

Measures the impact of MAP_POPULATE flag on model loading time and inference speed.
Tests both small models (fit in RAM) and large models (exceed RAM).

Usage:
    python3 run_experiments.py [--small-only] [--large-only]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from utils import (
    log,
    run_cmd,
    drop_page_cache,
    parse_llama_output,
    write_csv_header,
    append_csv_row,
    calculate_statistics
)


def load_settings():
    """Load experiment settings"""
    settings_path = Path(__file__).parent / "settings.json"
    with open(settings_path, 'r') as f:
        return json.load(f)


def run_single_iteration(binary_path, lib_path, model_path, prompt, n_tokens, run_num, result_dir):
    """Run a single inference iteration

    Args:
        binary_path: Path to llama-completion binary
        lib_path: Path to shared libraries directory
        model_path: Path to model file
        prompt: Input prompt
        n_tokens: Number of tokens to generate
        run_num: Run number
        result_dir: Results directory

    Returns:
        dict: Parsed metrics or None
    """
    log(f"Starting run {run_num}...")

    # Build command
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = lib_path

    cmd = [
        binary_path,
        '-m', str(model_path),
        '-p', prompt,
        '-n', str(n_tokens),
        '-no-cnv'
    ]

    # Run inference
    try:
        result = run_cmd(cmd, f"Run {run_num} inference", timeout=600, capture_output=True)
        output = result.stdout + result.stderr

        # Save full output
        log_path = result_dir / f"run_{run_num}.log"
        with open(log_path, 'w') as f:
            f.write(output)

        # Parse metrics
        metrics = parse_llama_output(output)

        if metrics:
            log(f"Run {run_num} completed: load={metrics['load_time_ms']:.2f}ms, "
                f"eval={metrics['eval_time_ms']:.2f}ms")
            return metrics
        else:
            log(f"Run {run_num}: Could not parse metrics", "WARN")
            return None

    except Exception as e:
        log(f"Run {run_num} failed: {e}", "ERROR")
        return None


def run_experiment_set(exp_name, binary_path, lib_path, model_path, settings, result_dir):
    """Run a complete set of experiments (multiple iterations)

    Args:
        exp_name: Experiment name (e.g., "exp1_small_prefetch_on")
        binary_path: Path to llama-completion binary
        lib_path: Path to shared libraries
        model_path: Path to model file
        settings: Settings dict
        result_dir: Results directory

    Returns:
        dict: Statistics
    """
    log(f"\n{'='*70}")
    log(f"Starting {exp_name}")
    log(f"Binary: {binary_path}")
    log(f"Model: {model_path}")
    log(f"Iterations: {settings['experiment']['num_iterations']}")
    log(f"{'='*70}\n")

    # Create CSV file
    csv_path = result_dir / f"{exp_name}.csv"
    write_csv_header(csv_path)

    # Run iterations
    num_iterations = settings['experiment']['num_iterations']
    prompt = settings['experiment']['prompt']
    n_tokens = settings['experiment']['tokens_to_generate']
    cooldown = settings['experiment']['cooldown_seconds']

    for i in range(1, num_iterations + 1):
        # Optional: Drop page cache before each run
        if settings['cleanup'].get('drop_caches', False):
            drop_page_cache()

        # Run iteration
        metrics = run_single_iteration(
            binary_path, lib_path, model_path,
            prompt, n_tokens, i, result_dir
        )

        # Save to CSV
        if metrics:
            append_csv_row(csv_path, i, metrics)
        else:
            # Write empty row to maintain run numbering
            append_csv_row(csv_path, i, {})

        # Cooldown between runs
        if i < num_iterations:
            log(f"Cooling down for {cooldown}s...")
            import time
            time.sleep(cooldown)

    # Calculate statistics
    stats = calculate_statistics(csv_path)
    if stats:
        log(f"\n{exp_name} Statistics:")
        log(f"  Load time: {stats['load']['mean']:.2f} ± {stats['load']['stdev']:.2f} ms "
            f"(range: {stats['load']['min']:.2f}-{stats['load']['max']:.2f})")
        log(f"  Eval time: {stats['eval']['mean']:.2f} ± {stats['eval']['stdev']:.2f} ms "
            f"(range: {stats['eval']['min']:.2f}-{stats['eval']['max']:.2f})")
        return stats
    else:
        log(f"{exp_name}: No valid statistics", "ERROR")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run MAP_POPULATE experiments")
    parser.add_argument('--small-only', action='store_true', help='Run only small model experiments')
    parser.add_argument('--large-only', action='store_true', help='Run only large model experiments')
    args = parser.parse_args()

    # Load settings
    settings = load_settings()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).parent / settings['paths']['results_dir'] / f"experiment_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    log(f"Results directory: {result_dir}")

    # Save configuration
    config_path = result_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(settings, f, indent=2)

    # Resolve paths
    paths = settings['paths']
    models_dir = Path(paths['models_dir'])
    bin_prefetch_on = Path(paths['build_prefetch_on']) / "llama-completion"
    bin_prefetch_off = Path(paths['build_prefetch_off']) / "llama-completion"
    lib_prefetch_on = Path(paths['build_prefetch_on'])
    lib_prefetch_off = Path(paths['build_prefetch_off'])

    # Verify binaries exist
    if not bin_prefetch_on.exists():
        log(f"Binary not found: {bin_prefetch_on}", "ERROR")
        sys.exit(1)
    if not bin_prefetch_off.exists():
        log(f"Binary not found: {bin_prefetch_off}", "ERROR")
        sys.exit(1)

    results = {}

    # === SMALL MODEL EXPERIMENTS ===
    if not args.large_only:
        small_model = models_dir / settings['experiment']['small_model']

        if not small_model.exists():
            log(f"Small model not found: {small_model}", "ERROR")
        else:
            # Experiment 1: Small + MAP_POPULATE=ON
            results['exp1'] = run_experiment_set(
                "exp1_small_prefetch_on",
                bin_prefetch_on, lib_prefetch_on,
                small_model, settings, result_dir
            )

            # Experiment 2: Small + MAP_POPULATE=OFF
            results['exp2'] = run_experiment_set(
                "exp2_small_prefetch_off",
                bin_prefetch_off, lib_prefetch_off,
                small_model, settings, result_dir
            )

    # === LARGE MODEL EXPERIMENTS ===
    if not args.small_only:
        large_model = models_dir / "gpt-oss-120b" / settings['experiment']['large_model']

        if not large_model.exists():
            log(f"Large model not found: {large_model}", "WARN")
            log("Skipping large model experiments")
        else:
            # Experiment 3: Large + MAP_POPULATE=ON
            log("\n" + "="*70)
            log("WARNING: Experiment 3 may thrash or fail (large model + eager loading)")
            log("="*70)
            results['exp3'] = run_experiment_set(
                "exp3_large_prefetch_on",
                bin_prefetch_on, lib_prefetch_on,
                large_model, settings, result_dir
            )

            # Experiment 4: Large + MAP_POPULATE=OFF
            results['exp4'] = run_experiment_set(
                "exp4_large_prefetch_off",
                bin_prefetch_off, lib_prefetch_off,
                large_model, settings, result_dir
            )

    # === SUMMARY ===
    log(f"\n{'='*70}")
    log("EXPERIMENT SUMMARY")
    log(f"{'='*70}")
    log(f"Results saved to: {result_dir}")

    for exp_name, stats in results.items():
        if stats:
            log(f"\n{exp_name}:")
            log(f"  Load: {stats['load']['mean']:.2f} ± {stats['load']['stdev']:.2f} ms")
            log(f"  Eval: {stats['eval']['mean']:.2f} ± {stats['eval']['stdev']:.2f} ms")

    log("\nAll experiments complete!")


if __name__ == '__main__':
    main()
