#!/usr/bin/env python3
"""
Experiment Runner - Data-driven from settings.json

Experiments are defined in settings.json under "experiments".
Each experiment specifies a model, binary (prefetch_on/off), runtime args,
and optional memory pressure.

Usage:
    sudo python3 run_experiments.py
"""

import os
import sys
import json
import time
import signal
import subprocess
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
    calculate_statistics,
    IOMonitor
)


def load_settings():
    """Load experiment settings"""
    settings_path = Path(__file__).parent / "settings.json"
    with open(settings_path, 'r') as f:
        return json.load(f)


def start_memory_pressure(mlock_tool_path, gib):
    """Start mlock_tool to create memory pressure.

    Returns the subprocess.Popen object (or None if gib == 0).
    """
    if gib <= 0:
        return None

    log(f"Starting memory pressure: locking {gib} GiB...")
    proc = subprocess.Popen(
        ["sudo", str(mlock_tool_path), str(gib)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Wait a bit for mlock to complete
    time.sleep(5)
    if proc.poll() is not None:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        log(f"mlock_tool exited prematurely! stdout: {stdout}, stderr: {stderr}", "ERROR")
        return None

    log(f"Memory pressure active: {gib} GiB locked (PID {proc.pid})")
    return proc


def stop_memory_pressure(proc):
    """Stop mlock_tool process."""
    if proc is None:
        return
    log(f"Stopping memory pressure (PID {proc.pid})...")
    try:
        subprocess.run(["sudo", "kill", str(proc.pid)], capture_output=True, timeout=5)
        proc.wait(timeout=10)
    except Exception:
        try:
            subprocess.run(["sudo", "kill", "-9", str(proc.pid)], capture_output=True, timeout=5)
            proc.wait(timeout=5)
        except Exception as e:
            log(f"Could not kill mlock_tool: {e}", "ERROR")
    log("Memory pressure released")


def run_single_iteration(binary_path, lib_path, model_path, prompt, n_tokens, run_num, result_dir, exp_name="", extra_args=None, prompt_file=None, blktrace_config=None, io_monitor_device=None):
    """Run a single inference iteration, optionally with blktrace capture and I/O monitoring.

    Args:
        blktrace_config: dict with 'block_device' and 'output_base_dir' keys, or None to disable.
        io_monitor_device: device name (e.g. 'nvme1n1') to monitor via /proc/diskstats, or None.
    """
    log(f"Starting run {run_num}...")

    cmd = [
        '/usr/bin/time', '-v',
        str(binary_path),
        '-m', str(model_path),
    ]
    if prompt_file:
        cmd.extend(['-f', str(prompt_file)])
    else:
        cmd.extend(['-p', prompt])
    cmd.extend([
        '-n', str(n_tokens),
        '-ngl', '0',
        '-no-cnv'
    ])
    if extra_args:
        cmd.extend(extra_args)

    # Set LD_LIBRARY_PATH so the correct libllama.so is loaded
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(lib_path)

    bt = None
    io_mon = None
    try:
        # Start I/O monitor if configured
        if io_monitor_device:
            io_mon = IOMonitor(device=io_monitor_device, interval_ms=25)
            io_mon.start()

        # Start blktrace if configured
        if blktrace_config:
            from blktrace_utils import BlktraceCapture
            bt_dir = Path(blktrace_config['output_base_dir']) / f"run_{run_num}"
            bt = BlktraceCapture(
                block_device=blktrace_config['block_device'],
                output_dir=bt_dir
            )
            bt.__enter__()

        # Use Popen when blktrace is enabled (need PID), subprocess.run otherwise
        if bt:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            bt.set_pid(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=3600)
                output = stdout.decode() + stderr.decode()
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                output = stdout.decode() + stderr.decode()
                log(f"Run {run_num}: inference timed out", "ERROR")
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                env=env
            )
            output = result.stdout + result.stderr

        # Stop blktrace (but defer CSV conversion — it needs RAM, which may be locked)
        if bt:
            bt.__exit__(None, None, None)
            bt = None  # prevent double-exit in except

        # Stop I/O monitor and save CSV
        if io_mon:
            io_mon.stop()
            io_csv_path = result_dir / f"{exp_name}_io_{run_num}.csv"
            n_samples = io_mon.to_csv(io_csv_path)
            log(f"I/O monitor: {n_samples} samples saved to {io_csv_path.name}")
            io_mon = None

        log_path = result_dir / f"{exp_name}_run_{run_num}.log"
        with open(log_path, 'w') as f:
            f.write(output)

        metrics = parse_llama_output(output)

        if metrics:
            wall = metrics.get('phase_total_wall_ms')
            gen = metrics.get('phase_generation_ms')
            faults = metrics.get('phase_total_faults', 'N/A')
            mlock = metrics.get('mlock_mib')
            mlock_str = f", mlock={mlock:.1f}MiB" if mlock else ""
            if wall is not None and gen is not None:
                log(f"Run {run_num} completed: wall={wall:.1f}ms, gen={gen:.1f}ms, faults={faults}{mlock_str}")
            else:
                log(f"Run {run_num} completed: eval={metrics.get('eval_time_ms', 'N/A')}ms, faults={faults}{mlock_str}")
            return metrics
        else:
            log(f"Run {run_num}: Could not parse metrics", "WARN")
            return None

    except Exception as e:
        log(f"Run {run_num} failed: {e}", "ERROR")
        if bt:
            try:
                bt.__exit__(None, None, None)
            except Exception:
                pass
        return None


def run_experiment_set(exp_name, binary_path, lib_path, model_path, defaults, cleanup, result_dir, extra_args=None, prompt_file=None, blktrace_config=None, io_monitor_device=None):
    """Run a complete set of experiments (multiple iterations)"""
    log(f"\n{'='*70}")
    log(f"Starting {exp_name}")
    log(f"Binary: {binary_path}")
    log(f"Lib: {lib_path}")
    log(f"Model: {model_path}")
    log(f"Args: {extra_args or []}")
    log(f"Prompt: {prompt_file or defaults['prompt']}")
    log(f"Iterations: {defaults['num_iterations']}")
    if blktrace_config:
        log(f"blktrace: {blktrace_config['block_device']} → {blktrace_config['output_base_dir']}")
    log(f"{'='*70}\n")

    csv_path = result_dir / f"{exp_name}.csv"
    write_csv_header(csv_path)

    num_iterations = defaults['num_iterations']
    prompt = defaults['prompt']
    n_tokens = defaults['tokens_to_generate']
    cooldown = defaults['cooldown_seconds']

    for i in range(1, num_iterations + 1):
        if cleanup.get('drop_caches', False):
            drop_page_cache()

        metrics = run_single_iteration(
            binary_path, lib_path, model_path,
            prompt, n_tokens, i, result_dir, exp_name=exp_name,
            extra_args=extra_args,
            prompt_file=prompt_file,
            blktrace_config=blktrace_config,
            io_monitor_device=io_monitor_device
        )

        if metrics:
            append_csv_row(csv_path, i, metrics)
        else:
            append_csv_row(csv_path, i, {})

        if i < num_iterations:
            log(f"Cooling down for {cooldown}s...")
            time.sleep(cooldown)

    stats = calculate_statistics(csv_path)
    if stats:
        log(f"\n{exp_name} Statistics:")
        log(f"  Load time: {stats['load']['mean']:.2f} +/- {stats['load']['stdev']:.2f} ms "
            f"(range: {stats['load']['min']:.2f}-{stats['load']['max']:.2f})")
        log(f"  Eval time: {stats['eval']['mean']:.2f} +/- {stats['eval']['stdev']:.2f} ms "
            f"(range: {stats['eval']['min']:.2f}-{stats['eval']['max']:.2f})")
        return stats
    else:
        log(f"{exp_name}: No valid statistics", "ERROR")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run experiments defined in settings.json")
    parser.parse_args()

    settings = load_settings()

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(__file__).parent / settings['paths']['results_dir'] / f"experiment_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    log(f"Results directory: {result_dir}")

    # Save configuration snapshot
    with open(result_dir / "config.json", 'w') as f:
        json.dump(settings, f, indent=2)

    # Resolve paths
    paths = settings['paths']
    models_dir = Path(paths['models_dir'])
    bin_dirs = {
        'prefetch_on': Path(paths['bin_prefetch_on']),
        'prefetch_off': Path(paths['bin_prefetch_off']),
    }
    if 'bin_current' in paths:
        bin_dirs['current'] = Path(paths['bin_current'])
    mlock_tool_path = Path(paths['mlock_tool'])

    # Verify binaries exist
    for name, bin_dir in bin_dirs.items():
        binary = bin_dir / "llama-completion"
        if not binary.exists():
            log(f"Binary not found: {binary}", "ERROR")
            sys.exit(1)
        log(f"Binary [{name}]: {binary}")

    # Resolve model paths
    model_paths = {}
    for key, rel_path in settings['models'].items():
        full_path = models_dir / rel_path
        model_paths[key] = full_path

    # Resolve blktrace config (optional)
    blktrace_config = None
    bt_settings = settings.get('blktrace')
    if bt_settings and bt_settings.get('enabled', False):
        blktrace_config = {
            'block_device': bt_settings['block_device'],
            'output_base_dir': bt_settings['output_dir'],
        }
        log(f"blktrace enabled: tracing {blktrace_config['block_device']}, output to {blktrace_config['output_base_dir']}")

    # Resolve I/O monitor config (optional)
    io_monitor_device = None
    io_settings = settings.get('io_monitor')
    if io_settings and io_settings.get('enabled', False):
        io_monitor_device = io_settings['device']
        log(f"I/O monitor enabled: polling /proc/diskstats for {io_monitor_device} every 25ms")

    # Run experiments from settings
    experiments = settings.get('experiments', [])
    if not experiments:
        log("No experiments defined in settings.json", "ERROR")
        sys.exit(1)

    log(f"Running {len(experiments)} experiment(s)...")

    # Group experiments: no-pressure first, then pressure
    no_pressure = [e for e in experiments if e.get('memory_pressure_gib', 0) == 0]
    pressure = [e for e in experiments if e.get('memory_pressure_gib', 0) > 0]

    results = {}
    mlock_proc = None

    try:
        # Phase 1: No-pressure experiments
        if no_pressure:
            log(f"\n{'#'*70}")
            log(f"PHASE 1: {len(no_pressure)} experiments WITHOUT memory pressure")
            log(f"{'#'*70}\n")

        for exp in no_pressure:
            name = exp['name']
            model_key = exp['model']
            binary_key = exp.get('binary', 'prefetch_off')
            extra_args = exp.get('args', [])
            prompt_file = exp.get('prompt_file')
            if prompt_file:
                prompt_file = Path(__file__).parent / prompt_file

            bin_dir = bin_dirs[binary_key]
            binary_path = bin_dir / "llama-completion"
            model_path = model_paths.get(model_key)

            if model_path is None or not model_path.exists():
                log(f"Model not found for '{model_key}': {model_path}", "ERROR")
                sys.exit(1)

            exp_defaults = dict(settings['defaults'])
            if 'tokens_to_generate' in exp:
                exp_defaults['tokens_to_generate'] = exp['tokens_to_generate']
            if 'num_iterations' in exp:
                exp_defaults['num_iterations'] = exp['num_iterations']
            results[name] = run_experiment_set(
                name, binary_path, bin_dir, model_path,
                exp_defaults, settings['cleanup'],
                result_dir, extra_args=extra_args,
                prompt_file=prompt_file,
                blktrace_config=blktrace_config,
                io_monitor_device=io_monitor_device
            )

        # Phase 2: Pressure experiments (grouped by pressure level)
        if pressure:
            # Group by pressure level
            pressure_levels = {}
            for exp in pressure:
                gib = exp['memory_pressure_gib']
                pressure_levels.setdefault(gib, []).append(exp)

            for gib, exps in sorted(pressure_levels.items()):
                log(f"\n{'#'*70}")
                log(f"PHASE 2: {len(exps)} experiments WITH {gib} GiB memory pressure")
                log(f"{'#'*70}\n")

                mlock_proc = start_memory_pressure(mlock_tool_path, gib)
                if mlock_proc is None:
                    log(f"Failed to start memory pressure at {gib} GiB, skipping these experiments", "ERROR")
                    continue

                for exp in exps:
                    name = exp['name']
                    model_key = exp['model']
                    binary_key = exp.get('binary', 'prefetch_off')
                    extra_args = exp.get('args', [])
                    prompt_file = exp.get('prompt_file')
                    if prompt_file:
                        prompt_file = Path(__file__).parent / prompt_file

                    bin_dir = bin_dirs[binary_key]
                    binary_path = bin_dir / "llama-completion"
                    model_path = model_paths.get(model_key)

                    if model_path is None or not model_path.exists():
                        log(f"Model not found for '{model_key}': {model_path}", "ERROR")
                        continue

                    exp_defaults = dict(settings['defaults'])
                    if 'tokens_to_generate' in exp:
                        exp_defaults['tokens_to_generate'] = exp['tokens_to_generate']
                    if 'num_iterations' in exp:
                        exp_defaults['num_iterations'] = exp['num_iterations']
                    results[name] = run_experiment_set(
                        name, binary_path, bin_dir, model_path,
                        exp_defaults, settings['cleanup'],
                        result_dir, extra_args=extra_args,
                        prompt_file=prompt_file,
                        blktrace_config=blktrace_config,
                        io_monitor_device=io_monitor_device
                    )

                stop_memory_pressure(mlock_proc)
                mlock_proc = None

    except KeyboardInterrupt:
        log("\nInterrupted by user!", "WARN")
    finally:
        if mlock_proc is not None:
            stop_memory_pressure(mlock_proc)

    # Post-process blktrace data (deferred — needs RAM that was locked during experiments)
    if blktrace_config:
        log(f"\n{'#'*70}")
        log("POST-PROCESSING: Converting blktrace data to CSV")
        log(f"{'#'*70}\n")
        from blktrace_utils import BlktraceCapture
        bt_base = Path(blktrace_config['output_base_dir'])
        for run_dir in sorted(bt_base.glob("run_*")):
            trace_files = list(run_dir.glob("trace.blktrace.*"))
            if trace_files and not (run_dir / "trace.csv").exists():
                log(f"Converting {run_dir.name}...")
                bt = BlktraceCapture(blktrace_config['block_device'], run_dir)
                try:
                    bt.to_csv()
                except Exception as e:
                    log(f"Failed to convert {run_dir.name}: {e}", "ERROR")
            elif (run_dir / "trace.csv").exists():
                log(f"{run_dir.name}: trace.csv already exists, skipping")

    # Summary
    log(f"\n{'='*70}")
    log("EXPERIMENT SUMMARY")
    log(f"{'='*70}")
    log(f"Results saved to: {result_dir}")

    for exp_name, stats in results.items():
        if stats:
            log(f"\n{exp_name}:")
            log(f"  Load: {stats['load']['mean']:.2f} +/- {stats['load']['stdev']:.2f} ms")
            log(f"  Eval: {stats['eval']['mean']:.2f} +/- {stats['eval']['stdev']:.2f} ms")

    log("\nAll experiments complete!")


if __name__ == '__main__':
    main()
