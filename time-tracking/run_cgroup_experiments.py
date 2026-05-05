#!/usr/bin/env python3
"""
run_cgroup_experiments.py — cgroup v2 experiment runner for BSC thesis work.

Replaces the old mlock_tool-based approach. Each iteration runs llama-completion
inside a fresh cgroup v2 scope with a kernel-enforced `memory.max` budget. The
slice is torn down completely between iterations so there is no state leakage:
no surviving slab, no leftover page cache from the previous run, no auxiliary
processes competing for the budget.

The Python harness and the I/O monitor (if enabled) run **outside** the cgroup.
Only llama-completion is constrained.

Designed against:
  - systemd v240+ with cgroup v2 unified hierarchy
  - swap disabled system-wide (`sudo swapoff -a`)
  - root privileges (run via `sudo`)

Usage:
    sudo python3 run_cgroup_experiments.py --settings settings_cgroup.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import (
    log,
    drop_page_cache,
    parse_llama_output,
    write_csv_header,
    append_csv_row,
    calculate_statistics,
    IOMonitor,
    CPUMonitor,
)


# Constants

# NOTE: slice name MUST NOT contain a hyphen. systemd interprets hyphens in
# slice/scope names as nested-hierarchy separators, so `bsc-experiment.slice`
# would actually live at `/sys/fs/cgroup/bsc.slice/bsc-experiment.slice/`.
# Use a single-component name to keep the cgroup tree flat.
SLICE_NAME      = "bscexp"
SLICE_UNIT      = f"{SLICE_NAME}.slice"
CGROUP_FS_ROOT  = Path("/sys/fs/cgroup")
SETTLE_AFTER_STOP_S   = 2
SETTLE_AFTER_DROP_S   = 5
RUN_TIMEOUT_S         = 14400  # 4 hours, generous for tight budgets


# Cgroup management

def cleanup_cgroup_slice():
    """Forcefully tear down the experiment slice and any scopes inside it.

    Idempotent: safe to call when no slice exists. Run before AND after each
    iteration to guarantee a fresh kernel cgroup state for the next run.
    """
    subprocess.run(
        ["systemctl", "stop", SLICE_UNIT],
        capture_output=True,
    )
    # Reset failed state in case the previous run was killed via OOM or signal
    subprocess.run(
        ["systemctl", "reset-failed", SLICE_UNIT],
        capture_output=True,
    )
    time.sleep(SETTLE_AFTER_STOP_S)


# Bash wrapper that runs the inner command, then reads cgroup memory stats
# from /proc/self/cgroup BEFORE the scope is destroyed by systemd. The wrapper
# runs INSIDE the cgroup itself, so it sees the same cgroup the workload was
# accounted to, even if the slice is removed by the time Python regains control.
#
# Stats are written to stdout as `BSC_CGROUP_*` lines that the parser scans.
# The wrapper preserves the inner command's exit code so failures still bubble up.
_CGROUP_STATS_WRAPPER = (
    '"$@"; '
    'RC=$?; '
    'cgpath=$(awk -F: \'{print $3}\' /proc/self/cgroup | head -1); '
    'if [ -d "/sys/fs/cgroup$cgpath" ]; then '
        '[ -f "/sys/fs/cgroup$cgpath/memory.peak" ] && '
            'echo "BSC_CGROUP_PEAK=$(cat /sys/fs/cgroup$cgpath/memory.peak)"; '
        '[ -f "/sys/fs/cgroup$cgpath/memory.current" ] && '
            'echo "BSC_CGROUP_CURRENT=$(cat /sys/fs/cgroup$cgpath/memory.current)"; '
        '[ -f "/sys/fs/cgroup$cgpath/memory.swap.peak" ] && '
            'echo "BSC_CGROUP_SWAP_PEAK=$(cat /sys/fs/cgroup$cgpath/memory.swap.peak)"; '
        '[ -f "/sys/fs/cgroup$cgpath/memory.events" ] && '
            'echo "BSC_CGROUP_EVENTS=$(tr \'\\n\' \' \' < /sys/fs/cgroup$cgpath/memory.events)"; '
        'echo "BSC_CGROUP_PATH=$cgpath"; '
    'fi; '
    'exit $RC'
)


def build_cgroup_cmd(budget_gib, inner_cmd):
    """Wrap inner_cmd in a fresh systemd-run --scope inside the experiment
    slice with a kernel-enforced memory budget. The inner command is run via
    a bash wrapper that captures cgroup memory stats from /proc/self/cgroup
    after the workload exits but before the scope is destroyed."""
    return [
        "systemd-run", "--scope",
        f"--slice={SLICE_UNIT}",
        "-p", f"MemoryMax={budget_gib}G",
        "-p", "MemorySwapMax=0",
        "-p", "MemoryAccounting=yes",
        "bash", "-c", _CGROUP_STATS_WRAPPER, "--",
    ] + inner_cmd


def parse_cgroup_stats_from_output(output):
    """Extract BSC_CGROUP_* markers from the run's combined stdout+stderr.

    Returns an empty dict if no markers are present (e.g., the wrapper failed
    or the cgroup hierarchy looks different than expected).
    """
    stats = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('BSC_CGROUP_PEAK='):
            try:
                stats['memory.peak'] = int(line.split('=', 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('BSC_CGROUP_CURRENT='):
            try:
                stats['memory.current'] = int(line.split('=', 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('BSC_CGROUP_SWAP_PEAK='):
            try:
                stats['memory.swap.peak'] = int(line.split('=', 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('BSC_CGROUP_EVENTS='):
            # Format from `tr \n ' '`: "low 0 high 0 max 0 oom 0 oom_kill 0"
            try:
                tokens = line.split('=', 1)[1].strip().split()
                for k, v in zip(tokens[0::2], tokens[1::2]):
                    stats[f'events.{k}'] = int(v)
            except (ValueError, IndexError):
                pass
        elif line.startswith('BSC_CGROUP_PATH='):
            stats['path'] = line.split('=', 1)[1].strip()
    return stats


# Diagnostic snapshots

MEMINFO_FIELDS = [
    'MemFree', 'MemAvailable', 'Cached', 'Active', 'Inactive',
    'Slab', 'SReclaimable', 'SUnreclaim', 'PageTables',
    'KernelStack', 'Mapped', 'AnonPages', 'Mlocked',
    'Dirty', 'Writeback', 'SwapTotal', 'SwapFree',
]


def snapshot_meminfo(result_dir, exp_name, run_num, phase):
    """Append a row to <exp_name>_meminfo.csv with /proc/meminfo + buddyinfo."""
    csv_path = result_dir / f"{exp_name}_meminfo.csv"

    meminfo = {}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    meminfo[key] = int(parts[1])
    except Exception as e:
        log(f"snapshot_meminfo: failed to read /proc/meminfo: {e}", "WARN")
        return

    buddyinfo = ""
    try:
        with open('/proc/buddyinfo') as f:
            buddyinfo = f.read().strip().replace('\n', ' | ')
    except Exception:
        pass

    write_header = not csv_path.exists()
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('run,phase,' + ','.join(f'{k}_kB' for k in MEMINFO_FIELDS) + ',buddyinfo\n')
        values = [str(meminfo.get(k, 0)) for k in MEMINFO_FIELDS]
        f.write(f'{run_num},{phase},' + ','.join(values) + f',"{buddyinfo}"\n')


# Fixed column order for the cgroup stats CSV — keeps the schema stable across
# runs even if some keys are missing in a particular iteration.
CGROUP_CSV_COLS = [
    'memory.peak',
    'memory.current',
    'memory.swap.peak',
    'events.low',
    'events.high',
    'events.max',
    'events.oom',
    'events.oom_kill',
    'events.oom_group_kill',
]


def append_cgroup_stats(result_dir, exp_name, run_num, stats):
    """Append cgroup memory.peak / memory.current / oom counters to a CSV.

    Uses a fixed column schema (CGROUP_CSV_COLS). Missing values are written
    as empty fields so the CSV stays parseable.
    """
    if not stats:
        return
    csv_path = result_dir / f"{exp_name}_cgroup.csv"
    write_header = not csv_path.exists()
    with open(csv_path, 'a') as f:
        if write_header:
            f.write('run,' + ','.join(CGROUP_CSV_COLS) + '\n')
        values = [str(run_num)]
        for col in CGROUP_CSV_COLS:
            v = stats.get(col)
            values.append('' if v is None else str(v))
        f.write(','.join(values) + '\n')


# Single iteration

def run_iteration(exp, iter_num, result_dir, settings):
    """Execute one iteration of an experiment.

    Strict per-iteration sequence:
       1. Forceful cgroup teardown   (defensive — should be a no-op)
       2. Drop page cache             (`echo 3 > drop_caches`)
       3. Sleep SETTLE_AFTER_DROP_S   (let kernel quiesce)
       4. Snapshot meminfo "pre"      (diagnostic)
       5. Start IOMonitor             (outside cgroup, optional)
       6. Run llama-completion        (INSIDE the cgroup, synchronous)
       7. Stop IOMonitor              (write CSV)
       8. Read cgroup memory.peak     (before slice is destroyed)
       9. Snapshot meminfo "post"     (diagnostic)
      10. Forceful cgroup teardown    (real cleanup for next iteration)
      11. Save full log, parse, return metrics
    """
    name = exp['name']
    budget_gib = exp['cgroup_budget_gib']
    defaults = settings['defaults']

    log_path = result_dir / f"{name}_run{iter_num}.log"

    # 1. Defensive cleanup
    cleanup_cgroup_slice()

    # 2. Drop caches
    drop_page_cache()

    # 3. Settle
    time.sleep(SETTLE_AFTER_DROP_S)

    # 4. Pre-run meminfo
    snapshot_meminfo(result_dir, name, iter_num, "pre")

    # Build inner command. Per-experiment 'bin_dir' overrides the default.
    bin_dir   = Path(exp.get('bin_dir', settings['paths']['bin_current']))
    bin_path  = bin_dir / "llama-completion"
    model_rel = settings['models'][exp['model']]
    model_path = Path(settings['paths']['models_dir']) / model_rel

    n_tokens = exp.get('tokens_to_generate', defaults['tokens_to_generate'])
    prompt   = exp.get('prompt', defaults['prompt'])

    # Optional per-experiment env vars (e.g. BSC_PIN_NO_EMBD=1). Injected via
    # `env K=V` prefix so they survive the systemd-run scope wrapper.
    env_prefix = []
    for k, v in exp.get('env_extra', {}).items():
        env_prefix += [f'{k}={v}']
    if env_prefix:
        env_prefix = ['env'] + env_prefix

    inner_cmd = env_prefix + [
        '/usr/bin/time', '-v',
        str(bin_path),
        '-m', str(model_path),
        '-p', prompt,
        '-n', str(n_tokens),
        '-ngl', '0',
        '-no-cnv',
    ] + exp.get('args', [])

    cmd = build_cgroup_cmd(budget_gib, inner_cmd)

    log(f"Iter {iter_num}/{exp.get('num_iterations', defaults['num_iterations'])}: "
        f"budget={budget_gib}G, args={exp.get('args', [])}")

    # 5. Start optional monitors (outside the cgroup)
    io_mon = None
    io_settings = settings.get('io_monitor', {})
    if io_settings.get('enabled', False):
        io_mon = IOMonitor(device=io_settings['device'], interval_ms=25)
        io_mon.start()

    cpu_mon = None
    cpu_settings = settings.get('cpu_monitor', {})
    if cpu_settings.get('enabled', False):
        cpu_mon = CPUMonitor(interval_ms=cpu_settings.get('interval_ms', 25))
        cpu_mon.start()

    # 6. Run llama-completion in the cgroup
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(bin_dir)

    output = ""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
            env=env,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        log(f"Iter {iter_num} TIMED OUT after {RUN_TIMEOUT_S}s", "ERROR")
        if e.stdout:
            output += e.stdout if isinstance(e.stdout, str) else e.stdout.decode(errors="replace")
        if e.stderr:
            output += e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace")

    # 7. Stop monitors (in reverse order)
    if io_mon is not None:
        io_mon.stop()
        io_csv = result_dir / f"{name}_io_run{iter_num}.csv"
        n_samples = io_mon.to_csv(io_csv)
        log(f"  io_monitor: {n_samples} samples → {io_csv.name}")

    if cpu_mon is not None:
        cpu_mon.stop()
        cpu_csv = result_dir / f"{name}_cpu_run{iter_num}.csv"
        n_samples = cpu_mon.to_csv(cpu_csv)
        log(f"  cpu_monitor: {n_samples} samples → {cpu_csv.name}")

    # 8. Extract cgroup stats from the wrapper's output (BSC_CGROUP_* markers).
    # The bash wrapper inside the cgroup printed memory.peak/current/events
    # before exiting, so we don't need to race against systemd tearing down
    # the slice — the data is already in `output`.
    cgroup_stats = parse_cgroup_stats_from_output(output)
    if cgroup_stats:
        peak_mib = cgroup_stats.get('memory.peak', 0) / (1024 * 1024)
        log(f"  cgroup memory.peak: {peak_mib:.1f} MiB")
    append_cgroup_stats(result_dir, name, iter_num, cgroup_stats)

    # 9. Post-run meminfo, then defensive teardown
    snapshot_meminfo(result_dir, name, iter_num, "post")
    cleanup_cgroup_slice()

    # 10. Save log and parse metrics
    with open(log_path, 'w') as f:
        f.write(output)

    metrics = parse_llama_output(output)
    if metrics:
        eval_ms  = metrics.get('eval_time_ms')
        wall_ms  = metrics.get('phase_total_wall_ms')
        gen_f    = metrics.get('phase_generation_faults')
        log(f"  done: eval={eval_ms} ms, wall={wall_ms} ms, gen_faults={gen_f}")
    else:
        log(f"  done: failed to parse metrics", "WARN")

    return metrics


# Experiment loop

def run_experiment(exp, settings, result_dir):
    """Run all iterations of one experiment configuration."""
    name      = exp['name']
    n_iters   = exp.get('num_iterations', settings['defaults']['num_iterations'])
    cooldown  = settings['defaults']['cooldown_seconds']
    budget    = exp['cgroup_budget_gib']

    log("")
    log("=" * 70)
    log(f"Experiment: {name}")
    log(f"  budget:     {budget} GiB (cgroup memory.max)")
    log(f"  iterations: {n_iters}")
    log(f"  args:       {' '.join(exp.get('args', []))}")
    log("=" * 70)

    csv_path = result_dir / f"{name}.csv"
    write_csv_header(csv_path)

    for i in range(1, n_iters + 1):
        metrics = run_iteration(exp, i, result_dir, settings)
        if metrics:
            append_csv_row(csv_path, i, metrics)
        else:
            append_csv_row(csv_path, i, {})

        if i < n_iters:
            log(f"  cooldown {cooldown}s ...")
            time.sleep(cooldown)

    stats = calculate_statistics(csv_path)
    if stats:
        log("")
        log(f"{name} stats:")
        log(f"  load: {stats['load']['mean']:.2f} ± {stats['load']['stdev']:.2f} ms "
            f"(range {stats['load']['min']:.2f}–{stats['load']['max']:.2f})")
        log(f"  eval: {stats['eval']['mean']:.2f} ± {stats['eval']['stdev']:.2f} ms "
            f"(range {stats['eval']['min']:.2f}–{stats['eval']['max']:.2f})")
    else:
        log(f"{name}: no valid statistics", "ERROR")
    return stats


# Pre-flight checks

def check_root():
    if os.geteuid() != 0:
        log("This script must be run as root (use sudo).", "ERROR")
        sys.exit(1)


def check_swap_off():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('SwapTotal:'):
                    swap_kb = int(line.split()[1])
                    if swap_kb > 0:
                        log(f"swap is enabled ({swap_kb} kB). Run `sudo swapoff -a` first.", "ERROR")
                        sys.exit(1)
                    return
    except Exception as e:
        log(f"Could not check swap state: {e}", "WARN")


def check_systemd_run():
    try:
        result = subprocess.run(
            ["systemd-run", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            log("systemd-run not available.", "ERROR")
            sys.exit(1)
    except Exception as e:
        log(f"systemd-run probe failed: {e}", "ERROR")
        sys.exit(1)


def check_cgroup_v2():
    if not (CGROUP_FS_ROOT / "cgroup.controllers").exists():
        log(f"{CGROUP_FS_ROOT} does not look like cgroup v2 unified hierarchy.", "ERROR")
        sys.exit(1)


def check_paths(settings):
    bin_path = Path(settings['paths']['bin_current']) / "llama-completion"
    if not bin_path.exists():
        log(f"Binary not found: {bin_path}", "ERROR")
        sys.exit(1)
    for model_key, rel in settings['models'].items():
        model_path = Path(settings['paths']['models_dir']) / rel
        if not model_path.exists():
            log(f"Model not found: {model_path}", "ERROR")
            sys.exit(1)


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Run cgroup-based experiments for SSD-backed MoE inference.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="settings_cgroup.json",
        help="Path to settings JSON (default: settings_cgroup.json next to this script)",
    )
    args = parser.parse_args()

    # Pre-flight
    check_root()
    check_systemd_run()
    check_cgroup_v2()
    check_swap_off()

    # Load settings
    settings_path = Path(args.settings)
    if not settings_path.is_absolute():
        settings_path = Path(__file__).parent / settings_path
    if not settings_path.exists():
        log(f"Settings file not found: {settings_path}", "ERROR")
        sys.exit(1)
    with open(settings_path) as f:
        settings = json.load(f)
    log(f"Settings: {settings_path}")

    # Verify binary + model paths
    check_paths(settings)

    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(settings['paths'].get('results_dir', 'results'))
    if not results_root.is_absolute():
        results_root = Path(__file__).parent / results_root
    result_dir = results_root / f"cgroup_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    log(f"Results: {result_dir}")

    # Save the exact configuration used
    with open(result_dir / "config.json", 'w') as f:
        json.dump(settings, f, indent=2)

    # Pre-flight cleanup
    log("Pre-flight cleanup of any leftover cgroup state ...")
    cleanup_cgroup_slice()

    experiments = settings.get('experiments', [])
    if not experiments:
        log("No experiments defined in settings", "ERROR")
        sys.exit(1)
    log(f"Running {len(experiments)} experiment(s)")

    results = {}
    try:
        for exp in experiments:
            results[exp['name']] = run_experiment(exp, settings, result_dir)
    except KeyboardInterrupt:
        log("Interrupted by user", "WARN")
    finally:
        cleanup_cgroup_slice()

    # Summary
    log("")
    log("=" * 70)
    log("EXPERIMENT SUMMARY")
    log("=" * 70)
    log(f"Results: {result_dir}")
    for name, stats in results.items():
        if stats:
            log("")
            log(f"{name}:")
            log(f"  load: {stats['load']['mean']:.2f} ± {stats['load']['stdev']:.2f} ms")
            log(f"  eval: {stats['eval']['mean']:.2f} ± {stats['eval']['stdev']:.2f} ms")
    log("")
    log("Done.")


if __name__ == '__main__':
    main()
