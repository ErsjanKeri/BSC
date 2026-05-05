#!/usr/bin/env python3
"""
Utility functions for MAP_POPULATE time-tracking experiments
"""

import json
import subprocess
import sys
import time
import threading
from pathlib import Path
from datetime import datetime


def log(message, level="INFO"):
    """Print timestamped log message

    Args:
        message: Message to print
        level: Log level (INFO, WARN, ERROR)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def run_cmd(cmd, description="", timeout=None, capture_output=True):
    """Run shell command and return output

    Args:
        cmd: Command string or list
        description: Description of command for logging
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        subprocess.CompletedProcess: Result
    """
    if description:
        log(f"Running: {description}")

    if isinstance(cmd, str):
        shell = True
    else:
        shell = False

    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout}s", "ERROR")
        raise
    except Exception as e:
        log(f"Command failed: {e}", "ERROR")
        raise


def drop_page_cache():
    """Drop page cache, dentries, and inodes (requires sudo)

    Uses echo 3 for aggressive cleanup:
    - 1 = page cache only
    - 2 = dentries and inodes
    - 3 = page cache + dentries + inodes (most thorough)

    Note: Will fail silently if no sudo access
    """
    try:
        log("Dropping caches (page cache + dentries + inodes)...")
        result = subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            log("Caches dropped successfully")
            return True
        else:
            log("Could not drop caches (no sudo access)", "WARN")
            return False
    except Exception as e:
        log(f"Could not drop caches: {e}", "WARN")
        return False


def parse_llama_output(output):
    """Parse llama-completion output for timing metrics

    Args:
        output: stdout from llama-completion

    Returns:
        dict: Parsed metrics or None if parsing failed
    """
    metrics = {
        # llama.cpp's own metrics (post-reset, kept for reference)
        'load_time_ms': None,
        'prompt_eval_time_ms': None,
        'eval_time_ms': None,
        'eval_tokens': None,
        'total_time_ms': None,
        'major_page_faults': None,
        'mlock_mib': None,
        # BSC phase timing (independent, never affected by reset)
        'phase_metadata_ms': None,
        'phase_mmap_ms': None,
        'phase_tensors_ms': None,
        'phase_pinning_ms': None,
        'phase_context_ms': None,
        'phase_warmup_ms': None,
        'phase_prompt_eval_ms': None,
        'phase_generation_ms': None,
        'phase_total_wall_ms': None,
        'phase_metadata_faults': None,
        'phase_mmap_faults': None,
        'phase_tensors_faults': None,
        'phase_pinning_faults': None,
        'phase_context_faults': None,
        'phase_warmup_faults': None,
        'phase_prompt_eval_faults': None,
        'phase_generation_faults': None,
        'phase_total_faults': None,
        'phase_t0_abs_us': None,
    }

    for line in output.split('\n'):
        if '[bsc_phases_json]' in line:
            # Parse the JSON block
            try:
                json_str = line.split('[bsc_phases_json]')[1].strip()
                phases = json.loads(json_str)
                metrics['phase_t0_abs_us'] = phases.get('t0_abs_us')
                metrics['phase_metadata_ms'] = phases.get('metadata_ms')
                metrics['phase_mmap_ms'] = phases.get('mmap_ms')
                metrics['phase_tensors_ms'] = phases.get('tensors_ms')
                metrics['phase_pinning_ms'] = phases.get('pinning_ms')
                metrics['phase_context_ms'] = phases.get('context_ms')
                metrics['phase_warmup_ms'] = phases.get('warmup_ms')
                metrics['phase_prompt_eval_ms'] = phases.get('prompt_eval_ms')
                metrics['phase_generation_ms'] = phases.get('generation_ms')
                metrics['phase_total_wall_ms'] = phases.get('total_wall_ms')
                metrics['phase_metadata_faults'] = phases.get('metadata_faults')
                metrics['phase_mmap_faults'] = phases.get('mmap_faults')
                metrics['phase_tensors_faults'] = phases.get('tensors_faults')
                metrics['phase_pinning_faults'] = phases.get('pinning_faults')
                metrics['phase_context_faults'] = phases.get('context_faults')
                metrics['phase_warmup_faults'] = phases.get('warmup_faults')
                metrics['phase_prompt_eval_faults'] = phases.get('prompt_eval_faults')
                metrics['phase_generation_faults'] = phases.get('generation_faults')
                metrics['phase_total_faults'] = phases.get('total_faults')
            except (json.JSONDecodeError, IndexError, ValueError):
                pass

        elif 'load time' in line and '[timing]' not in line:
            # Extract: "load time =    2073.36 ms"
            parts = line.split('=')
            if len(parts) >= 2:
                try:
                    metrics['load_time_ms'] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        elif 'prompt eval time' in line:
            # Extract: "prompt eval time =     196.08 ms /    12 tokens"
            parts = line.split('=')
            if len(parts) >= 2:
                try:
                    metrics['prompt_eval_time_ms'] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        elif 'eval time' in line and 'runs' in line:
            # Extract: "eval time =    6600.36 ms /    99 runs"
            parts = line.split('=')
            if len(parts) >= 2:
                try:
                    time_and_runs = parts[1].strip().split('/')
                    metrics['eval_time_ms'] = float(time_and_runs[0].strip().split()[0])
                    if len(time_and_runs) >= 2:
                        metrics['eval_tokens'] = int(time_and_runs[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        elif 'total time' in line:
            # Extract: "total time =    6818.39 ms /   111 tokens"
            parts = line.split('=')
            if len(parts) >= 2:
                try:
                    metrics['total_time_ms'] = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        elif 'Major (requiring I/O) page faults' in line:
            # Extract from /usr/bin/time -v output
            try:
                metrics['major_page_faults'] = int(line.strip().split(':')[-1].strip())
            except (ValueError, IndexError):
                pass

        elif 'pin_compute_weights:' in line and 'locked' in line:
            # Extract: "pin_compute_weights: attempted 470 tensors, locked 2979.7 MiB, failed 0 tensors, took 1128.45 ms"
            try:
                metrics['mlock_mib'] = float(line.split('locked')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    # Check if we got the critical metrics (either old or new system)
    has_old = metrics['load_time_ms'] is not None and metrics['eval_time_ms'] is not None
    has_new = metrics['phase_total_wall_ms'] is not None
    if has_old or has_new:
        return metrics
    else:
        return None


# CSV column definitions — single source of truth
CSV_COLUMNS = [
    'run',
    # llama.cpp metrics (post-reset, for reference)
    'load_time_ms', 'prompt_eval_time_ms', 'eval_time_ms', 'eval_tokens', 'total_time_ms',
    'major_page_faults', 'mlock_mib',
    # BSC phase durations (independent, never affected by reset)
    'phase_metadata_ms', 'phase_mmap_ms', 'phase_tensors_ms', 'phase_pinning_ms',
    'phase_context_ms', 'phase_warmup_ms', 'phase_prompt_eval_ms', 'phase_generation_ms',
    'phase_total_wall_ms',
    # BSC phase page faults
    'phase_metadata_faults', 'phase_mmap_faults', 'phase_tensors_faults', 'phase_pinning_faults',
    'phase_context_faults', 'phase_warmup_faults', 'phase_prompt_eval_faults', 'phase_generation_faults',
    'phase_total_faults',
    # Absolute timestamp for sanity checking
    'phase_t0_abs_us',
]


def write_csv_header(csv_path):
    """Write CSV header"""
    with open(csv_path, 'w') as f:
        f.write(','.join(CSV_COLUMNS) + '\n')


def append_csv_row(csv_path, run_num, metrics):
    """Append row to CSV"""
    def v(key):
        """Get metric value, converting None to empty string for CSV."""
        val = metrics.get(key)
        return '' if val is None else val

    values = [str(run_num)]
    for col in CSV_COLUMNS[1:]:  # skip 'run', already added
        values.append(str(v(col)))

    with open(csv_path, 'a') as f:
        f.write(','.join(values) + '\n')


def calculate_statistics(csv_path):
    """Calculate statistics from CSV file

    Args:
        csv_path: Path to CSV file

    Returns:
        dict: Statistics
    """
    import csv
    import statistics

    load_times = []
    eval_times = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['load_time_ms']:
                load_times.append(float(row['load_time_ms']))
            if row['eval_time_ms']:
                eval_times.append(float(row['eval_time_ms']))

    if not load_times or not eval_times:
        return None

    return {
        'load': {
            'mean': statistics.mean(load_times),
            'median': statistics.median(load_times),
            'stdev': statistics.stdev(load_times) if len(load_times) > 1 else 0,
            'min': min(load_times),
            'max': max(load_times),
            'count': len(load_times)
        },
        'eval': {
            'mean': statistics.mean(eval_times),
            'median': statistics.median(eval_times),
            'stdev': statistics.stdev(eval_times) if len(eval_times) > 1 else 0,
            'min': min(eval_times),
            'max': max(eval_times),
            'count': len(eval_times)
        }
    }


class IOMonitor:
    """Polls /proc/diskstats every 25ms on a background thread.

    Records read/write byte deltas for a specific block device.
    Zero memory overhead (~50 bytes per sample, stored in a list).
    Zero page cache impact (reads a tiny proc file).

    Usage:
        monitor = IOMonitor(device="nvme1n1", interval_ms=25)
        monitor.start()
        # ... run inference ...
        monitor.stop()
        monitor.to_csv("/path/to/output.csv")
    """

    def __init__(self, device="nvme1n1", interval_ms=25):
        self.device = device
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None

    def _read_stats(self):
        """Read sectors_read and sectors_written from /proc/diskstats."""
        with open('/proc/diskstats') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 14 and parts[2] == self.device:
                    return {
                        'sectors_read': int(parts[5]),
                        'sectors_written': int(parts[9]),
                    }
        return None

    def _poll(self):
        """Background polling loop."""
        prev = self._read_stats()
        if prev is None:
            return
        t0 = time.monotonic()
        prev_t = t0

        while self._running:
            time.sleep(self.interval)
            curr = self._read_stats()
            if curr is None:
                continue
            now = time.monotonic()

            # Actual elapsed time (not assumed interval — sleep is imprecise)
            dt = now - prev_t

            # Delta in bytes (sectors are 512 bytes)
            read_bytes = (curr['sectors_read'] - prev['sectors_read']) * 512
            write_bytes = (curr['sectors_written'] - prev['sectors_written']) * 512

            self.samples.append({
                'timestamp_s': now - t0,
                'dt_s': dt,
                'read_bytes': read_bytes,
                'write_bytes': write_bytes,
            })

            prev = curr
            prev_t = now

    def start(self):
        """Start background monitoring thread."""
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring and wait for thread to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def to_csv(self, path):
        """Write samples to CSV file."""
        path = Path(path)
        with open(path, 'w') as f:
            f.write("timestamp_s,dt_ms,read_bytes,write_bytes,read_mib_s,write_mib_s\n")
            for s in self.samples:
                dt = s['dt_s']
                if dt > 0:
                    read_mib_s = s['read_bytes'] / (1024 * 1024) / dt
                    write_mib_s = s['write_bytes'] / (1024 * 1024) / dt
                else:
                    read_mib_s = 0
                    write_mib_s = 0
                f.write(f"{s['timestamp_s']:.4f},{dt*1000:.1f},{s['read_bytes']},"
                        f"{s['write_bytes']},{read_mib_s:.1f},{write_mib_s:.1f}\n")
        return len(self.samples)


class CPUMonitor:
    """Polls /proc/stat every 25ms on a background thread.

    Records aggregate CPU-time deltas: user, nice, system, idle, iowait,
    irq, softirq. Computes percentages relative to total ticks per sample
    interval. Aggregate (all CPUs summed) — fine because the test machine
    is dedicated during cgroup runs.

    Used to plot kernel-vs-userspace CPU breakdown over time, supporting
    the Ch. 6 §6.2 claim that userspace I/O reduces kernel CPU.

    Zero memory overhead; reads /proc/stat (~few KB per sample).
    """

    def __init__(self, interval_ms=25):
        self.interval = interval_ms / 1000.0
        self.samples = []
        self._running = False
        self._thread = None

    def _read_stats(self):
        """Read aggregate CPU stats (first 'cpu' line of /proc/stat)."""
        with open('/proc/stat') as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != 'cpu':
            return None
        # Pad to ensure indices exist on older kernels
        vals = [int(x) for x in parts[1:]] + [0] * 10
        return {
            'user':    vals[0],
            'nice':    vals[1],
            'system':  vals[2],
            'idle':    vals[3],
            'iowait':  vals[4],
            'irq':     vals[5],
            'softirq': vals[6],
        }

    def _poll(self):
        prev = self._read_stats()
        if prev is None:
            return
        t0 = time.monotonic()
        prev_t = t0

        while self._running:
            time.sleep(self.interval)
            curr = self._read_stats()
            if curr is None:
                continue
            now = time.monotonic()
            dt = now - prev_t

            deltas = {k: curr[k] - prev[k] for k in curr}
            total = sum(deltas.values())
            if total > 0:
                pcts = {k: 100.0 * deltas[k] / total for k in deltas}
            else:
                pcts = {k: 0.0 for k in deltas}

            self.samples.append({
                'timestamp_s': now - t0,
                'dt_s': dt,
                **{f'{k}_pct': pcts[k] for k in pcts},
            })

            prev = curr
            prev_t = now

    def start(self):
        self.samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def to_csv(self, path):
        path = Path(path)
        with open(path, 'w') as f:
            f.write("timestamp_s,dt_ms,user_pct,nice_pct,system_pct,"
                    "idle_pct,iowait_pct,irq_pct,softirq_pct\n")
            for s in self.samples:
                f.write(
                    f"{s['timestamp_s']:.4f},{s['dt_s']*1000:.1f},"
                    f"{s['user_pct']:.2f},{s['nice_pct']:.2f},{s['system_pct']:.2f},"
                    f"{s['idle_pct']:.2f},{s['iowait_pct']:.2f},"
                    f"{s['irq_pct']:.2f},{s['softirq_pct']:.2f}\n"
                )
        return len(self.samples)
