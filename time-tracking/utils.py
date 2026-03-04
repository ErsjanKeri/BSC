#!/usr/bin/env python3
"""
Utility functions for MAP_POPULATE time-tracking experiments
"""

import subprocess
import sys
import time
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
        'load_time_ms': None,
        'prompt_eval_time_ms': None,
        'eval_time_ms': None,
        'eval_tokens': None,
        'total_time_ms': None
    }

    for line in output.split('\n'):
        if 'load time' in line:
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

    # Check if we got the critical metrics
    if metrics['load_time_ms'] is not None and metrics['eval_time_ms'] is not None:
        return metrics
    else:
        return None


def write_csv_header(csv_path):
    """Write CSV header

    Args:
        csv_path: Path to CSV file
    """
    with open(csv_path, 'w') as f:
        f.write("run,load_time_ms,prompt_eval_time_ms,eval_time_ms,eval_tokens,total_time_ms,exp_run_time_ms\n")


def append_csv_row(csv_path, run_num, metrics):
    """Append row to CSV

    Args:
        csv_path: Path to CSV file
        run_num: Run number
        metrics: Metrics dictionary
    """
    # Calculate exp_run_time = load_time + total_time (end-to-end performance)
    load_time = metrics.get('load_time_ms', None)
    total_time = metrics.get('total_time_ms', None)

    if load_time is not None and total_time is not None:
        exp_run_time = load_time + total_time
    else:
        exp_run_time = ''

    with open(csv_path, 'a') as f:
        f.write(f"{run_num},"
                f"{metrics.get('load_time_ms', '')},"
                f"{metrics.get('prompt_eval_time_ms', '')},"
                f"{metrics.get('eval_time_ms', '')},"
                f"{metrics.get('eval_tokens', '')},"
                f"{metrics.get('total_time_ms', '')},"
                f"{exp_run_time}\n")


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
