#!/usr/bin/env python3
"""
blktrace capture and analysis utilities for time-tracking experiments.

Adapted from disk-benchmarking/utils/analysis_tools.py with corrections:
- Action filtering: 'D' for dispatch (request count/sizes), 'C' for completion (throughput)
- PID filtering: isolate llama-completion process I/O
- Sector filtering: optional, applied in post-processing per-extent

Output goes to a separate disk (nvme0n1 mounted at /mnt/experiment_ssd)
to avoid interfering with model reads on nvme1n1 (system disk).

Usage:
    # As context manager around inference:
    with BlktraceCapture(block_device, output_dir) as bt:
        proc = subprocess.Popen([...])  # start inference
        bt.set_pid(proc.pid)            # record PID for filtering
        proc.wait()

    # Post-process:
    bt.to_csv()  # converts blkparse output to CSV
"""

import os
import re
import time
import subprocess
from pathlib import Path
from datetime import datetime


def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", flush=True)


class BlktraceCapture:
    """Context manager for blktrace capture.

    Starts blktrace on enter, stops on exit.
    Output files go to output_dir on a separate disk.
    """

    def __init__(self, block_device, output_dir, trace_name="trace"):
        """
        Args:
            block_device: e.g., "/dev/nvme1n1" (the disk to trace)
            output_dir: directory for trace files (should be on a DIFFERENT disk)
            trace_name: prefix for trace files (default: "trace")
        """
        self.block_device = block_device
        self.output_dir = Path(output_dir)
        self.trace_name = trace_name
        self.proc = None
        self.pid = None  # PID of the inference process (set later)

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Start blktrace
        self.proc = subprocess.Popen(
            ["blktrace", "-d", self.block_device, "-o", self.trace_name],
            cwd=str(self.output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(1)  # give blktrace time to start

        if self.proc.poll() is not None:
            stderr = self.proc.stderr.read().decode() if self.proc.stderr else ""
            raise RuntimeError(f"blktrace failed to start: {stderr}")

        log(f"blktrace started on {self.block_device} (PID {self.proc.pid}), output: {self.output_dir}")
        return self

    def set_pid(self, pid):
        """Record the inference process PID for later filtering.

        Note: For mmap-based inference, page faults come from worker threads
        with different TIDs. The recorded PID is the parent (often /usr/bin/time).
        For filtering, use process name ('llama-completio' or 'time') in post-processing,
        or filter by PID range (main_pid-1 to main_pid+20 covers all threads).
        """
        self.pid = pid

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc and self.proc.poll() is None:
            # Send SIGINT to stop blktrace gracefully
            self.proc.send_signal(2)  # SIGINT
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

        log("blktrace stopped")

        # Save PID for post-processing
        if self.pid is not None:
            pid_file = self.output_dir / "inference_pid.txt"
            with open(pid_file, 'w') as f:
                f.write(str(self.pid))

        return False  # don't suppress exceptions

    def to_csv(self, output_csv=None):
        """Convert blktrace binary files to CSV.

        Runs blkparse, then parses the text output into CSV format.
        Keeps ALL actions (D, C, Q, etc.) — filtering happens in analysis.

        Args:
            output_csv: output CSV path (default: output_dir/trace.csv)

        Returns:
            Path: output CSV file
        """
        if output_csv is None:
            output_csv = self.output_dir / "trace.csv"

        blkparse_output = self.output_dir / "blkparse_raw.txt"

        log("Running blkparse...")
        subprocess.run(
            ["blkparse", "-i", str(self.output_dir / self.trace_name),
             "-o", str(blkparse_output)],
            capture_output=True,
            check=True
        )

        log("Parsing blkparse output to CSV...")

        header = "device_major,device_minor,cpu,seq,timestamp,pid,action,rwbs,sector,size_sectors,size_bytes,process"
        lines = [header]

        with open(blkparse_output, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                if parts[0] in ("CPU", "Total"):
                    continue

                try:
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
                    size_sectors = parts[9]  # parts[8] is "+"
                    size_bytes = int(size_sectors) * 512

                    process = ""
                    for i in range(10, len(parts)):
                        if parts[i].startswith('[') and parts[i].endswith(']'):
                            process = parts[i][1:-1]
                            break

                    lines.append(f"{device_major},{device_minor},{cpu},{seq},{timestamp},{pid},{action},{rwbs},{sector},{size_sectors},{size_bytes},{process}")

                except (ValueError, IndexError):
                    continue

        with open(output_csv, 'w') as f:
            f.write('\n'.join(lines))

        n_rows = len(lines) - 1
        log(f"CSV saved: {output_csv} ({n_rows:,} rows)")

        return output_csv


def get_partition_offset(block_device, partition):
    """Get partition start sector offset.

    blktrace reports absolute sectors on the block device.
    filefrag reports sectors relative to the partition.
    We need the partition offset to convert between them.

    Args:
        block_device: e.g., "nvme1n1"
        partition: e.g., "nvme1n1p2"

    Returns:
        int: partition start sector
    """
    try:
        with open(f"/sys/block/{block_device}/{partition}/start") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0


def get_model_extents(model_path):
    """Get physical extent information for a model file.

    Returns list of (start_sector, end_sector) tuples in absolute device sectors.
    Useful for per-extent sector filtering in post-processing.

    Args:
        model_path: path to .gguf file

    Returns:
        list of (start_sector, end_sector) tuples
    """
    result = subprocess.run(
        ["filefrag", "-v", str(model_path)],
        capture_output=True, text=True, check=True
    )

    extent_pattern = r'^\s*\d+:\s+\d+\.\.\s*\d+:\s+(\d+)\.\.\s*(\d+):\s+\d+:'
    extents = []

    for line in result.stdout.split('\n'):
        match = re.search(extent_pattern, line)
        if match:
            start_block = int(match.group(1))
            end_block = int(match.group(2))
            # Convert filesystem blocks (4096 bytes) to sectors (512 bytes)
            start_sector = start_block * 8
            end_sector = (end_block + 1) * 8  # +1 because filefrag end is inclusive
            extents.append((start_sector, end_sector))

    # Sort by start sector
    extents.sort(key=lambda x: x[0])

    # Add partition offset for absolute device sectors
    # Detect which device/partition the file is on
    df_result = subprocess.run(
        ["df", str(model_path)], capture_output=True, text=True
    )
    # Parse partition from df output (e.g., "/dev/nvme1n1p2")
    df_lines = df_result.stdout.strip().split('\n')
    if len(df_lines) >= 2:
        partition_dev = df_lines[1].split()[0]  # e.g., "/dev/nvme1n1p2"
        dev_name = partition_dev.replace("/dev/", "")

        # Find parent block device and partition name
        # e.g., nvme1n1p2 -> block_device=nvme1n1, partition=nvme1n1p2
        import re as re2
        match = re2.match(r'(nvme\d+n\d+)(p\d+)?', dev_name)
        if match:
            block_dev = match.group(1)
            part_name = dev_name
            offset = get_partition_offset(block_dev, part_name)

            # Add offset to all extents
            extents = [(s + offset, e + offset) for s, e in extents]

    return extents
