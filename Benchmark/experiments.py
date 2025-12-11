#!/usr/bin/env python3
"""
SSD-Backed Inference Experiments - Complete runner

Usage:
    sudo python3 experiments.py
"""

import os
import sys
import subprocess
import time
import json
import shutil
import re
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
BSC_DIR = SCRIPT_DIR.parent
LLAMA_CLI = BSC_DIR / "llama.cpp" / "build" / "bin" / "llama-cli"
MODELS_DIR = BSC_DIR / "llama.cpp" / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

# System
TOTAL_RAM_GB = 30
OS_RESERVE_GB = 2
SWAP_FILE = "/swap.img"
SWAP_SIZE_GB = 50

# Models
MODELS = {
    "llama-2-7b": {"file": "llama-2-7b-chat.Q4_K_M.gguf", "size_gb": 3.9},
    "gpt-oss-20b": {"file": "gpt-oss-20b-F16.gguf", "size_gb": 13.0}
}

# Experiments: (model, ssd_percent, swappiness, tokens, name)
EXPERIMENTS = [
    ("llama-2-7b", 0, 0, 100, "baseline"),
    ("llama-2-7b", 50, 100, 100, "50percent"),
    ("llama-2-7b", 100, 100, 100, "100percent"),
    ("gpt-oss-20b", 0, 0, 100, "baseline"),
    ("gpt-oss-20b", 50, 100, 100, "50percent"),
    ("gpt-oss-20b", 100, 100, 100, "100percent"),
]

# ============================================================================
# HELPERS
# ============================================================================

def log(msg):
    """Simple timestamped logging"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run_cmd(cmd, check=True, capture=False):
    """Run shell command"""
    if capture:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
    else:
        return subprocess.run(cmd, shell=True, check=check)

# SETUP
def setup_environment():
    """Setup swap, compile tools (idempotent - safe to run multiple times)"""
    log("Setting up environment...")

    # Check sudo
    if os.geteuid() != 0:
        print("ERROR: Run with sudo: sudo python3 experiments.py")
        sys.exit(1)

    # Create or resize swap if needed
    swap_path = Path(SWAP_FILE)
    if swap_path.exists():
        current_size_gb = swap_path.stat().st_size / (1024 ** 3)
        if current_size_gb + 0.5 < SWAP_SIZE_GB:  # allow small slack
            log(f"Recreating swap: {current_size_gb:.1f}GB -> {SWAP_SIZE_GB}GB")
            run_cmd(f"swapoff {SWAP_FILE}", check=False)
            run_cmd(f"dd if=/dev/zero of={SWAP_FILE} bs=1G count={SWAP_SIZE_GB}")
            run_cmd(f"chmod 600 {SWAP_FILE}")
            run_cmd(f"mkswap {SWAP_FILE}")
    else:
        log(f"Creating {SWAP_SIZE_GB}GB swap file...")
        run_cmd(f"dd if=/dev/zero of={SWAP_FILE} bs=1G count={SWAP_SIZE_GB}")
        run_cmd(f"chmod 600 {SWAP_FILE}")
        run_cmd(f" {SWAP_FILE}")

    # Enable swap
    result = run_cmd("swapon --show", capture=True)
    if SWAP_FILE not in result.stdout:
        log("Enabling swap...")
        run_cmd(f"swapon {SWAP_FILE}")

    # Compile mem_locker
    mem_locker = BSC_DIR / "mem_locker"
    if not mem_locker.exists():
        log("Compiling mem_locker...")
        run_cmd(f"cd {BSC_DIR} && g++ -o mem_locker mlock_tool.cpp -pthread")

    # Check llama-cli
    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli not found at {LLAMA_CLI}")
        sys.exit(1)

    log("Environment ready")

# ============================================================================
# EXPERIMENT
# ============================================================================

def calculate_mlock(model_size_gb, ssd_percent):
    """Calculate memory to lock"""
    if ssd_percent == 0:
        return 0

    model_on_ssd = model_size_gb * (ssd_percent / 100.0)
    model_in_ram = model_size_gb - model_on_ssd
    mlock_gb = TOTAL_RAM_GB - OS_RESERVE_GB - model_in_ram
    mlock_gb = min(mlock_gb, TOTAL_RAM_GB - 2)
    mlock_gb = max(mlock_gb, 0)

    return int(mlock_gb)

def run_inference(model_file, output_dir, tokens):
    """Run llama.cpp inference and measure time"""
    log(f"Running inference ({tokens} tokens)...")

    inference_log = output_dir / "inference.log"
    start_time = time.time()

    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODELS_DIR / model_file),
        "-p", "Once upon a time",
        "-n", str(tokens),
        "--no-mmap",
        "--log-disable"
    ]

    try:
        with open(inference_log, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=14400)

        elapsed = time.time() - start_time
        tokens_per_sec = tokens / elapsed

        metrics = {
            "total_time_seconds": elapsed,
            "tokens_generated": tokens,
            "tokens_per_second": tokens_per_sec
        }

        with open(output_dir / "inference_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        log(f"Inference complete: {elapsed:.2f}s ({tokens_per_sec:.2f} t/s)")
        return metrics

    except subprocess.TimeoutExpired:
        log("ERROR: Inference timed out")
        return None
    except Exception as e:
        log(f"ERROR: {e}")
        return None

def generate_summary(output_dir, config, metrics):
    """Generate SUMMARY.md"""
    summary = output_dir / "SUMMARY.md"

    with open(summary, 'w') as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Model**: {config['model']} ({config['model_size_gb']:.1f} GB)\n")
        f.write(f"- **Scenario**: {config['scenario']} ({config['ssd_percent']}% on SSD)\n")
        f.write(f"- **Swappiness**: {config['swappiness']}\n")
        f.write(f"- **Memory locked**: {config['mlock_gb']} GB\n")
        f.write(f"- **Tokens**: {config['tokens']}\n\n")

        if metrics:
            f.write("## Performance\n\n")
            f.write(f"- **Time**: {metrics['total_time_seconds']:.2f}s\n")
            f.write(f"- **Speed**: {metrics['tokens_per_second']:.2f} tokens/sec\n\n")

        # Check for blktrace analysis
        pattern_file = output_dir / "access_pattern.json"
        if pattern_file.exists():
            with open(pattern_file) as pf:
                pattern = json.load(pf)
            f.write("## I/O Access Pattern\n\n")
            f.write(f"- **Reads**: {pattern.get('total_reads', 0):,}\n")
            f.write(f"- **Data**: {pattern.get('total_mb_read', 0):.2f} MB\n")
            f.write(f"- **Sequential**: {pattern.get('sequential_percent', 0):.1f}%\n\n")
        elif config.get("ssd_percent", 0) == 0:
            f.write("## I/O Access Pattern\n\n")
            f.write("*N/A - Baseline scenario has no disk I/O (100% in RAM)*\n\n")
        else:
            f.write("## I/O Access Pattern\n\n")
            f.write("*No blktrace data captured*\n\n")

        f.write("## Files\n\n")
        f.write("- `config.json` - Experiment configuration\n")
        f.write("- `inference_metrics.json` - Performance metrics\n")
        f.write("- `inference.log` - Full llama.cpp output\n")
        f.write("- `blktrace/` - Block I/O traces\n")
        f.write("- `page_faults.log` - Page fault counts\n")
        f.write("- `memory.csv`, `cpu.csv` - System metrics\n")
        f.write("- `thread_usage.log` - Per-thread CPU snapshots\n")

    log(f"Summary saved: {summary}")

def run_experiment(model, ssd_percent, swappiness, tokens, name):
    """Run single experiment with full monitoring"""

    # Ensure previous mem_locker isn't hanging around
    run_cmd("pkill -f mem_locker", check=False)

    # Setup
    model_config = MODELS[model]
    model_file = model_config["file"]
    model_size_gb = model_config["size_gb"]
    mlock_gb = calculate_mlock(model_size_gb, ssd_percent)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{timestamp}_{model}_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"  Experiment: {model} - {name}")
    print(f"  SSD: {ssd_percent}%, Swappiness: {swappiness}, mlock: {mlock_gb}GB")
    print("="*70 + "\n")

    # Save config
    config = {
        "model": model,
        "model_size_gb": model_size_gb,
        "scenario": name,
        "ssd_percent": ssd_percent,
        "swappiness": swappiness,
        "mlock_gb": mlock_gb,
        "tokens": tokens,
        "timestamp": timestamp
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Set swappiness
    log(f"Setting swappiness={swappiness}...")
    run_cmd(f"echo {swappiness} > /proc/sys/vm/swappiness")

    # Clear cache
    log("Clearing page cache...")
    run_cmd("echo 3 > /proc/sys/vm/drop_caches")
    time.sleep(2)

    # Start memory locker
    mlock_proc = None
    if mlock_gb > 0:
        log(f"Locking {mlock_gb}GB of memory...")
        mlock_log = output_dir / "mlock.log"
        mlock_proc = subprocess.Popen(
            [str(BSC_DIR / "mem_locker"), str(mlock_gb)],
            stdout=open(mlock_log, 'w'),
            stderr=subprocess.STDOUT
        )
        time.sleep(5)
    else:
        log("Skipping memory lock (baseline)")

    # Start monitoring
    log("Starting monitoring...")
    monitor_proc = subprocess.Popen(
        [str(SCRIPT_DIR / "utils" / "monitor_system.sh"), str(output_dir), "1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(2)

    # Start blktrace (to tmpfs)
    log("Starting blktrace...")
    trace_dir = Path(f"/dev/shm/blktrace_{os.getpid()}")
    trace_dir.mkdir(exist_ok=True)
    (output_dir / "blktrace").mkdir(exist_ok=True)

    # Detect swap device
    result = run_cmd(f"df {SWAP_FILE} | tail -1 | awk '{{print $1}}'", capture=True)
    device_partition = result.stdout.strip()
    swap_device = re.sub(r"p?\d+$", "", device_partition)
    if not swap_device or not Path(swap_device).exists():
        # Fall back to the partition itself if stripping failed
        swap_device = device_partition
    if not Path(swap_device).exists():
        # Last resort: try base nvme device
        swap_device = "/dev/nvme1n1"

    blktrace_proc = subprocess.Popen(
        ["blktrace", "-d", swap_device, "-o", "trace"],
        cwd=str(trace_dir),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(2)

    # Start thread utilization monitor
    thread_log = output_dir / "thread_usage.log"
    thread_mon_proc = subprocess.Popen(
        f"while true; do date +%s >> {thread_log}; "
        f"ps -eLo pid,comm,pcpu --sort=-pcpu | head -n 20 >> {thread_log}; "
        f"echo '---' >> {thread_log}; sleep 2; done",
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Start bpftrace
    log("Starting bpftrace...")
    bpf_script = """
        tracepoint:exceptions:page_fault_user {
            @faults[comm] = count();
        }
        interval:s:1 {
            time("%H:%M:%S ");
            print(@faults);
            clear(@faults);
        }
    """
    bpftrace_proc = subprocess.Popen(
        ["bpftrace", "-e", bpf_script],
        stdout=open(output_dir / "page_faults.log", 'w'),
        stderr=subprocess.PIPE
    )
    time.sleep(2)

    # Run inference
    metrics = run_inference(model_file, output_dir, tokens)

    # Stop everything
    log("Stopping monitoring...")

    if 'thread_mon_proc' in locals() and thread_mon_proc:
        thread_mon_proc.terminate()
        try:
            thread_mon_proc.wait(timeout=5)
        except:
            thread_mon_proc.kill()

    if bpftrace_proc:
        bpftrace_proc.terminate()
        try:
            bpftrace_proc.wait(timeout=5)
        except:
            bpftrace_proc.kill()

    if blktrace_proc:
        run_cmd("killall blktrace", check=False)
        try:
            blktrace_proc.wait(timeout=5)
        except:
            pass
        # Copy traces
        for f in trace_dir.glob("trace.blktrace.*"):
            shutil.copy2(f, output_dir / "blktrace" / f.name)
        shutil.rmtree(trace_dir, ignore_errors=True)

    if monitor_proc:
        monitor_proc.terminate()
        pid_file = output_dir / "monitor_pids.txt"
        if pid_file.exists():
            for pid in pid_file.read_text().strip().split('\n'):
                try:
                    os.kill(int(pid), 15)
                except:
                    pass

    if mlock_proc:
        mlock_proc.terminate()
        try:
            mlock_proc.wait(timeout=5)
        except:
            mlock_proc.kill()
    # Ensure no lingering lockers
    run_cmd("pkill -f mem_locker", check=False)

    # Analyze blktrace (stream to avoid huge intermediate files)
    if (output_dir / "blktrace" / "trace.blktrace.0").exists():
        log("Analyzing blktrace...")
        run_cmd(
            f"blkparse -i {output_dir}/blktrace/trace -o - | "
            f"python3 {SCRIPT_DIR}/utils/analyze_blktrace.py - {output_dir}/access_pattern.json",
            check=False
        )

    # Generate summary
    generate_summary(output_dir, config, metrics)

    log(f"Results saved to: {output_dir}\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  SSD-Backed Inference Experiments")
    print("="*70 + "\n")

    # Setup (idempotent)
    setup_environment()

    # Run all experiments
    for model, ssd_percent, swappiness, tokens, name in EXPERIMENTS:
        run_experiment(model, ssd_percent, swappiness, tokens, name)

    print("\n" + "="*70)
    print(f"  All experiments complete!")
    print(f"  Results in: {RESULTS_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
