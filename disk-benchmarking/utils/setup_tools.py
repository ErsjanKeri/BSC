"""
Setup and infrastructure tools for experiments

Includes:
- Logging utilities
- Command execution
- Cache management
- Memory locking setup
- tmpfs management
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime


def log(msg):
    """Timestamped logging"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, check=True, capture=False):
    """Run shell command

    Args:
        cmd: Command string to execute
        check: Raise exception on non-zero exit (default: True)
        capture: Capture and return stdout (default: False)

    Returns:
        str: stdout if capture=True, None otherwise
    """
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    else:
        subprocess.run(cmd, shell=True, check=check)


def check_root():
    """Ensure running as root"""
    if os.geteuid() != 0:
        print("ERROR: Must run as root (sudo)")
        import sys
        sys.exit(1)


def drop_caches():
    """Clear page cache, dentries, and inodes"""
    log("Dropping caches...")
    run_cmd("sync")
    run_cmd('sh -c "echo 3 > /proc/sys/vm/drop_caches"')
    log("Caches dropped")


def compile_mem_locker(mlock_bin, mlock_tool_cpp):
    """Compile mem_locker if needed

    Args:
        mlock_bin: Path to compiled binary
        mlock_tool_cpp: Path to source file
    """
    if mlock_bin.exists():
        log(f"mem_locker exists: {mlock_bin}")
        return

    if not mlock_tool_cpp.exists():
        print(f"ERROR: {mlock_tool_cpp} not found")
        import sys
        sys.exit(1)

    log(f"Compiling mem_locker...")
    run_cmd(f"g++ -o {mlock_bin} {mlock_tool_cpp} -pthread")
    log(f"Compiled: {mlock_bin}")


def mount_dedicated_tmpfs(tmpfs_mount, tmpfs_size_gb):
    """Mount dedicated tmpfs for blktrace (isolated from page cache)

    Args:
        tmpfs_mount: Mount point path
        tmpfs_size_gb: Size in GB

    Returns:
        Path: mount point path
    """
    mount_path = Path(tmpfs_mount)

    # Check if already mounted
    result = run_cmd(f"mount | grep {tmpfs_mount}", check=False, capture=True)
    if result:
        log(f"tmpfs already mounted at {tmpfs_mount}")
        return mount_path

    # Create mount point
    mount_path.mkdir(parents=True, exist_ok=True)

    # Mount tmpfs with fixed size
    log(f"Mounting dedicated tmpfs ({tmpfs_size_gb}GB) at {tmpfs_mount}...")
    run_cmd(f"mount -t tmpfs -o size={tmpfs_size_gb}G tmpfs {tmpfs_mount}")

    # Verify
    result = run_cmd(f"df -h {tmpfs_mount}", capture=True)
    log(f"tmpfs mounted successfully:\n{result}")

    return mount_path


def unmount_dedicated_tmpfs(tmpfs_mount):
    """Unmount dedicated tmpfs and free RAM

    Args:
        tmpfs_mount: Mount point path to unmount
    """
    log(f"Unmounting dedicated tmpfs at {tmpfs_mount}...")

    # Unmount
    run_cmd(f"umount {tmpfs_mount}", check=False)

    # Remove directory
    mount_path = Path(tmpfs_mount)
    if mount_path.exists() and mount_path.is_dir():
        try:
            mount_path.rmdir()
        except:
            pass  # May not be empty or may not exist

    log("tmpfs unmounted, 8GB RAM freed")
