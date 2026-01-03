# TUM Server Specifications

**Server**: cli-hiwi-02.dis.cit.tum.de

## Hardware Overview

### CPU
- **Model**: AMD Ryzen 7 7700X
- **Cores**: 8 cores / 16 threads
- **Clock**: 0.4 – 5.57 GHz
- **NUMA**: Single node

### Memory
- **Total RAM**: 30 GiB
- **Available**: ~27 GiB free
- **Swap**: 8.0 GiB configured (typically unused)

### GPU
- **Graphics**: AMD Raphael iGPU (integrated)
- **VRAM**: Shared with system RAM (no dedicated VRAM)
- **Note**: No NVIDIA driver; llama.cpp runs CPU-only

## Storage Configuration

### NVMe Drive 1: System Drive (Samsung 980 PRO)
- **Device**: `/dev/nvme1n1`
- **Model**: Samsung SSD 980 PRO 1TB
- **Serial**: S5GXNX1W517963T
- **Capacity**: 1.00 TB
- **Mount**: `/` (root filesystem)
- **Usage**: 523.61 GB used / 397 GB available (55% used)
- **Filesystem**: ext4, 915 GB total

**Partitions**:
- `nvme1n1p1`: 1.1 GB (boot/EFI, FAT32)
- `nvme1n1p2`: 915 GB (main OS, ext4)

### NVMe Drive 2: Experimental SSD (Western Digital)
- **Device**: `/dev/nvme0n1`
- **Model**: WUS4BB096D7P3E3 (Enterprise drive)
- **Serial**: A06A0AA2
- **Capacity**: 960.20 GB
- **Symlink**: `/blk/w0` → `/dev/disk/by-id/nvme-WUS4BB096D7P3E3_A06A0AA2`
- **Purpose**: Experimental SSD for thesis work (can be overwritten)
- **Target Performance**: ~80 GB/s bandwidth

⚠️ **Note**: Second SSD (`/blk/w0`) is shared - check Google Sheet before use!

### Block Devices
Both SSDs are exposed as block devices providing raw read/write access in fixed-size blocks (typically 4KB). This is the lowest-level interface the OS uses.

## Baseline I/O Statistics
```
Device: nvme0n1 (Experimental SSD)
- Read: 12.72 r/s, 50.90 kB/s
- Write: 0.38 w/s, 3.88 kB/s

Device: nvme1n1 (System SSD)
- Read: 0.03 r/s, 1.44 kB/s
- Write: 0.39 w/s, 9.50 kB/s
```

## Operating System

- **OS**: Linux 6.8.0-85-generic (Ubuntu)
- **Architecture**: x86_64
- **Sudo Access**: Granted for tracing (use carefully!)

## Thesis Context

### Research Objective
Optimize LLM inference when model weights are loaded from SSD storage rather than RAM.

### Current Performance Gap
- **Current**: llama.cpp achieves ~10 GB/s throughput
- **Target**: Saturate 80 GB/s SSD bandwidth
- **Suspected Issues**: Synchronous I/O, lack of prefetching, inefficient async I/O

### Key Research Questions
1. Are parameters accessed sequentially (layer-by-layer) or uniformly?
2. Can we implement a buffer manager for parameter loading?
3. How does memory pressure affect access patterns?

## Important Notes

### Use Stable Device IDs
Always use `/dev/disk/by-id/` paths, **NOT** `/dev/nvmeXnX` (which can change across reboots).

### Memory Concepts
- **Anonymous Memory**: Memory from `malloc()` with no backing file
  - Has no "home" on disk
  - Must be swapped to disk when evicted (slower than file-backed memory)
- **File-Backed Memory**: mmap'd files can be evicted without writing (just re-read from file)

### System Tuning
```bash
# Disable swap
sudo swapoff -a

# Clear page cache
sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"

# Set swappiness (0 = conservative, 100 = aggressive)
sudo sysctl vm.swappiness=0
```
