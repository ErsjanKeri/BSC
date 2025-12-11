# TUM Server Specifications - cli-hiwi-02.dis.cit.tum.de




## Questions: 
### free -h 
1) What exactly is Swap here? why is it partitioned to 8Gib 
2) Why is available and free as options? why is buffer/cache separately? is free and available not the same? 
3) 


## sudo swapoff -a -> proper 
## sudo bash -c "echo 1 > /proc/sys/vm/drop_caches"

## Two Physical SSDs 
nvme0n1: Western Digital 9.6TB enterprise drive (WUS4BB096D7P3E3)
nvme1n1: Samsung 980 PRO 1TB (your boot/OS drive)
-> connected via PCIe slots to motherboard, just blocks of flash storage cells 
Linux exposes each SSD as a block device (what is a block device? -> provide raw read/write access in fixed size blocks (typically 4KB), this is the lowest level interface the OS uses) -> are these blocks the same as the 128Kib block requests from the Cheops paper? 

3. Partitioning (dividing drives into sections)
Your 1TB Samsung drive is partitioned:

nvme1n1p1 = 1.1 GB partition (boot/EFI partition)
nvme1n1p2 = 915 GB partition (main OS partition)
Remaining ~8 GB = swap space (not shown in partition list, configured separately), ? but these are not 8 GB? 

4) is the nvm0n1 unpartitioned? what is partitioning and what does it do? can I partition only SSD or other stuff as well? 
Your setup:

nvme1n1p2 has an ext4 or xfs filesystem containing your OS (/)
nvme1n1p1 has a FAT32 filesystem for UEFI boot (/boot/efi)

The filesystem translates "open file X" into block device reads/writes.

Now: 
/blk/w0 -> is a symlink, symbolic lin pointing to the 9.6TB drive 




## System Information
- **Hostname**: cli-hiwi-02.dis.cit.tum.de
- **OS**: Linux 6.8.0-85-generic (Ubuntu)
- **CPU**: 16 cores (x86_64)

## Memory
- **Total RAM**: 30 GiB
- **Available RAM**: 28 GiB
- **Used RAM**: 1.8 GiB
- **Swap**: 8.0 GiB (unused)

## Storage

### NVMe Drive 1 (System Drive)
- **Device**: `/dev/nvme1n1` (Samsung SSD 980 PRO 1TB)
- **Model**: Samsung SSD 980 PRO 1TB
- **Serial**: S5GXNX1W517963T
- **Capacity**: 1.00 TB
- **Used**: 523.61 GB
- **Mount**: `/` (root filesystem)
- **Available Space**: 397G (55% used)
- **Filesystem**: 915G total

### NVMe Drive 2 (Fast Experimental SSD)
- **Device**: `/dev/nvme0n1` (WUS4BB096D7P3E3)
- **Model**: WUS4BB096D7P3E3
- **Serial**: A06A0AA2
- **Capacity**: 960.20 GB
- **Symlink**: `/blk/w0` → `/dev/disk/by-id/nvme-WUS4BB096D7P3E3_A06A0AA2`
- **Purpose**: Experimental SSD for thesis work (can be overwritten)
- **Performance Target**: ~80 GB/s bandwidth (to be tested)

## Current I/O Statistics (Baseline)
```
Device: nvme0n1 (Fast SSD)
- Read: 12.72 r/s, 50.90 kB/s
- Write: 0.38 w/s, 3.88 kB/s

Device: nvme1n1 (System SSD)
- Read: 0.03 r/s, 1.44 kB/s
- Write: 0.39 w/s, 9.50 kB/s
```

## Thesis Project Details

### Objective
Optimize DeepSeek inference when model weights are loaded from SSD storage.

### Current Problem
- llama.cpp only achieves 10 GB/s throughput
- Target: Saturate 80 GB/s SSD bandwidth
- Suspect: Synchronous I/O, lack of prefetching, inefficient async I/O

### Experimental Plan
1. Run inference on 16GB open-source model
2. Use BPF tracing to monitor I/O patterns
3. Force memory pressure to trigger SSD offloading
4. Analyze sequential vs uniform access patterns
5. Question CHEOPS paper claims about uniform parameter access

### Key Research Questions
1. Are parameters accessed sequentially (layer-by-layer) or uniformly?
2. Can we implement buffer manager for parameter loading?
3. How to optimize llama.cpp for this server?

## Important Notes
- **Second SSD** (`/blk/w0`) is shared - check Google Sheet before use
- **Use stable device IDs** (by-id paths, NOT `/dev/nvmeXnX`)
- **Sudo access** granted for tracing (be careful!)
- **No GPU** detected (CPU-only inference)
