# 9 December 2024 - Key Findings & Experimental Setup

## 1. llama.cpp Memory Management (CRITICAL)

### Model Loading: mmap() + Page Cache
**Location**: [llama.cpp/src/llama-mmap.cpp:276-298](llama.cpp/src/llama-mmap.cpp#L276-L298)

```cpp
addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);  // Hints sequential access
```

**How it works**:
1. Model file is **memory-mapped** (not read() into malloc'd buffer)
2. File contents are mapped into virtual address space
3. **NO physical RAM is allocated immediately** - pages loaded on-demand via page faults
4. Linux uses **page cache** to store mmap'd pages
5. Kernel expects **sequential access** (POSIX_FADV_SEQUENTIAL hint)

### Page Cache vs Available Memory (Your free -h Observation)

**Small model (llama-2-7b, 3.9GB)**:
- Fits entirely in page cache
- You saw: **Available** memory dropped
- Why: Kernel cached the entire model, reducing "available" RAM

**Large model (gpt-oss-20b, 13GB)**:
- Stored in **cache/buffers**, not consuming "available"
- Why: Linux page cache is **reclaimable** - kernel can evict pages if needed
- **Available** didn't drop because kernel counts cache as "available"

**Key insight**: `free -h` breakdown:
```
Total = Used + Free + Buffers/Cache
Available ≈ Free + Reclaimable Cache
```
When model is in cache: it's **used but reclaimable**.

---

## 2. Memory Locking (mlock)

### What is mlock()?
**System call**: `mlock(addr, size)` - pins virtual pages to physical RAM

**Effects**:
- ✅ Pages are **locked in RAM** - kernel cannot swap/evict them
- ✅ **Prevents page faults** for locked pages
- ✅ Makes memory **non-reclaimable**

### llama.cpp Built-in mlock Support
**Location**: [llama.cpp/src/llama-mmap.cpp:463-496](llama.cpp/src/llama-mmap.cpp#L463-L496)

llama.cpp can optionally mlock model weights to prevent swapping:
- Controlled by `--mlock` flag (or `use_mlock` in API)
- Requires sufficient `RLIMIT_MEMLOCK` (check with `ulimit -l`)
- Useful for **reducing inference latency** (no page faults)

**For our experiments**: We do NOT want llama.cpp to mlock - we WANT page faults!

---

## 3. Swap Space & Storage Allocation (CRITICAL!)

### Current Problem: Swap Size = 0?
**Check**: `swapon --show` or `free -h` (Swap line)

If **Swap = 0**:
- Kernel has **nowhere to evict pages** when RAM is full
- **Page faults won't trigger disk I/O** - kernel will just OOM kill
- ❌ **Our experiments won't work!**

### Solution: Enable Swap on System Drive

**Why we need swap**:
1. Force kernel to evict model pages to disk (SSD)
2. When llama.cpp accesses evicted pages → **page fault** → read from SSD
3. This is how we measure SSD-backed inference throughput

**Setup commands**:
```bash
# Create 50GB swap file on system drive (has 379GB free)
sudo dd if=/dev/zero of=/home/swapfile bs=1G count=50
sudo chmod 600 /home/swapfile
sudo mkswap /home/swapfile
sudo swapon /home/swapfile

# Verify
swapon --show
# Should show: /home/swapfile
```

**Note**: We use the system drive (`/`) which has plenty of space (379GB available)

---

## 4. Forcing SSD Access: The mlock Strategy

### Goal
Make llama.cpp inference use **SSD-backed pages** instead of RAM.

### Method: Pin Dummy Memory with mlock
Your tool: `mlock_tool.cpp` - allocates and locks N GB of memory

**What happens**:
1. mlock tool locks **22 GB** of RAM (for 30 GB total, 13 GB model)
2. Only **8 GB free** RAM remains (30 - 22 = 8)
3. When llama.cpp mmaps **13 GB model** → doesn't fit in remaining RAM
4. Kernel evicts model pages to **swap (on SSD)**
5. During inference → **page faults** → kernel reads from SSD swap
6. We measure **SSD throughput** during inference

**Formula**:
```
mlock_size = Total_RAM - Target_Free_RAM
Target_Free_RAM = Model_Size - Amount_to_Force_on_SSD

Examples:
- Force 100% on SSD: mlock 30 - (13 - 13) = 30 GB (leave 0 free)
  → DANGER: Will OOM, leave ~2GB for OS
- Force 100% on SSD (safe): mlock 28 GB (leave 2GB for OS)
- Force 50% on SSD: mlock 30 - 6.5 = 23.5 GB
- No SSD (baseline): mlock 0 GB
```

**Conservative formula**:
```
mlock_size = Total_RAM - OS_Reserve - RAM_for_model
Where:
  OS_Reserve = 2 GB (for kernel, processes)
  RAM_for_model = 0 to Model_Size (how much model should stay in RAM)

To force 100% model on SSD: RAM_for_model = 0
  → mlock_size = 30 - 2 - 0 = 28 GB
```

---

## 5. llama.cpp Setup (Version & Capabilities)

- **Version**: commit `b9a37717b` (version 7234)
- **Binary**: `/home/keri/BSC/llama.cpp/build/bin/llama-cli`
- ✅ **MLA support**: YES (DeepSeek2 architecture implemented)
- ✅ **MoE support**: YES (multiple architectures)
- **Available models**:
  - `llama-2-7b-chat.Q4_K_M.gguf`: 3.9 GB (standard MHA)
  - `gpt-oss-20b-F16.gguf`: 13 GB (unknown arch - **TODO: check**)

---

## 6. Experimental Setup - Step by Step

### Phase 0: Environment Preparation

**Step 1: Enable Swap on System Drive**
```bash
# Create 50GB swap on system drive
sudo dd if=/dev/zero of=/home/swapfile bs=1G count=50
sudo chmod 600 /home/swapfile
sudo mkswap /home/swapfile
sudo swapon /home/swapfile

# Verify
swapon --show
free -h
```

**Step 2: Compile mlock Tool**
```bash
cd /home/keri/BSC
g++ -o mem_locker mlock_tool.cpp
ulimit -l unlimited  # Allow unlimited mlock
```

**Step 3: Verify Model Architecture**
```bash
cd llama.cpp
./build/bin/llama-cli -m models/gpt-oss-20b-F16.gguf --help 2>&1 | head -20
# Look for: architecture, MoE, expert count
```

---

### Phase 1: Baseline (Model in RAM)

**Goal**: Measure inference speed when model is **fully in RAM** (no SSD access).

```bash
# Clear swap to ensure no pages on SSD
sudo swapoff -a
sudo swapon -a

# Ensure model is NOT mlocked by llama.cpp
./build/bin/llama-cli \
  -m models/gpt-oss-20b-F16.gguf \
  -p "Once upon a time" \
  -n 100 \
  --no-mlock \
  --log-disable 2>&1 | tee baseline_ram.log

# Extract: tokens/sec, total time
grep "tokens" baseline_ram.log
```

**Expected**:
- Model pages in RAM (page cache)
- **No SSD I/O** during inference
- **Fast inference** (baseline speed)

---

### Phase 2: Force SSD Access (Partial - 50%)

**Goal**: Force **half the model** (6.5 GB) to reside on SSD.

**Memory calculation**:
- Total RAM: 30 GB
- Model size: 13 GB
- OS reserve: 2 GB
- RAM for model: 6.5 GB (50%)
- **mlock size**: 30 - 2 - 6.5 = **21.5 GB**

```bash
# Start memory locker
./mem_locker 21 &  # Lock 21 GB
MLOCK_PID=$!

# Wait for memory to be locked
sleep 5

# Monitor SSD I/O
iostat -x 2 /dev/nvme0n1 > iostat_50percent.log &
IOSTAT_PID=$!

# Run inference
./build/bin/llama-cli \
  -m models/gpt-oss-20b-F16.gguf \
  -p "Once upon a time" \
  -n 100 \
  --no-mlock 2>&1 | tee inference_50percent.log

# Stop monitoring
kill $IOSTAT_PID $MLOCK_PID

# Analyze SSD throughput
grep "nvme0n1" iostat_50percent.log | awk '{print $4}' | sort -n | tail -10
# Column 4 = rkB/s (read throughput)
```

**Expected**:
- ~6.5 GB model pages swapped to SSD
- **Page faults** during inference → SSD reads
- **Slower than baseline** (due to SSD latency)
- SSD throughput: measure **rkB/s** during inference

---

### Phase 3: Force SSD Access (Full - 100%)

**Goal**: Force **entire model** (13 GB) to reside on SSD.

**Memory calculation**:
- **mlock size**: 30 - 2 - 0 = **28 GB**

```bash
# Lock 28 GB (leave only 2GB for OS)
./mem_locker 28 &
MLOCK_PID=$!

sleep 5

# Monitor SSD I/O
iostat -x 2 /dev/nvme0n1 > iostat_100percent.log &
IOSTAT_PID=$!

# Run inference
./build/bin/llama-cli \
  -m models/gpt-oss-20b-F16.gguf \
  -p "Once upon a time" \
  -n 100 \
  --no-mlock 2>&1 | tee inference_100percent.log

kill $IOSTAT_PID $MLOCK_PID
```

**Expected**:
- **All 13 GB** swapped to SSD
- **Maximum SSD I/O** during inference
- **Slowest inference** (everything from SSD)
- This measures **pure SSD throughput** for inference

---

### Phase 4: Access Pattern Analysis

**Goal**: Characterize if access is **sequential** or **random**.

```bash
# Lock 28 GB to force SSD
./mem_locker 28 &
MLOCK_PID=$!

# Trace block I/O
sudo blktrace -d /dev/nvme0n1 -o ssd_trace &
sleep 2

# Run inference
./build/bin/llama-cli \
  -m models/gpt-oss-20b-F16.gguf \
  -p "Test prompt" \
  -n 50 \
  --no-mlock

# Stop tracing
sudo killall blktrace
kill $MLOCK_PID

# Parse trace
blkparse ssd_trace -o trace_parsed.txt

# Analyze pattern
awk '/R/ {print $8, $10}' trace_parsed.txt > reads.txt
# Column 8 = sector offset
# Check if sectors are sequential (small gaps) or random (large jumps)

# Calculate sequentiality ratio
python3 -c "
import sys
sectors = [int(line.split()[0]) for line in open('reads.txt')]
gaps = [sectors[i+1] - sectors[i] for i in range(len(sectors)-1)]
sequential = sum(1 for g in gaps if abs(g) < 256)  # <128KB gap = sequential
print(f'Sequential: {sequential/len(gaps)*100:.1f}%')
"
```

**Expected results**:
- **Standard transformer (llama-2)**: High sequential % (~80%+)
  - Layer-by-layer processing → sequential parameter reads
- **MoE models**: Lower sequential % (if experts accessed sparsely)
  - Random expert selection → random parameter reads

---

## 7. Key Metrics to Measure

### Per Experiment Run:
1. **Inference speed**: tokens/second
2. **Total inference time**: seconds
3. **SSD throughput**: MB/s (from iostat column `rkB/s`)
4. **Total bytes read from SSD**: integrate iostat over time
5. **Memory usage**: `free -h` before/after
6. **Swap usage**: `swapon --show` before/after

### Access Pattern Metrics:
1. **Sequential read %**: From blktrace analysis
2. **Average read size**: From blktrace (small = random, large = sequential)
3. **Page fault count**: From `/proc/vmstat` (pgfault, pgmajfault)

### Comparison:
- **Baseline (RAM) vs 100% SSD**: Slowdown factor
- **Theoretical SSD max** (from spec sheet) vs **Actual throughput**
- **Sequential % for different models**: llama-2 vs gpt-oss-20b

---

## 8. Critical Checks Before Experiments

```bash
# 1. Swap is enabled
swapon --show
# Must show: /home/swapfile, 50G+

# 2. Sufficient mlock limit
ulimit -l
# Must show: unlimited (or >28GB in KB)

# 3. Model file exists
ls -lh llama.cpp/models/gpt-oss-20b-F16.gguf
# Must show: 13G

# 4. llama-cli doesn't use mlock by default
./build/bin/llama-cli --help | grep mlock
# Use --no-mlock flag to disable
```

---

## 9. Summary: Why This Works

1. **llama.cpp mmaps model file** → pages loaded on-demand (page faults)
2. **We mlock dummy memory** → forces kernel to evict model pages
3. **Model pages go to swap (on SSD)** → now model is "on disk"
4. **During inference** → llama.cpp accesses model parameters
5. **Page fault occurs** → kernel reads from SSD swap into RAM
6. **We measure SSD I/O** → this is the "SSD-backed inference" throughput
7. **Access pattern** (sequential vs random) affects readahead efficiency

**Goal**: Characterize how access patterns (sequential vs random) impact SSD-backed inference performance.

**Gabriel's hypothesis**: Real models (especially MoE) may have different access patterns than simple sequential models → affects optimal page size and readahead strategies.
