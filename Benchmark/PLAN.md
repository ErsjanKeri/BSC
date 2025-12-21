# Systematic Benchmark Improvement Plan

## Philosophy
- **Understand FIRST, implement SECOND**
- Each step must be **verified** before moving to next
- Gather sample data to confirm understanding
- No assumptions - check everything explicitly
- Build mental model piece by piece

---

## PHASE 0: Current State Discovery (1-2 hours)
**Goal: Know exactly what we have right now**

### Step 0.1: Map Physical Disks
**What:** Identify all storage devices and their characteristics
**How:**
```bash
# Run these commands and save output
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL
df -h
cat /proc/swaps
ls -la /dev/disk/by-id/
ls -la /dev/nvme*
```
**Output:** Create `hardware_inventory.txt` with:
- System disk (Samsung 980 PRO): device name, size, mountpoint
- Experimental disk (WD): device name, size, mountpoint, symlink details
- Current swap location and size
- Understand: Which is `/dev/nvme0n1` vs `/dev/nvme1n1`?

**Verify:** Can identify each physical disk by serial number/model

### Step 0.2: Locate Model Files
**What:** Find exactly where model weights are stored
**How:**
```bash
# Find model files
find ~/BSC -name "*.gguf" -o -name "*.bin" -exec ls -lh {} \;
df -h ~/BSC  # Which filesystem?
```
**Output:** Document in `hardware_inventory.txt`:
- Model file paths
- Which physical disk they're on
- Total size of models

**Verify:** Can explain which disk serves model files

### Step 0.3: Current Swap Investigation
**What:** Understand existing swap configuration
**How:**
```bash
cat /proc/swaps  # What's active?
swapon --show    # Details
ls -lh /swapfile*  # Common locations
df -h | grep -E 'nvme|sda'  # Where could swap be?
file /path/to/swap/if/found
```
**Output:** Document:
- Swap file location (which disk?)
- Swap size
- Swap priority
- Is it on same disk we want to measure?

**Verify:** Know exactly where swap writes go physically

### Step 0.4: Gabriel's Symlink Mystery
**What:** Find the "symlinked and mounted" second SSD
**How:**
```bash
# Look for symlinks pointing to storage
ls -la /mnt/
ls -la /media/
ls -la ~/
find /home -type l -ls 2>/dev/null | grep nvme
mount | grep nvme
```
**Output:** Document:
- What's the symlink path?
- What does it point to?
- What's mounted where?
- Why is it "4x slower"? (sequential vs random? interface?)

**Verify:** Can access the alternate SSD and understand its purpose

**CHECKPOINT 0:** Create `hardware_inventory.txt` with complete storage map
- All disks identified
- All mountpoints known
- Swap location clear
- Symlink structure understood

---

## PHASE 1: Fundamental Concepts (2-3 hours)
**Goal: Build mental models before touching code**

### Step 1.1: Linux Storage Stack Deep Dive
**What:** Understand the layers from application to SSD
**Topics:**
1. **Block devices** (`/dev/nvme0n1`, `/dev/nvme0n1p1`, etc.)
   - Disk vs partition
   - Device naming convention
   - Why `nvme0n1` vs `sda`?

2. **Filesystems** (ext4, xfs, tmpfs)
   - What lives on filesystem vs raw device
   - How files map to blocks
   - Inode concept

3. **Mounting**
   - What `mount` actually does
   - Why `/mnt/` and `/media/`
   - Bind mounts vs regular mounts

4. **Symlinks**
   - Hard vs soft links
   - When/why to use them
   - How they appear to applications

**Output:** Write `storage_concepts.md` explaining:
- Each layer of the stack
- How data flows from `write()` syscall to SSD
- Where blktrace sits in this stack

**Verify:** Can draw diagram of: Application → VFS → Filesystem → Block Layer → Device Driver → SSD

### Step 1.2: Swap System Understanding
**What:** How Linux swap actually works
**Topics:**
1. **Swap space types**
   - Swap partition vs swap file (what's the difference?)
   - Performance implications

2. **Swappiness parameter**
   - What does value 0-100 mean?
   - When does kernel swap vs keep in RAM?
   - How does this interact with mlock?

3. **Swap activation**
   - How to create swap file
   - How to activate/deactivate
   - Multiple swap areas (priority)

4. **What gets swapped?**
   - Anonymous pages vs file-backed pages
   - Why model parameters might swap
   - Why KV cache might swap

**Output:** Add to `storage_concepts.md`:
- Swap decision flow chart
- Formula for calculating needed swap size
- What swappiness=100 vs swappiness=0 means for our experiments

**Verify:** Can predict what happens with 7GB model + 30GB RAM + various swappiness

### Step 1.3: llama.cpp Memory Modes
**What:** Understand how llama.cpp loads and uses model weights
**Topics:**
1. **mmap mode** (default)
   - Memory-mapped files
   - On-demand page loading
   - What Linux does automatically
   - Benefit: No loading time
   - Cost: Page faults on access

2. **no_mmap mode**
   - Explicit file reads
   - All weights loaded to RAM upfront
   - More control, more memory pressure

3. **mlock mode**
   - Lock pages in RAM (prevent swapping)
   - Interaction with memory-mapped files
   - When useful vs harmful

**How:**
```bash
# Test each mode with small model
./llama-cli --help | grep -i mmap
./llama-cli --help | grep -i mlock

# Quick test runs to see behavior
./llama-cli -m model.gguf --mmap
./llama-cli -m model.gguf --no-mmap
./llama-cli -m model.gguf --mlock
```

**Output:** Add to `storage_concepts.md`:
- When each mode causes I/O
- Which mode is best for our experiments
- How to isolate parameter I/O from loading I/O

**Verify:** Can explain when SSD reads happen in each mode

### Step 1.4: Model Architecture Breakdown
**What:** Understand model memory composition
**Topics:**
1. **Model weights** (parameters)
   - Size for Llama-2-7B (7GB)
   - Size for OpenHermes-2.5-Mistral (how much?)
   - Are these loaded once or accessed repeatedly?

2. **KV cache**
   - What is it? (key-value cache for attention)
   - How big does it grow? (depends on context length)
   - Formula: `2 × n_layers × n_heads × d_head × context_length`
   - Is this in same file as weights or separate?

3. **Working memory**
   - Activations, intermediate computations
   - How much does llama.cpp need?

**How:**
```bash
# Check model details
./llama-cli -m llama-2-7b.gguf --verbose
# Look for n_layers, n_heads, context length in output
```

**Output:** Create `model_memory_breakdown.md`:
- Table: Component | Size | Access Pattern | Can Swap?
- For each model we'll test
- Calculation for KV cache size at different context lengths

**Verify:** Can calculate total memory needed for inference with N tokens

**CHECKPOINT 1:** Can explain entire flow from model file on disk to inference in RAM

---

## PHASE 2: Measurement Strategy Design (1-2 hours)
**Goal: Plan what/how to measure before writing code**

### Step 2.1: Define Measurement Targets
**What:** Decide exactly what I/O to capture
**Questions to answer:**
1. **Model loading I/O** (initial read of weights)
   - Do we want this? (YES - Gabriel said "model loading is also super important")
   - How to capture just this phase?
   - Separate experiment or part of inference?

2. **Parameter access during inference**
   - This is the core question (sequential vs random?)
   - How to isolate from KV cache I/O?
   - In mmap mode: these are page faults → swap reads

3. **KV cache I/O**
   - Gabriel: "It could be KV cache might need to use storage"
   - How much memory to lock so KV stays in RAM?
   - If it spills, what pattern?

**Output:** Create `measurement_plan.md`:
- Three experiment types:
  1. Model loading characterization
  2. Parameter access characterization (KV in RAM)
  3. Full memory pressure (everything competes)

**Verify:** Can explain what I/O each experiment type captures

### Step 2.2: Blktrace Setup Strategy
**What:** Solve the "tracing while writing traces" problem
**Options:**

**Option A: Write traces to other SSD**
- Pros: Completely separate I/O paths
- Cons: Slower disk, might impact experiment timing
- Decision: Need to measure overhead first

**Option B: Write traces to tmpfs (current)**
- Pros: Fast, no disk interference
- Cons: Limited by RAM, must copy out after
- Decision: Safe if we have RAM available

**Option C: Write traces to system disk**
- Pros: Fast SSD
- Cons: Same device we're measuring
- Decision: Only if we can filter out trace writes in analysis

**How to decide:**
```bash
# Test trace write overhead
# 1. Run inference WITHOUT blktrace → baseline tokens/sec
# 2. Run inference WITH blktrace to tmpfs → compare
# 3. Run inference WITH blktrace to other SSD → compare
# 4. Check trace file size to estimate RAM needed
```

**Output:** Add to `measurement_plan.md`:
- Chosen blktrace output location
- Estimated overhead percentage
- RAM budget for traces

**Verify:** Can run one inference with trace collection without crashing

### Step 2.3: Bandwidth Measurement Plan
**What:** Measure SSD bandwidth properly
**Two measurements needed:**

**A. Raw SSD capability**
```bash
# Sequential read test
fio --name=seq_read --rw=read --bs=128k --size=1G --numjobs=1 \
    --filename=/path/to/experimental/ssd/testfile

# Random read test  
fio --name=rand_read --rw=randread --bs=4k --size=1G --numjobs=1 \
    --filename=/path/to/experimental/ssd/testfile
```

**B. Actual bandwidth during inference**
- From blktrace data: sum of read sizes / time
- From iostat: real-time bandwidth monitoring
- Compare to theoretical max

**Output:** Add to `measurement_plan.md`:
- fio command templates
- Expected results (what's good? what's bad?)
- How to calculate from blktrace

**Verify:** Can measure baseline SSD performance

### Step 2.4: DuckDB Analysis Strategy
**What:** Plan how to analyze blktrace CSV data
**Schema design:**
```sql
CREATE TABLE block_io (
    timestamp DOUBLE,
    device VARCHAR,
    cpu INTEGER,
    pid INTEGER,
    action VARCHAR,
    rwbs VARCHAR,
    sector BIGINT,
    size_sectors INTEGER,
    process VARCHAR
);
```

**Queries to answer:**
1. **Sequential vs Random**
   ```sql
   -- Calculate gap between consecutive reads
   SELECT 
       sector,
       LAG(sector + size_sectors) OVER (ORDER BY timestamp) as prev_end,
       sector - LAG(sector + size_sectors) OVER (ORDER BY timestamp) as gap,
       CASE WHEN ABS(gap) < 256 THEN 'sequential' ELSE 'random' END as pattern
   FROM block_io
   WHERE rwbs LIKE '%R%'
   ```

2. **Access heat map** (which regions of disk?)
   ```sql
   SELECT sector / 1000000 as region_mb, COUNT(*) as access_count
   FROM block_io
   WHERE rwbs LIKE '%R%'
   GROUP BY region_mb
   ORDER BY region_mb
   ```

3. **Request size distribution**
   ```sql
   SELECT size_sectors * 512 / 1024 as size_kb, COUNT(*) 
   FROM block_io
   WHERE rwbs LIKE '%R%'
   GROUP BY size_kb
   ORDER BY size_kb
   ```

**Output:** Create `duckdb_analysis.sql` with all planned queries

**Verify:** Can load sample blktrace data into DuckDB

**CHECKPOINT 2:** Have complete measurement plan before writing code

---

## PHASE 3: Infrastructure Setup (2-3 hours)
**Goal: Prepare clean, correct environment**

### Step 3.1: Storage Infrastructure Setup
**What:** Configure disks, swap, mountpoints properly

**Task A: Identify and document disks**
```bash
# Create reference file
cat > ~/disk_map.txt << 'EOF'
System Disk (Samsung 980 PRO):
  Device: /dev/nvmeXnY
  Mount: /
  Use: OS, system files, model files
  Speed: XXX MB/s sequential

Experimental Disk (WD):
  Device: /dev/nvmeAnB  
  Mount: /mnt/experimental (or symlink location?)
  Use: Blktrace output, test data
  Speed: XXX MB/s sequential (slower)
EOF
```

**Task B: Setup swap on experimental disk**
```bash
# Decision: Put swap on disk we WANT to measure
# Why? We want to trace parameter reads from swap

# Create swap file (size = model size + KV cache estimate)
# For 20GB model: need ~25-30GB swap
sudo fallocate -l 30G /path/to/experimental/disk/swap_experiment
sudo chmod 600 /path/to/experimental/disk/swap_experiment
sudo mkswap /path/to/experimental/disk/swap_experiment

# Don't activate yet - we'll do this per-experiment
```

**Task C: Setup blktrace output location**
```bash
# Create directory on OTHER disk (or tmpfs)
# Decision based on Step 2.2 results
sudo mkdir -p /mnt/trace_output
# Or use /dev/shm for tmpfs
```

**Output:** Update `hardware_inventory.txt` with final configuration

**Verify:** Can activate/deactivate swap without errors

### Step 3.2: llama.cpp Configuration
**What:** Ensure we can run experiments with different memory modes

**Task A: Verify llama.cpp build**
```bash
cd ~/llama.cpp
git log -1  # Check version
./llama-cli --version
./llama-cli --help | grep -E 'mmap|mlock'
```

**Task B: Create wrapper scripts for each mode**
```bash
# scripts/run_mmap.sh
# scripts/run_nommap.sh  
# scripts/run_mlock.sh
```

**Task C: Test each mode with small model**
```bash
# Verify each mode works before experiments
./scripts/run_mmap.sh llama-2-7b.gguf "test prompt"
./scripts/run_nommap.sh llama-2-7b.gguf "test prompt"
./scripts/run_mlock.sh llama-2-7b.gguf "test prompt"
```

**Output:** Document working llama.cpp commands in `llama_modes.md`

**Verify:** Can run inference in all three modes

### Step 3.3: Monitoring Tools Setup
**What:** Ensure all measurement tools work

**Task A: Test blktrace on correct device**
```bash
# Find correct device for swap disk
SWAP_DEV=$(df /path/to/swap | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//')
echo "Will trace: $SWAP_DEV"

# Test blktrace for 10 seconds
sudo blktrace -d $SWAP_DEV -o test_trace -w 10
ls -lh test_trace*
blkparse -i test_trace -o test_parsed.txt
head -20 test_parsed.txt
```

**Task B: Test DuckDB CSV import**
```bash
# Convert sample trace to CSV
python3 utils/blktrace_to_csv.py test_parsed.txt test.csv

# Load into DuckDB
duckdb test.db << 'EOF'
CREATE TABLE block_io(...);
COPY block_io FROM 'test.csv' (HEADER);
SELECT COUNT(*) FROM block_io;
EOF
```

**Task C: Test iostat monitoring**
```bash
# Make sure we capture the right device
iostat -x 1 5 $SWAP_DEV
```

**Output:** Sample trace data verified in DuckDB

**Verify:** All tools work independently

**CHECKPOINT 3:** Clean environment ready for experiments

---

## PHASE 4: Pilot Experiments (3-4 hours)
**Goal: Small-scale tests to validate methodology**

### Step 4.1: Baseline Characterization
**What:** Establish ground truth with model in RAM

**Experiment 1: Pure RAM (no swap)**
```bash
# Configuration:
# - Model: llama-2-7b (7GB)
# - RAM: 30GB available
# - Swap: DISABLED
# - Mode: mmap (or no_mmap - test both)
# - Tokens: 100

# Expected: No disk I/O, pure performance baseline
```

**Measurements:**
- Tokens/sec (from llama.cpp output)
- Memory usage (from /proc/meminfo snapshots)
- Confirm zero swap I/O

**Output:** `results/pilot_01_baseline_ram/SUMMARY.md`

**Verify:** Get consistent tokens/sec across 3 runs

### Step 4.2: Model Loading Characterization
**What:** Measure just the model load phase

**Experiment 2: Cold start from disk**
```bash
# Configuration:
# - Clear page cache before each run
# - Model: llama-2-7b on experimental SSD
# - Mode: no_mmap (explicit load)
# - Trace just the loading phase

# Steps:
echo 3 > /proc/sys/vm/drop_caches  # Clear cache
# Start blktrace
# Start model load
# Stop blktrace after load completes
# Run inference (separate phase)
```

**Measurements:**
- Loading time
- Total bytes read
- Sequential vs random pattern during load
- Compare mmap vs no_mmap loading

**Output:** `results/pilot_02_model_loading/SUMMARY.md`

**Verify:** Loading pattern is highly sequential (expected)

### Step 4.3: Controlled Memory Pressure
**What:** Force model parameters to swap, keep KV in RAM

**Experiment 3: Partial swap**
```bash
# Configuration:
# - Model: llama-2-7b (7GB)
# - Lock KV cache in RAM (~2GB estimated)
# - Allow parameters to swap (5GB to SSD)
# - Swappiness: 100
# - Tokens: 100

# How to lock KV cache?
# Option 1: mlock enough RAM before inference (simplified)
# Option 2: Custom llama.cpp patch (future work)
```

**Measurements:**
- Parameter access pattern (should see disk I/O)
- Tokens/sec degradation
- Sequential vs random reads
- Page fault rate

**Output:** `results/pilot_03_partial_swap/SUMMARY.md`

**Verify:** Can see parameter access pattern on SSD

### Step 4.4: Full Memory Pressure
**What:** Everything competes for RAM

**Experiment 4: Aggressive swapping**
```bash
# Configuration:
# - Model: llama-2-7b
# - No memory locking
# - Swappiness: 100
# - Fill RAM with other data first
# - Tokens: 100

# Steps:
# 1. Allocate ~25GB with mem_locker (leave 5GB free)
# 2. Start inference (model + KV will fight for 5GB)
# 3. Observe what swaps
```

**Measurements:**
- KV cache vs parameter swap behavior
- Access pattern changes over time
- Performance degradation
- Which swaps first?

**Output:** `results/pilot_04_full_pressure/SUMMARY.md`

**Verify:** Understand what Linux chooses to swap

**CHECKPOINT 4:** Have working methodology with small model

---

## PHASE 5: Analysis Tools Development (2-3 hours)
**Goal: Build robust analysis pipeline**

### Step 5.1: Blktrace → CSV Converter
**What:** Reliable parser for blkparse output

**Requirements:**
- Stream processing (don't load entire file)
- Correct column parsing (fixed from current bug)
- CSV output with proper schema
- Handle large files (10GB+)

**Output:** `utils/blktrace_to_csv.py`

**Test:**
```bash
# Use pilot experiment traces
python3 utils/blktrace_to_csv.py \
    results/pilot_03_partial_swap/blktrace_parsed.txt \
    test_output.csv

# Verify schema
head test_output.csv
wc -l test_output.csv
```

**Verify:** CSV loads into DuckDB without errors

### Step 5.2: DuckDB Analysis Scripts
**What:** Standardized queries for all experiments

**Create:** `utils/analyze_duckdb.py`
```python
def analyze_access_pattern(csv_file):
    """
    Returns:
    - total_reads, total_mb
    - sequential_percent
    - request_size_distribution
    - temporal_pattern (access rate over time)
    - spatial_pattern (which disk regions?)
    """
```

**Output:** JSON results matching current format

**Test:** Run on all pilot experiment CSVs

**Verify:** Results match manual calculations

### Step 5.3: Summary Generator Rewrite
**What:** Concise table-based output

**New SUMMARY.md format:**
```markdown
# Experiment: llama-2-7b_partial_swap

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | llama-2-7b (7GB) |
| RAM Available | 30GB |
| Memory Locked | 2GB (KV cache) |
| Swappiness | 100 |
| Mode | mmap |

## Results
| Metric | Value |
|--------|-------|
| Tokens/sec | 45.2 |
| SSD Reads | 2,451,234 |
| Data Read | 4,823 MB |
| Sequential % | 67.3% |
| Avg Request | 128 KB |
| SSD Bandwidth | 82.4 MB/s |

## Pattern Classification
**MOSTLY SEQUENTIAL** - Parameter access shows predictable patterns with 67% sequential reads.

## Files
- config.json, metrics.json, block_io.csv, plots/
```

**Output:** Updated `utils/generate_summary.py`

**Verify:** All pilot summaries regenerated

**CHECKPOINT 5:** Analysis pipeline complete and tested

---

## PHASE 6: Full Experiment Suite (4-6 hours)
**Goal: Systematic testing with all models and configurations**

### Step 6.1: OpenHermes Model Acquisition
**What:** Download and prepare MoE model

**Tasks:**
```bash
# Find correct model
# OpenHermes-2.5-Mistral-7B or larger MoE variant?
# Verify model size and requirements

# Download
# Place on experimental SSD
# Verify model loads
```

**Output:** Model ready, size documented

### Step 6.2: Experiment Matrix Design
**What:** Define all experiment combinations

**Matrix:**
```
Models: [llama-2-7b, openhermes-mistral]
Memory Configs: [100% RAM, 50% RAM, 0% RAM]
Swappiness: [0, 60, 100]
llama.cpp Modes: [mmap, no_mmap]
Context Lengths: [512, 2048] (for KV cache sizing)
Tokens: [100, 500]
```

**Total experiments:** ~48 combinations
**Strategy:** Start with llama-2-7b full matrix, then openhermes subset

**Output:** `EXPERIMENT_MATRIX.md`

### Step 6.3: Automated Experiment Runner
**What:** Orchestrate all experiments reliably

**Features:**
- Sequential execution with cleanup
- Resume capability (if crash)
- Real-time progress tracking
- Automatic result validation

**Output:** Updated `experiments.py`

**Verify:** Dry-run mode works correctly

### Step 6.4: Execute Experiment Suite
**What:** Run all experiments (may take 8-24 hours)

**Process:**
1. Start with llama-2-7b matrix (24 experiments)
2. Review results
3. Adjust parameters if needed
4. Run openhermes matrix (24 experiments)

**Monitoring:**
- Check disk space (traces can be large)
- Monitor for errors
- Verify each result directory

**Output:** Complete `results/` directory

**CHECKPOINT 6:** All experiments complete

---

## PHASE 7: Analysis & Reporting (2-3 hours)
**Goal: Extract insights for thesis**

### Step 7.1: Comparative Analysis
**What:** Compare all experiments systematically

**Create:** `analysis/compare_all.py`
- Load all experiment results
- Generate comparison tables
- Find patterns across configurations
- Identify optimal settings

**Output:** 
- `analysis/COMPARISON_TABLE.md`
- `analysis/insights.md`

### Step 7.2: Answer Research Questions
**What:** Map results to thesis questions

**Questions:**
1. Is parameter access sequential or random?
2. Does access pattern differ by model architecture?
3. Impact of swappiness on performance?
4. Can we saturate SSD bandwidth?
5. KV cache vs parameter swap behavior?

**Output:** `RESEARCH_FINDINGS.md`

### Step 7.3: Visualization
**What:** Create plots for thesis

**Plots:**
- Tokens/sec vs memory configuration
- Access pattern heatmaps
- Sequential % by experiment
- Bandwidth utilization over time

**Output:** `plots/` directory

**CHECKPOINT 7:** Ready for thesis writing

---

## Success Criteria

### Phase 0: ✓ Know our hardware completely
### Phase 1: ✓ Can explain concepts to Gabriel
### Phase 2: ✓ Have clear measurement plan
### Phase 3: ✓ Tools work reliably
### Phase 4: ✓ Pilot data makes sense
### Phase 5: ✓ Analysis is reproducible
### Phase 6: ✓ Full dataset collected
### Phase 7: ✓ Insights extracted

---

## Risk Mitigation

**Risk 1:** Blktrace overhead too high
- **Mitigation:** Measure overhead first, adjust sampling rate

**Risk 2:** Traces too large (fill disk)
- **Mitigation:** Estimate size from pilots, clean between experiments

**Risk 3:** Results don't match expectations
- **Mitigation:** Validate methodology with known patterns first

**Risk 4:** Server instability
- **Mitigation:** Save intermediate results, checkpoint often

---

## Estimated Timeline

| Phase | Time | Critical Path |
|-------|------|---------------|
| Phase 0 | 2 hours | Yes - blocks everything |
| Phase 1 | 3 hours | Yes - need understanding |
| Phase 2 | 2 hours | Yes - wrong plan = wrong data |
| Phase 3 | 3 hours | Yes - broken tools = no data |
| Phase 4 | 4 hours | Yes - validates methodology |
| Phase 5 | 3 hours | Partial - can parallelize |
| Phase 6 | 12-24 hours | Yes - data collection |
| Phase 7 | 3 hours | Yes - thesis deliverable |

**Total:** 32-44 hours of active work + 12-24 hours compute time

---

## Next Immediate Action

**START HERE:**
```bash
# Phase 0, Step 0.1: Map Physical Disks
ssh hiwi-02
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL > ~/hardware_inventory.txt
df -h >> ~/hardware_inventory.txt
cat /proc/swaps >> ~/hardware_inventory.txt
```

**Then:** Review output together and understand each line before proceeding.
