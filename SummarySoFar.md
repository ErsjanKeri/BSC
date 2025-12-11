# Bachelor Thesis Progress Summary
**Student:** Roland (Ersjan Këri)  
**Supervisor:** Gabriel Haas (Viktor Leis Department, TUM)  
**Topic:** Optimizing DeepSeek Inference on SSDs - Addressing Parameter Loading Bottlenecks

**Current Status:** Achieved 10 GB/s throughput, targeting 80 GB/s SSD bandwidth saturation

---

## 1. Learning About Transformers and LLM Architecture

### 1.1 Fundamental Transformer Architecture

**What We Learned:**
- Transformers consist of stacked layers, each containing:
  - **Multi-Head Self-Attention (MHSA):** Uses Q, K, V parameter matrices for attention computation
  - **Feed-Forward Network (FFN):** Typically contains ~67% of total parameters, expands to 4× hidden dimension
- **Parameter organization matters for inference:**
  - Parameters are processed layer-by-layer sequentially
  - Each layer requires loading its weights from storage → compute → load next layer
  - This sequential access pattern is critical for understanding SSD optimization

**Key Insight for Thesis:**
The sequential layer processing creates a specific memory access pattern that directly impacts SSD utilization. Unlike training (which can batch-process), inference must proceed layer-by-layer for each token.

### 1.2 DeepSeek-V3 Specific Architecture

**Specifications Learned:**
- **Vocabulary size:** 128,000 tokens
- **Hidden dimension:** 7,168
- **Total parameters:** 671B (37B activated per token due to MoE)
- **Context window:** Up to 32,768 tokens
- **Tokenizer:** Byte-level BPE

**Embedding Process:**
```
Input text → Tokenizer → Token IDs [t₁, t₂, ..., tₙ]
Embedding Matrix E ∈ ℝ^(128000 × 7168)
Each token tᵢ → embedding vector eᵢ ∈ ℝ^7168
Result: Input matrix (n tokens × 7168 dimensions)
```

**Important Understanding:**
- One token per **row**, embedding dimensions as **columns**
- This organization enables efficient matrix operations
- The large embedding matrix (128K × 7168 = ~3.5GB in FP16) is one of the first components that must be loaded

### 1.3 Multi-Head Latent Attention (MLA) - DeepSeek's Innovation

**Critical Architecture Difference:**
DeepSeek uses MLA instead of standard Multi-Head Attention (MHA), which fundamentally changes memory and computation patterns.

**Standard MHA KV Cache per Token:**
```
128 heads × 56 dims/head × 2 (K,V) = 14,336 numbers per token
For 4K context: 14,336 × 4,000 = 57.3M numbers
```

**DeepSeek MLA KV Cache per Token:**
```
512 (compressed KV) + 64 (shared RoPE) = 576 numbers per token
For 4K context: 576 × 4,000 = 2.3M numbers
Memory reduction: 24.9× less!
```

**MLA Process:**
1. **Down-projection:** Compress keys/values into latent vector c_KV_t (dimension d_c = 512)
2. **Store compressed representation only**
3. **Matrix absorption trick:** Pre-combine decompression and attention matrices
4. **Direct computation:** Work in latent space without explicit decompression

**Why This Matters for Our Thesis:**
- **Critical concern:** llama.cpp may not properly implement MLA's latent space operations
- **Potential problem:** If llama.cpp treats MLA like standard MHA, it will:
  - Unnecessarily decompress cached vectors
  - Miss the computational efficiency gains
  - Fail to exploit the reduced memory bandwidth requirements
- **This could explain our 10 GB/s vs 80 GB/s gap**

### 1.4 Mixture of Experts (MoE) Architecture

**How MoE Works:**
- DeepSeek uses **DeepSeekMoE** with sparse expert activation
- **Example (gpt-oss-20b model we tested):**
  - Total parameters: 21B
  - **Active parameters per token: Only 3.6B** (due to expert sparsity)
  - Multiple expert networks, only ~2 activated per token

**Mixed Precision in MoE:**
- Expert layers: MXFP4 (4-bit mixed floating point)
- Attention layers: F16 (16-bit floating point)
- Embedding layers: F16/F32

**Implications for Parameter Loading:**
- **Not all parameters are accessed uniformly**
- Expert selection creates **sparse access patterns**
- Some experts may be "hot" (frequently used), others "cold"
- **This non-uniform access is poorly captured by dummy models in research papers**

**Critical Question for Our Work:**
Does llama.cpp properly handle:
1. Sparse expert loading (only loading activated experts)?
2. Expert caching strategies (keeping hot experts in memory)?
3. MoE-specific prefetching patterns?

---

## 2. Learning About Operating Systems and Memory Management

### 2.1 Memory Hierarchy and Storage Stack

**Complete Hierarchy (Latency and Bandwidth):**
```
CPU Registers:     <0.3 ns          (infinite bandwidth - on-chip)
L1 Cache:          ~1 ns            (~1 TB/s per core)
L2 Cache:          ~3-4 ns          (~500 GB/s per core)
L3 Cache:          ~12-20 ns        (~200 GB/s shared)
RAM (DRAM):        ~50-100 ns       (50-100 GB/s)
NVMe SSD:          ~10-100 μs       (3-80 GB/s)
SATA SSD:          ~50-100 μs       (~550 MB/s)
HDD:               ~5-10 ms         (~100-200 MB/s)
```

**Key Learning:**
- **Each jump down the hierarchy = 10-100× latency increase**
- **Bandwidth also decreases dramatically**
- **Our thesis focuses on the RAM ↔ SSD boundary** (the largest performance gap for large model inference)

### 2.2 Virtual Memory and Page Tables

**Virtual Memory Concepts:**
- Each process has its own **virtual address space** (up to 256 TB on x86-64)
- **MMU (Memory Management Unit)** translates virtual → physical addresses
- **TLB (Translation Lookaside Buffer):** Caches recent translations
- **Page tables:** 4-level hierarchy (PML4 → PDPT → PD → PT)

**Page Table Structure:**
```
Level 1 (PML4):  512 entries, each maps 512 GB
Level 2 (PDPT):  512 entries, each maps 1 GB
Level 3 (PD):    512 entries, each maps 2 MB
Level 4 (PT):    512 entries, each maps 4 KB (page size)
```

**Why Multi-Level Tables Are Efficient:**
- **Sparse allocation:** Only allocate page table entries for actually-used memory regions
- For a process using 100 GB: ~195 KB of page table metadata (not 200+ GB for flat tables)
- Page table pages themselves are stored in **RAM** and must stay pinned

**Demand Paging:**
- Physical pages allocated **only when accessed**
- Page faults trigger allocation
- Swapping to disk when RAM is full

**Relevance to Our Experiment:**
We plan to use memory locking with intentional swapping to observe **page access patterns** during model inference. Understanding demand paging is essential for this experiment.

### 2.3 System Calls and Kernel Architecture

**User Space vs Kernel Space:**
- **User space:** Where applications run (limited privileges)
- **Kernel space:** Privileged mode with hardware access
- **System calls:** Interface between user space and kernel

**Syscall Overhead:**
```
Total: 300-600 CPU cycles minimum
Components:
- Context switch: ~50-100 cycles
- Pipeline barrier: ~50-100 cycles
- TLB flush (KPTI mitigation): ~100-200 cycles
- Privilege level change: ~50-100 cycles
- Register saving/restoring: ~50-100 cycles
```

**Why This Matters:**
- Frequent small I/O operations incur syscall overhead repeatedly
- **Batching and buffering** can amortize syscall costs
- **mmap()** can reduce syscalls by mapping file directly to address space
- **io_uring** provides async I/O with minimal syscalls

### 2.4 Linux Storage Stack (7 Layers)

**Complete Data Flow:**
```
1. User Application
   ↓ (system calls: open, read, write, mmap)
2. Virtual File System (VFS) - abstraction layer
   ↓ (VFS operations: vfs_read, vfs_write)
3. File System Layer (ext4, XFS, etc.)
   ↓ (translates file offsets → block numbers)
4. Page Cache / Buffer Cache
   ↓ (caching layer in RAM)
5. Block Layer
   ↓ (submit_bio, I/O scheduler)
6. Device Driver Layer (NVMe driver)
   ↓ (nvme_queue_rq → hardware queues)
7. SSD Hardware
   ↓ (Flash Translation Layer, NAND flash, DMA)
```

**Critical Interfaces:**
- **read()/write():** Copy data from kernel space → user space (requires buffer copy)
- **mmap():** Map file into virtual address space (zero-copy possible, but page faults on first access)
- **submit_bio():** Submit I/O request to block layer
- **DMA:** Device writes directly to RAM, bypassing CPU

**Block vs Page Concepts:**
- **Filesystem block:** Typically 4 KiB (fs allocation unit)
- **Memory page:** 4 KiB (virtual memory unit)
- **I/O request size:** Variable (often 128 KiB or larger for sequential reads)
- **Block layer operations:** Work with arbitrary-sized I/O requests

**Why Understanding This Matters:**
- **Page cache behavior** determines whether reads hit RAM or SSD
- **I/O scheduler** can merge/reorder requests for better SSD utilization
- **Request size** dramatically affects throughput (larger = better for sequential)
- Each layer adds overhead and potential optimization opportunities

### 2.5 mmap() vs read() - Critical Distinction

**read() behavior:**
```
1. Syscall into kernel
2. Kernel reads from page cache (or SSD if not cached)
3. Kernel **copies data** from page cache → user buffer
4. Return to user space
```

**mmap() behavior:**
```
1. Syscall to map file into virtual address space
2. Returns immediately (no data transfer yet!)
3. First access triggers **page fault**
4. Kernel loads page into page cache
5. MMU maps virtual address → physical page in cache
6. Future accesses are just memory reads (no syscalls!)
```

**Trade-offs:**
- **mmap():** Zero-copy, but page faults on first access can be expensive
- **read():** Explicit copy, but predictable behavior and easier prefetching
- **mmap() with madvise():** Can provide hints for sequential/random access patterns

**For Our Thesis:**
Understanding mmap() is critical because we suspect llama.cpp uses it for model loading. We need to trace page fault patterns to see if access is sequential or random.

---

## 3. The CHEOPS Paper: I/O Characteristics of LLM Inference

### 3.1 Paper Overview and Methodology

**CHEOPS Paper Goal:**
Characterize I/O patterns when offloading LLM model weights and KV cache to NVMe SSDs

**Experimental Setup:**
- **Platform:** Google Cloud Platform (GCP) g2-standard-32 instances
- **Problem:** GCP uses slow network-attached storage
- **Solution:** NVMeVirt - emulates NVMe SSD using **RAM** with configurable latency
- **Achieved latency:** 9.3 μs (similar to real NVMe)
- **Frameworks tested:** DeepSpeed-MII, FlexGen

**Why NVMeVirt in RAM:**
- Provides **controllable, consistent** environment
- Eliminates real SSD variability
- Allows precise I/O tracing without GCP storage bottlenecks

### 3.2 Key Findings from CHEOPS

**Performance Results:**
```
Hardware capability: 16.9 GiB/s (real NVMe SSD bandwidth)

Framework Performance:
- DeepSpeed: 4.9 GiB/s (29% of hardware capacity)
- FlexGen:  2.6 GiB/s (15% of hardware capacity)

Average I/O request size:
- DeepSpeed: 128 KiB
- FlexGen:  32 KiB
```

**Critical Observation:**
Even on hardware capable of 16.9 GiB/s, existing frameworks achieve only 2.6-4.9 GiB/s. This suggests **software bottlenecks**, not hardware limitations.

### 3.3 Gabriel's Skepticism About CHEOPS

**Gabriel's + ME Main Concerns:**

**1. Model Access Pattern**
- **Question:** Do real models (especially MoE) have different access patterns?
- **Hypothesis:** Expert sparsity might create "hot" vs "cold" parameter regions
- Real attention patterns might access KV cache non-uniformly
- The request sizes were so large it should not play a role?

**2. GPU Memory Transfer Inefficiency**
- CHEOPS measured GPU offloading: GPU → RAM → SSD
- This **double-step transfer** is inherently slower than CPU offloading: RAM → SSD
- **Why?** GPU memory must first copy to RAM before SSD can DMA it
- CPU offloading would be faster but wasn't thoroughly tested

**3. Access Pattern Analysis**
- Paper shows uniform I/O operations at block layer
- **But:** Is this representative of real model inference? is this not sequential? what the heck does uniform mean then?  (MAIN POINT)
- MoE expert selection might create **random access patterns**
- KV cache access patterns vary by token position (CDF graph shows non-uniformity)

**4. Methodology Concerns**
- Using RAM-based NVMeVirt might not capture:
  - Real SSD characteristics (wear leveling, garbage collection) X -> actually Gabriel mentioned that these page sizes were so large randomness would play no role, if they were small then perhaps 
  - Channel parallelism effects
  - Multi-queue scheduling behavior
  - Approach what they are doing is weird 


### 3.4 What CHEOPS Did Well

**Valuable Contributions:**
1. **I/O tracing methodology:** Using blktrace/bpftrace to observe block layer
2. **KV cache access patterns:** Demonstrated non-uniform access (input tokens accessed 256×, output tokens decrease linearly)
3. **Batch size effects:** Showed how concurrent requests affect I/O
4. **Framework comparison baseline:** Provides quantitative data on DeepSpeed/FlexGen performance

**Important for Our Work:**
- Their **tracing methodology** is directly applicable to our experiments
- The **128 KiB request size** finding suggests optimal buffering strategies
- Their **low utilization** validates our hypothesis that software is the bottleneck

---

## 4. Our Experimental Server Setup

### 4.1 Server Specifications (cli-hiwi-02.dis.cit.tum.de)

**CPU:**
- AMD Ryzen 7 7700X
- 16 cores (8 physical cores × 2 threads)
- Base clock: 4.5 GHz, boost up to 5.4 GHz

**Memory:**
- 30 GiB RAM total
- ~27 GiB available after OS overhead

**Storage:**

**Primary (System) Drive:**
```
Device: /dev/nvme0n1 (Samsung 980 PRO)
Capacity: 1TB
Model: Samsung SSD 980 PRO 1TB
Interface: NVMe PCIe 4.0 x4
Usage: Operating system (Ubuntu), root filesystem
Mounted: /
```

**Secondary (Experimental) Drive:**
```
Device: /dev/nvme1n1 (Western Digital)
Capacity: 960GB (894 GiB usable)
Model: WUS4BB096D7P3E3
Interface: NVMe PCIe
Bandwidth capacity: ~80 GB/s (claimed)
Usage: Model storage, experimental data
Mount point: /blk/w0
Current status: No filesystem (raw block device)
```

**Network:**
- Access via proxy jump through i13vm10.in.tum.de
- Requires TUM VPN connection

**Graphics:**
- No discrete GPU
- Integrated AMD graphics only
- **Limitation:** Cannot test GPU offloading, only CPU inference

### 4.2 Initial Experiments on Server

**Experiment 1: Llama-2 7GB Model (RAM-only)**
```
Command: ./llama-cli -m Llama-2-7b-hf-Q4_K_M.gguf --prompt "Write a poem"
Model size: 7GB
Memory available: 27 GiB
Result: Model loaded entirely in RAM
Performance: ~13.5 tokens/second
Observation: Baseline performance when no SSD access needed
```

**Experiment 2: gpt-oss-20b Model (12.83GB)**
```
Command: ./llama-cli -m gpt-oss-20b-q4.gguf --prompt "Explain MoE"
Model size: 12.83GB
Parameters: 21B total, 3.6B activated per token
Architecture: Mixture of Experts (MoE)
Result: Still fits in available RAM
Performance: TBD (not measured yet)
Key finding: Model uses mixed precision (MXFP4 for experts, F16 for attention)
```

**Memory Usage Observations:**
```
$ free -h
              total   used   free   shared  buff/cache  available
Mem:           30Gi   XXGi   XXGi      XXMi       XXGi      27Gi

After loading model:
- "used" increases by model size
- buff/cache remains relatively stable
- mmap'd files show up in buff/cache, not "used"
```

**Key Learning:**
Both test models fit comfortably in RAM, so we haven't yet forced SSD parameter loading. Need to test larger models or artificially limit available memory to trigger SSD access patterns.

### 4.3 Network Configuration and SSH Setup

**Connection Path:**
```
Your Machine
    ↓ (eduVPN - TUM network)
Proxy Server: i13vm10.in.tum.de
    ↓ (ProxyJump)
Target Server: cli-hiwi-02.dis.cit.tum.de
```

**SSH Config:**
```
Host i13vm10
    HostName i13vm10.in.tum.de
    User keri
    IdentityFile ~/.ssh/id_tum

Host hiwi-02
    HostName cli-hiwi-02.dis.cit.tum.de
    User keri
    ProxyJump i13vm10
    IdentityFile ~/.ssh/id_tum
```

**Access Method:**
```bash
# Connect via proxy jump
ssh hiwi-02

# Direct scp through proxy
scp -o ProxyJump=i13vm10 file.txt hiwi-02:/path/
```

---

## 5. Current Understanding: Key Technical Concepts

### 5.1 Why Existing Frameworks Underperform

**Identified Bottlenecks:**

**1. Synchronous I/O Operations**
- Frameworks use blocking read() calls
- CPU idle while waiting for SSD
- No pipelining of I/O and computation

**2. Lack of Prefetching**
- Next layer's parameters not loaded until current layer finishes
- SSD idle during computation phase
- Miss opportunity for I/O-computation overlap

**3. Inefficient I/O Interfaces**
- Traditional read()/write() syscalls
- Not using modern async I/O (io_uring)
- Missing zero-copy opportunities

**4. Small I/O Request Sizes**
- FlexGen: 32 KiB requests
- DeepSpeed: 128 KiB requests
- **Larger sequential requests** could saturate SSD bandwidth better

**5. Suboptimal Buffering**
- No intelligent buffer management
- Page cache thrashing possible
- No consideration of access patterns

### 5.2 Potential Optimization Strategies

**Buffer Manager Approach:**
1. **Prediction:** Prefetch next layers before needed
2. **Double buffering:** Load layer N+1 while computing layer N
3. **LRU caching:** Keep frequently-accessed parameters in RAM
4. **Expert-aware caching:** For MoE, cache "hot" experts

**I/O Optimization:**
1. **Async I/O:** Use io_uring for non-blocking reads
2. **Large requests:** Issue 128 KiB - 1 MiB sequential reads
3. **Direct I/O:** Bypass page cache for large streaming reads
4. **mmap with madvise:** Provide sequential/random hints to kernel

**Architecture-Specific:**
1. **MLA support:** Implement proper latent space operations
2. **MoE optimizations:** Sparse expert loading, expert caching
3. **KV cache management:** Exploit non-uniform access patterns

### 5.3 Sparsity as Optimization Opportunity

**What is Sparsity:**
- Many model weights or activations are zero (or near-zero)
- Can skip computation and memory transfer for zero values
- Especially prominent in:
  - MoE architectures (inactive experts)
  - Pruned models
  - Quantized models with threshold masking

**Potential Gains:**
- **2-5× speedup** possible by exploiting sparsity
- **Beyond I/O optimization** - computational efficiency
- **Requires:** Sparse matrix formats, sparse kernels

**Unexploited in Current Frameworks:**
- CHEOPS paper notes this as "further opportunity"
- llama.cpp likely doesn't exploit sparsity
- DeepSpeed/FlexGen treat all parameters uniformly

---

## 6. Planned Next Steps: Memory Locking Experiment

### 6.1 Experimental Goal

**Primary Question:**
What are the **page access patterns** when loading model parameters from SSD?
- **Uniform access:** All pages equally likely → random access optimization
- **Sequential access:** Pages accessed in order → sequential prefetching optimization

**Why This Matters:**
- Determines optimal buffer management strategy
- Informs prefetching algorithm design
- Validates or refutes CHEOPS findings about sequential I/O

### 6.2 Experimental Design

**Setup:**
1. **Select large model** that exceeds available RAM:
   - Option A: Use full DeepSeek model (if available)
   - Option B: Artificially limit available memory with cgroup
   - Goal: Force parameters to be loaded from SSD

2. **Memory locking:**
   ```bash
   # Use mlock() or mlockall() to pin critical pages in RAM
   # Force other pages to be swapped to disk
   # Simulate extreme memory pressure
   ```

3. **Tracing tools:**
   ```bash
   # Trace page faults
   sudo bpftrace -e 'tracepoint:exceptions:page_fault_user { ... }'
   
   # Trace block I/O
   sudo blktrace /dev/nvme1n1
   
   # Monitor I/O bandwidth
   iostat -x 1
   
   # Trace mmap operations
   strace -e mmap,mprotect,madvise ./llama-cli ...
   ```

4. **Run inference:**
   ```bash
   ./llama-cli -m large_model.gguf --prompt "test" -n 100
   ```

### 6.3 Data Collection

**Metrics to Measure:**
- **Page fault frequency:** How often are pages loaded from SSD?
- **Page fault addresses:** Are they sequential or random?
- **I/O request sizes:** What block sizes does block layer see?
- **I/O throughput:** Actual GB/s achieved during inference
- **I/O latency distribution:** min/median/max/p99 latencies
- **Temporal patterns:** I/O bursts vs steady streaming?

**Expected Patterns:**

**If Sequential:**
```
Page faults: 0x1000, 0x2000, 0x3000, 0x4000...
I/O requests: Large sequential reads (128 KiB - 1 MiB)
Optimization: Sequential prefetching, large buffers
```

**If Random:**
```
Page faults: 0x1000, 0x5000, 0x2000, 0x9000...
I/O requests: Small scattered reads (4 KiB - 32 KiB)
Optimization: Cache hot pages, predictive loading based on model architecture
```

**If Hybrid (MoE-specific):**
```
Layer weights: Sequential (loaded once per layer)
Expert weights: Random (depends on routing decisions)
Optimization: Different strategies for different parameter types
```

### 6.4 Analysis Plan

**Step 1: Visualize Access Patterns**
- Plot page fault addresses over time
- Create heatmap of accessed memory regions
- Identify hot/cold regions

**Step 2: Compare to Model Architecture**
- Map memory addresses → model components (embeddings, layers, experts)
- Correlate access patterns with inference stages
- Validate if experts are accessed sparsely

**Step 3: Identify Bottlenecks**
- Measure actual SSD utilization %
- Calculate wasted bandwidth (compare to 80 GB/s capacity)
- Identify I/O stalls (gaps between requests)

**Step 4: Design Optimizations**
- Propose buffer management strategy based on patterns
- Estimate speedup from prefetching/caching
- Prioritize optimization opportunities

---

## 7. Open Questions and Concerns

### 7.1 llama.cpp Implementation Uncertainties

**Critical Unknowns:**

**1. MLA (Multi-Head Latent Attention) Support**
- **Question:** Does llama.cpp implement DeepSeek's latent space compression?
- **Risk:** If not, it may decompress unnecessarily → wasted bandwidth
- **How to verify:**
  - Examine llama.cpp source code for MLA-specific kernels
  - Compare memory bandwidth with/without MLA model
  - Test if KV cache size matches MLA predictions (576 vs 14,336 per token)

**2. MoE (Mixture of Experts) Handling**
- **Question:** Does llama.cpp load all experts or only activated ones?
- **Risk:** Loading all experts wastes 10× more bandwidth than needed
- **How to verify:**
  - Monitor I/O during inference with MoE model
  - Check if I/O size matches total parameters or only activated ones
  - Examine llama.cpp MoE implementation

**3. Parameter Loading Strategy**
- **Question:** Does llama.cpp use mmap, read(), or custom I/O?
- **Current hypothesis:** Likely uses mmap for simplicity
- **How to verify:** strace the llama-cli process during model loading

**4. KV Cache Management**
- **Question:** How does llama.cpp handle KV cache with limited memory?
- **Scenarios:**
  - Offload old tokens to SSD?
  - Recompute on demand?
  - Use paged attention-style blocks?
- **How to verify:** Test long context inference with memory pressure

### 7.2 Experimental Challenges

**Challenge 1: No GPU**
- Cannot test GPU offloading scenarios
- Cannot compare CPU vs GPU parameter loading
- Limited to CPU inference only

**Challenge 2: Model Availability**
- Full DeepSeek-V3 (671B) too large for our setup
- Need appropriately-sized model (30-100 GB range)
- Should be MoE-based to match thesis focus

**Challenge 3: Measurement Accuracy**
- Page faults during first inference ≠ steady-state behavior
- Need multiple runs, warm vs cold cache comparisons
- Tracing overhead may affect performance

**Challenge 4: Isolating Variables**
- Many potential bottlenecks (CPU, RAM bandwidth, SSD, software)
- Need controlled experiments to isolate I/O effects
- Requires comparison: RAM-only vs SSD-based inference

### 7.3 Theoretical Questions to Resolve

**1. Optimal I/O Request Size**
- CHEOPS found 128 KiB effective
- But is this optimal for 80 GB/s SSD?
- Larger requests (1-4 MiB) might saturate better

**2. Prefetching Feasibility**
- Can we predict next layers accurately enough?
- What about MoE expert activation (depends on input content)?
- Trade-off: aggressive prefetch → waste bandwidth on wrong experts

**3. Buffer Management Strategy**
- How much RAM to reserve for buffering?
- LRU vs. model-aware caching?
- Static allocation vs. dynamic adjustment?

**4. Async I/O Effectiveness**
- Would io_uring really help llama.cpp?
- Computation time vs. I/O time overlap?
- Is the bottleneck I/O wait or computation?

---

## 8. Key Insights and Takeaways

### 8.1 Major Findings So Far

**1. Software is the Bottleneck, Not Hardware**
- CHEOPS: 16.9 GiB/s capable → only 2.6-4.9 GiB/s achieved
- Our setup: 80 GB/s capable → only 10 GB/s observed
- **Conclusion:** Massive optimization opportunity in software

**2. Architecture Matters**
- DeepSeek's MLA reduces KV cache by 24.9×
- MoE sparsity reduces active parameters by ~5-10×
- **If llama.cpp doesn't exploit these, it wastes bandwidth**

**3. Access Patterns are Non-Uniform**
- KV cache: input tokens accessed 256×, output tokens decay
- MoE: only 2/N experts activated per token
- **Uniform caching strategies will perform suboptimally**

**4. I/O and Computation Can Overlap**
- Current sequential approach: load → compute → load → compute
- Potential: load layer N+1 **while** computing layer N
- **Double buffering could provide significant speedup**

### 8.2 What Makes This Thesis Valuable

**1. Real-World Focus**
- Not just theoretical analysis
- Actual deployment on TUM servers
- Practical optimization of widely-used tool (llama.cpp)

**2. Novel Problem Scope**
- MLA and MoE together create unique challenges
- Existing research (CHEOPS) used dummy models
- We test real trained models with real access patterns

**3. System-Level Optimization**
- Spans multiple layers: application, OS, storage hardware
- Requires understanding entire stack
- Potential for 5-10× speedup (10 GB/s → 50-80 GB/s)

**4. Bridge Research and Practice**
- Academic rigor (measurements, analysis)
- Practical impact (llama.cpp improvements)
- Can contribute back to open-source community

### 8.3 What We Need to Do Next

**Immediate (Next 1-2 Weeks):**
1. ✅ Understand transformer architecture → **DONE**
2. ✅ Learn OS memory management → **DONE**
3. ✅ Set up server access and environment → **DONE**
4. ⏳ Run memory locking experiment → **IN PROGRESS**
5. ⏳ Trace page access patterns → **NEXT**
6. ⏳ Verify llama.cpp MLA/MoE handling → **NEXT**

**Medium-term (Weeks 3-6):**
1. Analyze access pattern data
2. Design buffer management strategy
3. Prototype optimization (prefetching/caching)
4. Benchmark improved implementation

**Long-term (Weeks 7-12):**
1. Full implementation and testing
2. Comparison with baseline llama.cpp
3. Write thesis
4. Prepare presentation

---

## 9. Technical Context: Related Concepts

### 9.1 PagedAttention and vLLM

**What We Learned:**
- vLLM uses OS-style virtual memory for KV cache management
- Block-level allocation (not monolithic arrays)
- Enables copy-on-write for beam search, parallel sampling
- **But:** vLLM doesn't support SSD offloading (CPU memory only)

**Relevance:**
- Shows modern techniques for memory management
- Inspiration for buffer manager design
- But architecturally different from our parameter loading problem

### 9.2 Attention Mechanism Variants

**Standard Multi-Head Attention (MHA):**
- Q, K, V ∈ ℝ^(n_h × d_h) per token
- KV cache: 2 × n_h × d_h per token
- Example: 128 heads × 56 dims × 2 = 14,336 per token

**Grouped-Query Attention (GQA):**
- Share K, V across groups of Q heads
- Reduces KV cache by group factor (e.g., 8× if 8 heads per group)

**Multi-Query Attention (MQA):**
- Single K, V shared across all Q heads
- Minimal KV cache: 2 × d_h per token
- Used in models like Falcon, PaLM

**Multi-Head Latent Attention (MLA) - DeepSeek:**
- Compress K, V into latent vector c_KV
- Ultra-low KV cache: d_c + d_R_h per token (e.g., 576)
- Most aggressive compression, highest complexity

**Why This Matters:**
Understanding different attention mechanisms helps us:
- Identify whether llama.cpp implements MLA correctly
- Estimate expected memory bandwidth requirements
- Design appropriate caching strategies

### 9.3 Quantization and Mixed Precision

**Common Formats:**
- **FP32:** 32-bit floating point (full precision)
- **FP16:** 16-bit floating point (half precision)
- **INT8:** 8-bit integer quantization
- **INT4/MXFP4:** 4-bit formats for extreme compression

**DeepSeek's Mixed Precision:**
- Attention: FP16
- Embeddings: FP16/FP32
- Experts (MoE): MXFP4

**Implications:**
- Parameter size varies by layer
- I/O bandwidth requirements depend on quantization
- Need to account for mixed precision in buffer sizing

---

## 10. Summary: Where We Stand

### 10.1 Knowledge Acquired

**✅ Transformer Architecture:**
- Understand layer-by-layer processing
- Know DeepSeek-V3 specifications (671B, MLA, MoE)
- Grasp MLA's latent space compression (24.9× reduction)
- Understand MoE sparse activation patterns

**✅ Operating Systems:**
- Virtual memory, page tables, MMU/TLB
- System calls, user/kernel space
- Storage stack (7 layers, APIs at each level)
- mmap vs read, page cache behavior

**✅ I/O and Storage:**
- NVMe architecture and protocols
- Sequential vs random access trade-offs
- I/O request sizes and their impact
- Modern async I/O (io_uring)

**✅ Research Context:**
- CHEOPS paper findings and methodology
- Existing framework limitations (DeepSpeed, FlexGen)
- Gabriel's concerns about dummy models vs real models

**✅ Practical Setup:**
- Server access configured (proxy jump)
- llama.cpp built and tested
- Initial experiments with 7B and 20B models
- Environment ready for larger-scale testing

### 10.2 Critical Questions to Answer

**Primary Research Question:**
**How can we optimize llama.cpp parameter loading to saturate 80 GB/s SSD bandwidth for DeepSeek inference?**

**Sub-questions:**
1. **Access patterns:** Sequential, random, or hybrid?
2. **llama.cpp gaps:** Does it properly support MLA and MoE?
3. **Buffer management:** What strategy maximizes throughput?
4. **Prefetching:** Can we predict and overlap I/O with computation?
5. **Real vs dummy models:** Do trained weights have different access patterns?

### 10.3 Hypothesis

**Our Working Hypothesis:**
llama.cpp achieves only 10 GB/s (vs. 80 GB/s capacity) because:

1. **Synchronous I/O:** No overlapping of I/O with computation
2. **No prefetching:** Waits for each layer to finish before loading next
3. **Suboptimal MLA handling:** Doesn't exploit latent space compression
4. **Uniform parameter loading:** Treats all parameters equally, ignores MoE sparsity
5. **Small I/O requests:** Not issuing large enough sequential reads

**If We Optimize:**
- **Async I/O + prefetching:** Could achieve 2-3× speedup
- **MLA-aware operations:** Another 2× from reduced bandwidth needs
- **MoE-aware loading:** Another 2× from loading only active experts
- **Combined:** Potential 8-12× speedup → 80-120 GB/s effective bandwidth

### 10.4 Next Immediate Action

**This Week:**
1. **Design memory locking experiment**
   - Choose appropriate model size
   - Set up tracing tools (bpftrace, blktrace, iostat)
   - Prepare scripts for data collection

2. **Run initial trace**
   - Execute inference with memory pressure
   - Collect page fault data
   - Measure I/O patterns

3. **Analyze results**
   - Determine if access is sequential or random
   - Identify hot/cold memory regions
   - Correlate with model architecture

4. **Verify llama.cpp behavior**
   - Examine source code for MLA/MoE handling
   - Use strace to understand I/O operations
   - Compare observed behavior with expected patterns

**Deliverable:**
Clear characterization of access patterns with data to support buffer manager design decisions.

---

## Appendix: Quick Reference

### Server Connection
```bash
ssh hiwi-02  # After configuring ProxyJump in ~/.ssh/config
```

### Useful Commands
```bash
# Check memory
free -h

# Check storage
lsblk
df -h

# Monitor I/O
iostat -x 1

# Trace I/O
sudo blktrace /dev/nvme1n1

# Run llama.cpp
./llama-cli -m model.gguf --prompt "test"

# Check processes
htop

# System calls
strace -e open,read,mmap ./llama-cli ...
```

### Key Files
- Models: `/home/keri/models/`
- llama.cpp: `/home/keri/llama.cpp/`
- Experimental drive: `/blk/w0/`

### Important Numbers
- Available RAM: 27 GiB
- SSD capacity (target): 80 GB/s
- Current throughput: 10 GB/s
- Target improvement: 8× (to 80 GB/s)