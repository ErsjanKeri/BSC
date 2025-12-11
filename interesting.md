

KV cache/activations: anonymous. Without swap, they must stay in RAM; if RAM is full you hit OOM. With swap, they can be paged out, but performance tanks. ML frameworks sometimes implement explicit offload to SSD (e.g., paged attention, CPU/NVMe offload) using their own files/mappings; the kernel’s generic swap is not optimized for high-performance KV offload.


Llama.cpp sometimes allocates not in buff/cache, has lock on the memory for small models 


Our machine specs: 

CPU: AMD Ryzen 7 7700X (8 cores / 16 threads) shown in lscpu; clocks 0.4–5.57 GHz; single NUMA node.

RAM: 30 Gi total, ~27 Gi free, no swap configured (free -h).

GPU: no NVIDIA driver (nvidia-smi absent); lspci shows an AMD Raphael iGPU, so there’s no dedicated VRAM—graphics would share system RAM. The current llama.cpp build only ships CPU libs (libggml-cpu.so), so llama-cli will run on CPU



Way to grok llama.cpp

Start with the docs: read README.md, then docs/BUILD.md, docs/run.md, and docs/gguf.md to nail formats, build flags, and runtime options.
Learn the mental layers: CLI frontends (llama-cli, llama-run), model loader (GGUF parsing, tensor init), compute graph build (attention/FFN), backend kernels (ggml CPU/GPU), and schedulers/sampling. Keep this stack in mind as you read.
Do a guided code tour (in this order):
src/llama.cpp/src/llama.h – API and high-level flow.
src/llama-impl.h – internal structures.
ggml/src/ggml.c (CPU) and ggml/src/ggml-cuda.cpp/ggml-hip.cpp – tensor ops.
common/ utils (tokenizer, sampling).
examples/ (small, focused demos) to see usage patterns.
Run small experiments while reading: llama-bench to see threading/backends, llama-gguf to inspect headers, llama-cli -m <model> -t <threads> to connect code to behavior. Add -d (debug logging) and --verbose-prompt to watch the pipeline.
Study GGUF and quantization: understand tensor layouts, n_params, and per-tensor types; skim docs/quantization.md and the conversion scripts.
Use tracing/profiling to link code to perf: build with -DLLAMA_PERF=ON, run llama-bench --dump-timings, and skim the kernel dispatch in ggml to see which ops dominate.
Read issues/PRs for design intent: search for “design”, “roadmap”, and backend discussions; skim recent PRs touching ggml and llama.cpp for rationale and constraints.
If you like video, pick one good deep dive (e.g., a maintainer talk on ggml/llama.cpp internals) but rely on code/doc reading for fidelity.
Take notes in a map: for each subsystem, write a short “responsibility + key files + hot paths + configs” summary; this cements understanding better than passive watching.



1) flow: 
in llama.cpp/common/arg.cpp

**common_params** -> struct that holds all parsed values, its the output 
**llama_example** ex -> an enum enum which tells the parser which binary is using it (llama-cli, server etc), options can be gated on this 
**common_params_context** -> a small wrapper that holds a pointer to common_params, the ex enum, and a vector of common_arg entries (option definitions)
-> parser is shared across many tools, its not just a special keyword 
**add_opt(common_arg(...))** -> just pushes an option definition into the contexts list 
-> common_arg(...) takes the flag name, a metavar string for help "FNAME" and a handler callable, the handler is often a lambda like function which does a work. 

Parsing flow (simplified):

1) common_params_parser_init(params, ex, print_usage) builds ctx_arg containing a pointer to params, the ex enum, the options vector (initially empty), and any hooks like print_usage.
2) It calls add_opt(...) many times to register options into ctx_arg.options.
3) Later, common_params_parse(argc, argv, params, ex, print_usage) walks argv, looks up each flag in ctx_arg.options, and invokes the handler (lambda or function pointer) to write into params. That’s why params is passed by reference; the lambda edits it in place.
4) After parsing, params is filled; ctx_arg was just the parser’s internal bookkeeping.

"Each common_arg registered via add_opt defines how to handle one flag: names, help text, and a handler function (lambda or function pointer) that writes into the common_params struct. During parsing, the code walks argv, finds the matching common_arg, and invokes its handler with the provided value(s), mutating the shared common_params instance. So the flow is: build the option table → iterate args → dispatch to the option’s handler → handler updates params." 

common_parser_init 
```C++
add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? "model path from which to load base model"
            : "model path to load",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}).set_env("LLAMA_ARG_MODEL"));
```



Swappiness as an Experimental Variable
Scenario 1: Swappiness = 100 (Aggressive)
Behavior: Kernel swaps pages proactively
Result: Model pages may get swapped even if there's still some RAM available
Use case: Tests worst-case SSD performance (maximum swapping)
Access pattern: Might show more random access as pages swap in/out dynamically
Scenario 2: Swappiness = 0 (Conservative)
Behavior: Kernel avoids swapping until absolutely necessary (RAM truly full)
Result: Model stays in RAM as long as possible, only swaps when forced by mlock
Use case: Tests on-demand paging (swap only when accessed)
Access pattern: Might show more sequential access as pages load in order
Scenario 3: Swappiness = 60 (Default)
Behavior: Balanced approach
Result: Middle ground between the two
Use case: Real-world baseline
This Could Be a Key Finding!
Gabriel might be interested in: "How does swappiness affect access patterns?" Hypothesis:
High swappiness (100) → More page thrashing → More random access
Low swappiness (0) → On-demand only → More sequential access
This could affect optimal prefetching strategies!
Should We Add This as a Parameter?
I can easily add --swappiness as an experimental parameter:
# Current experiments
./run_experiment.py --model gpt-oss-20b --scenario 100percent --tokens 100

# With swappiness control
./run_experiment.py --model gpt-oss-20b --scenario 100percent --swappiness 0
./run_experiment.py --model gpt-oss-20b --scenario 100percent --swappiness 60
./run_experiment.py --model gpt-oss-20b --scenario 100percent --swappiness 100
This would give you 3x the experiments but potentially much richer data:
2 models × 3 scenarios × 3 swappiness values = 18 experiments total
Question: Do you want me to:
Keep it simple (swappiness=100 for all experiments, as originally planned)
Add swappiness as parameter (run 18 experiments instead of 6)
Just document it (so you can manually test different values later if needed)
What do you think? Given your tight timeline, option 1 or 3 might be best to start, but option 2 could provide valuable insights for your thesis!



Traces writing was slowing us down insanely, because we were writing in disk as well (and traces final output was like 1.1 GiB), instead store/buffer traces in RAM to be faster 

# sudo sysctl vm.swappiness=0 

CPU utilisation?
