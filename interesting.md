
Our machine specs: 

CPU: AMD Ryzen 7 7700X (8 cores / 16 threads) shown in lscpu; clocks 0.4–5.57 GHz; single NUMA node.

RAM: 30 Gi total, ~27 Gi free, no swap configured (free -h).

GPU: no NVIDIA driver (nvidia-smi absent); lspci shows an AMD Raphael iGPU, so there’s no dedicated VRAM—graphics would share system RAM. The current llama.cpp build only ships CPU libs (libggml-cpu.so), so llama-cli will run on CPU





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

deterministic prefetching might be killer for MoE 
1) PowerInfer -> (sparsity hot/cold neurons) ReLU activisation keep GPU rest RAM, what if we do same but RAM/SSD? 
2) Buffer for internal memory -> would require us to write our own virtual memory manager inside the application 
3) io_uring 
4) flash attention 
-> thread to prefetch into memory???

It requires deep kernel knowledge (io_uring).
It requires deep engine knowledge (ggml internals).
It offers marginal gains for dense models (limited by SSD speed).

"Anonymous Memory" as memory that has no home on the hard drive.

Anonymous Memory (malloc) is like a Whiteboard.
The Source: It comes from nowhere. It is just blank, empty space given to you.

The "Name": It has no name. You cannot look it up in the library catalog (the file system). It is just "Generic Whiteboard #5."

If RAM is full (Eviction): You cannot just erase the whiteboard, because that information exists nowhere else! If you erase it, the data is lost forever.

The Solution (Swap): To free up space, you must take a photo of the whiteboard and store it in a special "junk drawer" (the Swap partition). This is slow (writing to disk).