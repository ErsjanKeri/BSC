# BSC thesis: SSD-backed sparse expert inference

Research workspace for the BSc thesis **"Tensor-Level Access Pattern Analysis and Optimization for SSD-Backed Sparse Expert Architectures"**, supervised by Prof. Dr. Viktor Leis and advised by Gabriel Haas at TUM.

The thesis investigates LLM inference when MoE model weights live on NVMe SSD rather than RAM, on a fork of llama.cpp with custom tensor-tracing instrumentation, an io_uring + O_DIRECT expert loader, and a multi-stage MoE pipeline. The primary research artifact is `thesis/main.pdf`.

## Layout

```
BSC/
├── thesis/                  LaTeX thesis (chapters, scripts, figures, bib). main.tex builds main.pdf.
├── time-tracking/           cgroup v2 wall-clock harness + canonical results.
│   ├── results/
│   │   ├── cgroup_20260501_191023/   GPT-OSS-20B canonical (96 runs, 32 configs × 3 iters)
│   │   └── cgroup_20260502_140144/   GPT-OSS-120B retry canonical (72 runs, 24 configs × 3 iters)
│   ├── settings_thesis_final.json           20B sweep configuration
│   ├── settings_thesis_final_120b_retry.json  120B retry configuration
│   ├── run_cgroup_experiments.py            harness (drops caches, wraps in cgroup, captures meminfo)
│   └── utils.py                             shared helpers
├── tensor-tracing/          Per-token tensor access tracer + parsers + visualizers.
│   ├── traces/                              fresh canonical traces (multi-topic prompt, 20B)
│   ├── tools/                               parse_*.py + simulate_*.py
│   ├── webui/                               React frontend (1-2 tok visualization)
│   └── desktopui/                           C++ ImGui frontend (100+ tok visualization)
├── generation_outputs/      Deterministic seed=42 generation text from the canonical sweep, both models.
├── journal/                 Dated research narrative (Dec 2025 – May 2026). INDEX.md is the entry point.
└── docs/                    Reference material (llama.cpp internals, related work, server setup).
```

## Reproducing the headline numbers

See `thesis/chapters/A_software_artifacts.tex` (Appendix A) for the full reproduction recipe. Summary:

```bash
# Build the fork
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-completion bsc-pack-experts -j$(nproc)

# Produce the .bscexp sidecar pack file for each model
./build/bin/bsc-pack-experts <model.gguf> -o <model.gguf>.bscexp --verify

# Run the canonical sweeps (each takes a few hours)
cd ~/BSC/time-tracking
sudo python3 run_cgroup_experiments.py --settings settings_thesis_final.json
sudo python3 run_cgroup_experiments.py --settings settings_thesis_final_120b_retry.json
```

## Build the thesis

```bash
cd ~/BSC/thesis && make            # produces main.pdf via pdflatex + biber
```

Figures and tables are regenerated from `_meta/final_canon.json`:

```bash
python3 scripts/build_final_canon.py    # rebuild oracle from results CSVs
python3 scripts/figures_canon.py        # regenerate figures/canon/*.pdf
python3 scripts/tables_canon.py         # regenerate _meta/canon_tables.tex
```

## Hardware

TUM lab server (`cli-hiwi-02.dis.cit.tum.de`): AMD Ryzen 7 7700X (8c/16t), 30 GiB DDR5, 2× NVMe SSDs over PCIe 4.0. Models stored on `nvme1n1`; OS on `nvme0n1`. The 30 GiB main memory is `2.2×` smaller than the 65.4 GB GPT-OSS-120B on-disk footprint, which is the cross-model stress condition the thesis builds on.
