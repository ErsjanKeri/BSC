# Analysis and Optimization of LLM Inference on Fast NVMe Storage

LaTeX source for the Bachelor's thesis investigating Mixture-of-Experts language model inference on a single workstation when the model file does not fit in RAM, using io_uring + O_DIRECT to read expert weights from NVMe on demand.

- Author: Ersjan Keri
- Supervisor: Prof. Dr. Viktor Leis
- Advisor: Gabriel Haas
- Institution: Technische Universität München, School of Computation, Information and Technology (Informatics)

## Build

```
make            # produces build/main.pdf
```

Requires `latexmk`, `pdflatex`, and `biber`. The thesis uses the [TUM-Dev/tum-thesis-latex](https://github.com/TUM-Dev/tum-thesis-latex) template.

## Layout

```
chapters/        chapter sources (.tex)
pages/           front matter (cover, title, abstract, ...)
figures/         TikZ sources and generated PDFs
scripts/         Python that derives numbers and figures from canonical data
_meta/           auto-generated tables and the canonical numbers oracle
bibliography.bib bibliography
main.tex         document entry point
settings.tex     LaTeX configuration
```

The numerical claims in the thesis are derived from canonical sweep data published in the parent BSC research workspace (`../time-tracking/results/cgroup_2026050{1,2}_*`). `scripts/derive_numbers.py` reproduces every cited value from the byte-level source data; `scripts/build_final_canon.py`, `scripts/figures_canon.py`, and `scripts/tables_canon.py` regenerate the figures and tables from `_meta/final_canon.json`.

## Reproducing the headline numbers

See the appendix (`chapters/A_software_artifacts.tex`, §A.5) for the five-step recipe.
