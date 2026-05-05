#!/usr/bin/env python3
"""Three MoE FFN graph variants — minimal, same layout, one clear diff.

Each panel has the same vertical flow with short labels. The DIFFERENCE
between panels is what's highlighted:
  - LEFT (mmap):    orange mul_mat_id nodes, weights from mmap
  - MIDDLE (overlap): SAME orange nodes, PLUS teal callbacks inserted
                     between them (weights from io_uring cache)
  - RIGHT (v3):     the orange+swiglu+sum block REPLACED by ONE green box
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "plots"

C = {
    "io":       "#D8E8FF",    # inputs/outputs
    "std":      "#E8E8E8",    # standard ops (reshape, swiglu, sum)
    "mmid":     "#F4A261",    # mul_mat_id
    "cb":       "#6FC3BF",    # our callback
    "fused":    "#4CAF50",    # our fused op
}


def box(ax, x, y, w, h, text, color, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor="#333", facecolor=color))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=("bold" if bold else "normal"))
    return (x, y, w, h)


def flow(ax, a, b, color="#555"):
    ax.add_patch(FancyArrowPatch(
        (a[0] + a[2]/2, a[1]),           # bottom center of a
        (b[0] + b[2]/2, b[1] + b[3]),    # top center of b
        arrowstyle="-|>", mutation_scale=14, color=color, linewidth=1.3))


def base_canvas(ax, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10, fontweight="bold")


def _mmap(ax):
    base_canvas(ax, "Stock (mmap)")
    i   = box(ax, 2, 19.5, 6, 1.2, "x   [2880, 1]", C["io"], bold=True)
    u   = box(ax, 2, 17, 6, 1.5, "mul_mat_id   (up)", C["mmid"], bold=True)
    g   = box(ax, 2, 14.5, 6, 1.5, "mul_mat_id   (gate)", C["mmid"], bold=True)
    s   = box(ax, 2, 12.3, 6, 1.2, "swiglu", C["std"])
    d   = box(ax, 2, 9.8, 6, 1.5, "mul_mat_id   (down)", C["mmid"], bold=True)
    w   = box(ax, 2, 7.5, 6, 1.2, "× router_weights + Σ", C["std"])
    o   = box(ax, 2, 5, 6, 1.2, "moe_out   [2880, 1]", C["io"], bold=True)
    for a, b_ in [(i, u), (u, g), (g, s), (s, d), (d, w), (w, o)]:
        flow(ax, a, b_)
    ax.text(5, 3.3, "3 mul_mat_id ops\nweights via mmap",
            ha="center", fontsize=10, style="italic")


def _overlap(ax):
    base_canvas(ax, "io_uring + overlap")
    i   = box(ax, 2, 20, 6, 1.2, "x   [2880, 1]", C["io"], bold=True)
    cb1 = box(ax, 2, 18.2, 6, 1.3, "callback:  load up+gate\n(io_uring submit + wait)", C["cb"], fs=8, bold=True)
    u   = box(ax, 2, 16.2, 6, 1.3, "mul_mat_id   (up)\n[reads from uring cache]", C["mmid"], fs=8, bold=True)
    g   = box(ax, 2, 14.2, 6, 1.3, "mul_mat_id   (gate)\n[reads from uring cache]", C["mmid"], fs=8, bold=True)
    s   = box(ax, 2, 12.5, 6, 1.1, "swiglu", C["std"])
    cb2 = box(ax, 2, 10.7, 6, 1.3, "callback:  wait for down\n(io_uring residual wait)", C["cb"], fs=8, bold=True)
    d   = box(ax, 2, 8.7, 6, 1.3, "mul_mat_id   (down)\n[reads from uring cache]", C["mmid"], fs=8, bold=True)
    w   = box(ax, 2, 7, 6, 1.1, "× router_weights + Σ", C["std"])
    o   = box(ax, 2, 4.8, 6, 1.2, "moe_out   [2880, 1]", C["io"], bold=True)
    for a, b_ in [(i, cb1), (cb1, u), (u, g), (g, s), (s, cb2), (cb2, d), (d, w), (w, o)]:
        flow(ax, a, b_)
    ax.text(5, 3.3, "SAME 3 mul_mat_id ops\n+ 2 callback nodes inserted\n(down I/O overlaps with up+gate compute)",
            ha="center", fontsize=10, style="italic")


def _v3(ax):
    base_canvas(ax, "io_uring + pipeline v3")
    i = box(ax, 2, 19.5, 6, 1.2, "x   [2880, 1]", C["io"], bold=True)
    f = box(ax, 2, 9.5,  6, 9, "ONE fused op\n\n`ggml_map_custom3`\n\nreplaces up + gate +\nswiglu + down + Σ\n\ndispatches experts\nin first-ready order,\ninterleaves io_uring\nwaits per-expert",
            C["fused"], fs=10, bold=True)
    o = box(ax, 2, 5,    6, 1.2, "moe_out   [2880, 1]", C["io"], bold=True)
    flow(ax, i, f)
    flow(ax, f, o)
    ax.text(5, 3.3, "1 fused custom op\nreplaces all 3 mul_mat_id ops\n(per-expert I/O scheduling)",
            ha="center", fontsize=10, style="italic")


def main():
    plt.rcParams.update({"savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(18, 11))
    _mmap(axes[0])
    _overlap(axes[1])
    _v3(axes[2])

    legend = [
        mpatches.Patch(color=C["io"],    label="inputs / outputs"),
        mpatches.Patch(color=C["std"],   label="standard ops"),
        mpatches.Patch(color=C["mmid"],  label="ggml_mul_mat_id (MoE op)"),
        mpatches.Patch(color=C["cb"],    label="our callback (ggml_map_custom*)"),
        mpatches.Patch(color=C["fused"], label="our fused custom op"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.01), fontsize=10, frameon=True)
    fig.suptitle("Per-layer MoE FFN graph — three implementations (decode, n_tokens=1)",
                 fontsize=14, y=0.97, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"07_graph_variants.{ext}")
    print("wrote 07_graph_variants")
    plt.close(fig)


if __name__ == "__main__":
    main()
