#!/usr/bin/env python3
"""mul_mat_id mechanics — minimal, one clear idea.

Single panel. Shows: one call → picks 4 matrices from a bank of 128 → does 4
matrix-vector multiplies → stacks into output.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "plots"

C = {
    "io":    "#D8E8FF",
    "bank":  "#FFF2D0",
    "mmid":  "#F4A261",
    "expert":"#FFD89A",
    "out":   "#D8FFEA",
}


def box(ax, x, y, w, h, text, color, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
        linewidth=1.3, edgecolor="#333", facecolor=color))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=("bold" if bold else "normal"))
    return (x, y, w, h)


def arrow(ax, a, b, color="#555", dashed=False, width=1.2):
    kw = {"arrowstyle": "-|>", "mutation_scale": 12, "color": color, "linewidth": width}
    if dashed:
        kw["linestyle"] = (0, (4, 2))
    ax.add_patch(FancyArrowPatch(a, b, **kw))


def main():
    plt.rcParams.update({"savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 24); ax.set_ylim(0, 14); ax.set_aspect("equal"); ax.axis("off")

    # TOP: the 3 inputs
    x = box(ax, 0.5, 11.5, 5, 1.8,
            "x\n[2880, 1]\n(hidden state\nof ONE token)", C["io"], fs=10, bold=True)
    sel = box(ax, 9, 11.5, 5, 1.8,
            "selected_experts\n[4]\ne.g. {71, 4, 108, 33}\n(from router)", C["io"], fs=10, bold=True)
    bank = box(ax, 17, 11.5, 6.5, 1.8,
            "up_exps (weight bank)\n[2880, 2880, 128]\n= stack of 128 matrices",
            C["bank"], fs=10, bold=True)

    # THE op — center
    op = box(ax, 5.5, 8.5, 13, 1.8,
            "ggml_mul_mat_id(up_exps, x, selected_experts)",
            C["mmid"], fs=12, bold=True)
    arrow(ax, (3, 11.5), (9, 10.3))
    arrow(ax, (11.5, 11.5), (11.5, 10.3))
    arrow(ax, (20.2, 11.5), (15, 10.3))

    # INSIDE: 4 picked matrices side by side
    ax.text(12, 7.7, "inside: for i = 0..3, picks W = up_exps[:,:,selected_experts[i]]  and computes  W @ x",
            ha="center", fontsize=10, style="italic")

    # Show only i=0 and i=3 with "..." in between. Cleaner.
    # i=0
    box(ax, 1, 4.8, 3, 2.5,
        "W = up_exps[:,:,71]\n[2880, 2880]\n(expert 71)", C["expert"], fs=9, bold=True)
    ax.text(4.2, 6.05, "@", fontsize=18, ha="left", va="center")
    box(ax, 4.8, 5.1, 1.6, 1.9, "x\n[2880, 1]", C["io"], fs=9)
    ax.text(6.6, 6.05, "=", fontsize=18, ha="left", va="center")
    box(ax, 7.1, 5.1, 2, 1.9, "out[:, 0]\n[2880]", C["out"], fs=9, bold=True)
    ax.text(5, 3.6, "i = 0", fontsize=11, ha="center", style="italic")

    # ellipsis
    ax.text(11.5, 6.05, "· · ·", fontsize=28, ha="center", va="center")
    ax.text(11.5, 3.6, "i = 1, 2", fontsize=11, ha="center", style="italic")

    # i=3
    box(ax, 13.5, 4.8, 3, 2.5,
        "W = up_exps[:,:,33]\n[2880, 2880]\n(expert 33)", C["expert"], fs=9, bold=True)
    ax.text(16.7, 6.05, "@", fontsize=18, ha="left", va="center")
    box(ax, 17.3, 5.1, 1.6, 1.9, "x\n[2880, 1]", C["io"], fs=9)
    ax.text(19.1, 6.05, "=", fontsize=18, ha="left", va="center")
    box(ax, 19.6, 5.1, 2, 1.9, "out[:, 3]\n[2880]", C["out"], fs=9, bold=True)
    ax.text(17.5, 3.6, "i = 3", fontsize=11, ha="center", style="italic")

    # Arrow from op down to the matrix row
    arrow(ax, (12, 8.5), (12, 7.4), width=1.5)

    # BOTTOM: stacked output
    out = box(ax, 8, 1.2, 8, 1.9,
        "stack the 4 results along dim-1\n\nOutput: [2880, 4, 1]",
        C["out"], fs=11, bold=True)
    # dotted arrows from each visible "out" to the stacked output
    for src_x in (8.1, 20.6):
        ax.add_patch(FancyArrowPatch(
            (src_x, 5.1), (12, 3.1),
            arrowstyle="-|>", mutation_scale=9, color="#888", linewidth=0.9,
            linestyle=(0, (3, 2))))

    ax.text(12, 0.5,
        "→  ONE call processes ALL 4 selected experts internally. "
        "No graph boundary between the 4 iterations.",
        ha="center", fontsize=10, style="italic")

    fig.suptitle("`ggml_mul_mat_id` in decode (n_tokens = 1)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"08_mul_mat_id_decode.{ext}")
    print("wrote 08_mul_mat_id_decode")
    plt.close(fig)


if __name__ == "__main__":
    main()
