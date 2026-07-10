"""Degradation & divergence curves for the quantization and flow-pruning
studies (ideas 2 & 3)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
SURF, GRID = "#fcfcfb", "#e1e0d9"
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"

q = json.load(open("quant_lm_results.json"))
f = json.load(open("flow_prune_results.json"))


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8)


fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.2, 4.0), dpi=170)
fig.patch.set_facecolor(SURF)

# ---- panel 1: quantization ----
bits = [b for b in (12, 10, 8, 6, 5, 4, 3, 2)]
agree = [q[str(b)]["agree"] for b in bits]
slot = [q[str(b)]["posslot"] for b in bits]
base = q["float32"]["agree"]
a1.plot(bits, agree, "-o", color=BLUE, lw=2, ms=4,
        label="teacher agreement")
a1.plot(bits, slot, "-o", color=YELLOW, lw=2, ms=4,
        label="pos-slot 2AFC")
a1.axhline(base, color=BLUE, lw=1, ls=":")
a1.annotate("float32 baseline", (11.8, base + 0.02), fontsize=7.5,
            color=INK2, ha="right")
a1.annotate("10-bit: lossless\n(and 5× sparser)", (10, base),
            textcoords="offset points", xytext=(-4, -34), fontsize=7.5,
            color=INK2)
a1.set_xlabel("bits per weight (integer-simplex rows)", fontsize=8.5,
              color=INK2)
a1.set_ylabel("accuracy", fontsize=8.5, color=INK2)
a1.set_xlim(12.6, 1.4)
a1.set_ylim(-0.05, 1.1)
style(a1, "Quantization degradation")
a1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left")

# ---- panel 2: flow pruning, accuracies ----
fr = [float(k) for k in f]
agree = [f[k]["agree"] for k in f]
slot = [f[k]["posslot"] for k in f]
a2.plot([x * 100 for x in fr], agree, "-o", color=BLUE, lw=2, ms=4,
        label="teacher agreement")
a2.plot([x * 100 for x in fr], slot, "-o", color=YELLOW, lw=2, ms=4,
        label="pos-slot 2AFC")
a2.axhline(base, color=BLUE, lw=1, ls=":")
a2.annotate("the syntax circuit rides the\nhighest-flow edges: 1.00\n"
            "through 99% demolition", (72, 0.86), fontsize=7.5, color=INK2)
a2.set_xlabel("% of edges pruned (by activation flow)", fontsize=8.5,
              color=INK2)
a2.set_ylabel("accuracy", fontsize=8.5, color=INK2)
a2.set_ylim(-0.05, 1.1)
style(a2, "Flow-pruning degradation")
a2.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center left")

# ---- panel 3: flow pruning, divergence ----
kl = [f[k]["kl"] for k in f]
a3.plot([x * 100 for x in fr], kl, "-o", color=VIOLET, lw=2, ms=4)
a3.annotate("saturates: outputs fully\ndecoupled from full model",
            (80, 14.2), fontsize=7.5, color=INK2)
a3.set_xlabel("% of edges pruned", fontsize=8.5, color=INK2)
a3.set_ylabel("KL(full ‖ pruned), nats", fontsize=8.5, color=INK2)
style(a3, "Divergence from the full student")

fig.tight_layout()
fig.savefig("fig_degradation.png", facecolor=SURF, bbox_inches="tight")
print("wrote fig_degradation.png")
