"""Report figures for the unit-net / j-carve day: distillation training
journeys, the carve pruning curves, and the dragon-neuron intervention."""
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
SURF, GRID = "#fcfcfb", "#e1e0d9"
C = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT, labelsize=8)


def parse(log, pat=r"step\s+(\d+).*agree(?:@1\(teacher\))?=([\d.]+)"):
    steps, agree = [], []
    for line in open(log):
        m = re.search(pat, line)
        if m:
            steps.append(int(m.group(1)))
            agree.append(float(m.group(2)))
    return steps, agree


# ---- fig 1: the distillation journey ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=170)
fig.patch.set_facecolor(SURF)
runs = [("phase1.log", "mirror descent, hot schedule (UNK-poisoned)", C[5]),
        ("phase1b.log", "mirror descent, cooled (UNK-poisoned)", C[2]),
        ("phase1_sig.log", "Adam+reparam, sigmoid (clean targets)", C[1]),
        ("phase1_relu2.log", "Adam+reparam, hardened ReLU", C[0]),
        ("phase1_sel.log", "hardened ReLU + ownership loss", C[4])]
for log, label, color in runs:
    try:
        s, a = parse(log)
        a1.plot(s, a, color=color, lw=1.8, label=label)
    except FileNotFoundError:
        pass
a1.set_xlabel("training step", fontsize=9, color=INK2)
a1.set_ylabel("teacher agreement@1", fontsize=9, color=INK2)
style(a1, "Distillation journey (UNK-poisoned runs' 33% was artifact)")
a1.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper left")

# concept selectivity trajectory
s, sel = parse("phase1_sel.log", r"step\s+(\d+).*concept_sel=([+-][\d.]+)")
s0, sel0 = parse("phase1_relu2.log", r"step\s+(\d+).*mean_sel=([+-][\d.]+)")
a2.plot(s0, sel0, color=C[0], lw=1.8, label="no ownership loss")
a2.plot(s, sel, color=C[4], lw=1.8, label="targeted ownership loss")
a2.axhline(0, color=MUT, lw=1, ls=":")
a2.set_xlabel("training step", fontsize=9, color=INK2)
a2.set_ylabel("concept selectivity margin", fontsize=9, color=INK2)
style(a2, "Designed carrier units: selectivity crosses zero and soars")
a2.legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.tight_layout()
fig.savefig("fig_distill.png", facecolor=SURF, bbox_inches="tight")
print("wrote fig_distill.png")

# ---- fig 2: carve pruning curves ----
res = json.load(open("../j-carve/carve_results.json"))
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=170)
fig.patch.set_facecolor(SURF)
chance = {"sentiment": 0.5, "topic": 1 / 3, "pos-slot": 0.5, "pronoun": 0.5}
for i, (task, r) in enumerate(res.items()):
    keeps = [c["keep"] for c in r["curve"]]
    accs = [c["acc"] for c in r["curve"]]
    ax.semilogx(keeps, accs, "-o", color=C[i], lw=2, ms=4, label=task)
    ax.axhline(chance[task], color=C[i], lw=0.8, ls=":", alpha=0.5)
ax.set_xlabel("fraction of network kept (log)", fontsize=9, color=INK2)
ax.set_ylabel("2AFC accuracy", fontsize=9, color=INK2)
ax.invert_xaxis()
ax.set_ylim(-0.05, 1.05)
style(ax, "Carving curves: pos-slot holds 100% down to 10% of the network;\n"
          "dotted = chance; other tasks capacity-gated (student at chance "
          "even uncarved)")
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
fig.tight_layout()
fig.savefig("../j-carve/fig_carve.png", facecolor=SURF, bbox_inches="tight")
print("wrote ../j-carve/fig_carve.png")

# ---- fig 3: the dragon neuron ----
fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=170)
fig.patch.set_facecolor(SURF)
labels = ["baseline", "unit 35\nablated", "unit 35\nboosted"]
pd = [0.0002, 0.0000, 1.0000]
bars = ax.bar(labels, pd, color=[C[0], C[2], C[5]], width=0.55)
for b, v in zip(bars, pd):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.4f}",
            ha="center", fontsize=9, color=INK2)
ax.set_ylabel("p(' dragon')", fontsize=9, color=INK2)
ax.set_ylim(0, 1.15)
style(ax, "One designed neuron, total causal control "
          "(every effect predicted exactly)")
fig.tight_layout()
fig.savefig("fig_dragon.png", facecolor=SURF, bbox_inches="tight")
print("wrote fig_dragon.png")
