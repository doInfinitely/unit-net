# Unit-Nets: One Day of Training Without Gradients in a Constrained Weight Geometry

*Remy & Claude — 2026-07-07, `~/Code/dual-projection-mnist/`*

## 1. The task

Train an MNIST classifier under hard structural constraints ("dual-projection"
geometry, hereafter a **unit-net**):

- Every layer has **two nonnegative matrices**: excitatory `P` and inhibitory
  `N`. Pre-activation `z = Pa − Na`; the same input may project to the same
  neuron through both (dual projections).
- Each neuron's positive weights form a **convex combination**: every row of
  `P` sums to exactly 1.
- Every inhibitory weight is bounded: `N ∈ [0, 1]`.
- All activations lie in `[0, 1]` for all inputs — a theorem, not an empirical
  check, since inputs in `[0,1]` + convex rows force `z ≤ 1`.
- A constant-1 feature is appended at every layer (the only bias mechanism).

Architecture throughout: `784 (+1) → 64 (+1) → 10`, ReLU, 101,780 parameters.
Verification is artifact-only (`model.npz`, structural gates + hidden-split
accuracy); every model produced today passes the structural gates.

The session's driving question, posed at the start: **can simulated annealing
with beam search — moves of ±δ on an edge, parallel row perturbation with
renormalization, and moving δ between two edges that project to the same
neuron — replace backprop in this regime, and at what resource cost?**

## 2. Baselines, and the pathologies of the geometry

The oracle method is backprop through a constraint-satisfying
reparameterization (`P = row-softmax(W)`, `N = sigmoid(V)`). Getting it to
work surfaced four traps, each a direct consequence of the geometry:

1. **Dead-ReLU initialization.** `Pa ≤ 1` by construction but `Na` can reach
   ~n_in. `N` must initialize near zero or every ReLU is dead and no gradient
   flows.
2. **The weight-decay trap.** Decay on `V` pulls `sigmoid(V)` toward 0.5 —
   *growing* inhibition and killing the network. Regularize `N` itself (L1),
   never its reparameterization.
3. **Vanishing logit spread.** Convex rows are averaging operators; output
   spread is tiny. Cross-entropy needs a logit scale of 60–100 to produce
   usable gradients.
4. **Margin-memorization collapse.** Per-synapse `N` is bounded but a row of
   `N` can sum to hundreds; late training memorizes with huge negative
   margins and test accuracy collapses (97 → 80% within epochs). Validation
   selection is mandatory.

Tuned baselines: **97.2%** test (H=64), **97.7%** (H=256), ~5 s wall,
~4×10¹¹ FLOPs. An *unconstrained* twin, fairly tuned, reaches **97.7%**
(H=64) / **98.2%** (H=128, parameter-matched): the constraints cost gradient
descent ~0.5–1 point of ceiling and considerable fussiness, but produce the
smallest generalization gap on the board (0.2 vs 0.5).

## 3. Annealing with beam search

The annealer keeps a beam of B=8 networks; each step spawns 31 mutants per
member, scores all 256 candidates on a shared minibatch (one batched GPU
matmul), and Boltzmann-selects survivors at an annealed temperature. The move
vocabulary grew through the day — all constraint-preserving by construction:

| Move | Action |
|---|---|
| A | one inhibitory weight ±δ, clamped to [0,1] |
| B | perturb a neuron's whole positive row, renormalize |
| C | move δ between two positive weights of one neuron (sum conserved) |
| D | Gaussian-blob mass tilt on a first-layer row (radius annealed 6→0.8 px) |
| E | heat-kernel blob on the *activation-correlation graph* (Hebbian topology) |
| F | blob over an imposed 8×8 hidden-neuron lattice |
| G | blob-pair transfer: take under blob 1, deposit as blob 2 (→ C as radius→0) |

Key findings, in the order they were earned:

- **Selection noise, not landscape ruggedness, caused every plateau.** A
  256-example scoring batch has CE noise comparable to real candidate
  differences; growing the batch over the run (256→2048) turned a hard
  plateau at 76% into monotone climbing.
- **Move design dominates schedule tuning.** Ablations: A+C alone matched the
  full A+B+C set at 1/6 the compute; whole-row noise (B) is dead weight.
- **Structured proposals beat temperature.** The blob move (D) added +3.2
  points at zero cost by injecting locality through the *proposal
  distribution* — convolution's inductive bias without weight sharing.
- **Topology beats geometry.** Imposing an 8×8 lattice on the hidden layer
  (F) reached 90.4%; deriving the topology from the network's own activity
  (E, heat kernel `exp(−tL)` on the correlation graph, re-fit every 1k steps)
  reached **91.4% — the session's gradient-free accuracy record.** Hebb's
  principle applied to the search operator rather than the weights.
- **Coarse-to-fine is the universal schedule.** Blob radii, edit counts,
  temperatures, and even the move *mix* (blob-heavy → transfer-heavy) all
  want to anneal together; a mix-anneal run hit 90.8% with no hidden-layer
  topology at all.

![First-layer receptive fields](filters.png)
*First-layer P-rows. Backprop learns sparse stroke templates; plain annealing
leaves salt-and-pepper; topology-aware moves grow dense, contiguous fields.*

## 4. Event-driven scoring (delta evaluation)

A mutation touches one row, so scoring it needs no full forward pass: cache
the parents' activations, propagate only the rank-1 consequence of the edit
(`dz_r = a·ΔW_row`, then one outer product through the output column). Exact
(self-checked against full forwards), ~50× fewer FLOPs, and mutants become
3 KB edit descriptors instead of 400 KB weight copies. This moved the
annealing-vs-backprop resource gap from ~6,700× to ~140×, and every
subsequent experiment ran on this engine.

## 5. The gradient bridge: mirror descent and its degradations

"Compute the gradient of the transfer" turns out to be classical: the optimal
transfer direction per row is pairwise Frank-Wolfe, and applying it to all
coordinates multiplicatively is **exponentiated gradient / mirror descent**
(`row ← row·e^{−ηg}`, renormalize) — the natural gradient method for simplex
geometry, no reparameterization, boundary reachable without softmax
saturation. Results form a clean information ladder (H=64, matched budgets):

| Update rule | Gradient info used per row | Test |
|---|---|---|
| random proposals + selection (lattice annealer) | none | 90.8% |
| top-k pairwise Frank-Wolfe, k=32 | 64 coords, uniform | 92.4% |
| signSGD, fixed δ (Remy's hybrid) | 1 bit/weight | 92.7% |
| mirror descent (full magnitudes) | all | 95.0% |
| Adam + reparameterization | all + curvature memory | 97.2% |

Two null results with content: heat-kernel *smoothing* of gradients is worth
≈nothing (+0.4 max) — the spatial prior that gave blind search +3.2 points is
redundant once real gradients exist; and gradient-*informed proposals*
(REINFORCE over source/sink features) **learned Hebb's rule from reward alone**
(sink policy → high pre-post correlation, source → anti-correlation) yet
slightly hurt accuracy: pointing all proposals in the greedy direction
collapses the diversity that selection feeds on.

## 6. Planners: the row solver and MCTS

**Exact block solve.** Given hidden features, the entire output layer is
jointly convex over the constraint set. Solving it: finished annealed models
gained only +0.3–0.4% (their readouts were already block-optimal), random
convex features + solved readout = **chance** (9.8% — under convex rows,
random hidden features are informationless; every accuracy point is earned
feature learning), and solving *during* training collapsed the search
(80.7%, or chance if done often): at a block optimum the loss saturates and
the selection landscape flattens — *a perfect readout is a bad teacher*.

**MCTS over mutants** (adapting Remy's The_Monte MCTS engine — `tree.py`/
`ucb.py` vendored). Edit-level trees fail structurally: without re-rooting,
progressive widening spends any budget within ~10 edits of init (finished at
chance); with commits, lookahead costs 5× throughput for ~1 level of
foresight; a tuned cold-UCB tree tunnels to depth 123 and 39% val but remains
~10× behind flat annealing per FLOP. **Macro-MCTS** fixes both failure modes
at once: each node is a *chunk of delta-annealing under a chosen strategy*
(operator mix × step scale), so the tree plans the schedule, not the synapses.
Result: **87.1% / 90.0% val at matched FLOPs — parity with the hand-tuned
annealer — with the schedule discovered by search**: coarse blobs alternating
with functional-graph moves early, inhibition mid-run, ultra-fine transfers
in the endgame. It independently rediscovered the hand-designed curriculum,
including the bandit's late-inhibition insight, plus a novel
feature/readout interleaving. Scaled to 10 macros and 3× budget: 89.6% /
91.7% val.

An online bandit over move types (reward = measured Δloss per proposal, free
under delta scoring) matches hand-tuned mixes with zero tuning and surfaced
the late-inhibition schedule first.

## 7. Quantization: the finite-lattice study

Fixing the update quantum δ makes the weight space a finite lattice — finite
but astronomically so (δ=1/32 ⇒ ~10^81,211 networks; naming one takes only
~270 kbits, which is the honest size of the learning problem). Head-to-head,
STE gradients vs a lattice-native annealer (moves = exact δ-token transfers):

| δ | levels/weight | STE backprop | lattice annealer |
|---|---|---|---|
| — | ∞ | 97.2% | 87.4% |
| 0.02 | 51 | 96.9% | 90.0% |
| 0.05 | 21 | 96.8% | **90.8%** |
| 0.1 | 11 | 95.4% | 88.8% |
| 0.2 | 6 | 94.3% | 84.9% |
| 0.5 | 3 | 84.5% | 38.4% |

![Quantization crossover](quant_curve.png)

No crossover — but quantization **helps** gradient-free search absolutely
(sweet spot at 21 levels: best flat-annealing result of the day, with three
moves) and training natively on the lattice crushes post-hoc rounding
(84.9% vs 66.3% at δ=0.2).

## 8. The analog-noise regime

Simulated device physics: write noise (every *written* weight lands as
`w·e^{σξ}`; you only pay for rows you write) and read noise (`z += σξ` on
every training forward). Sweep at read σ=0.05:

| write σ | mirror descent | signSGD δ=10⁻⁴ | lattice annealer |
|---|---|---|---|
| 0 | 93.9 | 91.3 | 66.3 |
| 0.05 | 91.5 | 90.5 | 60.4 |
| 0.1 | 85.8 | **86.6** | 54.5 |
| 0.2 | 48.6 | **51.9** | 39.5 |
| 0.4 | 31.9 | 17.7 | 31.4 |

![Analog noise sweep](noise_curve.png)

The symmetric law: **gradients are robust to read noise and fragile to write
noise; selection is the reverse** — each method fails when the channel it
depends on loses precision. The sign hybrid with δ matched to the write-noise
floor overtakes full mirror descent at moderate noise: *on noisy hardware the
update quantum must exceed the write-noise floor; precision below the floor
is worse than a blunt step.* The annealer's read-noise fragility (90.8 → 66.3
from read noise alone) is the open weakness: selection consumes loss
differences ~100× smaller than gradient methods' batch-averaged signal.

## 9. Interpretability

The constrained geometry's clearest win. Because every row is a budgeted
convex combination minus bounded inhibition, magnitudes have fixed units
end-to-end: composed products `W₁·W₀` are directly comparable digit templates,
and "strong path" means something absolute. Artifacts built:

- **Topology diagram** — <https://claude.ai/code/artifact/b9efec99-0490-4f2b-a8ad-f2339871fc84>
- **Interactive 3D network** (three models, top-k slider, hover cones,
  click-to-inspect weight images and composed receptive fields) —
  <https://claude.ai/code/artifact/0fdeb625-9496-4c2b-8316-04eef2251318>
- Live MCTS dashboard (`mcts_viz.py`, force-directed tree, live-tunable
  hyperparameters) served during tree-search runs.

![3D render](render_3d.png)
*W = P−N in 3D: backprop wires sparse and surgical; the annealed
functional-graph model suppresses whole hidden communities.*

![Accuracy vs compute](comparison.png)

## 10. Final standings

| Method | Test | FLOPs | Wall |
|---|---|---|---|
| Unconstrained MLP (H=128, Adam) | 98.2% | 4×10¹¹ | 2 s |
| Backprop + reparam (H=256) | 97.7% | 1.6×10¹² | 8 s |
| Backprop + reparam (H=64) | 97.2% | 4×10¹¹ | 5 s |
| Mirror descent (native simplex) | 95.0% | 1.7×10¹² | 18 s |
| signSGD fixed-δ | 92.7% | 8×10¹¹ | 6 s |
| Anneal + functional graph | 91.4% | 2.7×10¹⁵ | 562 s |
| **Lattice annealer δ=0.05** | **90.8%** | **1.0×10¹³** | 187 s |
| Composite AGGCEH (delta eval) | 89.4% | 1.5×10¹³ | 227 s |
| Macro-MCTS (10 strategies) | 89.6% | 3.5×10¹³ | 770 s |
| Edit-level MCTS (best) | ~39% val | 1.7×10¹² | killed |

**Thesis of the day:** optimization methods are ranked by the information
rate and precision of the channel they consume — full gradient magnitudes >
gradient signs > candidate rankings — and the winner at any operating point
is the method whose channel assumptions the substrate can still honor. Clean
digital hardware honors Adam's; quantized and write-noisy substrates
progressively hand the advantage to sign-steps and selection. Structural
priors (blobs, topology, Hebbian graphs) are worth bits exactly when the
search signal is information-poor, and near nothing once gradients flow.
Deliberation (solvers, tree search) pays at the strategy level, never at the
synapse level.

## Appendix: code map

`common.py` (verifier-faithful semantics) · `train_backprop.py` (reparam +
STE) · `train_unconstrained.py` · `train_eg.py` (mirror descent / FW / top-k /
sign, noise sim) · `train_anneal.py` (full-forward annealer, moves A–F) ·
`train_anneal_delta.py` (delta eval, moves A–H, bandit, policy, lattice,
noise sim) · `row_solver.py` · `mcts_mutants.py` + `mcts_macro.py` +
`monte/` (vendored from Remy's The_Monte) + `mcts_viz.py` /
`mcts_dashboard.html` · `quantize_check.py` · `viz_filters.py`, `analyze.py`,
`viz3d_gen.py`, `viz_print.py` · models `model_*.npz` (all pass structural
gates) · logs for every run.
