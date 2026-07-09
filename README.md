# unit-net: distilling an open-weight LLM into a constrained network with a readable workspace

Successor project to `~/Code/dual-projection-mnist` (see
`docs_prior_session/report.pdf` for the full prior study). A **unit-net** is
a network in the dual-projection constrained geometry:

- two nonnegative matrices per layer, excitatory `P` and inhibitory `N`;
  pre-activation `z = Pa − Na`
- every row of `P` sums to exactly 1 (convex combination — a unit budget of
  excitation per neuron)
- `N ∈ [0,1]` per synapse; all activations in `[0,1]` by construction

## Why: the J-space connection

Anthropic's J-lens result (July 2026; paper *"Verbalizable Representations
Form a Global Workspace in Language Models"*, code
`anthropics/jacobian-lens`) reads what an internal activation "is disposed to
make the model say" via a **fitted linear transport**:

```
lens_l(h) = unembed( J_l · h ),   J_l = E[ ∂h_final / ∂h_l ]
```

`J_l` is a corpus-averaged Jacobian — an *approximation*, because a
transformer's true layer-to-output map is wildly input-dependent. The
project's hypothesis (Remy's): **in a unit-net the J-lens is exact and
unit-normalized.** With ReLU activations, the true Jacobian from layer `l`
to the output is precisely the product of active-path effective matrices:

```
∂z_out / ∂h_l = W_L^(a) · … · W_{l+1}^(a),   W^(a) = (P − N) masked to active units
```

Every entry of every factor is bounded (`P` rows convex, `N ≤ 1`), so path
products have fixed units: "strong path" is an absolute, comparable
statement. J-space — the workspace of concepts with high verbalizable
disposition — should be **directly detectable**: prime the network with an
input, and the concept's representation shows up as bounded high-mass paths
emanating from it to the output layer. No lens fitting, no averaging: read
the wiring. (This is exactly the composed-receptive-field view from the
prior session's 3D explorer, generalized from digit templates to token
dispositions.)

## Plan

**Phase 0 — pipe-cleaning (`phase0_charlm.py`, runnable now).** Char-level
next-token unit-net trained by mirror descent (the prior session's best
native-geometry method, 95% of oracle). One-hot inputs are *naturally* in
{0,1} — a unit-net's first layer over one-hots reads as "each token excites a
budgeted distribution of features," which is the interpretability story from
day one. Proves sequence prediction works in the geometry.

**Phase 1 — distillation.** Teacher: an open-weights decoder (Qwen2.5-0.5B —
same family as the jacobian-lens examples). Student: K-token context window
of one-hot (or frozen-teacher-embedding, min-max-normalized to [0,1])
inputs → 2–4 unit-net layers → vocabulary logits. Loss: KL to teacher
next-token distribution (logit-scale trick from the prior session), mirror
descent + projected GD. Corpus: TinyStories or similar narrow distribution
first — the constraint geometry costs capacity, so start where a small dense
model is already competent.

**Phase 2 — exact J-lens.** Implement `jlens_exact.py`: for a primed input,
compute the masked path product from every hidden unit to the unembedding;
rank hidden units by bounded output disposition. Compare against the fitted
Jacobian lens (run `anthropics/jacobian-lens` machinery on the student):
prediction — they agree, but the exact version needs no corpus and
decomposes into enumerable bounded paths.

**Phase 3 — workspace experiments.** The Anthropic result's signatures,
replayed in the readable substrate: prime a concept, verify it is (a)
reportable — decodable from path mass before output, (b) controllable —
clamp/edit the specific rows carrying it (surgical, since inhibition is
bounded and budgeted), (c) causally tied to behavior — token-level path
ablations with fixed-unit effect sizes.

## Prior-session tools that carry over

- `unitnet.py` (this repo): the constrained-layer library — forward
  semantics, structural checks, mirror-descent/signSGD steps — distilled from
  the prior project's `common.py`/`train_eg.py`.
- The annealing/bandit/macro-MCTS toolkit remains relevant for
  hardware-regime fine-tuning (see prior report §7–8: quantization helps
  gradient-free search; sign-steps win under write noise).
- Interpretability: 3D path explorer (`viz3d_gen.py` pattern) extends
  directly to token → feature → vocabulary cones.

## Environment

Torch 2.12 cu130 via `/home/remy/Code/tiny-tessarachnid/.venv`; 2× RTX 3090
(GPU 1 usually idle). Teacher models via HuggingFace.
