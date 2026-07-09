"""Phase 2: exact J-lens vs corpus-averaged (Anthropic-style) lens.

Anthropic's J-lens fits J_l = E[dh_final/dh_l] over a corpus because a
transformer's layer-to-output map is input-dependent. In a unit-net the true
Jacobian is exact per input: J_exact(x) = W_out · diag(relu mask(x)) · W_h,
every entry bounded (convex rows, N<=1) so path mass has fixed units.

This script measures what averaging loses: reconstruct output dispositions
with (a) the exact per-input lens (sanity: identical to the true forward)
and (b) the corpus-averaged lens, and compare top-k token overlap. Then it
demos exact path attribution: for a prompt, which (position, context token)
paths carry the predicted token.
"""
import os
import torch
from unitnet import TokenUnitLM

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1"
M = 512          # evaluation prompts
TOPK = 5


def main():
    torch.manual_seed(0)
    model = TokenUnitLM.load(os.path.join(HERE, "model_phase1_adam.npz"), DEV)
    vm = torch.load(os.path.join(HERE, "vocab_map.pt"), weights_only=True)
    ids = torch.load(os.path.join(HERE, "tinystories_tokens.pt"),
                     weights_only=True).to(DEV)
    reduced = vm["reduced"].to(DEV)
    inv = vm["inv"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(vm["teacher"])

    V, K, E = model.V, model.K, model.E
    H = model.trunk.Ps[0].shape[0]
    W_h = (model.trunk.Ps[0] - model.trunk.Ns[0])      # (H, KE+1)
    W_o = (model.trunk.Ps[1] - model.trunk.Ns[1])      # (V, H+1)
    Wh_x, wh_c = W_h[:, :K * E], W_h[:, K * E]
    Wo_h, wo_c = W_o[:, :H], W_o[:, H]

    # sample eval windows from the tail of the corpus
    n_val = len(ids) // 20
    val = ids[-n_val:]
    ix = torch.randint(0, len(val) - K - 1, (M,))
    ctx = torch.stack([val[i:i + K] for i in ix])
    red = reduced[ctx]

    (eacts, feat, tacts), z_true = model.forward(red)
    hidden = tacts[1]                                   # (M, H)
    if model.act == "sigmoid":
        mask = hidden * (1 - hidden)   # exact: d sigma(z) = a(1-a)
    else:
        mask = (hidden > 0).float()

    # ---- pass 1: corpus-averaged lens ----
    # J_avg = E[ Wo_h · diag(mask) · Wh_x ],  b_avg = E[const paths]
    J_avg = torch.zeros(V, K * E, device=DEV)
    b_avg = torch.zeros(V, device=DEV)
    for i in range(M):
        Ji = Wo_h @ (mask[i].unsqueeze(1) * Wh_x)
        J_avg += Ji / M
        b_avg += (Wo_h @ (mask[i] * wh_c) + wo_c) / M

    # ---- pass 2: reconstruction quality ----
    z_avg = feat @ J_avg.T + b_avg
    # exact lens sanity on a few inputs
    errs = []
    for i in range(8):
        Ji = Wo_h @ (mask[i].unsqueeze(1) * Wh_x)
        bi = Wo_h @ (mask[i] * wh_c) + wo_c
        errs.append((feat[i] @ Ji.T + bi - z_true[i]).abs().max().item())
    print(f"exact-lens reconstruction max err (8 samples): {max(errs):.2e} "
          f"(should be ~float eps: the lens IS the forward)")

    def topk_overlap(a, b, k=TOPK):
        ta = a.topk(k, dim=1).indices
        tb = b.topk(k, dim=1).indices
        return (ta.unsqueeze(2) == tb.unsqueeze(1)).any(2).float().mean(1)

    dv = mask.std(0)
    print(f"gating: {int((dv > 0.01).sum())}/{H} units with input-dependent "
          f"transport (std of activation derivative > 0.01)")
    ov = topk_overlap(z_true, z_avg)
    agree = (z_true.argmax(1) == z_avg.argmax(1)).float().mean().item()
    print(f"averaged lens vs truth over {M} prompts: "
          f"top-{TOPK} overlap {ov.mean().item():.3f} "
          f"(min {ov.min().item():.2f}), argmax agreement {agree:.3f}")
    print("=> even with ONE relu layer, corpus averaging loses per-input "
          "structure; the unit-net's exact lens loses nothing, by "
          "construction, at zero fitting cost.")

    # ---- exact path attribution demo ----
    print("\npath attribution (exact, fixed units): "
          "prediction <- (position, context token) paths")
    emb_W = (model.emb.Ps[0] - model.emb.Ns[0])         # (E, V+1)
    shown = 0
    for i in range(M):
        if shown >= 4:
            break
        c = int(z_true[i].argmax())
        if c == 0:
            continue  # skip UNK-argmax prompts for the demo
        shown += 1
        ctx_toks = [tok.decode([int(t)]) for t in ctx[i]]
        pred = tok.decode([int(inv[c])]) if c > 0 else "<unk>"
        Ji = Wo_h @ (mask[i].unsqueeze(1) * Wh_x)       # (V, KE)
        contrib = Ji[c] * feat[i]                        # (KE,)
        per_pos = contrib.view(K, E).sum(1)
        top_pos = per_pos.argsort(descending=True)[:3]
        parts = ", ".join(
            f"pos {p}({ctx_toks[p]!r}):{per_pos[p]:+.4f}"
            for p in top_pos.tolist())
        print(f"  {''.join(ctx_toks)!r} -> {pred!r}  via  {parts}")


if __name__ == "__main__":
    main()
