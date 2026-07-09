"""Phase 3: workspace experiments in the readable substrate.

The Anthropic J-space signatures, replayed where every path is bounded:
(a) REPORTABLE — a concept's pre-output disposition z_c decomposes into
    bounded per-unit contributions and predicts the teacher's probability
    of that concept, before any output is produced.
(b) CONTROLLABLE — clamp the specific hidden units carrying the concept;
    the effect on z_c is EXACTLY predictable (linear output layer + fixed
    units), unlike lens-guided steering in a transformer.
(c) CAUSAL — boosting the concept's carrier units changes generation.
"""
import os
import torch
from unitnet import TokenUnitLM

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1"
M = 768
CONCEPTS = [" Lily", " dragon", " happy", " scared", " park", " mom"]


def main():
    torch.manual_seed(0)
    model = TokenUnitLM.load(os.path.join(HERE, "model_phase1_adam.npz"), DEV)
    vm = torch.load(os.path.join(HERE, "vocab_map.pt"), weights_only=True)
    ids = torch.load(os.path.join(HERE, "tinystories_tokens.pt"),
                     weights_only=True).to(DEV)
    reduced = vm["reduced"].to(DEV)
    inv = vm["inv"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(vm["teacher"])
    teacher = AutoModelForCausalLM.from_pretrained(
        vm["teacher"], torch_dtype=torch.bfloat16).to(DEV).eval()

    V, K = model.V, model.K
    H = model.trunk.Ps[0].shape[0]
    W_o = (model.trunk.Ps[1] - model.trunk.Ns[1])
    Wo_h = W_o[:, :H]

    n_val = len(ids) // 20
    val = ids[-n_val:]
    ix = torch.randint(0, len(val) - K - 1, (M,))
    ctx = torch.stack([val[i:i + K] for i in ix])
    red = reduced[ctx]
    (_, feat, tacts), z = model.forward(red)
    hidden = tacts[1]
    with torch.no_grad():
        t_logits = teacher(input_ids=ctx).logits[:, -1, :].float()
        t_logp = torch.log_softmax(t_logits, dim=1)

    print("=== (a) REPORTABLE: concept disposition z_c predicts teacher "
          "log-prob of the concept ===")
    for w in CONCEPTS:
        tid = tok(w).input_ids[0]
        c = int(reduced[tid])
        if c == 0:
            print(f"  {w!r}: not in reduced vocab, skipped")
            continue
        zc = z[:, c]
        tp = t_logp[:, tid]
        # Spearman via rank correlation
        rz = zc.argsort().argsort().float()
        rt = tp.argsort().argsort().float()
        rho = torch.corrcoef(torch.stack([rz, rt]))[0, 1].item()
        print(f"  {w!r:10s} spearman(z_c, teacher logp) = {rho:+.3f}")

    print("\n=== (b) CONTROLLABLE: clamp the concept's carrier units; "
          "measured effect == predicted effect (exact intervention "
          "calculus) ===")
    w = " dragon"
    tid = tok(w).input_ids[0]
    c = int(reduced[tid])
    # carrier units by SELECTIVITY: path to c minus strongest competing
    # path (raw weight picks units that push whole token clusters)
    comp = Wo_h.clone(); comp[c] = -1e9
    selectivity = Wo_h[c] - comp.max(dim=0).values
    carriers = selectivity.argsort(descending=True)[:5]
    print(f"  concept {w!r}: top carrier units "
          f"{carriers.tolist()} with path weights "
          f"{[round(float(Wo_h[c, u]), 3) for u in carriers]}")
    i = int(z[:, c].argsort(descending=True)[M // 20])  # a mid-high prompt
    h0 = hidden[i].clone()
    for mode, hval in [("ablate->0", 0.0), ("boost->1", 1.0)]:
        h1 = h0.clone()
        h1[carriers] = hval
        dz_pred = (Wo_h[c, carriers] * (hval - h0[carriers])).sum()
        ones = torch.ones(1, 1, device=DEV)
        z0 = torch.cat([h0.unsqueeze(0), ones], 1) @ W_o.T
        z1 = torch.cat([h1.unsqueeze(0), ones], 1) @ W_o.T
        dz_meas = (z1 - z0)[0, c]
        p0 = torch.softmax(z0 * 60, 1)[0, c].item()
        p1 = torch.softmax(z1 * 60, 1)[0, c].item()
        print(f"  {mode:10s} dz_c predicted {dz_pred:+.4f} measured "
              f"{dz_meas:+.4f} | p({w.strip()}) {p0:.4f} -> {p1:.4f}")

    print("\n=== (c) CAUSAL: boosted carriers change generation ===")
    prompt = "Once upon a time, there was a big"
    pids = tok(prompt).input_ids[-K:]
    pids = [0] * (K - len(pids)) + pids
    for boost in [False, True]:
        redq = [int(reduced[t]) for t in pids]
        outs = []
        for _ in range(12):
            rt_ = torch.tensor([redq], device=DEV)
            (_, f_, ta_), z_ = model.forward(rt_)
            if boost:
                h_ = ta_[1][0].clone()
                h_[carriers] = 1.0
                ones = torch.ones(1, 1, device=DEV)
                z_ = torch.cat([h_.unsqueeze(0), ones], 1) @ W_o.T
            r = int(z_[0].argmax())
            outs.append(int(inv[r]) if r > 0 else tok.eos_token_id)
            redq = redq[1:] + [r]
        tag = "boosted " if boost else "baseline"
        print(f"  {tag}: {tok.decode(outs)!r}")


if __name__ == "__main__":
    main()
