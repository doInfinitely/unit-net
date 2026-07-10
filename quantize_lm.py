"""Idea 2: native quantization of the unit-net student.

The geometry quantizes natively: a convex P row IS a fixed-point
probability distribution — b bits/weight means integer numerators summing
to exactly 2^b (largest-remainder), i.e. exact integer arithmetic with a
shared denominator; N entries quantize to the uniform 2^b-level grid on
[0,1]. Sweep b and measure teacher-agreement + pos-slot 2AFC degradation.
"""
import json
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1"
BITS = [12, 10, 8, 6, 5, 4, 3, 2]


def quantize_P(P, b):
    D = 2 ** b
    scaled = P * D
    base = torch.floor(scaled)
    short = (D - base.sum(dim=1)).round().long()
    rem = scaled - base
    order = rem.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(P.shape[1], device=P.device)
                   .expand_as(order))
    base += (ranks < short.unsqueeze(1)).float()
    return base / D            # rows sum to exactly 1, entries k/2^b


def quantize_N(N, b):
    L = 2 ** b - 1
    return (N * L).round() / L


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "j-carve"))
    from unitnet import TokenUnitLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import probes as probes_mod

    model = TokenUnitLM.load(os.path.join(HERE, "model_phase1_adam.npz"), DEV)
    vm = torch.load(os.path.join(HERE, "vocab_map.pt"), weights_only=True)
    reduced = vm["reduced"].to(DEV)
    tok = AutoTokenizer.from_pretrained(vm["teacher"])
    teacher = AutoModelForCausalLM.from_pretrained(
        vm["teacher"], dtype=torch.bfloat16).to(DEV).eval()
    ids = torch.load(os.path.join(HERE, "tinystories_tokens.pt"),
                     weights_only=True).to(DEV)
    top_d = vm["inv"][1:].to(DEV)
    K = model.K
    n_val = len(ids) // 20
    val = ids[-n_val:]
    g = torch.Generator(device=DEV).manual_seed(0)
    ix = torch.randint(0, len(val) - K - 1, (1024,), generator=g, device=DEV)
    ctx = torch.stack([val[i:i + K] for i in ix])
    red = reduced[ctx]
    with torch.no_grad():
        t_arg = teacher(input_ids=ctx).logits[:, -1, :].float()[:, top_d]\
            .argmax(1) + 1

    # pos-slot probes
    rows = probes_mod.build()["pos-slot"]
    def enc(text):
        i = tok(text).input_ids[-K:]
        return [0] * (K - len(i)) + [int(reduced[x]) for x in i]
    pr_ctx = torch.tensor([enc(c) for c, _, _ in rows], device=DEV)
    def rid(ws):
        return [int(reduced[tok(w).input_ids[0]]) for w in ws]
    ans = [(rid(c), rid(w)) for _, c, w in rows]

    def metrics(m):
        with torch.no_grad():
            _, z = m.forward(red)
            agree = (z.argmax(1) == t_arg).float().mean().item()
            _, zp = m.forward(pr_ctx)
            q = torch.softmax(zp * 60, 1)
            acc = sum(int(q[i, c].sum() > q[i, w].sum())
                      for i, (c, w) in enumerate(ans)) / len(ans)
        return agree, acc

    a0, p0 = metrics(model)
    print(f"float32 baseline: agree={a0:.3f} pos-slot={p0:.2f}", flush=True)
    out = {"float32": {"agree": a0, "posslot": p0}}
    for b in BITS:
        q = TokenUnitLM(model.V, model.K, model.E,
                        [model.trunk.Ps[0].shape[0]], device=DEV)
        q.act = q.emb.act = q.trunk.act = model.act
        q.emb.Ps[0] = quantize_P(model.emb.Ps[0], b)
        q.emb.Ns[0] = quantize_N(model.emb.Ns[0], b)
        for i in range(2):
            q.trunk.Ps[i] = quantize_P(model.trunk.Ps[i], b)
            q.trunk.Ns[i] = quantize_N(model.trunk.Ns[i], b)
        legal = not (q.emb.check() + q.trunk.check())
        a, p = metrics(q)
        nz = sum(int((P > 0).sum()) for P in
                 [q.emb.Ps[0]] + q.trunk.Ps)
        print(f"b={b:2d} bits: agree={a:.3f} pos-slot={p:.2f} "
              f"legal={legal} nonzero P entries={nz:,}", flush=True)
        out[b] = {"agree": a, "posslot": p, "nonzeroP": nz}
    json.dump(out, open(os.path.join(HERE, "quant_lm_results.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
