"""Idea 3: activation-flow pruning.

Run inference over a corpus sample; for every edge accumulate its
activation flow E[|w_ij| * a_j] (how much signal actually crosses it).
That is the importance ranking. Sweep a flow floor: prune all edges below
it (N entries zeroed; P entries zeroed then rows renormalized — the pruned
model stays a legal unit-net) and measure divergence (KL vs the full
student) and degradation (teacher agreement, pos-slot 2AFC) as the floor
rises. Task-agnostic counterpart of j-carve's task-specific path mass.
"""
import json
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1"
PRUNE_FRACTIONS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]


def main():
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
    K, V, E = model.K, model.V, model.E
    H = model.trunk.Ps[0].shape[0]
    n_val = len(ids) // 20
    val = ids[-n_val:]
    g = torch.Generator(device=DEV).manual_seed(0)
    ix = torch.randint(0, len(val) - K - 1, (2048,), generator=g, device=DEV)
    ctx = torch.stack([val[i:i + K] for i in ix])
    red = reduced[ctx]
    with torch.no_grad():
        t_arg = teacher(input_ids=ctx).logits[:, -1, :].float()[:, top_d]\
            .argmax(1) + 1
        (eacts, feat, tacts), z_full = model.forward(red)
        q_full = torch.softmax(z_full * 60, 1)

    # ---- activation flow per edge ----
    # emb layer inputs are one-hots: flow through emb edge (e, t) =
    # |W[e,t]| * P(token t appears); position-agnostic token frequencies:
    tok_freq = torch.bincount(red.flatten(), minlength=V).float()
    tok_freq = tok_freq / tok_freq.sum()
    a_emb = torch.cat([tok_freq, torch.ones(1, device=DEV) / K])  # const col
    flows = {}
    for name, net, li, a_mean in [
        ("emb", model.emb, 0, a_emb),
        ("trunk0", model.trunk, 0,
         torch.cat([feat.mean(0), torch.ones(1, device=DEV)])),
        ("trunk1", model.trunk, 1,
         torch.cat([tacts[1].mean(0), torch.ones(1, device=DEV)]))]:
        W = (net.Ps[li] - net.Ns[li]).abs()
        flows[name] = W * a_mean.unsqueeze(0)

    rows = probes_mod.build()["pos-slot"]
    def enc(text):
        i = tok(text).input_ids[-K:]
        return [0] * (K - len(i)) + [int(reduced[x]) for x in i]
    pr_ctx = torch.tensor([enc(c) for c, _, _ in rows], device=DEV)
    def rid(ws):
        return [int(reduced[tok(w).input_ids[0]]) for w in ws]
    ans = [(rid(c), rid(w)) for _, c, w in rows]

    all_flow = torch.cat([f.flatten() for f in flows.values()])
    out = {}
    for frac in PRUNE_FRACTIONS:
        floor = all_flow.quantile(frac)
        m = TokenUnitLM(V, K, E, [H], device=DEV)
        m.act = m.emb.act = m.trunk.act = model.act
        pruned = 0
        for name, net_src, net_dst, li in [
                ("emb", model.emb, m.emb, 0),
                ("trunk0", model.trunk, m.trunk, 0),
                ("trunk1", model.trunk, m.trunk, 1)]:
            keep = flows[name] >= floor
            pruned += int((~keep).sum())
            P = net_src.Ps[li] * keep
            # rows that lost everything keep their single largest entry
            dead = P.sum(1) == 0
            if dead.any():
                top1 = net_src.Ps[li][dead].argmax(1)
                P[dead, top1] = net_src.Ps[li][dead, top1]
            net_dst.Ps[li] = P / P.sum(1, keepdim=True)
            net_dst.Ns[li] = net_src.Ns[li] * keep
        with torch.no_grad():
            _, z = m.forward(red)
            q = torch.softmax(z * 60, 1)
            kl = (q_full * (torch.log(q_full + 1e-9)
                            - torch.log(q + 1e-9))).sum(1).mean().item()
            agree = (z.argmax(1) == t_arg).float().mean().item()
            _, zp = m.forward(pr_ctx)
            qp = torch.softmax(zp * 60, 1)
            acc = sum(int(qp[i, c].sum() > qp[i, w].sum())
                      for i, (c, w) in enumerate(ans)) / len(ans)
        legal = not (m.emb.check() + m.trunk.check())
        print(f"prune {frac:.0%} of edges (floor={floor:.2e}): "
              f"KL={kl:.3f} agree={agree:.3f} pos-slot={acc:.2f} "
              f"legal={legal}", flush=True)
        out[frac] = {"kl": kl, "agree": agree, "posslot": acc}
    json.dump(out, open(os.path.join(HERE, "flow_prune_results.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
