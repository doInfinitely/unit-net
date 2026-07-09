"""Controlled experiment: does engineered unit-ownership decouple a concept
from the distributed predictive flow (lower reportable correlation)?

Crossed design: model A owns {Lily, dragon, happy}, model B owns
{scared, park, mom}; every concept is measured owned and unowned on a shared
eval set, serving as its own control.
"""
import os
import torch
from unitnet import TokenUnitLM

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1"
M = 2048
CONCEPTS = [" Lily", " dragon", " happy", " scared", " park", " mom"]
OWNED_A = {" Lily", " dragon", " happy"}


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def main():
    torch.manual_seed(0)
    vm = torch.load(os.path.join(HERE, "vocab_map.pt"), weights_only=True)
    ids = torch.load(os.path.join(HERE, "tinystories_tokens.pt"),
                     weights_only=True).to(DEV)
    reduced = vm["reduced"].to(DEV)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(vm["teacher"])
    teacher = AutoModelForCausalLM.from_pretrained(
        vm["teacher"], dtype=torch.bfloat16).to(DEV).eval()

    models = {"A": TokenUnitLM.load(os.path.join(HERE, "model_ownA.npz"), DEV),
              "B": TokenUnitLM.load(os.path.join(HERE, "model_ownB.npz"), DEV)}
    K = models["A"].K
    n_val = len(ids) // 20
    val = ids[-n_val:]
    ix = torch.randint(0, len(val) - K - 1, (M,))
    ctx = torch.stack([val[i:i + K] for i in ix])
    red = reduced[ctx]
    with torch.no_grad():
        t_logp = torch.log_softmax(
            teacher(input_ids=ctx).logits[:, -1, :].float(), dim=1)

    z = {}
    sel = {}
    for name, m in models.items():
        (_, _, tacts), zz = m.forward(red)
        z[name] = zz
        H = m.trunk.Ps[0].shape[0]
        Wo = (m.trunk.Ps[1] - m.trunk.Ns[1])[:, :H]
        v2, i2 = Wo.topk(2, dim=0)
        sel[name] = lambda c, Wo=Wo, v2=v2, i2=i2: float(
            (Wo[c] - torch.where(i2[0] == c, v2[1], v2[0])).max())

    print(f"{'concept':10s} {'owner':6s} | {'rho owned':>9s} {'rho unowned':>11s}"
          f" | {'sel owned':>9s} {'sel unowned':>11s}")
    diffs = []
    for w in CONCEPTS:
        tid = tok(w).input_ids[0]
        c = int(reduced[tid])
        owner = "A" if w in OWNED_A else "B"
        other = "B" if owner == "A" else "A"
        rho_own = spearman(z[owner][:, c], t_logp[:, tid])
        rho_un = spearman(z[other][:, c], t_logp[:, tid])
        print(f"{w!r:10s} {owner:6s} | {rho_own:+9.3f} {rho_un:+11.3f}"
              f" | {sel[owner](c):+9.3f} {sel[other](c):+11.3f}")
        diffs.append(rho_own - rho_un)
    d = torch.tensor(diffs)
    print(f"\npaired effect (rho_owned - rho_unowned): "
          f"mean {d.mean():+.3f}, per-concept {[round(x,3) for x in diffs]}")
    print("negative mean => ownership decouples the concept from the "
          "predictive flow; ~zero => the 1-run observation was noise")


if __name__ == "__main__":
    main()
