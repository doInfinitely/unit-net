"""Phase 0: char-level language model as a unit-net, trained by mirror
descent. Proves sequence prediction works in the constrained geometry and
demos the exact J-lens: prime with a context, read which hidden units carry
strong bounded paths to which output tokens.

One-hot inputs are naturally in {0,1}; the first layer's rows are budgeted
distributions over (position, char) features — interpretable from step one.
"""
import argparse
import os
import time
import urllib.request
import torch
from unitnet import UnitNet, save

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "tinyshakespeare.txt")
URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--eta-p", type=float, default=0.5)
    ap.add_argument("--eta-n", type=float, default=0.01)
    ap.add_argument("--logit-scale", type=float, default=30.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="model_phase0.npz")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    if not os.path.exists(CORPUS):
        print("downloading tinyshakespeare...")
        urllib.request.urlretrieve(URL, CORPUS)
    text = open(CORPUS).read()
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], device=dev)
    n_val = len(data) // 20
    train, val = data[:-n_val], data[-n_val:]
    K = args.context
    print(f"corpus {len(text):,} chars, vocab {V}, context {K}, "
          f"input dim {K*V}")

    net = UnitNet([K * V, args.hidden, V], act="relu", device=dev, gen=gen)

    def batch(src, bs):
        ix = torch.randint(0, len(src) - K - 1, (bs,), generator=gen,
                           device=dev)
        ctx = torch.stack([src[i:i + K] for i in ix])          # (bs, K)
        x = torch.zeros(bs, K * V, device=dev)
        x.scatter_(1, ctx + torch.arange(K, device=dev) * V, 1.0)
        return x, src[ix + K]

    t0 = time.time()
    for step in range(args.steps):
        x, y = batch(train, args.batch)
        _, z = net.forward(x)
        loss = torch.nn.functional.cross_entropy(z * args.logit_scale, y)
        loss.backward()
        net.mirror_step(args.eta_p * 0.5 * (1 + torch.cos(
            torch.tensor(3.14159 * step / args.steps)).item()), args.eta_n)
        if (step + 1) % 1000 == 0:
            with torch.no_grad():
                xv, yv = batch(val, 4096)
                _, zv = net.forward(xv)
                vloss = torch.nn.functional.cross_entropy(
                    zv * args.logit_scale, yv)
                bpc = vloss.item() / torch.log(torch.tensor(2.0)).item()
                acc = (zv.argmax(1) == yv).float().mean().item()
            print(f"step {step+1:5d} val_bpc={bpc:.3f} "
                  f"next-char acc={acc:.3f} t={time.time()-t0:.0f}s "
                  f"constraints={'OK' if not net.check() else 'FAIL'}",
                  flush=True)

    # ---- sample ----
    ctx = "The king "[-K:].ljust(K)
    idx = [stoi.get(c, 0) for c in ctx]
    out = ctx
    for _ in range(200):
        x = torch.zeros(1, K * V, device=dev)
        for p, c in enumerate(idx):
            x[0, p * V + c] = 1.0
        with torch.no_grad():
            _, z = net.forward(x)
        probs = torch.softmax(z[0] * args.logit_scale, dim=0)
        c = torch.multinomial(probs, 1, generator=gen).item()
        out += chars[c]
        idx = idx[1:] + [c]
    print("\nsample:\n" + out + "\n")

    # ---- exact J-lens demo: strong paths from hidden units to tokens ----
    prime = "MENENIUS:"[-K:].ljust(K)
    x = torch.zeros(1, K * V, device=dev)
    for p, c in enumerate(prime):
        x[0, p * V + stoi.get(c, 0)] = 1.0
    acts, z = net.forward(x)
    lenses = net.jlens_exact(x)
    J0 = lenses[0]  # (V_out, K*V) exact input->output jacobian... layer 0
    h = acts[1][0]  # hidden activations
    top_units = h.argsort(descending=True)[:5]
    print(f"primed with {prime!r}; top hidden units and their strongest "
          f"bounded paths to output tokens (exact J-lens, fixed units):")
    Jh = lenses[1] if 1 in lenses else None
    W_out = (net.Ps[-1] - net.Ns[-1])[:, :-1]  # (V, H)
    for u in top_units.tolist():
        disp = W_out[:, u] * h[u]  # bounded contribution of unit u to each z
        top_tok = disp.argsort(descending=True)[:4]
        toks = ", ".join(f"{chars[t]!r}:{disp[t]:+.3f}" for t in
                         top_tok.tolist())
        print(f"  unit {u:3d} (a={h[u]:.3f}) -> {toks}")
    save(net, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
