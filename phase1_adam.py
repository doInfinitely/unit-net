"""Phase 1 (Adam variant): same distillation, same unit-net architecture,
but trained through the constraint-satisfying reparameterization
(P = row-softmax, N = sigmoid) with Adam — the prior session's oracle
recipe. Exported weights are identical unit-net structure; the exact J-lens
is a property of the architecture, not the trainer.
"""
import argparse
import os
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


class ReparamLM(torch.nn.Module):
    def __init__(self, V, K, E, H, act="relu", n_cap=2.0):
        super().__init__()
        self.V, self.K, self.E, self.H = V, K, E, H
        self.act_name = act
        self.act = torch.relu if act == "relu" else torch.sigmoid
        self.n_cap = n_cap
        self.We = torch.nn.Parameter(torch.randn(E, V + 1) * 3.0)
        self.Ve = torch.nn.Parameter(torch.full((E, V + 1), -8.0))
        self.Wh = torch.nn.Parameter(torch.randn(H, K * E + 1) * 3.0)
        self.Vh = torch.nn.Parameter(torch.full((H, K * E + 1), -8.0))
        self.Wo = torch.nn.Parameter(torch.randn(V, H + 1) * 2.0)
        self.Vo = torch.nn.Parameter(torch.full((V, H + 1), -8.0))

    def mats(self):
        out = []
        for W, Vv in ((self.We, self.Ve), (self.Wh, self.Vh),
                      (self.Wo, self.Vo)):
            P = torch.softmax(W, dim=1)
            N = torch.sigmoid(Vv)
            # differentiable inhibition budget: row sums capped at n_cap so
            # no neuron can be buried below zero for every input (the relu
            # death mechanism is N overgrowth)
            scale = (self.n_cap / N.sum(dim=1, keepdim=True)).clamp(max=1.0)
            out.append((P, N * scale))
        return out

    def forward(self, red_ids):
        bs = red_ids.shape[0]
        (Pe, Ne), (Ph, Nh), (Po, No) = self.mats()
        x = torch.zeros(bs * self.K, self.V, device=red_ids.device)
        x.scatter_(1, red_ids.reshape(-1, 1), 1.0)
        ones = torch.ones(bs * self.K, 1, device=x.device)
        e = self.act(torch.cat([x, ones], 1) @ (Pe - Ne).T)
        feat = e.reshape(bs, self.K * self.E)
        ones = torch.ones(bs, 1, device=x.device)
        zh = torch.cat([feat, ones], 1) @ (Ph - Nh).T
        self.last_zh = zh
        h = self.act(zh)
        self.last_h = h
        z = torch.cat([h, ones], 1) @ (Po - No).T
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--embed", type=int, default=96)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--l1-n", type=float, default=1e-3)
    ap.add_argument("--act", default="relu")
    ap.add_argument("--n-cap", type=float, default=2.0)
    ap.add_argument("--lam-live", type=float, default=1.0)
    ap.add_argument("--live-q", type=float, default=0.05)
    ap.add_argument("--lam-sparse", type=float, default=3e-3)
    ap.add_argument("--lam-sel", type=float, default=0.0,
                    help="targeted ownership: push concept tokens to own "
                         "dedicated carrier units (exclusive-max margin)")
    ap.add_argument("--sel-words", default=" Lily, dragon, happy, scared,"
                    " park, mom")
    ap.add_argument("--logit-scale", type=float, default=60.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="model_phase1_adam.npz")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16).to(dev).eval()

    ids = torch.load(os.path.join(HERE, "tinystories_tokens.pt"),
                     weights_only=True)
    vm = torch.load(os.path.join(HERE, "vocab_map.pt"), weights_only=True)
    reduced, inv = vm["reduced"], vm["inv"]
    top_d = inv[1:].to(dev)
    V, K, H, E = args.vocab, args.context, args.hidden, args.embed
    ids = ids.to(dev)
    reduced_d = reduced.to(dev)
    n_val = len(ids) // 20
    tr_ids, va_ids = ids[:-n_val], ids[-n_val:]

    model = ReparamLM(V, K, E, H, act=args.act,
                      n_cap=args.n_cap).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"ReparamLM (Adam) [table {V}->{E} x{K}; {K*E}->{H}->{V}]",
          flush=True)

    def batch(src, bs):
        ix = torch.randint(0, len(src) - K - 1, (bs,), generator=gen,
                           device=dev)
        ctx = torch.stack([src[i:i + K] for i in ix])
        return ctx, reduced_d[ctx], reduced_d[src[ix + K]]

    @torch.no_grad()
    def teacher_probs(ctx):
        out = teacher(input_ids=ctx).logits[:, -1, :].float()
        # restrict + renormalize over the real reduced tokens; the UNK slot
        # gets ZERO target mass (aggregating the tail made UNK the argmax
        # ~1/3 of the time and poisoned the whole distillation)
        sub = torch.softmax(out[:, top_d], dim=1)
        return torch.cat([torch.zeros_like(sub[:, :1]), sub], dim=1)

    sel_ids = [int(reduced[tok(w).input_ids[0]])
               for w in args.sel_words.split(",")]
    sel_ids = [c for c in sel_ids if c > 0]
    print(f"ownership targets (reduced ids): {sel_ids}", flush=True)
    t0 = time.time()
    best_agree, best_sd = -1.0, None
    for step in range(args.steps):
        ctx, red, y = batch(tr_ids, args.batch)
        p_t = teacher_probs(ctx)
        z = model(red)
        logq = torch.log_softmax(z * args.logit_scale, dim=1)
        loss = -(p_t * logq).sum(1).mean() \
            + torch.nn.functional.nll_loss(logq, y)  # hard-target mix
        loss = loss + args.l1_n * (torch.sigmoid(model.Ve).mean()
                                   + torch.sigmoid(model.Vh).mean()
                                   + torch.sigmoid(model.Vo).mean())
        # liveness: each hidden unit should fire on >= live_q of the batch
        alive = torch.sigmoid(model.last_zh * 20).mean(0)
        loss = loss + args.lam_live * torch.relu(args.live_q - alive).mean()
        # selectivity pressure: sparse hidden activity
        loss = loss + args.lam_sparse * model.last_h.mean()
        if args.lam_sel:
            (Po_, No_) = model.mats()[2]
            Wo_ = (Po_ - No_)[:, :model.H]              # (V, H)
            vals, idxs = Wo_.topk(2, dim=0)              # (2, H)
            margins = []
            for c in sel_ids:
                # competitor strength per unit, excluding token c itself
                comp = torch.where(idxs[0] == c, vals[1], vals[0])
                margins.append((Wo_[c] - comp).max())
            loss = loss - args.lam_sel * torch.stack(margins).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                ctx, red, y = batch(va_ids, 1024)
                p_t = teacher_probs(ctx)
                z = model(red)
                agree = (z.argmax(1) == p_t.argmax(1)).float().mean().item()
                acc = (z.argmax(1) == y).float().mean().item()
                q = torch.softmax(z * args.logit_scale, 1)
                kl = (p_t * (torch.log(p_t + 1e-9) - torch.log(q + 1e-9))
                      ).sum(1).mean().item()
            if agree > best_agree:
                best_agree = agree
                best_sd = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}
            with torch.no_grad():
                hb = model.last_h
                dead = int((hb.max(0).values <= 0).sum())
                (Po_, No_) = model.mats()[2]
                Wo = (Po_ - No_)[:, :H]
                v2, i2 = Wo.topk(2, dim=0)
                sel = torch.stack(
                    [(Wo[c] - torch.where(i2[0] == c, v2[1], v2[0])).max()
                     for c in sel_ids])
            print(f"step {step+1:5d} KL={kl:.3f} agree={agree:.3f} "
                  f"acc={acc:.3f} best={best_agree:.3f} dead={dead} "
                  f"concept_sel={sel.mean().item():+.4f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_sd)
    with torch.no_grad():
        (Pe, Ne), (Ph, Nh), (Po, No) = model.mats()
        # exact renorm + export in TokenUnitLM format
        arrs = {"V": np.array(V), "K": np.array(K), "E": np.array(E),
                "act": np.array(args.act), "n_trunk": np.array(2)}
        for name, P, N in (("emb", Pe, Ne), ("t0", Ph, Nh), ("t1", Po, No)):
            P = (P.double() / P.double().sum(1, keepdim=True)).float()
            key = {"emb": ("emb_P", "emb_N"), "t0": ("tP0", "tN0"),
                   "t1": ("tP1", "tN1")}[name]
            arrs[key[0]] = P.cpu().numpy()
            arrs[key[1]] = N.clamp(0, 1).cpu().numpy()
        np.savez(os.path.join(HERE, args.out), **arrs)
    print(f"saved {args.out} (best agree={best_agree:.3f})")


if __name__ == "__main__":
    main()
