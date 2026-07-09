"""Phase 1: distill Qwen2.5-0.5B into a unit-net on TinyStories.

Teacher: open-weights decoder, run online per batch (bf16). Student: K-token
one-hot context -> unit-net (dual projections, convex rows, bounded
inhibition) -> logits over a reduced top-V vocabulary. Loss: KL from the
teacher's next-token distribution restricted+renormalized to the reduced
vocab. Trained by mirror descent (native geometry, no reparameterization).
"""
import argparse
import os
import time
import torch
from unitnet import TokenUnitLM

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--eta-p", type=float, default=0.5)
    ap.add_argument("--eta-n", type=float, default=0.01)
    ap.add_argument("--logit-scale", type=float, default=40.0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="model_phase1.npz")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16).to(dev).eval()
    print(f"teacher {args.teacher} loaded "
          f"({sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params)",
          flush=True)

    # ---- tokenize corpus once, cache ----
    cache = os.path.join(HERE, "tinystories_tokens.pt")
    if os.path.exists(cache):
        ids = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        text = open(os.path.join(HERE, "tinystories_valid.txt")).read()
        ids = []
        CH = 500_000
        for i in range(0, len(text), CH):
            ids.extend(tok(text[i:i + CH]).input_ids)
        ids = torch.tensor(ids, dtype=torch.long)
        torch.save(ids, cache)
    print(f"corpus: {len(ids):,} teacher tokens", flush=True)

    # ---- reduced vocab: top-V by frequency; slot 0 = UNK ----
    vmap_path = os.path.join(HERE, "vocab_map.pt")
    vt = len(tok)  # includes added special tokens (<|endoftext|> etc.)
    counts = torch.bincount(ids, minlength=vt)
    top = counts.argsort(descending=True)[:args.vocab - 1]
    reduced = torch.full((vt,), 0, dtype=torch.long)  # UNK=0
    reduced[top] = torch.arange(1, args.vocab)
    inv = torch.zeros(args.vocab, dtype=torch.long)
    inv[1:] = top
    torch.save({"reduced": reduced, "inv": inv,
                "teacher": args.teacher}, vmap_path)
    cover = counts[top].sum().item() / counts.sum().item()
    print(f"reduced vocab {args.vocab} covers {cover:.1%} of corpus tokens",
          flush=True)

    V, K, H = args.vocab, args.context, args.hidden
    ids = ids.to(dev)
    reduced_d = reduced.to(dev)
    top_d = top.to(dev)  # the V-1 real teacher ids (slot 0 is UNK)
    n_val = len(ids) // 20
    tr_ids, va_ids = ids[:-n_val], ids[-n_val:]

    net = TokenUnitLM(V, K, args.embed, [H], device=dev, gen=gen)
    n_params = sum(P.numel() + N.numel() for m in net.nets()
                   for P, N in zip(m.Ps, m.Ns))
    print(f"student TokenUnitLM [table {V}->{args.embed} shared x{K}; "
          f"trunk {K*args.embed} -> {H} -> {V}] "
          f"params={n_params/1e6:.1f}M", flush=True)

    def batch(src, bs):
        ix = torch.randint(0, len(src) - K - 1, (bs,), generator=gen,
                           device=dev)
        ctx = torch.stack([src[i:i + K] for i in ix])       # teacher ids
        return ctx, reduced_d[ctx], reduced_d[src[ix + K]]

    @torch.no_grad()
    def teacher_probs(ctx):
        out = teacher(input_ids=ctx).logits[:, -1, :].float()
        # restrict + renormalize over the real reduced tokens; the UNK slot
        # gets ZERO target mass (aggregating the tail made UNK the argmax
        # ~1/3 of the time and poisoned the whole distillation)
        sub = torch.softmax(out[:, top_d], dim=1)
        return torch.cat([torch.zeros_like(sub[:, :1]), sub], dim=1)

    t0 = time.time()
    best_agree, best_state = -1.0, None
    for step in range(args.steps):
        frac = step / max(1, args.steps - 1)
        eta = args.eta_p * 0.5 * (1 + torch.cos(
            torch.tensor(3.14159 * frac)).item())
        ctx, red, y = batch(tr_ids, args.batch)
        p_t = teacher_probs(ctx)
        _, z = net.forward(red)
        logq = torch.log_softmax(z * args.logit_scale, dim=1)
        loss = -(p_t * logq).sum(dim=1).mean()          # CE(p_teacher, q)
        loss.backward()
        net.step(eta, args.eta_n)
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                ctx, red, y = batch(va_ids, 1024)
                p_t = teacher_probs(ctx)
                _, z = net.forward(red)
                q = torch.softmax(z * args.logit_scale, dim=1)
                agree = (z.argmax(1) == p_t.argmax(1)).float().mean().item()
                acc = (z.argmax(1) == y).float().mean().item()
                kl = (p_t * (torch.log(p_t + 1e-9)
                             - torch.log(q + 1e-9))).sum(1).mean().item()
            if agree > best_agree:
                best_agree = agree
                best_state = [(m.Ps[i].detach().clone(),
                               m.Ns[i].detach().clone())
                              for m in net.nets()
                              for i in range(len(m.Ps))]
            print(f"step {step+1:5d} KL={kl:.3f} "
                  f"agree@1(teacher)={agree:.3f} next-tok acc={acc:.3f} "
                  f"best={best_agree:.3f} t={time.time()-t0:.0f}s "
                  f"constraints={'OK' if not net.check() else 'FAIL'}",
                  flush=True)

    if best_state is not None:  # restore best-agreement checkpoint
        flat = [(m, i) for m in net.nets() for i in range(len(m.Ps))]
        for (m, i), (P, N) in zip(flat, best_state):
            m.Ps[i] = P.requires_grad_()
            m.Ns[i] = N.requires_grad_()
    net.save(os.path.join(HERE, args.out))
    print(f"saved {args.out}")

    # quick sample: greedy continuation in reduced-token space
    prompt = "Once upon a time, there was a little"
    pids = tok(prompt).input_ids[-K:]
    pids = [0] * (K - len(pids)) + pids
    red = [int(reduced[i]) for i in pids]
    outs = []
    for _ in range(30):
        rt = torch.tensor([red], device=dev)
        with torch.no_grad():
            _, z = net.forward(rt)
        r = int(z[0].argmax())
        outs.append(int(inv[r]) if r > 0 else tok.eos_token_id)
        red = red[1:] + [r]
    print("student continuation:", repr(tok.decode(outs)))


if __name__ == "__main__":
    main()
