"""Idea 1: weight-analysis TRANSPLANT instead of distillation.

Constrain Qwen2.5-0.5B in place, per Remy's procedure: in every linear
layer's weight row, (a) normalize the positive entries to sum to 1
(mode 'sum': divide by their sum; mode 'softmax': softmax over the positive
entries), (b) divide all negative entries by the magnitude of the row's
most negative entry (so negatives land in [-1, 0], i.e. legal N).

This yields a weight-constrained transformer (unit-net weight law; the
attention/norm ops stay as-is). Measure argmax agreement with the original
model on held-out TinyStories windows — the number to beat is the
distilled student's 33%. Also: a layer-wise sweep (constrain only the
first L blocks) to see which layers tolerate the constraint.
"""
import argparse
import copy
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def constrain_(W, mode):
    """In-place unit-net row law on a weight matrix (out_features, in)."""
    pos = W.clamp(min=0)
    neg = W.clamp(max=0)
    if mode == "softmax":
        # softmax over positive entries only, zeros stay zero
        m = pos > 0
        e = torch.where(m, (pos - pos.max(dim=1, keepdim=True).values).exp(),
                        torch.zeros_like(pos))
        pos = e / e.sum(dim=1, keepdim=True).clamp(min=1e-12)
    else:  # 'sum'
        pos = pos / pos.sum(dim=1, keepdim=True).clamp(min=1e-12)
    nmax = (-neg).max(dim=1, keepdim=True).values.clamp(min=1e-12)
    neg = neg / nmax
    W.copy_(pos + neg)


LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def transplant(model, mode, n_layers=None):
    """Constrain the first n_layers blocks (all if None). Returns model."""
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            if n_layers is not None and li >= n_layers:
                break
            for name, mod in layer.named_modules():
                if any(name.endswith(n) for n in LINEAR_NAMES):
                    constrain_(mod.weight.data, mode)
    return model


@torch.no_grad()
def agreement(model, ref_logits, ctx, top_d, bs=64):
    both, red_both = 0, 0
    for i in range(0, len(ctx), bs):
        out = model(input_ids=ctx[i:i + bs]).logits[:, -1, :].float()
        both += (out.argmax(1) == ref_logits[i:i + bs].argmax(1)).sum().item()
        red_both += (out[:, top_d].argmax(1)
                     == ref_logits[i:i + bs][:, top_d].argmax(1)).sum().item()
    return both / len(ctx), red_both / len(ctx)


def constrain_folded_(W_src, W_dst, dst_is_col=True):
    """Exact-compensation law: divide each row of W_src by its positive sum
    (positives then sum to 1); negatives land in [-1,0] iff |neg|max <=
    pos_sum (clip + count violations). The lost scale s folds EXACTLY into
    the corresponding columns of W_dst (linear path: v->o, up->down)."""
    with torch.no_grad():
        s = W_src.clamp(min=0).sum(dim=1).clamp(min=1e-12)
        W_src.div_(s.unsqueeze(1))
        clipped = int((W_src < -1).sum())
        W_src.clamp_(min=-1)
        if dst_is_col:
            W_dst.mul_(s.unsqueeze(0).to(W_dst.dtype))
    return clipped


def transplant_folded(model):
    """Constrain v_proj and up_proj with exact scale-folding into o_proj /
    down_proj. Returns total clipped negatives."""
    clips = 0
    cfg = model.config
    n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads
    for layer in model.model.layers:
        sa, mlp = layer.self_attn, layer.mlp
        # v -> o : attention mixes values linearly. GQA: each kv head's
        # value rows feed n_rep query heads' o_proj columns — replicate the
        # per-row scales accordingly before folding.
        W = sa.v_proj.weight.data
        with torch.no_grad():
            s = W.clamp(min=0).sum(dim=1).clamp(min=1e-12)
            W.div_(s.unsqueeze(1))
            if sa.v_proj.bias is not None:  # v = Wx + b: scale b too
                sa.v_proj.bias.data.div_(s.to(sa.v_proj.bias.dtype))
            clips += int((W < -1).sum())
            W.clamp_(min=-1)
            s_exp = s.view(cfg.num_key_value_heads, hd)
            s_exp = s_exp.repeat_interleave(n_rep, dim=0).flatten()
            sa.o_proj.weight.data.mul_(
                s_exp.unsqueeze(0).to(sa.o_proj.weight.dtype))
        # up -> down : enters the SwiGLU product linearly
        clips += constrain_folded_(mlp.up_proj.weight.data,
                                   mlp.down_proj.weight.data)
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--layerwise", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)

    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16).to(dev).eval()

    ids = torch.load(os.path.join(HERE, "tinystories_tokens.pt"),
                     weights_only=True).to(dev)
    vm = torch.load(os.path.join(HERE, "vocab_map.pt"), weights_only=True)
    top_d = vm["inv"][1:].to(dev)
    K = 12
    n_val = len(ids) // 20
    val = ids[-n_val:]
    ix = torch.randint(0, len(val) - K - 1, (args.n_eval,),
                       generator=torch.Generator().manual_seed(0))
    ctx = torch.stack([val[i:i + K] for i in ix])

    with torch.no_grad():
        ref_logits = torch.cat(
            [ref(input_ids=ctx[i:i + 64]).logits[:, -1, :].float()
             for i in range(0, len(ctx), 64)])

    n_blocks = len(ref.model.layers)
    for mode in ("sum", "softmax"):
        m = copy.deepcopy(ref)
        transplant(m, mode)
        full, red = agreement(m, ref_logits, ctx, top_d)
        print(f"transplant mode={mode:8s} ALL {n_blocks} blocks: "
              f"agreement full-vocab {full:.3f}, reduced-vocab {red:.3f} "
              f"(distilled student reference: 0.33 reduced)", flush=True)
        del m
        torch.cuda.empty_cache()

    m = copy.deepcopy(ref)
    clips = transplant_folded(m)
    full, red = agreement(m, ref_logits, ctx, top_d)
    nv = sum(l.self_attn.v_proj.weight.numel() + l.mlp.up_proj.weight.numel()
             for l in ref.model.layers)
    print(f"FOLDED transplant (v_proj + up_proj, exact compensation): "
          f"agreement full {full:.3f}, reduced {red:.3f}; "
          f"clipped negatives {clips}/{nv} ({100*clips/nv:.3f}%)", flush=True)
    del m
    torch.cuda.empty_cache()

    if args.layerwise:
        for L in (1, 2, 4, 8, 12, 16, 20, 24):
            if L > n_blocks:
                break
            m = copy.deepcopy(ref)
            transplant(m, "sum", n_layers=L)
            full, red = agreement(m, ref_logits, ctx, top_d)
            print(f"  first {L:2d}/{n_blocks} blocks constrained: "
                  f"full {full:.3f}, reduced {red:.3f}", flush=True)
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
