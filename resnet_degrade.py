"""Quantization + flow-pruning degradation curves for the unit-net
transplanted ResNet18 (per-filter integer-simplex rows; pruning re-folds
the post-prune renormalization scale into BN so only the pruning itself
changes the function)."""
import json
import os
import torch
import torchvision
from torchvision import transforms as T
from resnet_transplant import (transplant_resnet, evaluate, WNIDS)

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda:1"
BITS = [14, 12, 10, 8, 6, 4, 3, 2]
PRUNE = [0.5, 0.8, 0.9, 0.95, 0.98]


def conv_bn_pairs(model):
    pairs = [(model.conv1, model.bn1)]
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            pairs.append((block.conv1, block.bn1))
            pairs.append((block.conv2, block.bn2))
            if block.downsample is not None:
                pairs.append((block.downsample[0], block.downsample[1]))
    return pairs


def quantize_row_simplex(flat, b):
    """flat: (out, k) rows with pos parts summing to 1. Quantize pos part to
    integer numerators / 2^b (largest remainder), neg part to 2^b grid."""
    D = 2 ** b
    pos = flat.clamp(min=0)
    neg = flat.clamp(max=0)
    scaled = pos * D
    base = torch.floor(scaled)
    short = (D - base.sum(1)).round().long()
    rem = scaled - base
    order = rem.argsort(1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(flat.shape[1], device=flat.device)
                   .expand_as(order))
    base += (ranks < short.unsqueeze(1)).float()
    negq = -((-neg) * (D - 1)).round() / (D - 1)
    return base / D + negq


def refold_(conv, bn, s):
    """Fold a fresh per-filter scale s into the BN (exact, incl. eps)."""
    eps = bn.eps
    var = bn.running_var.data
    bn.weight.data.mul_(torch.sqrt(var / s**2 + eps) * s
                        / torch.sqrt(var + eps))
    bn.running_mean.data.div_(s)
    bn.running_var.data.div_(s**2)


def fresh():
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(DEV).eval()
    transplant_resnet(m)
    return m


def main():
    tf = T.Compose([T.Resize(180), T.CenterCrop(160), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    ds = torchvision.datasets.ImageFolder(
        os.path.join(HERE, "imagenette2-160", "val"), transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)

    base = fresh()
    acc0, _ = evaluate(base, loader, DEV)
    print(f"transplanted baseline: top-1 {acc0:.4f}", flush=True)
    out = {"baseline": acc0, "quant": {}, "prune": {}}

    for b in BITS:
        m = fresh()
        with torch.no_grad():
            for conv, bn in conv_bn_pairs(m):
                W = conv.weight.data
                flat = quantize_row_simplex(W.view(W.shape[0], -1), b)
                W.copy_(flat.view_as(W))
        acc, agr = evaluate(m, loader, DEV, ref_model=base)
        print(f"quant b={b:2d}: top-1 {acc:.4f} agreement {agr:.4f}",
              flush=True)
        out["quant"][b] = {"acc": acc, "agree": agr}
        del m
        torch.cuda.empty_cache()

    # ---- activation flows via hooks ----
    m = fresh()
    flows = {}
    handles = []
    with torch.no_grad():
        acts = {}
        def mk(name):
            def hook(mod, inp, _):
                a = inp[0].abs().mean(dim=(0, 2, 3))          # per in-channel
                acts[name] = acts.get(name, 0) + a
            return hook
        for i, (conv, _) in enumerate(conv_bn_pairs(m)):
            handles.append(conv.register_forward_hook(mk(i)))
        for j, (x, _) in enumerate(loader):
            m(x.to(DEV))
            if j >= 3:
                break
        for h in handles:
            h.remove()
        for i, (conv, _) in enumerate(conv_bn_pairs(m)):
            W = conv.weight.data.abs()                        # (o,i,kh,kw)
            flows[i] = W * acts[i].view(1, -1, 1, 1)
    all_flow = torch.cat([f.flatten() for f in flows.values()])

    for frac in PRUNE:
        floor = all_flow.quantile(frac)
        m = fresh()
        with torch.no_grad():
            for i, (conv, bn) in enumerate(conv_bn_pairs(m)):
                keep = flows[i] >= floor
                W = conv.weight.data
                W.mul_(keep)
                flat = W.view(W.shape[0], -1)
                s = flat.clamp(min=0).sum(1).clamp(min=1e-12)
                flat.div_(s.unsqueeze(1))                     # re-legalize
                flat.clamp_(min=-1)
                refold_(conv, bn, s)                          # exact re-fold
        acc, agr = evaluate(m, loader, DEV, ref_model=base)
        print(f"prune {frac:.0%}: top-1 {acc:.4f} agreement {agr:.4f}",
              flush=True)
        out["prune"][frac] = {"acc": acc, "agree": agr}
        del m
        torch.cuda.empty_cache()

    json.dump(out, open(os.path.join(HERE, "resnet_degrade.json"), "w"),
              indent=1)
    print("wrote resnet_degrade.json")


if __name__ == "__main__":
    main()
