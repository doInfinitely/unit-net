"""Folded unit-net transplant of a pretrained ResNet18.

Every conv in ResNet is followed by BatchNorm, and ReLU is positively
homogeneous — so per-filter scales fold EXACTLY: divide each conv filter
(one output channel) by its positive weight sum, then absorb the scale in
the BN that consumes it (running_mean /= s, running_var /= s^2, and an
exact epsilon-correction gamma *= sqrt(var/s^2+eps)*s/sqrt(var+eps)).
Negatives land in [-1,0] wherever |neg|max <= pos_sum (clip + count).

Result: a ResNet whose conv weights all obey the unit-net row law
(positives sum to 1, negatives bounded by -1) with — up to clips —
IDENTICAL function. Measured on Imagenette (real ImageNet images,
10 classes): top-1 accuracy of both models + argmax agreement.
The final fc has no BN to fold into: constrained-unfolded as a measured
ablation (--fc).
"""
import argparse
import os
import torch
import torchvision
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
# imagenette class dirs -> imagenet class indices
WNIDS = ["n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
         "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"]
IMNET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def transplant_conv_bn_(conv, bn):
    """Unit-net row law on conv filters, exactly folded into the BN."""
    W = conv.weight.data                       # (out, in, kh, kw)
    out = W.shape[0]
    flat = W.view(out, -1)
    s = flat.clamp(min=0).sum(dim=1).clamp(min=1e-12)
    flat.div_(s.unsqueeze(1))
    clipped = int((flat < -1).sum())
    flat.clamp_(min=-1)
    if conv.bias is not None:
        conv.bias.data.div_(s)
    eps = bn.eps
    var = bn.running_var.data
    # exact compensation including epsilon
    bn.weight.data.mul_(torch.sqrt(var / s**2 + eps) * s
                        / torch.sqrt(var + eps))
    bn.running_mean.data.div_(s)
    bn.running_var.data.div_(s**2)
    return clipped, flat.numel()


def transplant_resnet(model, do_fc=False):
    clips = total = 0
    pairs = [(model.conv1, model.bn1)]
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            pairs.append((block.conv1, block.bn1))
            pairs.append((block.conv2, block.bn2))
            if block.downsample is not None:
                pairs.append((block.downsample[0], block.downsample[1]))
    for conv, bn in pairs:
        c, n = transplant_conv_bn_(conv, bn)
        clips += c
        total += n
    if do_fc:  # no BN downstream: unfolded (function-changing) constraint
        W = model.fc.weight.data
        s = W.clamp(min=0).sum(dim=1).clamp(min=1e-12)
        W.div_(s.unsqueeze(1))
        W.clamp_(min=-1)
        model.fc.bias.data.div_(s)
    return clips, total, len(pairs)


@torch.no_grad()
def evaluate(model, loader, dev, ref_model=None):
    correct = n = agree = 0
    for x, y in loader:
        x = x.to(dev)
        pred = model(x).argmax(1).cpu()
        # map imagenette label -> imagenet index
        target = torch.tensor([IMNET_IDX[t] for t in y])
        correct += (pred == target).sum().item()
        if ref_model is not None:
            agree += (pred == ref_model(x).argmax(1).cpu()).sum().item()
        n += len(y)
    return correct / n, (agree / n if ref_model is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--fc", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)

    tf = T.Compose([T.Resize(180), T.CenterCrop(160), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    ds = torchvision.datasets.ImageFolder(
        os.path.join(HERE, "imagenette2-160", "val"), transform=tf)
    assert [ds.classes[i] for i in range(10)] == WNIDS
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)

    ref = torchvision.models.resnet18(weights="IMAGENET1K_V1")\
        .to(dev).eval()
    acc_ref, _ = evaluate(ref, loader, dev)
    print(f"original resnet18: top-1 {acc_ref:.4f} "
          f"({len(ds)} imagenette val images)", flush=True)

    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").to(dev).eval()
    clips, total, n_pairs = transplant_resnet(m, do_fc=args.fc)
    acc, agr = evaluate(m, loader, dev, ref_model=ref)
    pct = sum(p.numel() for p in
              [c.weight for c, _ in [(m.conv1, m.bn1)]])  # placeholder
    conv_params = sum(c.weight.numel() for c in m.modules()
                      if isinstance(c, torch.nn.Conv2d))
    all_params = sum(p.numel() for p in m.parameters())
    print(f"FOLDED transplant ({n_pairs} conv-bn pairs"
          f"{' + fc unfolded' if args.fc else ''}): "
          f"top-1 {acc:.4f}, agreement with original {agr:.4f}, "
          f"clipped negatives {clips:,}/{total:,} ({100*clips/total:.3f}%)",
          flush=True)
    print(f"unit-net-legal weights: {conv_params:,} conv params of "
          f"{all_params:,} total ({100*conv_params/all_params:.1f}%)",
          flush=True)
    torch.save(m.state_dict(), os.path.join(HERE, "resnet18_unitnet.pt"))
    print("saved resnet18_unitnet.pt")


if __name__ == "__main__":
    main()
