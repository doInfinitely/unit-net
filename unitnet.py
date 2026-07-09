"""unit-net core: constrained layers (dual projections, convex excitatory
rows, bounded inhibition), structural verification, native-geometry
optimizer steps (mirror descent / signSGD), and the exact J-lens.

Distilled from ~/Code/dual-projection-mnist (common.py, train_eg.py);
see docs_prior_session/report.pdf.
"""
import torch


def apply_activation(z, name):
    if name == "relu":
        return torch.clamp(z, min=0)
    if name == "clip":
        return torch.clamp(z, 0, 1)
    if name == "sigmoid":
        return torch.sigmoid(z)
    raise ValueError(name)


class UnitNet:
    """A stack of dual-projection layers. Weights are held natively (no
    reparameterization); constraints are maintained by the update rules."""

    def __init__(self, sizes, act="relu", device="cpu", gen=None,
                 append_one=True, init_scale=0.5):
        """init_scale: softmax temperature of the random init. ~0.5 gives
        near-uniform rows (fine for small fan-in); use 2-4 for very wide
        inputs so rows start concentrated enough to produce usable logit
        spread (prior session: averaging operators crush signal)."""
        self.sizes = sizes
        self.act = act
        self.append_one = append_one
        self.Ps, self.Ns = [], []
        for nin, nout in zip(sizes[:-1], sizes[1:]):
            k = nin + (1 if append_one else 0)
            W = torch.randn(nout, k, generator=gen,
                            device=device) * init_scale
            self.Ps.append(torch.softmax(W, dim=1).requires_grad_())
            # N starts at zero: P@a <= 1 but N@a can reach n_in (prior
            # session: nonzero N init kills every ReLU)
            self.Ns.append(torch.zeros(nout, k, device=device)
                           .requires_grad_())

    def forward(self, x, read_noise=0.0):
        """x: (batch, sizes[0]) in [0,1]. Returns (activations list incl.
        input, final pre-activation z)."""
        a, acts = x, [x]
        z = None
        for P, N in zip(self.Ps, self.Ns):
            if self.append_one:
                ones = torch.ones(a.shape[0], 1, device=a.device,
                                  dtype=a.dtype)
                a = torch.cat([a, ones], dim=1)
            z = a @ (P - N).T
            if read_noise:
                z = z + read_noise * torch.randn_like(z)
            a = apply_activation(z, self.act)
            acts.append(a)
        return acts, z

    @torch.no_grad()
    def mirror_step(self, eta_p, eta_n, write_noise=0.0):
        """Exponentiated gradient on P rows (native simplex geometry),
        projected GD on N. Call after loss.backward()."""
        for P, N in zip(self.Ps, self.Ns):
            P.mul_(torch.exp(-eta_p * P.grad))
            P.div_(P.sum(dim=1, keepdim=True))
            N.sub_(eta_n * N.grad)
            N.clamp_(0, 1)
            if write_noise:
                P.mul_(torch.exp(write_noise * torch.randn_like(P)))
                P.div_(P.sum(dim=1, keepdim=True))
                N.mul_(torch.exp(write_noise * torch.randn_like(N)))
                N.clamp_(0, 1)
            P.grad = None
            N.grad = None

    @torch.no_grad()
    def sign_step(self, delta):
        """Fixed-quantum sign steps (prior session: most write-noise-robust
        gradient method when delta >= the device's write-noise floor)."""
        for P, N in zip(self.Ps, self.Ns):
            P.sub_(delta * P.grad.sign())
            P.clamp_(min=0)
            P.div_(P.sum(dim=1, keepdim=True))
            N.sub_(delta * N.grad.sign())
            N.clamp_(0, 1)
            P.grad = None
            N.grad = None

    @torch.no_grad()
    def check(self, tol_sum=1e-6, tol_n=1e-9):
        problems = []
        for i, (P, N) in enumerate(zip(self.Ps, self.Ns)):
            if P.min() < -1e-12:
                problems.append(f"layer {i}: negative P")
            if N.min() < -1e-12 or N.max() > 1 + tol_n:
                problems.append(f"layer {i}: N outside [0,1]")
            if (P.sum(dim=1) - 1).abs().max() > tol_sum:
                problems.append(f"layer {i}: P row sum off")
        return problems

    @torch.no_grad()
    def jlens_exact(self, x):
        """The exact J-lens for one input (batch of 1): for every layer l,
        the true Jacobian dz_out/dh_l = product of active-masked effective
        matrices. Every entry bounded (convex P rows, N<=1): path mass has
        fixed units. Returns {layer_index: (n_out, n_l) matrix}.

        Contrast: Anthropic's fitted lens J_l = E[dh_final/dh_l] averages
        this over a corpus; here it is exact per input, no fitting."""
        acts, _ = self.forward(x)
        L = len(self.Ps)
        lenses = {}
        # J from layer l activations to final z, built back-to-front
        J = None
        for l in range(L - 1, -1, -1):
            W = (self.Ps[l] - self.Ns[l])
            if self.append_one:
                W = W[:, :-1]  # constant column carries no h-dependence
            if l == L - 1:
                Jl = W.clone()
            else:
                a = acts[l + 1][0]
                if self.act == "sigmoid":  # exact: d sigma = a(1-a)
                    d = a * (1 - a)
                else:                      # relu gate
                    d = (a > 0).to(W.dtype)
                Jl = J @ (d.unsqueeze(1) * W)
            lenses[l] = Jl
            J = Jl
        return lenses


class TokenUnitLM:
    """Unit-net language model: a shared token table (each feature unit is a
    convex budget over the vocabulary — a legible 'attention dictionary')
    feeding a unit-net trunk over the concatenated per-position features.

    emb:   UnitNet([V, E])      shared across the K positions
    trunk: UnitNet([K*E, ..., V])
    """

    def __init__(self, V, K, E, trunk_sizes, device="cpu", gen=None,
                 init_scale=1.5, act="relu"):
        self.V, self.K, self.E, self.act = V, K, E, act
        self.emb = UnitNet([V, E], act=act, device=device, gen=gen,
                           init_scale=init_scale)
        self.trunk = UnitNet([K * E] + trunk_sizes + [V], act=act,
                             device=device, gen=gen, init_scale=init_scale)

    def forward(self, red_ids, read_noise=0.0):
        """red_ids: (bs, K) reduced token ids. Returns (feat, z)."""
        bs = red_ids.shape[0]
        x = torch.zeros(bs * self.K, self.V, device=red_ids.device)
        x.scatter_(1, red_ids.reshape(-1, 1), 1.0)
        acts, _ = self.emb.forward(x, read_noise)
        feat = acts[-1].reshape(bs, self.K * self.E)
        tacts, z = self.trunk.forward(feat, read_noise)
        return (acts, feat, tacts), z

    def step(self, eta_p, eta_n):
        self.emb.mirror_step(eta_p, eta_n)
        self.trunk.mirror_step(eta_p, eta_n)

    def check(self):
        return self.emb.check() + self.trunk.check()

    def nets(self):
        return [self.emb, self.trunk]

    def save(self, path):
        import numpy as np
        arrs = {"V": np.array(self.V), "K": np.array(self.K),
                "E": np.array(self.E), "act": np.array(self.act),
                "emb_P": self.emb.Ps[0].detach().cpu().numpy(),
                "emb_N": self.emb.Ns[0].detach().cpu().numpy(),
                "n_trunk": np.array(len(self.trunk.Ps))}
        for i, (P, N) in enumerate(zip(self.trunk.Ps, self.trunk.Ns)):
            arrs[f"tP{i}"] = P.detach().cpu().numpy()
            arrs[f"tN{i}"] = N.detach().cpu().numpy()
        np.savez(path, **arrs)

    @classmethod
    def load(cls, path, device="cpu"):
        import numpy as np
        d = np.load(path)
        V, K, E = int(d["V"]), int(d["K"]), int(d["E"])
        act = str(d["act"]) if "act" in d else "relu"
        nt = int(d["n_trunk"])
        sizes = [d[f"tP{i}"].shape[0] for i in range(nt - 1)]
        self = cls(V, K, E, sizes, device=device, act=act)
        self.emb.Ps[0] = torch.tensor(d["emb_P"], device=device)
        self.emb.Ns[0] = torch.tensor(d["emb_N"], device=device)
        for i in range(nt):
            self.trunk.Ps[i] = torch.tensor(d[f"tP{i}"], device=device)
            self.trunk.Ns[i] = torch.tensor(d[f"tN{i}"], device=device)
        return self


def save(net, path):
    import numpy as np
    arrs = {"num_layers": np.array(len(net.Ps)),
            "activations": np.array([net.act] * len(net.Ps))}
    for i, (P, N) in enumerate(zip(net.Ps, net.Ns)):
        arrs[f"P{i}"] = P.detach().cpu().double().numpy()
        arrs[f"N{i}"] = N.detach().cpu().double().numpy()
    np.savez(path, **arrs)
