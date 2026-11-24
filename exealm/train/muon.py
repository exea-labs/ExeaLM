"""Muon (Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/): momentum, then each weight matrix's update is
replaced by the nearest semi-orthogonal matrix, so the step moves every direction of the matrix by about the same amount
instead of mostly the few directions that dominate the gradient.

The update is scaled by 0.2 * sqrt(max(rows, cols)), as in Liu et al. 2025, "Muon is Scalable for LLM Training"
(arXiv 2502.16982). That gives it about the RMS of an AdamW update, so both kinds of parameter share one learning rate,
schedule and weight decay, and every training loop here (which sets one lr on every param group) works unchanged.

Muon only fits weight matrices. Groups with use_muon=False take plain AdamW steps (the same math as torch.optim.AdamW),
so embeddings, the output head, norms, biases and per-slot scalars stay on AdamW inside the same optimizer, and one
state dict holds everything a checkpoint, a resume or the search's slot resets (exealm/model/search.py) need.
"""
import torch


def orthogonalize(G, steps=5):
    """About U V^T for G = U S V^T: Newton-Schulz in bfloat16 with Jordan's quintic coefficients. Singular values that
    are not tiny end between about 0.7 and 1.2 rather than at exactly 1, which trains as well and needs only 5 steps.
    The fused addmm calls round to bfloat16 half as often as the same sums written out (1.1% against 2.7% off a 32-bit
    run, measured on 256 x 680); they are also what torch.optim.Muon uses."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / X.norm().clamp(min=1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = torch.addmm(X, torch.addmm(A, A, A, beta=b, alpha=c), X, beta=a)
    if tall:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-3, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5, betas=(0.9, 0.95),
                 eps=1e-8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, betas=betas, eps=eps, use_muon=False))
        for g in self.param_groups:
            if g["use_muon"] and any(p.dim() != 2 for p in g["params"]):
                raise ValueError("Muon groups take 2-D weight matrices only")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g["use_muon"]:
                    if not st:
                        st["momentum_buffer"] = torch.zeros_like(p)
                    buf = st["momentum_buffer"]
                    buf.lerp_(p.grad, 1 - g["momentum"])
                    upd = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                    upd = orthogonalize(upd, g["ns_steps"]) * (0.2 * max(p.shape) ** 0.5)
                    p.mul_(1 - lr * wd)
                    p.add_(upd, alpha=-lr)
                else:
                    if not st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    b1, b2 = g["betas"]
                    st["step"] += 1
                    st["exp_avg"].lerp_(p.grad, 1 - b1)
                    st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (st["exp_avg_sq"].sqrt() / (1 - b2 ** st["step"]) ** 0.5).add_(g["eps"])
                    p.mul_(1 - lr * wd)
                    p.addcdiv_(st["exp_avg"], denom, value=-lr / (1 - b1 ** st["step"]))
        return loss
