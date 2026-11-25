"""torch.compile must not change a TONN layer's gradients. Its forward updates the `scale` buffer, and the compiler's
partitioner may recompute a read of that buffer in the backward pass (pytorch/pytorch#201089): with an in-place update
the backward would divide by the new scale instead of the one the forward used."""
import copy
import sys

import pytest
import torch

from exealm import ExeaLM, MemoryContext, get_config
from exealm.model.tonn import TONNLayer

TYPES = [("D", 1), ("M", 3), ("MI", 0), ("MIM", 64), ("C", 4), ("G", 0)]


def a_layer_that_has_run():
    """A layer whose plan is cached, as when pretrain.py compiles the model after the search, with scales measured on a
    3x larger batch so that this batch moves them."""
    cfg = get_config("tiny", slots=len(TYPES), d=8 * len(TYPES))
    torch.manual_seed(0)
    layer = TONNLayer(cfg, [dict(tmpl=t, lag=lag, lam0=0.5) for t, lag in TYPES], 0)
    with torch.no_grad():
        layer(3 * torch.randn(4, 40, cfg.d), MemoryContext())
    return layer, torch.randn(4, 40, cfg.d), torch.randn(4, 40, cfg.d)


def run(layer, x, w, backend):
    layer = copy.deepcopy(layer)
    x = x.clone().requires_grad_()
    f = layer if backend is None else torch.compile(layer, backend=backend)
    y = f(x, MemoryContext())
    (y * w).sum().backward()
    return y.detach(), layer.scale.clone(), {"input": x.grad, **{n: p.grad for n, p in layer.named_parameters()}}


# aot_eager_decomp_partition partitions as inductor does but generates no code, so it needs no C++ compiler.
CASES = [("aot_eager_decomp_partition", False), ("aot_eager_decomp_partition", True),
         pytest.param("inductor", True, marks=pytest.mark.skipif(sys.platform == "win32", reason="needs a C++ compiler"))]


@pytest.mark.parametrize("backend,new_slots", CASES)
def test_compiled_gradients_match_eager(backend, new_slots):
    layer, x, w = a_layer_that_has_run()
    if new_slots:                                              # what a search replacement leaves: no measurement yet
        layer.scale.zero_()
    y, scale, grads = run(layer, x, w, None)
    torch._dynamo.reset()
    yc, scale_c, grads_c = run(layer, x, w, backend)
    torch._dynamo.reset()
    torch.testing.assert_close(yc, y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scale_c, scale, rtol=1e-5, atol=1e-5)
    for n, g in grads.items():
        assert (g is None) == (grads_c[n] is None), n
        if g is not None:
            torch.testing.assert_close(grads_c[n], g, rtol=1e-4, atol=1e-5 * float(g.abs().max()), msg=n)


def a_small_model():
    cfg = get_config("tiny", n_layers=3, attn_layers=(1,), pinned={}, menu=[t for t, _ in TYPES], swarm_size=1)
    torch.manual_seed(0)
    row = [dict(tmpl=t, lag=lag, lam0=0.5) for t, lag in TYPES] + [dict(tmpl="D", lag=2, lam0=0.5)] * (cfg.slots - len(TYPES))
    specs = [row, row]                                         # the two TONN layers
    return ExeaLM(cfg, specs=specs).train(), torch.randint(0, cfg.vocab_size, (2, 49), generator=torch.Generator().manual_seed(1))


def stream_step(model, x, windows=3):
    """pretrain.py's train_loss with --stream: windows read in order with one state dict, one backward through all."""
    state, T, losses = {}, (x.shape[1] - 1) // windows, []
    for w in range(windows):
        s = slice(w * T, (w + 1) * T)
        losses.append(model(x[:, :-1][:, s], MemoryContext(stream=state), x[:, 1:][:, s])[1])
    torch.stack(losses).mean().backward()


def test_compiled_stream_training_matches_eager():
    """Each window updates the scales before the backward of all windows runs. An update in place changed a tensor
    the compiled backward had saved ("modified by an inplace operation")."""
    model, x = a_small_model()
    compiled = copy.deepcopy(model)
    stream_step(model, x)
    torch._dynamo.reset()
    stream_step(torch.compile(compiled, backend="aot_eager_decomp_partition"), x)
    torch._dynamo.reset()
    for (n, p), q in zip(model.named_parameters(), compiled.parameters()):
        assert (p.grad is None) == (q.grad is None), n
        if p.grad is not None:
            torch.testing.assert_close(q.grad, p.grad, rtol=1e-4, atol=1e-5 * float(p.grad.abs().max()), msg=n)
    for a, b in zip(model.buffers(), compiled.buffers()):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_a_tonn_layer_compiles_as_one_graph():
    """No graph break in a training forward: a break splits every TONN layer into frames that dynamo compiles one per
    layer, and past its recompile limit (8) the remaining layers run eager."""
    layer, x, _ = a_layer_that_has_run()
    torch._dynamo.reset()
    torch.compile(layer, backend="aot_eager", fullgraph=True)(x.requires_grad_(), MemoryContext()).sum().backward()
    torch._dynamo.reset()
