"""TONN's architecture search: while the model trains, find out what each circuit is worth and replace useless ones.

One search round:
  1. Worth. On held-out "selection" batches, switch off all slots of one type in one layer at once and measure how much
     the loss rises. That rise, split among those slots, is each slot's worth (in bits per token). A whole group is
     switched off together because one slot's effect in a wide layer is mostly noise.
  2. Steer. Types that are worth more per slot get drawn more often for new circuits (80%), with 20% spread evenly so
     every type keeps getting tried.
  3. Replace. The lowest-worth slots (only ones worth nothing or less) get a new circuit: half the time a mutation of a
     useful slot, otherwise a fresh draw. New circuits get `grace` rounds before they can be judged.
  4. The floor. A layer never drops below cfg.pinned of a type (one RG and one AG): retrieval and the agent board only
     become useful once their encoder and board have trained, so early worth would remove them too soon.

Rounds run every `every` steps during the first `until` fraction of training, replacing a fraction sigma that shrinks
to zero. After that the architecture is frozen. For the full run the plan is to search during a pilot and then freeze
(docs/training.md).
"""
import random
from collections import Counter

import torch

from .tonn import rand_spec, slot_cost


@torch.no_grad()
def group_worth(model, batches):
    """Worth (bits per token) of every slot, from switching off each (layer, type) group. batches: list of
    (x, y, mem). Returns W (n_layers, K), {type: [worth per slot of each group]} and the base loss in bits."""
    was = model.training
    model.eval()

    def bits():
        return sum(float(model(x, mem, y)[1]) for x, y, mem in batches) / len(batches) / 0.6931471805599453

    base = bits()
    W = torch.zeros(len(model.layers), max(l.K for l in model.layers))
    per_type = {}
    for i, lay in enumerate(model.layers):
        groups = {}
        for k, s in enumerate(lay.specs):
            groups.setdefault(s["tmpl"], []).append(k)
        for t, ks in groups.items():
            lay.mask[ks] = 0.0
            rise = bits() - base
            lay.mask[ks] = 1.0
            W[i, ks] = rise / len(ks)
            per_type.setdefault(t, []).append(rise / len(ks))
    model.train(was)
    return W, per_type, base


def type_worth_per_layer(model, W, t):
    """Mean worth (bits per token) of the slots of type t in each layer, None for a layer without one. For "AG" this is
    how much each layer's agent board is worth: the loss rise when that layer stops reading and posting."""
    out = []
    for i, lay in enumerate(model.layers):
        ks = [k for k, s in enumerate(lay.specs) if s["tmpl"] == t]
        out.append(round(float(W[i, ks].mean()), 5) if ks else None)
    return out


def steer_weights(per_type, menu, explore=0.2):
    tw = {t: max(0.0, sum(v) / len(v)) for t, v in per_type.items()}
    mean = sum(tw.values()) / max(1, len(tw))
    w = {t: tw.get(t, mean) for t in menu}
    tot = sum(w.values()) or 1.0
    return {t: (1 - explore) * w[t] / tot + explore / len(w) for t in w}


def reset_adam_rows(opt, lay, k):
    """Clear the optimizer's running statistics for one slot's rows (its old circuit's no longer apply)."""
    rows, cols = slice(k * 3 * lay.h, (k + 1) * 3 * lay.h), slice(k * lay.h, (k + 1) * lay.h)
    for p, sl in ((lay.inp.weight, (rows,)), (lay.inp.bias, (rows,)), (lay.out.weight, (slice(None), cols)),
                  (lay.lam, (k,)), (lay.beta, (k,))):
        st = opt.state.get(p) if opt is not None else None
        for name in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            if st and name in st:
                st[name][sl] = 0


def denoise_step(model, opt, W, sigma, rng, T, weights=None, protected=frozenset(), only_useless=True, p_mutate=0.5):
    """Replace the lowest worth-per-cost fraction sigma of all slots, keeping every layer's pinned floor.
    Returns [(layer, slot, new spec)]."""
    cfg = model.cfg
    slots = [(i, k) for i, lay in enumerate(model.layers) for k in range(lay.K)]
    score = {(i, k): float(W[i, k]) / slot_cost(model.layers[i].specs[k], cfg, T) for i, k in slots}
    order = sorted(slots, key=score.get)
    good = order[len(order) // 2:]
    counts = [Counter(s["tmpl"] for s in lay.specs) for lay in model.layers]
    n = int(round(sigma * len(slots)))
    changed = []
    for i, k in order:
        if len(changed) >= n:
            break
        if (i, k) in protected or (only_useless and float(W[i, k]) > 0):
            continue
        old = model.layers[i].specs[k]["tmpl"]
        if counts[i][old] <= cfg.pinned.get(old, 0):
            continue                                                      # the floor
        if rng.random() < p_mutate and good:
            gi, gk = rng.choice(good)
            spec = dict(model.layers[gi].specs[gk])
            if rng.random() < 0.2:
                spec["tmpl"] = rand_spec(rng, cfg, weights=weights)["tmpl"]
            spec["lag"] = rng.choice(cfg.lags)
            spec["lam0"] = max(-0.97, min(0.97, spec["lam0"] + rng.uniform(-0.2, 0.2)))
        else:
            spec = rand_spec(rng, cfg, weights=weights)
        counts[i][old] -= 1
        counts[i][spec["tmpl"]] += 1
        model.layers[i].set_spec(k, spec)
        reset_adam_rows(opt, model.layers[i], k)
        changed.append((i, k, spec))
    return changed


class Searcher:
    """Runs search rounds on a schedule. Call maybe_round() once per training step."""

    def __init__(self, every=500, until=0.5, sigma0=0.15, grace=2, p_mutate=0.5, explore=0.2, seed=0):
        self.every, self.until, self.sigma0, self.grace = every, until, sigma0, grace
        self.p_mutate, self.explore = p_mutate, explore
        self.rng = random.Random(seed)
        self.round, self.born = 0, {}

    def due(self, step, total):
        return step > 0 and step % self.every == 0 and step < self.until * total

    def maybe_round(self, model, opt, step, total, batches, T):
        if not self.due(step, total):
            return None
        sigma = self.sigma0 * (1 - step / (self.until * total))
        W, per_type, base = group_worth(model, batches)
        board = type_worth_per_layer(model, W, "AG")                       # before any slot is replaced
        weights = steer_weights(per_type, model.cfg.menu, self.explore)
        protected = {s for s, r in self.born.items() if self.round - r < self.grace}
        changed = denoise_step(model, opt, W, sigma, self.rng, T, weights, protected, p_mutate=self.p_mutate)
        for i, k, _ in changed:
            self.born[(i, k)] = self.round
        self.round += 1
        worth = {t: round(sum(v) / len(v), 5) for t, v in per_type.items()}
        return dict(step=step, sigma=round(sigma, 4), selection_bits=round(base, 4), replaced=len(changed),
                    type_worth=dict(sorted(worth.items(), key=lambda kv: -kv[1])),
                    board_worth=board)
