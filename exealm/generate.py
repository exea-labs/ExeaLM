"""Generate for several rows at once, in lockstep: every row gets one new token per step.

With swarm=True the rows are one swarm of agents. They share the in-batch board (exealm/model/board.py), so while
they write, each agent hears what the others posted in earlier chunks. This is how the model is trained (every row of
a swarm reads the same number of tokens), so it needs no persistent board and no extra passes.

    from exealm.generate import generate
    new = generate(model, [ids_a, ids_b, ids_c, ids_d], max_new_tokens=64, swarm=True, stop=[eot])

It re-reads the whole context for every new token (see ExeaLM.generate): fine for demos, slow for long outputs.
"""
from dataclasses import replace

import torch

from .memory import MemoryContext


@torch.no_grad()
def generate(model, prompts, max_new_tokens, temperature=1.0, top_k=None, stop=(), pad=None, swarm=False, mem=None,
             max_context=None, generator=None):
    """prompts: one list of token ids per row (per agent). Returns one list of new token ids per row.

    temperature 0 picks the most likely token. A row ends at its first token in `stop` (not returned); it then keeps
    receiving `pad` so the other rows can go on, and the loop ends when every row has ended. Prompts of different
    lengths are padded on the left with `pad` (default: the first stop token, else 0) so all rows end at the same
    position. max_context keeps only the last so many tokens as input. mem: a MemoryContext to use instead (a
    datastore, a persistent board); swarm=True sets its swarm_size to the number of rows.
    """
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("every row needs at least one prompt token")
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens and temperature must not be negative")
    stop = set(int(s) for s in stop)
    if pad is None:
        pad = min(stop) if stop else 0
    dev = next(model.parameters()).device
    width = max(len(p) for p in prompts)
    idx = torch.tensor([[pad] * (width - len(p)) + [int(t) for t in p] for p in prompts], dtype=torch.long, device=dev)
    mem = mem if mem is not None else MemoryContext()
    if swarm:
        mem = replace(mem, swarm_size=len(prompts))
    out = [[] for _ in prompts]
    done = [False] * len(prompts)
    for _ in range(max_new_tokens):
        x = idx if max_context is None else idx[:, -max_context:]
        m = mem
        if mem.doc_ids is not None:                        # retrieval skips the document being read: extend its id
            ids = mem.doc_ids
            if ids.shape[1] < idx.shape[1]:
                ids = torch.cat([ids, ids[:, -1:].expand(-1, idx.shape[1] - ids.shape[1])], 1)
            m = replace(mem, doc_ids=ids[:, -x.shape[1]:])
        logits = model(x, m)[0][:, -1].float()
        if temperature == 0:
            nxt = logits.argmax(-1)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = logits.topk(min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(1)
        for i, t in enumerate(nxt.tolist()):
            if done[i]:
                nxt[i] = pad
            elif t in stop:
                done[i] = True
                nxt[i] = pad
            else:
                out[i].append(t)
        idx = torch.cat([idx, nxt[:, None]], 1)
        if all(done):
            break
    return out
