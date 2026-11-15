"""Lockstep generation for several rows (agents) at once."""
import pytest
import torch
import torch.nn as nn

from exealm import ExeaLM, MemoryContext, generate, get_config


class Counter(nn.Module):
    """A stand-in model: the next token is always (last token + 1) mod V. Remembers the memory contexts it was given."""

    def __init__(self, vocab=10):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.vocab, self.mems, self.widths = vocab, [], []

    def forward(self, idx, mem=None, targets=None):
        self.mems.append(mem)
        self.widths.append(idx.shape[1])
        return nn.functional.one_hot((idx + 1) % self.vocab, self.vocab).float() * 10, None


def test_rows_stop_on_their_own_and_the_rest_go_on():
    model = Counter()
    out = generate(model, [[1], [5]], max_new_tokens=20, temperature=0, stop=[4, 9], pad=0)
    assert out == [[2, 3], [6, 7, 8]]                      # the stop token itself is not returned
    assert len(model.mems) == 4                            # ended as soon as both rows had stopped


def test_short_prompts_are_padded_on_the_left_and_the_token_budget_holds():
    model = Counter()
    out = generate(model, [[1, 2, 3], [7]], max_new_tokens=2, temperature=0)
    assert out == [[4, 5], [8, 9]]
    assert model.widths == [3, 4]


def test_sampling_options():
    model = Counter()
    g = torch.Generator().manual_seed(0)
    assert generate(model, [[1]], 3, temperature=1.0, top_k=1, generator=g) == [[2, 3, 4]]
    generate(model, [[1, 2, 3, 4, 5]], 3, temperature=0, max_context=2)
    assert model.widths[-3:] == [2, 2, 2]
    with pytest.raises(ValueError):
        generate(model, [[1], []], 3)
    with pytest.raises(ValueError):
        generate(model, [[1]], 3, temperature=-1)


def test_swarm_flag_makes_the_rows_one_swarm_and_extends_document_ids():
    model = Counter()
    generate(model, [[1], [2], [3]], 2, temperature=0, swarm=True)
    assert all(m.swarm_size == 3 for m in model.mems)
    generate(model, [[1], [2]], 1, temperature=0)
    assert model.mems[-1].swarm_size == 1
    mem = MemoryContext(doc_ids=torch.tensor([[7, 7], [8, 8]]))
    generate(model, [[1, 2], [3, 4]], 3, temperature=0, mem=mem)
    assert model.mems[-1].doc_ids.tolist() == [[7, 7, 7, 7], [8, 8, 8, 8]]


def trained_scales(seed=0):
    """A tiny model whose circuit scales have been measured on a past batch, as after any training (before that,
    a new circuit is scaled by the batch it sees, so rows are not independent yet)."""
    torch.manual_seed(seed)
    model = ExeaLM(get_config("tiny", swarm_size=1))
    model.train()
    model(torch.randint(0, 256, (4, 32)))
    return model.eval()


def test_solo_rows_in_one_batch_write_what_they_write_alone():
    model = trained_scales()
    a, b = torch.randint(0, 256, (2, 20)).tolist()
    both = generate(model, [a, b], 6, temperature=0)
    assert both == [generate(model, [a], 6, temperature=0)[0], generate(model, [b], 6, temperature=0)[0]]


def test_agents_of_a_swarm_hear_each_other_while_writing():
    """Change only agent 0's prompt: agent 1's next-token scores move if they are a swarm, and do not if solo."""
    model = trained_scales()
    with torch.no_grad():
        for lay in model.layers:                           # an untrained board is nearly silent: turn it up
            lay.ag.post_b.fill_(4.0)
    a, a2, b = torch.randint(0, 256, (3, 24))

    def scores_of_agent_1(first, swarm):
        seen = []
        hook = model.head.register_forward_hook(lambda mod, inp, out: seen.append(out[1, -1].clone()))
        generate(model, [first.tolist(), b.tolist()], 1, temperature=0, swarm=swarm)
        hook.remove()
        return seen[0]

    assert torch.equal(scores_of_agent_1(a, False), scores_of_agent_1(a2, False))
    assert not torch.allclose(scores_of_agent_1(a, True), scores_of_agent_1(a2, True))
