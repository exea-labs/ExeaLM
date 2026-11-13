"""The retrieval (RG) and agent-board (AG) circuits: no peeking at the future, no looking up the document being read,
agents only hear what their swarm posted earlier, and gradients reach the model's own retrieval encoder."""
import torch

from exealm import AgentBoard, Datastore, ExeaLM, MemoryContext, get_config


def setup(swarm_size=1, B=2, T=40, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = get_config("tiny", swarm_size=swarm_size, **kw)
    m = ExeaLM(cfg)
    ds = Datastore(torch.randint(0, cfg.vocab_size, (50, cfg.rag_chunk_len)), torch.arange(50) % 7)
    ds.refresh(m.encoder, m.emb)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    docs = torch.full((B, T), 3)
    mem = MemoryContext(datastore=ds, doc_ids=docs, swarm_size=swarm_size)
    m.train()
    m(x, mem)                                    # set each slot's running output scale
    m.eval()
    return cfg, m, x, mem


import pytest  # noqa: E402


@pytest.mark.parametrize("query", ["last", "mean"])
def test_retrieval_and_board_are_causal_for_one_agent(query):
    cfg, m, x, mem = setup(rag_query=query)
    t = 21
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5)
    assert not torch.allclose(a[0, t:], b[0, t:])


def test_swarm_members_only_hear_posts_from_earlier_chunks():
    cfg, m, x, mem = setup(swarm_size=2)
    t = 13                                       # inside chunk 1 (chunks of 8)
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    c_end = (t // cfg.chunk + 1) * cfg.chunk     # the post made in t's chunk is readable from here on
    assert torch.allclose(a[0, :t], b[0, :t], atol=1e-5)
    assert torch.allclose(a[1, :c_end], b[1, :c_end], atol=1e-5)
    assert not torch.allclose(a[1, c_end:], b[1, c_end:])      # the other agent does hear it later


def test_separate_swarms_do_not_hear_each_other():
    cfg, m, x, mem = setup(swarm_size=1)
    x2 = x.clone()
    x2[0] = (x2[0] + 1) % cfg.vocab_size
    with torch.no_grad():
        assert torch.allclose(m(x, mem)[0][1], m(x2, mem)[0][1], atol=1e-5)


def test_datastore_never_returns_the_document_being_read():
    keys = torch.nn.functional.normalize(torch.randn(100, 8), dim=-1)
    ds = Datastore(torch.zeros(100, 4, dtype=torch.long), torch.arange(100) % 10)
    ds.keys = keys
    q = keys[:20]                                # each query's best match is itself, in document i % 10
    idx, ok = ds.search(q, 3, exclude=torch.arange(20) % 10)
    assert ok.all()
    assert not (ds.doc_ids[idx] == (torch.arange(20) % 10).view(-1, 1)).any()


def test_gradients_reach_the_models_own_retrieval_encoder():
    cfg, m, x, mem = setup()
    m.train()
    m(x, mem, x)[1].backward()
    assert m.encoder.key.weight.grad is not None and m.encoder.key.weight.grad.abs().sum() > 0
    assert m.layers[0].rg.alpha.grad is not None and m.layers[0].inp.weight.grad.abs().sum() > 0


def test_persistent_board_shares_messages_between_agents():
    cfg, m, x, mem = setup()
    board = AgentBoard()
    with torch.no_grad():
        alone = m(x[1:], MemoryContext(board=board, agent_ids=torch.tensor([1])))[0]
        m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0]), write_board=True))    # agent 0 posts
        n = sum(len(c) for c in board.channels.values())
        m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0]), write_board=True))    # same text again
        assert sum(len(c) for c in board.channels.values()) == n                              # replaced, not duplicated
        heard = m(x[1:], MemoryContext(board=board, agent_ids=torch.tensor([1])))[0]        # agent 1 reads
        own = m(x[:1], MemoryContext(board=board, agent_ids=torch.tensor([0])))[0]
        own_alone = m(x[:1], MemoryContext())[0]
    assert n == cfg.n_layers * (x.shape[1] // cfg.chunk)     # one AG slot per layer, one post per full chunk
    assert not torch.allclose(alone, heard)                   # agent 1 heard agent 0
    assert torch.allclose(own, own_alone, atol=1e-5)          # an agent does not re-read its own old posts


def test_swarm_batches_split_and_mix():
    import numpy as np
    from exealm.data.loader import SyntheticShards
    data = SyntheticShards(n_tokens=20_000)
    x, y, d = data.batch(8, 16, np.random.default_rng(0), swarm_size=4, policy="split", mask_seams=False)
    for s in range(2):                                    # each swarm reads one stretch of text, in order
        rows = x[4 * s:4 * s + 4]
        assert torch.equal(rows[1:, 0], y[4 * s:4 * s + 3, -1])
    xm, ym, _ = data.batch(8, 16, np.random.default_rng(0), swarm_size=4, policy="split")
    assert torch.equal(xm, x)                             # same windows; only the seam targets are hidden
    seam = torch.zeros(8, 16, dtype=torch.bool)
    seam[[0, 1, 2, 4, 5, 6], -1] = True
    assert (ym[seam] == -100).all() and torch.equal(ym[~seam], y[~seam])
    x, y, d = data.batch(8, 16, np.random.default_rng(0), swarm_size=4, policy="mixed")
    assert x.shape == (8, 16) and d.shape == (8, 16)
    _, y1, _ = data.batch(8, 16, np.random.default_rng(0), swarm_size=1, policy="split")
    assert (y1 >= 0).all()                                # a lone agent has no seam


def test_the_split_seam_target_really_leaks_through_the_board():
    """Why batch() masks it: agent 0's last prediction changes when agent 1's first token does."""
    cfg, m, _, _ = setup(swarm_size=2)
    x = torch.randint(0, cfg.vocab_size, (2, 40))
    x2 = x.clone()
    x2[1, 0] = (x2[1, 0] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, MemoryContext(swarm_size=2))[0], m(x2, MemoryContext(swarm_size=2))[0]
    assert not torch.allclose(a[0, -1], b[0, -1])


def test_datastore_returns_k_results_even_when_the_own_document_dominates():
    keys = torch.nn.functional.normalize(torch.randn(200, 8), dim=-1)
    doc = torch.zeros(200, dtype=torch.long)
    doc[190:] = torch.arange(1, 11)                       # 190 chunks of document 0, 10 chunks of other documents
    ds = Datastore(torch.zeros(200, 4, dtype=torch.long), doc)
    ds.keys = keys
    idx, ok = ds.search(keys[:5], 4, exclude=torch.zeros(5, dtype=torch.long))
    assert ok.all() and (ds.doc_ids[idx] != 0).all()


def test_larger_retrieval_keys_work_and_stay_causal():
    cfg, m, x, mem = setup(rag_key_dim=24)
    assert m.encoder.key.out_features == 24 and mem.datastore.keys.shape[1] == 24
    x2 = x.clone()
    x2[0, 21] = (x2[0, 21] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    assert torch.allclose(a[:, :21], b[:, :21], atol=1e-5)


def test_lexical_keys_match_chunks_that_share_rare_words():
    cfg = get_config("tiny", rag_lexical=16)
    m = ExeaLM(cfg)
    common = torch.full((8,), 5)
    a = common.clone(); a[2], a[5] = 200, 201                 # two chunks sharing two rare tokens
    b = common.clone(); b[1], b[6] = 200, 201
    c = common.clone(); c[3], c[4] = 150, 151                 # a chunk with different rare tokens
    store = torch.stack([a, b, c] + [torch.full((8,), 5)] * 20)
    m.encoder.set_idf(store)
    assert m.encoder.idf[200] > m.encoder.idf[5]
    lex = lambda t: torch.nn.functional.normalize((m.encoder.lex[t] * m.encoder.idf[t].unsqueeze(-1)).sum(0), dim=-1)
    assert float(lex(a) @ lex(b)) > float(lex(a) @ lex(c))
    ds = Datastore(store, torch.arange(len(store)))
    ds.refresh(m.encoder, m.emb)
    assert ds.keys.shape[1] == cfg.h + 16


def test_swarm_text_scores_only_the_targets_that_count():
    """experiments/swarm_text.py averages over scored targets, so the hidden seam targets don't count as zero loss."""
    import math
    import numpy as np
    import importlib.util
    import pathlib
    from exealm.data.loader import SyntheticShards
    path = pathlib.Path(__file__).parents[1] / "experiments" / "swarm_text.py"
    spec = importlib.util.spec_from_file_location("swarm_text", path)
    swarm_text = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swarm_text)
    score = swarm_text.score
    torch.manual_seed(0)
    cfg = get_config("tiny", swarm_size=2)
    m = ExeaLM(cfg)
    x, y, _ = SyntheticShards(n_tokens=20_000).batch(4, 24, np.random.default_rng(0), swarm_size=2, policy="split")
    assert (y[[0, 2], -1] == -100).all()
    got = score(m, [(x, y)], 2, 2, torch.device("cpu"))["per_agent"]
    with torch.no_grad():
        nll = torch.nn.functional.cross_entropy(m.eval()(x, MemoryContext(swarm_size=2))[0].transpose(1, 2), y,
                                                reduction="none", ignore_index=-100)
    rows = nll.sum(1) / (y != -100).sum(1) / math.log(2)
    assert np.allclose(got, [float(rows[[0, 2]].mean()), float(rows[[1, 3]].mean())], atol=1e-4)


def _randomize_pools(m):
    """chunk_pool='attn' starts as the mean (zero weights); give the pools real weights so the test means something."""
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "pool" in name:
                p.normal_(0, 1.0)


@pytest.mark.parametrize("kw", [dict(chunk_pool="attn"), dict(board_chunk=4), dict(chunk_pool="attn", board_chunk=4)])
def test_pooling_and_board_chunk_stay_causal(kw):
    cfg, m, x, mem = setup(**kw)
    _randomize_pools(m)
    t = 21
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5)
    assert not torch.allclose(a[0, t:], b[0, t:])


@pytest.mark.parametrize("kw", [dict(chunk_pool="attn"), dict(board_chunk=4), dict(chunk_pool="attn", board_chunk=4)])
def test_pooled_posts_are_heard_only_after_their_chunk(kw):
    cfg, m, x, mem = setup(swarm_size=2, **kw)
    _randomize_pools(m)
    t = 13
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x, mem)[0], m(x2, mem)[0]
    bc = cfg.board_chunk or cfg.chunk
    c_end = (t // bc + 1) * bc                   # the post made in t's board chunk is readable from here on
    assert torch.allclose(a[0, :t], b[0, :t], atol=1e-5)
    assert torch.allclose(a[1, :c_end], b[1, :c_end], atol=1e-5)


def test_a_prior_agent_is_heard_from_the_first_token_and_hears_no_one():
    """board_prior: row 0 of a swarm finished reading before row 1 started. Row 1 hears all of row 0's posts from its
    first token on; row 0 never hears row 1; row 1 still cannot see its own future."""
    cfg, m, x, mem = setup(swarm_size=2, board_chunk=4)
    mem.board_prior = True
    _randomize_pools(m)
    T = x.shape[1]
    with torch.no_grad():
        base = m(x, mem)[0]
        x0 = x.clone()
        x0[0, T - 6] = (x0[0, T - 6] + 1) % cfg.vocab_size              # late in row 0's text
        a = m(x0, mem)[0]
        x1 = x.clone()
        x1[1, 9] = (x1[1, 9] + 1) % cfg.vocab_size
        b = m(x1, mem)[0]
    assert not torch.allclose(base[1, :8], a[1, :8])                    # row 1 hears the end of row 0 at once
    assert torch.allclose(base[0], b[0], atol=1e-5)                     # row 0 never hears row 1
    assert torch.allclose(base[1, :9], b[1, :9], atol=1e-5)             # and row 1 is still causal


def test_persistent_board_reads_are_cached_until_the_next_post():
    board = AgentBoard()
    key, val, log_w = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4), torch.zeros(1, 2, 1)
    board.post(0, [0], key, val, log_w, torch.tensor([7]), n_complete=2)
    first = board.read(0, [0], "cpu")
    assert board.read(0, [0], "cpu") is first                          # cached
    assert first[0].shape == (1, 2, 4) and first[3].tolist() == [[7, 7]]
    other = AgentBoard()
    other.post(0, [0], key, val, log_w, torch.tensor([8]), n_complete=1)
    board.merge(other)
    after = board.read(0, [0], "cpu")
    assert after[0].shape == (1, 3, 4) and sorted(after[3][0].tolist()) == [7, 7, 8]


def test_a_keyword_library_feeds_the_retrieval_circuit_and_stays_causal():
    from exealm.memory.lexical import LexicalDatastore
    torch.manual_seed(0)
    cfg = get_config("tiny", swarm_size=1)
    m = ExeaLM(cfg)
    C = cfg.chunk
    lib = torch.randint(0, cfg.vocab_size, (30, cfg.rag_chunk_len))
    ds = LexicalDatastore(lib, torch.arange(30))
    x = torch.randint(0, cfg.vocab_size, (2, 5 * C))
    x[0, :C] = lib[7, :C]                                               # chunk 0 repeats library chunk 7 word for word
    mem = MemoryContext(datastore=ds, doc_ids=torch.full(x.shape, 99))
    m.train()
    m(x, mem)                                                           # set each slot's running output scale
    m.eval()
    idx, ok = ds.search_tokens(x[:, :C], 3)
    assert idx[0, 0] == 7 and ok[0, 0]
    with torch.no_grad():
        a = m(x, mem)[0]
        x2 = x.clone()
        x2[1, 2 * C + 3] = (x2[1, 2 * C + 3] + 1) % cfg.vocab_size
        b = m(x2, mem)[0]
        none = m(x, MemoryContext())[0]
    assert torch.allclose(a[:, :2 * C + 3], b[:, :2 * C + 3], atol=1e-5)   # a later token changes nothing before it
    assert not torch.allclose(a[0, C:], none[0, C:])                   # and the library is read from chunk 1 on
