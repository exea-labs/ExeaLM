"""Opt-in author identity on in-batch and persistent board values."""
from dataclasses import asdict

import torch

from exealm import AgentBoard, ExeaLM, MemoryContext, get_config
from exealm.hub import build_model
from exealm.model.board import BoardCircuit


def test_in_batch_author_values_are_distinct_and_receive_writer_specific_gradients():
    cfg = get_config("tiny", board_identity_size=4, board_read_self=False)
    board = BoardCircuit(cfg)
    values = torch.zeros(4, 2 * cfg.chunk, 1, cfg.h)
    streams = (values, torch.ones_like(values), torch.ones_like(values))
    torch.testing.assert_close(board.author_values, torch.zeros_like(board.author_values))

    def read(ids=None):
        return board(streams, MemoryContext(swarm_size=2, agent_ids=ids), layer=0, slots=[0])

    out = read()
    for reader, writer in [(0, 1), (1, 0), (2, 1)]:
        grad, = torch.autograd.grad(out[reader, cfg.chunk:].sum(), board.author_values, retain_graph=True)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum(1).nonzero().flatten().tolist() == [writer]

    with torch.no_grad():
        board.author_values[0].fill_(1)
        board.author_values[1].fill_(2)
        board.author_values[2].fill_(3)
        default = read()
        torch.testing.assert_close(default, read(torch.tensor([0, 1, 0, 1])))
        different = read(torch.tensor([2, 1, 0, 1]))
        assert default[:, :cfg.chunk].count_nonzero() == 0
        assert not torch.equal(default[0, cfg.chunk:], default[1, cfg.chunk:])
        assert not torch.equal(default[1, cfg.chunk:], different[1, cfg.chunk:])
        torch.testing.assert_close(default[[0, 2, 3]], different[[0, 2, 3]])
        assert read(torch.tensor([99, 1, -1, 1]))[[1, 3]].count_nonzero() == 0


def test_persistent_author_ids_condition_values_at_read_time():
    cfg = get_config("tiny", board_identity_size=3, board_read_self=False)
    circuit = BoardCircuit(cfg)
    stream = torch.ones(1, cfg.chunk, 1, cfg.h)
    streams = (torch.zeros_like(stream), stream, stream)

    def stored(author):
        board = AgentBoard()
        shape = (1, 1, 1, cfg.h)
        board.post(0, [0], torch.full(shape, cfg.h ** -0.5), torch.ones(shape), torch.zeros(1, 1, 1),
                   torch.tensor([author]), n_complete=1)
        return board

    def read(board, reader=2):
        return circuit(streams, MemoryContext(board=board, agent_ids=torch.tensor([reader])), layer=0, slots=[0])

    author0, author1, unknown = map(stored, (0, 1, 99))
    unsigned = read(unknown).detach()
    torch.testing.assert_close(read(author0), read(author1))
    with torch.no_grad():
        circuit.author_values[0].fill_(1)
        circuit.author_values[1].fill_(2)
    torch.testing.assert_close(author0.read(0, [0], "cpu")[1], author1.read(0, [0], "cpu")[1])
    assert not torch.equal(read(author0), read(author1))
    torch.testing.assert_close(read(unknown), unsigned)
    assert read(author0, reader=0).count_nonzero() == 0
    grad, = torch.autograd.grad(read(author1).sum(), circuit.author_values)
    assert grad.abs().sum(1).nonzero().flatten().tolist() == [1]


def test_legacy_checkpoint_and_standalone_swarm_of_one_keep_their_outputs():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    baseline = ExeaLM(cfg).eval()
    old_config = asdict(cfg)
    del old_config["board_identity_size"]
    assert not any("author_values" in key for key in baseline.state_dict())
    checkpoint = dict(cfg=old_config, specs=baseline.specs, model=baseline.state_dict(), kind="tonn")
    legacy = build_model(checkpoint).eval()
    enabled = ExeaLM(get_config("tiny", board_identity_size=2), baseline.specs).eval()
    enabled.load_state_dict(baseline.state_dict(), strict=False)
    x = torch.randint(0, cfg.vocab_size, (2, 2 * cfg.chunk))
    with torch.no_grad():
        batch = MemoryContext(swarm_size=2)
        assigned = MemoryContext(swarm_size=2, agent_ids=torch.tensor([1, 0]))
        torch.testing.assert_close(legacy(x, assigned)[0], baseline(x, batch)[0], rtol=0, atol=0)
        torch.testing.assert_close(enabled(x, assigned)[0], baseline(x, batch)[0], rtol=0, atol=0)
        for layer in enabled.layers:
            layer.ag.author_values.fill_(1)
        saved = dict(checkpoint, cfg=asdict(enabled.cfg), model=enabled.state_dict())
        restored = build_model(saved).eval()
        torch.testing.assert_close(restored(x, batch)[0], enabled(x, batch)[0], rtol=0, atol=0)
        solo = MemoryContext(swarm_size=1, agent_ids=torch.tensor([1, 0]))
        torch.testing.assert_close(enabled(x, solo)[0], baseline(x, solo)[0], rtol=0, atol=0)
