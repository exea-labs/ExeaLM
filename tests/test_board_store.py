"""Persistent board merges obey the same retention policy as posts."""
import torch

from exealm import AgentBoard


def post(board, author, value, slots=(0,)):
    key = torch.full((1, 1, len(slots), 4), float(value))
    board.post(0, slots, key, key, torch.zeros(1, 1, len(slots)), torch.tensor([author]), 1)


def test_merge_retains_newest_messages_per_channel_and_invalidates_reads():
    board, incoming = AgentBoard(capacity=2), AgentBoard(capacity=10)
    post(board, 1, 1, (0, 1))
    post(board, 2, 2)
    before = board.read(0, [0, 1], 'cpu')
    post(incoming, 1, 10)  # replacement must be newer than author 2
    post(incoming, 3, 3)
    clock = board.clock
    board.merge(incoming)
    assert list(board.channels[0, 0]) == [(1, 0), (3, 0)]
    assert list(board.channels[0, 1]) == [(1, 0)]
    after = board.read(0, [0, 1], 'cpu')
    assert after is not before
    assert after[0][0, :, 0].tolist() == [10, 3]
    assert after[4].tolist() == [[True, True], [True, False]]
    assert list(incoming.channels[0, 0]) == [(1, 0), (3, 0)]
    assert board.clock == clock + 1
    post(board, 4, 4)
    assert list(board.channels[0, 0]) == [(3, 0), (4, 0)]


def test_merge_larger_source_and_self_merge():
    board, incoming = AgentBoard(capacity=2), AgentBoard(capacity=5)
    for author in range(5):
        post(incoming, author, author, (0, 1))
    board.merge(incoming)
    assert all(list(c) == [(3, 0), (4, 0)] for c in board.channels.values())
    assert all(len(c) == 5 for c in incoming.channels.values())
    expected = board.read(0, [0, 1], 'cpu')
    board.merge(board)
    assert all(torch.equal(a, b) for a, b in zip(expected, board.read(0, [0, 1], 'cpu')))


def test_zero_capacity_merge_retains_no_messages():
    board, incoming = AgentBoard(capacity=0), AgentBoard()
    post(incoming, 1, 1)
    assert board.read(0, [0], 'cpu') is None
    board.merge(incoming)
    assert board.read(0, [0], 'cpu') is None
    assert len(incoming.channels[0, 0]) == 1
