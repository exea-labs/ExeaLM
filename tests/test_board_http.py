"""Real HTTP and spawned model processes exercise the shared board."""
import multiprocessing as mp
import socket
import time
from urllib.error import HTTPError, URLError

import pytest
import torch

pytest.importorskip('fastapi')
pytest.importorskip('uvicorn')
from exealm import AgentBoard, ExeaLM, MemoryContext, get_config
from exealm.memory.board_http import RemoteAgentBoard


def serve(sock):
    import uvicorn
    from exealm.memory.board_server import create_app
    uvicorn.Server(uvicorn.Config(create_app(capacity=3, token='test'), log_level='error')).run(sockets=[sock])


@pytest.fixture
def remote():
    ctx = mp.get_context('spawn')
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen()
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    proc = ctx.Process(target=serve, args=(sock,))
    proc.start()
    board = RemoteAgentBoard(url, token='test', timeout=1)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                board.read(0, [0], 'cpu')
                break
            except (URLError, TimeoutError):
                if not proc.is_alive() or time.monotonic() > deadline:
                    raise
                time.sleep(.05)
        yield board
    finally:
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        sock.close()


def post(board, author):
    key = torch.full((1, 1, 1, 8), float(author))
    board.post(0, [0], key, key, torch.zeros(1, 1, 1), torch.tensor([author]), 1)


def test_remote_matches_local_retention_merge_and_isolation(remote):
    local = AgentBoard(capacity=3)
    assert remote.read(0, [0], 'cpu') is None
    for author in [1, 2, 3, 1, 4]:
        post(remote, author)
        post(local, author)
    for a, b in zip(remote.read(0, [0, 1], 'cpu'), local.read(0, [0, 1], 'cpu')):
        assert torch.equal(a, b)
    isolated = RemoteAgentBoard(remote.url.rsplit('/boards/', 1)[0], 'other', token='test')
    assert isolated.read(0, [0], 'cpu') is None
    isolated.merge(remote)
    assert torch.equal(isolated.read(0, [0], 'cpu')[0], local.read(0, [0], 'cpu')[0])
    bad = RemoteAgentBoard(remote.url.rsplit('/boards/', 1)[0], token='wrong')
    with pytest.raises(HTTPError) as e:
        bad.read(0, [0], 'cpu')
    assert e.value.code == 401
    with pytest.raises(HTTPError) as e:
        remote._request('merge', {'messages': [dict(layer=0, slot=0, author=7, chunk=0,
                                                  key=[1], val=[1], log_w=0)]})
    assert e.value.code == 422
    assert remote.read(0, [0], 'cpu')[3].tolist() == [[3, 1, 4]]


def model_agent(url, mode, queue):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ExeaLM(get_config('tiny', swarm_size=1)).eval()
    x = torch.randint(0, 256, (1, 24))
    remote = RemoteAgentBoard(url.rsplit('/boards/', 1)[0], token='test')
    with torch.no_grad():
        if mode == 'post':
            model(x, MemoryContext(board=remote, agent_ids=torch.tensor([0]), write_board=True))
            queue.put(True)
        else:
            alone = model(x, MemoryContext())[0]
            heard = model(x, MemoryContext(board=remote, agent_ids=torch.tensor([1])))[0]
            local = AgentBoard(capacity=3)
            model(x, MemoryContext(board=local, agent_ids=torch.tensor([0]), write_board=True))
            expected = model(x, MemoryContext(board=local, agent_ids=torch.tensor([1])))[0]
            queue.put((torch.allclose(heard, expected, atol=1e-6), not torch.allclose(alone, heard)))


def test_two_spawned_tiny_agents_share_board_with_local_logit_parity(remote):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    try:
        for mode in ['post', 'read']:
            proc = ctx.Process(target=model_agent, args=(remote.url, mode, queue))
            proc.start()
            try:
                result = queue.get(timeout=30)
                proc.join(10)
                assert proc.exitcode == 0
                assert result is True if mode == 'post' else result == (True, True)
            finally:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(10)
    finally:
        queue.close()
        queue.join_thread()


def test_bad_merge_is_atomic_and_concurrent_posts_are_not_lost(remote):
    from concurrent.futures import ThreadPoolExecutor
    msg = dict(layer=0, slot=0, author=1, chunk=0, key=[1.] * 8, val=[1.] * 8, log_w=0.)
    bad = dict(msg, author=2, key=[2.], val=[2.])
    with pytest.raises(HTTPError) as e:
        remote._request('merge', {'messages': [msg, bad]})
    assert e.value.code == 422
    assert remote.read(0, [0], 'cpu') is None
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda a: post(remote, a), [1, 2, 3]))
    result = remote.read(0, [0], 'cpu')
    assert sorted(result[3][0].tolist()) == [1, 2, 3]
    assert result[4].all()
