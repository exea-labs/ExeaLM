import numpy as np
import pytest
import torch

faiss = pytest.importorskip('faiss')
from exealm import Datastore, ExeaLM, MemoryContext, get_config
from exealm.memory.faiss import FaissDatastore


@pytest.fixture(autouse=True)
def threads():
    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    yield
    faiss.omp_set_num_threads(old)


def store(doc_ids=None, nprobe=1):
    torch.manual_seed(10)
    keys = torch.nn.functional.normalize(torch.randn(256, 8), dim=-1)
    docs = torch.arange(256) % 4 if doc_ids is None else doc_ids
    ds = FaissDatastore(torch.zeros(256, 4, dtype=torch.long), docs,
                        nlist=4, pq_m=2, nbits=2, nprobe=nprobe)
    ds.keys = keys
    ds.rebuild_index()
    return ds


@pytest.mark.parametrize('backend', ['exact', 'faiss'])
def test_exclusion_and_insufficient_candidates(backend):
    docs = torch.zeros(256, dtype=torch.long)
    docs[-2:] = 1
    ds = store(docs)
    if backend == 'exact':
        exact = Datastore(ds.tokens, docs)
        exact.keys = ds.keys
        ds = exact
    q = ds.keys[:3]
    idx, ok = ds.search(q, 4, exclude=torch.tensor([0, 1, 5]))
    assert ok.sum(1).tolist() == [2, 4, 4]
    assert not (docs[idx[0, ok[0]]] == 0).any()
    assert not (docs[idx[1, ok[1]]] == 1).any()
    assert idx.min() >= 0 and idx.max() < 256


def test_all_excluded_empty_queries_and_large_k():
    ds = store(torch.zeros(256, dtype=torch.long))
    idx, ok = ds.search(ds.keys[:1], 500, torch.zeros(1, dtype=torch.long))
    assert idx.shape == ok.shape == (1, 256) and not ok.any()
    assert ds.search(ds.keys[:0], 4)[0].shape == (0, 4)
    assert ds.search(ds.keys[:1], 0)[0].shape == (1, 0)
    with pytest.raises(ValueError):
        ds.search(ds.keys[:1], -1)


def test_full_probe_agrees_with_direct_pq_index_and_rebuilds():
    ds = store(nprobe=4)
    q = ds.keys[:10]
    _, candidates = ds.index.search(q.numpy(), 16, params=faiss.IVFPQSearchParameters(nprobe=4))
    scores = torch.einsum('md,mcd->mc', q, ds.keys[torch.from_numpy(candidates)])
    expected = np.take_along_axis(candidates, scores.argsort(dim=1, descending=True)[:, :4].numpy(), axis=1)
    idx, ok = ds.search(q, 4)
    assert ok.all() and np.array_equal(idx.numpy(), expected)
    old = ds.index
    ds.keys = ds.keys.flip(0)
    ds.rebuild_index()
    assert ds.index is not old
    _, candidates = ds.index.search(q.numpy(), 16, params=faiss.IVFPQSearchParameters(nprobe=4))
    scores = torch.einsum('md,mcd->mc', q, ds.keys[torch.from_numpy(candidates)])
    expected = np.take_along_axis(candidates, scores.argsort(dim=1, descending=True)[:, :4].numpy(), axis=1)
    assert np.array_equal(ds.search(q, 4)[0].numpy(), expected)
    ds.keys[0, 0] = float('nan')
    with pytest.raises(ValueError):
        ds.rebuild_index()
    with pytest.raises(RuntimeError):
        ds.search(q, 4)


def test_refresh_model_forward_and_encoder_gradients():
    torch.manual_seed(3)
    model = ExeaLM(get_config('tiny', swarm_size=1))
    ds = FaissDatastore(torch.randint(0, 256, (256, 8)), torch.arange(256) % 7,
                        nlist=4, pq_m=2, nbits=2)
    ds.refresh(model.encoder, model.emb, step=12)
    assert ds.refreshed_at == 12
    assert model.encoder.training
    old = ds.index
    ds.refresh(model.encoder, model.emb, step=13)
    assert ds.index is not old
    x = torch.randint(0, 256, (2, 24))
    mem = MemoryContext(datastore=ds, doc_ids=torch.full(x.shape, 3))
    loss = model(x, mem, x)[1]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.key.weight.grad.abs().sum() > 0
    assert ds.to('cpu') is ds
    assert ds.search(ds.keys[:1], 4)[0].device.type == 'cpu'


def test_exhaustive_candidate_reranking_matches_exact_search():
    ds = store(nprobe=4)
    ds.rerank = 256
    exact = Datastore(ds.tokens, ds.doc_ids)
    exact.keys = ds.keys
    q = ds.keys[:16]
    exclude = ds.doc_ids[:16]
    idx, ok = ds.search(q, 4, exclude)
    expected, expected_ok = exact.search(q, 4, exclude)
    assert torch.equal(ok, expected_ok) and torch.equal(idx, expected)
