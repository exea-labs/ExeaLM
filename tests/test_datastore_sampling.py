from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exealm.memory.datastore import Datastore


def shards(lengths):
    return SimpleNamespace(tokens=np.arange(sum(lengths), dtype=np.uint16),
                           offsets=np.cumsum([0] + lengths[:-1]) if lengths else np.array([], dtype=np.int64))


def test_exact_length_document_and_final_window_can_be_sampled():
    ds = Datastore.from_shards(shards([4]), 3, 4)
    assert ds.tokens.tolist() == [[0, 1, 2, 3]] * 3
    assert ds.doc_ids.tolist() == [0, 0, 0]
    ds = Datastore.from_shards(shards([5]), 100, 4)
    assert set(map(tuple, ds.tokens.tolist())) == {(0, 1, 2, 3), (1, 2, 3, 4)}


def test_samples_valid_windows_uniformly_not_documents():
    # Valid starts are 2 (doc 1), and 6, 7, 8 (doc 2). Doc 0 is too short.
    src = shards([2, 4, 6])
    ds = Datastore.from_shards(src, 20_000, 4, seed=5)
    starts, counts = np.unique(ds.tokens[:, 0].numpy(), return_counts=True)
    assert starts.tolist() == [2, 6, 7, 8]
    assert np.all(np.abs(counts - 5000) < 500)
    for row, doc in zip(ds.tokens.tolist(), ds.doc_ids.tolist()):
        assert row == list(range(row[0], row[0] + 4))
        start = src.offsets[doc]
        end = src.offsets[doc + 1] if doc + 1 < len(src.offsets) else len(src.tokens)
        assert start <= row[0] and row[-1] < end
    same = Datastore.from_shards(src, 20_000, 4, seed=5)
    assert torch.equal(ds.tokens, same.tokens) and torch.equal(ds.doc_ids, same.doc_ids)


@pytest.mark.parametrize('lengths', [[], [2], [2, 2, 2, 2]])
def test_no_valid_window_fails_instead_of_rejection_loop(lengths):
    with pytest.raises(ValueError, match='no document'):
        Datastore.from_shards(shards(lengths), 10, 4)


@pytest.mark.parametrize('n_chunks,chunk_len', [(0, 4), (-1, 4), (1, 0), (1, -1)])
def test_invalid_sizes_are_rejected(n_chunks, chunk_len):
    with pytest.raises(ValueError, match='positive'):
        Datastore.from_shards(shards([10]), n_chunks, chunk_len)
