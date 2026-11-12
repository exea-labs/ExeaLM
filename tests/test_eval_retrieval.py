import torch

from exealm import Datastore, ExeaLM, MemoryContext, get_config
from exealm.eval.show_retrieval import main, retrievals


def test_retrievals_reports_every_rg_slot_and_never_the_document_being_read():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    m = ExeaLM(cfg)
    ds = Datastore(torch.randint(0, cfg.vocab_size, (60, cfg.rag_chunk_len)), torch.arange(60) % 6)
    ds.refresh(m.encoder, m.emb)
    T = 4 * cfg.chunk
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    recs = retrievals(m, ids, MemoryContext(datastore=ds, doc_ids=torch.full((1, T), 2)))
    n_rg = sum(s["tmpl"] == "RG" for row in m.specs for s in row)
    assert len(recs) == n_rg * 3                                   # chunks 1, 2, 3 retrieve; chunk 0 cannot
    for r in recs:
        assert 1 <= r["chunk"] <= 3 and len(r["hits"]) == cfg.rag_topk
        assert all(int(ds.doc_ids[i]) != 2 for i, _ in r["hits"])
        assert abs(sum(w for _, w in r["hits"]) - 1) < 1e-4
    assert all(lay.rg.record is False for lay in m.layers)       # switched back off


def test_show_retrieval_runs_on_synthetic_data(capsys):
    main(["--preset", "tiny", "--data", "synthetic", "--datastore_chunks", "200", "--top", "2"])
    out = capsys.readouterr().out
    assert "=== chunk 1" in out and "layer 0 slot" in out


def test_hybrid_layers_are_reported_by_their_model_layer_number():
    torch.manual_seed(0)
    cfg = get_config("tiny", n_layers=3, attn_layers=(1,))                # TONN layers are model layers 0 and 2
    m = ExeaLM(cfg)
    ds = Datastore(torch.randint(0, cfg.vocab_size, (60, cfg.rag_chunk_len)), torch.arange(60) % 6)
    ds.refresh(m.encoder, m.emb)
    T = 3 * cfg.chunk
    mem = MemoryContext(datastore=ds, doc_ids=torch.full((1, T), 2))
    recs = retrievals(m, torch.randint(0, cfg.vocab_size, (1, T)), mem)
    assert {r["layer"] for r in recs} == {0, 2}
