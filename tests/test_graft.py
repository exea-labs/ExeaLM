"""Grafting memory slots onto a model pretrained without them (exealm.train.graft), then training from it (--init)."""
import json

import pytest
import torch

from exealm import Datastore, ExeaLM, MemoryContext, get_config
from exealm.data.loader import SyntheticShards
from exealm.train import pretrain
from exealm.train.graft import contribution, graft
from exealm.train.pretrain import load_model

CORE = dict(pinned={}, menu=("D", "I", "GI", "M", "MI", "MIM", "EP", "G", "C", "SM", "OS"), swarm_size=1)


def trained(tmp_path):
    out = tmp_path / "core"
    cfg = tmp_path / "core.json"
    cfg.write_text(json.dumps(CORE))
    pretrain.main(["--preset", "tiny", "--data", "synthetic", "--seq", "32", "--batch", "4", "--steps", "3",
                   "--eval_every", "3", "--search", "0", "--device", "cpu", "--config", str(cfg), "--out", str(out)])
    return out / "model.pt"


def test_a_graft_replaces_the_weakest_circuits_and_barely_changes_the_model(tmp_path):
    path = trained(tmp_path)
    ck = torch.load(path, weights_only=False)
    before, _ = load_model(path)
    new = graft(ck, {"RG": 1, "AG": 1}, {"rag_lexical": 16})
    for r in new["grafted"]["replaced"]:
        layer = before.layers[r["layer"]]
        others = [k for k in range(layer.K) if k not in {x["slot"] for x in new["grafted"]["replaced"]
                                                         if x["layer"] == r["layer"]}]
        assert contribution(layer, r["slot"]) <= min(contribution(layer, k) for k in others)
    torch.save(new, tmp_path / "grafted.pt")
    after, ck2 = load_model(tmp_path / "grafted.pt")
    assert all(sum(s["tmpl"] == t for s in row) == 1 for row in after.specs for t in ("RG", "AG"))
    assert after.encoder is not None and ck2["cfg"]["rag_lexical"] == 16 and ck2["cfg"]["pinned"] == {"RG": 1, "AG": 1}
    x = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(0))
    before.eval()
    after.eval()
    with torch.no_grad():
        a, b = before(x)[0], after(x)[0]
    data = SyntheticShards()
    ds = Datastore.from_shards(data, 200, after.cfg.rag_chunk_len, seed=1)
    ds.refresh(after.encoder, after.emb)
    with torch.no_grad():
        c = after(x, MemoryContext(datastore=ds, doc_ids=torch.full((2, 32), 10 ** 6)))[0]
    assert torch.isfinite(c).all()
    assert (a.softmax(-1) - b.softmax(-1)).abs().max() < 0.05                # only the weakest circuits are gone


def test_training_continues_from_a_graft_with_the_library_attached(tmp_path):
    path = trained(tmp_path)
    new = graft(torch.load(path, weights_only=False), {"RG": 1})
    torch.save(new, tmp_path / "grafted.pt")
    out = tmp_path / "post"
    pretrain.main(["--init", str(tmp_path / "grafted.pt"), "--data", "synthetic", "--seq", "32", "--batch", "4",
                   "--steps", "2", "--eval_every", "2", "--search", "0", "--device", "cpu", "--datastore_chunks", "300",
                   "--out", str(out)])
    model, ck = load_model(out / "model.pt")
    assert all(any(s["tmpl"] == "RG" for s in row) for row in model.specs)
    with pytest.raises(ValueError, match="memory circuits"):
        graft(torch.load(path, weights_only=False), {"SM": 1})
