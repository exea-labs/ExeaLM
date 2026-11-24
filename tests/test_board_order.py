"""board_order: "time" (an agent hears every teammate's posts from earlier chunks) against "doc" (an agent hears
every post of the teammates before it in the swarm, from its first token, and nothing from those after it)."""
import json

import pytest
import torch

from exealm import ExeaLM, MemoryContext, get_config
from exealm.data.loader import SyntheticShards
from exealm.model.board import BoardCircuit
from exealm.train import pretrain


def outputs(order, x, swarm=4):
    torch.manual_seed(0)
    cfg = get_config("tiny", board_order=order, swarm_size=swarm)
    model = ExeaLM(cfg).eval()
    with torch.no_grad():
        return model(x, MemoryContext(swarm_size=swarm))[0], cfg.chunk


def changed(order, row, T=32):
    """Which rows' predictions, and from which position, move when one row's tokens change."""
    x = torch.randint(0, 256, (4, T), generator=torch.Generator().manual_seed(1))
    y = x.clone()
    y[row] = torch.randint(0, 256, (T,), generator=torch.Generator().manual_seed(2))
    a, C = outputs(order, x)
    b, _ = outputs(order, y)
    moved = (a - b).abs().amax(-1) > 1e-5                                  # (rows, positions)
    return moved, C


def test_a_doc_order_board_hears_the_text_before_from_the_first_token_and_never_the_text_after():
    moved, C = changed("doc", row=1)
    assert not moved[0].any()                                               # row 0 never hears row 1
    assert moved[2, :C].any() and moved[3, :C].any()                        # rows after it hear it at once
    moved, _ = changed("doc", row=3)
    assert not moved[:3].any()                                              # nobody hears the last row


def test_a_time_order_board_hears_nothing_in_the_first_chunk_and_hears_every_teammate_after_it():
    moved, C = changed("time", row=1)
    assert not moved[[0, 2, 3], :C].any()                                   # nothing posted yet
    assert moved[0, C:].any() and moved[2, C:].any()                        # the row before hears it too


def test_alone_the_two_orders_are_the_same_model():
    x = torch.randint(0, 256, (2, 32), generator=torch.Generator().manual_seed(3))
    a, _ = outputs("time", x, swarm=1)
    b, _ = outputs("doc", x, swarm=1)
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="board_order"):
        BoardCircuit(get_config("tiny", board_order="space"))


def test_doc_order_training_keeps_the_seam_targets(tmp_path, monkeypatch):
    seen = []
    real = SyntheticShards.batch

    def spy(self, *args, **kw):
        seen.append(kw.get("mask_seams", True))
        return real(self, *args, **kw)

    monkeypatch.setattr(SyntheticShards, "batch", spy)
    for order in ("time", "doc"):
        cfg = tmp_path / f"{order}.json"
        cfg.write_text(json.dumps({"board_order": order}))
        seen.clear()
        pretrain.main(["--preset", "tiny", "--data", "synthetic", "--seq", "32", "--batch", "8", "--steps", "2",
                       "--eval_every", "2", "--search", "0", "--device", "cpu", "--swarm_policy", "split",
                       "--config", str(cfg), "--out", str(tmp_path / order)])
        assert seen and all(m == (order == "time") for m in seen)
