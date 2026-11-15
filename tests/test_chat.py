"""The chat template, the chat demo and the one-line checkpoint loader."""
import sys
import types

import pytest
import torch

import exealm
from exealm import ExeaLM, Llama, get_config, hub
from exealm.chat import CHAT_TEMPLATE, Bytes, answer, chat_prompt, main
from exealm.train.pretrain import make_optimizer, save

TALK = [dict(role="system", content="Be brief."), dict(role="user", content="What is 2 + 2?\nこんにちは"),
        dict(role="assistant", content="4"), dict(role="user", content="def f(x):\n    return x")]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    pytest.importorskip("tokenizers")
    import gzip
    import json
    from exealm.data.tokenizer import train
    d = tmp_path_factory.mktemp("tok")
    with gzip.open(d / "a.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps(dict(text=f"What is {i} + 2? こんにちは def f(x): return x. Be brief. {i}")) + "\n")
    return train([str(d / "a.jsonl.gz")], 400, str(d / "tokenizer"))


def test_chat_prompt_ends_with_the_assistant_header(tok):
    ids = chat_prompt(TALK, tok)
    start, end, asst = (tok.token_to_id(t) for t in ("<|im_start|>", "<|im_end|>", "<|assistant|>"))
    assert ids[-2:] == [start, asst] and ids[0] == start
    assert ids.count(start) == 5 and ids.count(end) == 4
    assert chat_prompt(TALK, tok, add_generation_prompt=False) == ids[:-2]
    with pytest.raises(ValueError, match="unknown role"):
        chat_prompt([dict(role="robot", content="hi")], tok)


def test_jinja_template_gives_the_same_tokens(tok):
    jinja2 = pytest.importorskip("jinja2")
    for gen in (True, False):
        text = jinja2.Template(CHAT_TEMPLATE).render(messages=TALK, add_generation_prompt=gen)
        assert tok.encode(text).ids == chat_prompt(TALK, tok, add_generation_prompt=gen)


def checkpoint(tmp_path, kind="tonn", vocab=256):
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=vocab)
    model = ExeaLM(cfg) if kind == "tonn" else Llama(cfg)
    path = str(tmp_path / f"{kind}.pt")
    save(path, model, make_optimizer(model, 1e-3, 0.1), 0, cfg, kind)
    return model, path


@pytest.mark.parametrize("kind", ["tonn", "llama"])
def test_from_pretrained_reads_both_checkpoint_formats(tmp_path, kind):
    model, path = checkpoint(tmp_path, kind)
    x = torch.randint(0, 256, (2, 16))
    want = model.eval()(x)[0]
    loaded, none = exealm.from_pretrained(path, tokenizer=False)
    assert none is None and not loaded.training and type(loaded) is type(model)
    assert torch.equal(loaded(x)[0], want)
    # the experiment scripts' format: no "kind", the config as a plain dict
    bare = str(tmp_path / "bare.pt")
    torch.save(dict(model=model.state_dict(), cfg=vars(model.cfg), step=3, specs=getattr(model, "specs", None)), bare)
    again, _ = exealm.from_pretrained(bare, tokenizer=False)
    assert type(again) is type(model) and torch.equal(again(x)[0], want)


def test_from_pretrained_downloads_by_name_and_checks_the_tokenizer(tmp_path, tok, monkeypatch):
    model, path = checkpoint(tmp_path, vocab=tok.get_vocab_size())
    torch.save(dict(model=model.state_dict(), cfg=vars(model.cfg), specs=model.specs), path)
    tok.save(str(tmp_path / "tokenizer.json"))
    asked = []

    def fake_download(filename, revision=None):
        asked.append(filename)
        return path if filename.endswith(".pt") else str(tmp_path / "tokenizer.json")

    monkeypatch.setattr(hub, "_download", fake_download)
    loaded, t = exealm.from_pretrained("17m/some_run")
    assert asked == ["checkpoints/17m/some_run.pt", "tokenizer/tokenizer.json"]
    assert t.get_vocab_size() == loaded.cfg.vocab_size
    _, small = checkpoint(tmp_path, "llama", vocab=300)
    monkeypatch.setattr(hub, "_download", lambda f, revision=None: small if f.endswith(".pt") else
                        str(tmp_path / "tokenizer.json"))
    with pytest.raises(ValueError, match="tokenizer has"):
        exealm.from_pretrained("17m/other.pt")


def test_demo_answers_once_alone_and_as_a_swarm(tmp_path, tok, capsys):
    _, path = checkpoint(tmp_path, vocab=tok.get_vocab_size())
    tok.save(str(tmp_path / "tokenizer.json"))
    common = ["--ckpt", path, "--tokenizer", str(tmp_path / "tokenizer.json"), "--max_new_tokens", "5"]
    assert len(main(common + ["--prompt", "What is 2 + 2?", "--system", "Be brief."])) == 1
    assert capsys.readouterr().out.startswith("model> ")
    replies = main(common + ["--prompt", "What is 2 + 2?", "--swarm", "3"])
    assert len(replies) == 3
    assert [line[:8] for line in capsys.readouterr().out.splitlines()][:1] == ["agent 0>"]


def test_demo_continues_raw_bytes_for_a_tiny_checkpoint(tmp_path):
    model, path = checkpoint(tmp_path)
    assert len(main(["--ckpt", path, "--tokenizer", "none", "--raw", "--prompt", "abc", "--max_new_tokens", "4"])) == 1
    with pytest.raises(SystemExit):
        main(["--ckpt", path, "--tokenizer", "none", "--prompt", "abc"])
    b = Bytes()
    assert b.decode(b.encode("こんにちは").ids) == "こんにちは"
    out = answer(model.eval(), b, ["ab", "cd"], agents=2, raw=True, max_new_tokens=3, temperature=0)
    assert len(out) == 2


def test_wandb_logging_flattens_records_and_keeps_the_run_id(tmp_path, monkeypatch):
    from exealm.train import pretrain
    rec = dict(event="eval", step=20, train_loss=5.5, lr=0.001, note="text", board=dict(read_nothing=0.9,
               read_nothing_per_layer=[0.8, None]), ok=True)
    assert pretrain.wandb_record(rec) == {"eval/train_loss": 5.5, "eval/lr": 0.001, "eval/board/read_nothing": 0.9,
                                          "eval/board/read_nothing_per_layer/0": 0.8, "eval/ok": True}
    calls = dict(init=[], log=[], finished=0)

    class Run:
        id = "abc123"

        def log(self, data, step):
            calls["log"].append((step, data))

        def finish(self):
            calls["finished"] += 1

    fake = types.ModuleType("wandb")
    fake.init = lambda **kw: calls["init"].append(kw) or Run()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    out = str(tmp_path / "run")
    args = ["--preset", "tiny", "--steps", "8", "--eval_every", "4", "--search", "0", "--device", "cpu", "--out", out,
            "--wandb", "exealm-test"]
    pretrain.main(args + ["--save_every", "4", "--stop_at", "4"])
    assert calls["init"][0]["project"] == "exealm-test" and calls["init"][0]["id"] is None
    assert calls["init"][0]["name"] == "run" and calls["init"][0]["config"]["params"] > 0
    assert [s for s, _ in calls["log"]] == [4, 4] and "eval/val_bits_per_token" in calls["log"][0][1]
    assert calls["finished"] == 1
    pretrain.main(args + ["--resume", "auto"])
    assert calls["init"][1]["id"] == "abc123" and calls["finished"] == 2
    assert [s for s, _ in calls["log"]][-1] == 8
