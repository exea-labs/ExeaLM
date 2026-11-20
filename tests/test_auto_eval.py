"""Evaluating every checkpoint of a run as it is written."""
import json

import pytest
import torch

from exealm import Llama, get_config
from exealm.eval import auto
from exealm.train.pretrain import make_optimizer, save


def test_table_and_numbers_need_no_harness():
    result = dict(results=dict(hellaswag={"acc,none": 0.25, "acc_stderr,none": 0.01, "alias": "hellaswag"},
                               wikitext={"word_perplexity,none": 900.0, "bits_per_byte,none": 2.0}))
    m = auto.numbers(result)
    assert m == dict(hellaswag=dict(acc=0.25, acc_stderr=0.01), wikitext=dict(word_perplexity=900.0, bits_per_byte=2.0))
    recs = [dict(step=200, metrics=m), dict(step=100, metrics=dict(hellaswag=dict(acc=0.2))),
            dict(step=100, metrics=dict(hellaswag=dict(acc=0.3)))]
    assert auto.table(recs).splitlines() == [
        "| step | hellaswag | wikitext |", "|---:|---|---|",
        "| 100 | 0.3000 (acc) |  |", "| 200 | 0.2500 (acc) | 900.0000 (word_perplexity) |"]


@pytest.fixture
def run(tmp_path, short_cache):
    pytest.importorskip("lm_eval")
    pytest.importorskip("tokenizers")
    import yaml
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    old_threads = torch.get_num_threads()
    torch.manual_seed(1)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["a b c hello world" * 20],
                            trainers.BpeTrainer(vocab_size=270, min_frequency=2, special_tokens=["<|endoftext|>"],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(tmp_path / "tokenizer.json"))
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "data.jsonl").write_text("".join(json.dumps(dict(question="hello", choices=[" world", " a"], answer=0))
                                              + "\n" for _ in range(3)), encoding="utf-8", newline="\n")
    (tasks / "fixture.yaml").write_text(yaml.safe_dump(dict(
        task="exealm_auto_fixture", dataset_path="json",
        dataset_kwargs=dict(data_files={"test": str(tasks / "data.jsonl")}, cache_dir=short_cache),
        test_split="test", output_type="multiple_choice", doc_to_text="question", doc_to_target="answer",
        doc_to_choice="choices", num_fewshot=0)), encoding="utf-8", newline="\n")
    cfg = get_config("tiny", vocab_size=tok.get_vocab_size())
    model = Llama(cfg)

    def write(name, step):
        save(str(tmp_path / "run" / name), model, make_optimizer(model, 0.001, 0.1), step, cfg, "llama")

    yield tmp_path, write
    torch.set_num_threads(old_threads)


def test_each_checkpoint_is_scored_once_and_the_table_grows(run):
    tmp, write = run
    args = ["--run", str(tmp / "run"), "--tokenizer", str(tmp), "--tasks", "exealm_auto_fixture",
            "--include_path", str(tmp / "tasks"), "--limit", "2", "--max_length", "16"]
    write("ckpt.pt", 100)
    first = auto.main(args)
    assert [r["step"] for r in first] == [100] and first[0]["file"] == "ckpt.pt"
    assert first[0]["metrics"]["exealm_auto_fixture"]["acc"] in (0.0, 1.0) and len(first[0]["checkpoint_sha256"]) == 64
    assert [r["step"] for r in auto.main(args)] == [100]                 # nothing new: no second record
    write("ckpt.pt", 200)                                               # training rewrote the checkpoint
    (tmp / "run" / "broken.pt").write_bytes(b"not a checkpoint")
    assert [r["step"] for r in auto.main(args)] == [100, 200]
    write("model.pt", 300)
    done = auto.main(args + ["--watch", "0.01"])                        # exits once the final model is scored
    assert [(r["step"], r["file"]) for r in done] == [(100, "ckpt.pt"), (200, "ckpt.pt"), (300, "model.pt")]
    rows = (tmp / "run" / "evals.md").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "| step | exealm_auto_fixture |" and [r.split("|")[1].strip() for r in rows[2:]] == ["100", "200", "300"]
    more = auto.main(args[:-4] + ["--limit", "3", "--max_length", "16"])     # another limit is another evaluation
    assert len(more) == 5


def test_a_final_model_at_the_same_step_as_the_last_checkpoint_is_scored_once(run):
    tmp, write = run
    write("ckpt.pt", 50)
    write("model.pt", 50)
    recs = auto.main(["--run", str(tmp / "run"), "--tokenizer", str(tmp), "--tasks", "exealm_auto_fixture",
                      "--include_path", str(tmp / "tasks"), "--limit", "1", "--max_length", "16", "--watch", "0.01"])
    assert len(recs) == 1 and recs[0]["step"] == 50
