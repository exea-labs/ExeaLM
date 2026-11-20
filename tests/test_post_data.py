"""Post-training data: rows of the open datasets become the files the SFT, DPO and GRPO loops read."""
import gzip
import json

import pytest

from exealm.data import post

U, A = dict(role="user", content="q"), dict(role="assistant", content="a")


def test_sft_rows_in_three_forms():
    assert post.to_messages(dict(messages=[U, A]))["messages"] == [U, A]
    got = post.to_messages(dict(conversations=[{"from": "system", "value": "s"}, {"from": "human", "value": "q"},
                                               {"from": "gpt", "value": "a"}]))["messages"]
    assert [m["role"] for m in got] == ["system", "user", "assistant"]
    got = post.to_messages(dict(instruction="質問", context="文脈", response="答え"))["messages"]
    assert got == [dict(role="user", content="質問\n\n文脈"), dict(role="assistant", content="答え")]
    assert post.to_messages(dict(instruction="q", input="", output="a"))["messages"][0]["content"] == "q"


@pytest.mark.parametrize("row, why", [
    (dict(messages=[U]), "does not end with an answer"),
    (dict(messages=[dict(role="robot", content="x"), A]), "unknown role"),
    (dict(messages=[U, dict(role="assistant", content=" ")]), "empty content"),
    (dict(messages=[U, dict(role="assistant", content="a", function_calls="[...]")]), "function call"),
    (dict(text="plain text"), "unknown form"),
    (dict(instruction="q", response=None), "empty instruction or answer"),
])
def test_sft_rows_that_are_dropped_say_why(row, why):
    with pytest.raises(post.Drop, match=why):
        post.to_messages(row)


def test_preferences_from_two_conversations_or_two_strings():
    extra = dict(annotations=None, audio=None, refusal=None)                 # Dolci rows carry many empty fields
    row = dict(chosen=[dict(U, **extra), dict(role="assistant", content="good", **extra)],
               rejected=[dict(U, **extra), dict(role="assistant", content="bad", **extra)])
    assert post.to_preference(row) == dict(messages=[U], chosen="good", rejected="bad")
    assert post.to_preference(dict(prompt="q", chosen="good", rejected="bad")) == dict(messages=[U], chosen="good", rejected="bad")
    with pytest.raises(post.Drop, match="differ before"):
        post.to_preference(dict(chosen=[U, A], rejected=[dict(role="user", content="other"), dict(role="assistant", content="b")]))
    with pytest.raises(post.Drop, match="the same"):
        post.to_preference(dict(chosen=[U, A], rejected=[U, A]))
    with pytest.raises(post.Drop, match="no prompt"):
        post.to_preference(dict(chosen="a", rejected="b"))


def test_tasks_keep_rows_with_an_exact_answer_only():
    row = dict(messages=[dict(role="user", content="Question: 2+2?")], ground_truth=" 4 ", dataset="gsm8k",
               constraint_type=None, constraint=None)
    assert post.to_task(row) == dict(messages=row["messages"], answer="4", reward="math")
    with pytest.raises(post.Drop, match="constraint"):
        post.to_task(dict(row, constraint_type="keywords", constraint="use the word x"))
    with pytest.raises(post.Drop, match="no answer"):
        post.to_task(dict(row, ground_truth=""))


def test_convert_all_counts_what_it_drops_and_the_loops_can_read_the_files(tmp_path):
    rows = [dict(messages=[U, A], source="part-a"), dict(messages=[U]), dict(text="x"), dict(messages=[U, A])]
    out, report = post.convert_all(rows, "sft", "repo/name")
    assert report == dict(read=4, kept=2, dropped={"does not end with an answer": 1, "unknown form": 1})
    assert [r["source"] for r in out] == ["part-a", "repo/name"]
    out, report = post.convert_all(rows * 5, "sft", "repo/name", max_rows=3)
    assert report["kept"] == 3 and len(out) == 3
    post.write(str(tmp_path / "s.jsonl.gz"), out)
    from exealm.train import dpo, sft
    assert len(sft.load_conversations(str(tmp_path / "s.jsonl.gz"))[0]) == 3
    prefs, _ = post.convert_all([dict(chosen=[U, A], rejected=[U, dict(role="assistant", content="b")])], "pref", "r")
    post.write(str(tmp_path / "p.jsonl.gz"), prefs)
    assert dpo.load_preferences(str(tmp_path / "p.jsonl.gz")) == ([([U], "a", "b")], 0)
    with gzip.open(tmp_path / "p.jsonl.gz", "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["source"] == "r"


def test_every_source_names_its_license_and_revision(capsys):
    for name, s in post.SOURCES.items():
        assert s["kind"] in post.CONVERT and s["license"] and s["revision"], name
    post.main(["--list"])
    assert "ultrafeedback" in capsys.readouterr().out
