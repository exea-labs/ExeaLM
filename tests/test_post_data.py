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


THOUGHT = "<think>\nLet me add them. 2 + 2 is 4.\n</think>"
MATH = dict(expected_answer="4", problem_type="has_answer_extracted", problem_source="MATH_training_set",
            generation_model="DeepSeek-R1", pass_rate_72b_tir="1/32", problem="What is 2 + 2?",
            generated_solution=THOUGHT + "The sum is \\boxed{4}.", inference_mode="cot", used_in_kaggle=False)
PROGRAM = "print(sum(map(int, input().split())))"
CODE = dict(id="x1", input="Print the sum of two numbers.", output=THOUGHT + f"\n\n```python\n{PROGRAM}\n```",
            source="codeforces", license="cc-by-4.0", dataset="code_contests", split="train", difficulty="7",
            solution=PROGRAM)
SKY = "<think> Short light scatters more.\n</think>\n\nRayleigh scattering."      # OpenThoughts3: a space after <think>
OT3 = dict(difficulty=7, source="stackexchange-physics", domain="science",
           conversations=[{"from": "human", "value": "Why is the sky blue?"}, {"from": "gpt", "value": SKY}])


def test_reasoning_rows_in_three_sets_get_one_thinking_form():
    want = "<think>\nLet me add them. 2 + 2 is 4.\n</think>\n\n"
    got = post.with_thinking(post.to_messages(MATH)["messages"])
    assert got == [dict(role="user", content="What is 2 + 2?"),
                   dict(role="assistant", content=want + "The sum is \\boxed{4}.")]
    got = post.with_thinking(post.to_messages(CODE)["messages"])
    assert got[-1]["content"] == want + f"```python\n{PROGRAM}\n```"
    got = post.with_thinking(post.to_messages(OT3)["messages"])
    want = "<think>\nShort light scatters more.\n</think>\n\nRayleigh scattering."
    assert got == [dict(role="user", content="Why is the sky blue?"), dict(role="assistant", content=want)]


def test_thinking_can_be_stripped_or_capped():
    msgs = post.to_messages(MATH)["messages"]
    assert post.with_thinking(msgs, "strip")[-1]["content"] == "The sum is \\boxed{4}."
    assert post.with_thinking(msgs, "keep", max_think_chars=100) == post.with_thinking(msgs)
    with pytest.raises(post.Drop, match="longer than --max_think_chars"):
        post.with_thinking(msgs, "strip", max_think_chars=10)
    with pytest.raises(ValueError):
        post.with_thinking(msgs, "drop")


@pytest.mark.parametrize("answer, why", [
    ("The sum is 4.", "no thinking section"),
    ("<think>\nLet me add 2 and 2, which", "thinking never ends"),                  # the generation was cut off
    ("<think>a</think>b</think>c", "more than one thinking section"),
    ("<think>a</think><think>b</think>c", "more than one thinking section"),
    ("<think> \n</think>4", "empty thinking"),
    ("<think>2 + 2</think>\n\n", "no answer after the thinking"),
])
def test_broken_reasoning_rows_are_dropped_and_say_why(answer, why):
    with pytest.raises(post.Drop, match=why):
        post.with_thinking(post.to_messages(dict(MATH, generated_solution=answer))["messages"])


def test_code_rows_without_a_license_or_a_question_are_dropped():
    with pytest.raises(post.Drop, match="no license"):
        post.to_messages(dict(CODE, license="-", dataset="-", split="-"))
    with pytest.raises(post.Drop, match="question is not in the row"):
        post.to_messages(dict(CODE, input="-", index="12", dataset="taco"))
    with pytest.raises(post.Drop, match="empty question"):
        post.to_messages(dict(MATH, problem=" "))


def test_convert_all_selects_rows_and_applies_the_thinking_options():
    other = dict(OT3, domain="math")
    cut = dict(OT3, conversations=[OT3["conversations"][0], {"from": "gpt", "value": "<think> Short light"}])
    out, report = post.convert_all([OT3, other, cut, OT3], "sft", "open-thoughts/OpenThoughts3-1.2M",
                                   keep=("domain", ("science",)), think="strip")
    assert report == dict(read=4, kept=2, dropped={"domain not selected": 1, "thinking never ends (cut off)": 1})
    assert out[0]["messages"] == [dict(role="user", content="Why is the sky blue?"),
                                  dict(role="assistant", content="Rayleigh scattering.")]
    assert out[0]["source"] == "stackexchange-physics"
    safety = dict(messages=[U, A], source=post.SAFETY[0])
    out, report = post.convert_all([dict(safety, source="ai2-adapt-dev/flan_v2_converted"), safety], "sft", "r",
                                   keep=("source", post.SAFETY))
    assert report["kept"] == 1 and out[0]["source"] == post.SAFETY[0]


def test_streamed_files_and_thinking_options_reach_the_report(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def rows(repo, *args, **kwargs):
        assert repo == "open-thoughts/OpenThoughts3-1.2M" and not args
        assert kwargs == dict(data_files=post.shards(110, 119, 120), split="train", streaming=True)
        yield from [OT3, dict(OT3, domain="code")]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=rows))
    report = post.main(["openthoughts3-science", "--max_think_chars", "500", "--out", str(tmp_path)])
    assert report["kept"] == 1 and report["think"] == "keep" and report["max_think_chars"] == 500
    assert json.loads((tmp_path / "openthoughts3-science.report.json").read_text(encoding="utf-8")) == report
    from exealm.train import sft
    assert len(sft.load_conversations(str(tmp_path / "openthoughts3-science.jsonl.gz"))[0]) == 1
    with pytest.raises(SystemExit):
        post.main(["tulu3-sft", "--think", "strip", "--out", str(tmp_path)])
