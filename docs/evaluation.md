# Checkpoint evaluation

Install the optional harness dependencies:

```bash
pip install -e '.[eval]'
python -m exealm.eval.harness --ckpt runs/pretrain/model.pt \
  --tokenizer tokenizer --tasks hellaswag --limit 20 --out runs/hellaswag.json
```

Use the **exact tokenizer used to train the checkpoint**. A directory containing
`tokenizer.json` or a JSON file is accepted. The adapter checks vocabulary size and
the end-of-text token and records SHA-256 hashes of both files in the results. Equal
vocabulary sizes alone cannot identify a correct tokenizer, so check the recorded
hash against the training artifact. The project's public tokenizer is at
[exea-labs/ExeaLM](https://huggingface.co/exea-labs/ExeaLM/tree/main/tokenizer).
The published 32k tokenizer currently has 32,768 entries.

Both TONN and Llama checkpoints saved by `exealm.train.pretrain.save` are supported.
Evaluation loads the saved architecture and weights, uses eval/inference mode, and
discards optimizer state. Requests run individually as solo agents, with no external
retrieval datastore or persistent board. A model's own causal AG circuit remains
part of its architecture. Examples never communicate with one another.

The adapter uses the harness's context/continuation tokenization convention,
including moving trailing context whitespace into the continuation. Scores sum
next-token log probabilities only for continuation targets; the greedy flag checks
every target against the argmax. Empty continuations have likelihood zero. Inputs
use at most `--max_length` tokens (default 1024). Long contexts are left-truncated;
continuations longer than the limit are scored in disjoint target blocks instead
of dropping targets. Rolling likelihood uses the harness's disjoint rolling windows
and scores each document independently, including its first token conditioned on
end-of-text. These are full-prefix forward passes, with no recurrent/KV cache.

Generation is greedy and supports `until` (string or list of strings),
`max_gen_toks`, `do_sample=False`, and temperature zero. It stops at end-of-text,
the earliest decoded stop string, or the token budget. Stop strings can span token
boundaries and are excluded from the returned text. Context is bounded at every
step. Sampling options are rejected explicitly; no chat template is supplied by
this adapter. Tasks requiring those features need a separate implementation.
Harness tasks that execute untrusted generated code are not enabled automatically.

Multiple tasks may be comma-separated. `--include_path DIR` loads custom harness
task YAML files. `--threads` defaults to four, and `--device` defaults to CPU.
`--out` must be a new JSON file; when omitted, results are printed. JSON task
metadata is serialized with the harness's own serializer, including callable task
preprocessors. CI exercises a local multiple-choice task and the complete module
entry point without downloading a benchmark.

The implementation follows the official
[model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md)
and [model API](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py).
The optional dependency is limited to the validated 0.4.x API family.

## Reproducible integration smoke test

```bash
python -m experiments.harness_smoke --out runs/harness-smoke --limit 20 \
  --tokenizer_revision 36ce8b6fc71a4d05e57c8ac800985c81d9f6863d
```

This downloads only the project's tokenizer, creates a tiny TONN checkpoint with
three optimization steps on independent uniform token IDs, and runs 20 HellaSwag
validation examples through the same CLI entry point. It uses no benchmark training
data and downloads no model weights. It verifies integration, not useful language
model quality. The output directory must be new. The source record includes the
resolved tokenizer revision and model configuration; results include file hashes.
A recorded run is in `results/harness_smoke/`. The actual checkpoint and downloaded
data stay outside version control.
