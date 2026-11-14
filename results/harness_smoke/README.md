# Harness integration smoke result

The module CLI completed a 20-example HellaSwag validation run using a tiny TONN
checkpoint and the project's published 32,768-token tokenizer. The checkpoint was
trained for three steps on independent uniform random token IDs, not benchmark
text. This is a plumbing check, not a meaningful model-quality estimate.

Recorded result: accuracy 0.30, length-normalized accuracy 0.35; effective 20 of
10,042 validation examples. Bootstrap resampling was disabled for this smoke run.
Evaluation used CPU, four threads, max_length=256, zero-shot prompts, and solo
agents with no external datastore or persistent board. The summary records exact
checkpoint/tokenizer hashes, tokenizer revision, model configuration and package
versions. No checkpoint, tokenizer or benchmark text is committed.

Reproduce with a new directory:

```bash
python -m experiments.harness_smoke --out runs/harness-smoke --limit 20 \
  --tokenizer_revision 36ce8b6fc71a4d05e57c8ac800985c81d9f6863d
```

The same module entry point was separately invoked as:

```bash
python -m exealm.eval.harness --ckpt CHECKPOINT --tokenizer TOKENIZER_JSON \
  --tasks hellaswag --limit 20 --max_length 256 --threads 4 --out NEW_RESULT_JSON
```

Offline tests compare continuation scores to direct forwards, validate bounded
scoring without dropping long targets, count every rolling target, and exercise
empty prompts, whitespace boundaries, EOS, stop strings spanning tokens, token
limits and vocabulary mismatches. A local task runs through simple_evaluate, and
an offline CLI task with callable metadata verifies result serialization. CI installs
the eval extra so these tests run instead of being skipped.
