"""Evaluation.

Built in: bits per token / per byte on held-out data (exealm/train/pretrain.py reports it during training).

Planned (each is a good first project):
  * English: HellaSwag, ARC-Easy, PIQA, WinoGrande, LAMBADA through EleutherAI's lm-evaluation-harness
    (https://github.com/EleutherAI/lm-evaluation-harness). A 0.1B model scores near chance on MMLU, so don't use it
    to compare small models.
  * Japanese: llm-jp-eval (https://github.com/llm-jp/llm-jp-eval), JCommonsenseQA, JNLI.
  * Retrieval (RG): questions whose answers are in the datastore but not in the prompt; compare with RG switched off.
  * Agents (AG): swarm tasks where each agent sees part of the information; solo vs swarms of 2, 4, 8.
  * Speed: tokens per second for training and generation against the Llama baseline of the same size. Built:
    python -m exealm.eval.speed --preset small (exealm/eval/speed.py).

  Built: lm-evaluation-harness adapter, python -m exealm.eval.harness --ckpt PATH --tasks hellaswag --limit 20.
  See docs/evaluation.md for tokenizer requirements, supported requests and limits.
"""
