"""Load a checkpoint in one line, from a local file or from the Hugging Face repo.

    import exealm
    model, tok = exealm.from_pretrained("17m/exealm_together_gated_seed0")

The name is a path under checkpoints/ in https://huggingface.co/exea-labs/ExeaLM (without ".pt"), or a local .pt file.
Downloads need `pip install huggingface_hub`; the tokenizer needs `pip install tokenizers`. These are research
checkpoints of a base model (results/README.md says what each one is): they continue text, they do not chat yet.
"""
import os

import torch

from .config import ModelConfig
from .model import ExeaLM, Llama

REPO = "exea-labs/ExeaLM"
TOKENIZER = "tokenizer/tokenizer.json"


def build_model(ck):
    """The model a checkpoint dict holds. Reads both formats: exealm.train.pretrain.save (which records the kind of
    model) and the experiment scripts' (which don't: a state dict with TONN layers or a chunk encoder is an ExeaLM)."""
    cfg = ModelConfig.from_dict(ck["cfg"])
    kind = ck.get("kind")
    if kind is None:
        kind = "tonn" if any(k.startswith(("layers.", "encoder.")) for k in ck["model"]) else "llama"
    model = ExeaLM(cfg, ck.get("specs")) if kind == "tonn" else Llama(cfg)
    model.load_state_dict(ck["model"])
    if getattr(model, "encoder", None) is not None:
        model.encoder.idf_ready = True                     # keep the word rarities saved in the checkpoint
    return model


def _download(filename, revision=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub to download checkpoints, or pass a local .pt file") from e
    return hf_hub_download(REPO, filename, revision=revision)


def from_pretrained(name, device="cpu", tokenizer=True, revision=None):
    """Returns (model in eval mode, tokenizer). tokenizer: True for the project's published 32k tokenizer, a path to
    a tokenizer.json (or its folder), or False for None (for example a `tiny` checkpoint trained on bytes)."""
    if os.path.exists(name):
        ck = torch.load(name, map_location="cpu", weights_only=False)        # your own file: may hold training state
    else:
        if name.endswith(".pt"):
            name = name[:-3]
        path = _download((name if name.startswith("checkpoints/") else f"checkpoints/{name}") + ".pt", revision)
        ck = torch.load(path, map_location="cpu", weights_only=True)         # downloaded: tensors and plain data only
    model = build_model(ck).to(device).eval()
    tok = None
    if tokenizer:
        from .data.tokenizer import load
        tok = load(_download(TOKENIZER, revision) if tokenizer is True else tokenizer)
        if tok.get_vocab_size() != model.cfg.vocab_size:
            raise ValueError(f"the tokenizer has {tok.get_vocab_size()} tokens but the checkpoint was trained with "
                             f"{model.cfg.vocab_size}: pass the tokenizer it was trained with, or tokenizer=False")
    return model, tok
