"""Publish the study's finished models to a public Hugging Face repo: weights only (safetensors, 32-bit), each model's
config and training log, the study's tokenizer and a model card. The checkpoints the runs keep also hold AdamW's
state; a release does not need it. Folders whose name starts with "_" (void or set-aside runs) and learning-rate checks
are never published.

    python experiments/release/export_hf.py runs/exealm_100m_4.1x_s0 runs/qwen_100m_4.1x_s0 ... \\
        --repo exea-labs/exealm-nairr --stage /path/stage --upload

Load one with the repo's code:

    import json; from safetensors.torch import load_file
    from exealm.config import ModelConfig; from exealm.model import ExeaLM, Llama
    c = json.load(open("exealm_100m_4.1x_s0/config.json")); cfg = ModelConfig.from_dict(c["cfg"])
    model = ExeaLM(cfg, c["specs"]) if c["kind"] == "tonn" else Llama(cfg)
    model.load_state_dict(load_file("exealm_100m_4.1x_s0/model.safetensors"))
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def export(run, stage):
    name = os.path.basename(run.rstrip("/"))
    if name.startswith("_") or name.startswith("lr_"):
        raise SystemExit(f"{name}: not published (void, set aside, or a learning-rate check)")
    ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu", weights_only=False)
    out = os.path.join(stage, name)
    os.makedirs(out, exist_ok=True)
    weights = {k: v.detach().float().contiguous().clone() for k, v in ck["model"].items()}   # clone: no shared memory
    save_file(weights, os.path.join(out, "model.safetensors"), metadata={"format": "pt", "run": name})
    st = ck.get("state") or {}
    cfg = dict(run=name, kind=ck["kind"], step=ck["step"], total_steps=st.get("total_steps"), cfg=ck["cfg"],
               specs=ck.get("specs"), params=sum(v.numel() for v in weights.values()))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    if os.path.exists(os.path.join(run, "log.jsonl")):
        shutil.copy(os.path.join(run, "log.jsonl"), os.path.join(out, "log.jsonl"))
    return name, cfg["params"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--repo", default="exea-labs/exealm-nairr")
    ap.add_argument("--stage", required=True, help="a folder to write the release into before uploading")
    ap.add_argument("--tokenizer", default="data/exealm-pretrain/slice-30b-r2/tokenizer")
    ap.add_argument("--card", default="experiments/release/MODEL_CARD.md")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.stage, exist_ok=True)
    done = []
    for run in a.runs:
        if not os.path.exists(os.path.join(run, "model.pt")):
            print(f"{run}: no model.pt, skipped", flush=True)
            continue
        done.append(export(run, a.stage))
        print(f"exported {done[-1][0]} ({done[-1][1] / 1e6:.1f}M parameters)", flush=True)
    if os.path.isdir(a.tokenizer):
        shutil.copytree(a.tokenizer, os.path.join(a.stage, "tokenizer"), dirs_exist_ok=True)
    shutil.copy(a.card, os.path.join(a.stage, "README.md"))
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(folder_path=a.stage, repo_id=a.repo, repo_type="model",
                          commit_message=f"{len(done)} models from the NAIRR study")
        print(f"uploaded {len(done)} models to https://huggingface.co/{a.repo}", flush=True)


if __name__ == "__main__":
    main()
