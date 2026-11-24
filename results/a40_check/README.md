# First runs on NVIDIA GPUs: two rented A40s, 18 November 2026

One RunPod machine with two NVIDIA A40s (46 GB each), PyTorch 2.9.0 with CUDA 13.0, branch `h100-port` at commit
423f795. An A40 is not the H100 the Expanse nodes have: these runs check that the code is right on CUDA, not how fast
an H100 is.

The two-GPU runs here used the branch's first multi-GPU trainer (`exealm/train/multi.py`, where `--batch` was the rows
of each GPU). On 22 November 2026 the branch took the DDP trainer from `main` instead, where `--batch` is the total
over the GPUs, so the two-GPU command below would now be `--batch 64`. Nothing here has been measured again with it.

## `python experiments/gpu_check.py`

Every required check passed; the full output is in [gpu_check.txt](gpu_check.txt).

- A layer of each of the 12 circuit types, an attention block and a Gated DeltaNet block, on the GPU in 32-bit
  against the CPU: the largest difference is 5.2e-07 of the largest value (the gated running sum, GI).
- The same layers in bf16 autocast against 32-bit on the GPU: at most 5.9e-03 (GI again; Gated DeltaNet 5.3e-03).
- Whole models (ExeaLM with swarms of 4, Llama, Qwen3.5-style), loss and gradient size: within 1.0e-05 of the CPU in
  32-bit, and bf16 within 3.2e-04 of 32-bit.
- 150 training steps with the search and a datastore: 5.6048 to 2.683 validation bits per token. Stopped at step 70
  and resumed: identical to the last bit.
- Two processes on two GPUs, 60 steps: in sync at all three evaluations, 6.9634 to 4.6467 bits. Stopped at step 30
  and resumed: identical to the last bit.

The first attempt failed inside the check itself: it reused one input tensor for the CPU and the GPU pass, and on a
GPU the copy of a tensor that already requires a gradient is not a leaf, so its gradient was missing. A CPU cannot
show this, because moving a tensor to the device it is already on returns the same tensor. Fixed in 423f795.

## The real training loop on one GPU and on two

`python -m exealm.train.pretrain --preset 100m --data synthetic --seq 512 --batch 32 --steps 150 --eval_every 50
--eval_batches 2 --search 0`, alone on GPU 0 and under `torchrun --nproc_per_node 2`. The made-up corpus has 256
tokens, so the model has 73.5M parameters instead of 94.3M. bf16, fused AdamW, swarms of 4 with the board on, no
datastore.

| GPUs | Tokens per step | Tokens per second, steps 51 to 100 | Steps 101 to 150 | Peak memory, GPU 0 | Processes in sync |
|---|---|---|---|---|---|
| 1 | 16,384 | 39,252 | 39,322 | 11.8 GB | |
| 2 | 32,768 | 73,058 | 73,060 | 11.8 GB | yes, at all three evaluations |

Two A40s train 1.86 times as many tokens per second as one.

The same command with `--preset 500m --batch 16` on one A40 (456.5M parameters with the 256-token corpus): 10,245
tokens per second over steps 51 to 100 and a peak of 23.1 GB. The machine was stopped before that run's last 50
steps and before its two-GPU partner, the throughput script and its compile and TF32 switches had run.
