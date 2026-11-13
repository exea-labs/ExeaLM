# Architecture: TONN

## The shape of the model

```
tokens -> embedding -> [ TONN layer -> gated MLP ] x 14 -> norm -> next-token probabilities
```

The `100m` preset (`exealm/config.py`): width 640, 14 layers, 20 circuits per layer of 32 channels each, MLP width
1,704, a 49,152-token vocabulary with tied input and output embeddings. That comes to 101M parameters, 31M of them in
the embedding table. The Llama-style baseline with the same width, depth and MLP has 100M.

## The hybrid (the default)

In the `100m` and `small` presets every third layer is a TONN layer and the others are full attention layers
(`attn_layers`). TONN's starting circuits use short convolutions 4 times as often and gated units and delays a quarter
as often (`init_weights`). Pure TONN is still available as `100m-tonn` and `small-tonn`.

The 2:1 layout came from equal-compute tests:
- Half attention, half TONN won at equal steps at every size but lost at equal training time at 48M, because TONN
  layers are slower per step.
- Two attention layers per TONN layer kept the quality and won at equal training time at 48M too.

Why: on real tokens at 13M parameters pure TONN trailed a same-size Llama, by 0.18 bits per byte on code. Code needs
exact copying of names, which full attention does well. The hybrid keeps attention for that and lets TONN's
convolutions, state spaces and memories do the rest. It beat Llama on every kind of text with both seeds
(results/README.md, sections 7 and 8). With a random starting mix the hybrid depended on the seed: one seed drew
mostly convolutions and won, the other drew mostly gated units and lost on code. Favouring convolutions fixed that.

## A TONN layer

1. Normalize the residual stream and project it, in **one** matrix, into 20 slots × 3 streams (A, B, C) × 32 channels.
2. Each slot runs its circuit on its streams. Slots of the same type run together as one batched operation.
3. Each slot's output is divided by its running average size, so no slot drowns out the others. The average comes
   from past batches only, so a prediction never depends on other rows in the batch.
4. Project all slots back to the residual stream in **one** matrix and add.

Since 20 × 32 = 640 = the model width, the two projections cost the same as attention's four projections.

## The circuit menu (`exealm/model/circuits.py`)

| Type | What it computes | Think of it as |
|---|---|---|
| D | A, `lag` steps back | looking back a fixed distance |
| I | leaky running sum of A | a fading memory |
| GI | running sum of A, with a forget gate from B | an LSTM-style memory |
| M | A (lagged) × B | "this AND that" |
| MI | running sum of A (lagged) × B | counting co-occurrences |
| MIM | C × running sum of A (lagged) × B | an induction-head-like memory (linear attention) |
| EP | softmax-weighted average of the past | soft pooling |
| G | silu(A) × B | a gated unit, per position |
| C | silu(A + B lagged + C lagged twice) | a short convolution (local n-grams) |
| RP | rotary attention | ordinary attention, one head of 32 channels |
| SM | selective state space | Mamba-like memory |
| OS | oscillating memory | S4D-like, picks up periodic patterns |
| RG | retrieval from the datastore | the model's own RAG ([memory.md](memory.md)) |
| AG | agent board, read and write | talking to other agents ([memory.md](memory.md)) |

Each slot also has a time lag (from 0 to 128) and a decay per channel (how fast it forgets), both set when the circuit
is placed. The decay is then learned.

## The search (`exealm/model/search.py`)

The architecture is not fixed by hand. Every few hundred steps early in training:

1. **Worth.** On held-out "selection" text, switch off all slots of one type in one layer and measure how much the
   loss rises, in bits per token. Split that among the slots. Switching off a whole group at once matters: in a wide
   layer, one slot's effect is mostly noise.
2. **Steer.** New circuits are drawn in proportion to how much each type is worth per slot (80%), plus an even 20% so
   every type keeps getting tried.
3. **Replace.** Slots worth nothing or less are replaced, lowest worth per unit of compute first: half the time by a
   variant of a useful slot, otherwise by a fresh draw. New circuits get a grace period of two rounds.
4. **Floor.** A layer never drops below one RG and one AG (`cfg.pinned`).

The fraction replaced shrinks to zero, and the architecture is then frozen for the rest of training.

Every round's record in the training log (`"event": "search"` in `log.jsonl`) includes the architecture it leaves
behind. `python experiments/plot_search.py <log.jsonl>` draws it, layer by layer, over training (results/README.md,
section 11).

## What we know, what we don't

Known, from small byte-level tests (4.85M parameters, 25 to 82 MB of English web text):
- TONN layers alone **tie** a same-size transformer. With an MLP after every layer, TONN **wins** by 0.05 to 0.08 bits
  per byte (see the README's table). Two seeds of TONN differ by 0.03.
- When we switch parts off: the MLP matters most; then selective state spaces (SM), induction-like memories (MIM) and
  oscillating memory (OS). Delays (D), gated units (G) and attention (RP) matter little at that size.
- The search adds a little (0.005 to 0.01 bits) on top of a good random draw.
- TONN trains about 2.7 times slower per step than the transformer on the same hardware (unoptimized kernels).

Not known yet (each is a great project):
- Does the lead hold at 20M, 50M and 100M parameters, with a real tokenizer instead of bytes?
- Do RG and AG earn their place? Compare with them switched off (`--datastore_chunks 0`, `pinned={}`).
- Which circuit types are dead weight at scale? Could a new type beat them?
- How fast can TONN get with fused kernels for `decay_scan` and `gated_scan`?
