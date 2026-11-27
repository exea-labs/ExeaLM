# The brain: reading a long text window by window

A language model reads a fixed window, 512 tokens here, and forgets everything before it. ExeaLM's TONN layers are
recurrences: running sums, a Mamba-like selective state space (SM), oscillators (OS) and a gated delta-rule matrix
memory (GD), which is a weight matrix rewritten at every token. Read window by window with their state carried from one
window to the next, they remember the whole text at the same cost per token, however long it is. That is the brain.

## How it works

Pass the same dict in `MemoryContext(stream=...)` with every window of the same rows, in order:

```python
state = {}
for window in windows:                       # (B, 512) tokens each, in the order of the text
    logits, loss = model(window, MemoryContext(stream=state), targets)
```

- **TONN layers** carry three things: the last 256 positions of their circuit inputs (the longest lag is 128 and the
  short convolution looks back twice its lag), the final state of every running sum and SM scan, and the oscillator
  and delta-rule states. The two scans and the delta rule take a starting state (`exealm/model/scan.py`,
  `exealm/model/delta.py`), so a TONN model reads a text in windows exactly as in one long pass
  (`tests/test_streaming.py`, every streamable circuit type, windows shorter than the longest lag).
- **Attention layers** keep the keys and values of the window before, as Transformer-XL does: the first tokens of a
  window still see the text just before it. A Llama baseline gets the same cache, so the comparison is fair. Attention
  remembers one window; the TONN state remembers everything.
- **The gradient** flows through the carried state. `python -m exealm.train.pretrain ... --stream 4` trains on rows of
  4 windows read in order (truncated backpropagation through time, as recurrent models are trained).
- Not yet streamed: rotary-attention circuits (RP, never in the starting circuits; they refuse) and the memory circuits
  (board, library, weight-alter), which read only their own window. The Qwen3.5-style baseline's Gated DeltaNet layers
  carry their matrix state from window to window when it is read as a stream (results/expanse/README.md, section 10).

## The test

`experiments/finalize`, stage 7, on one MI300X at 0.1B, 3,000 steps of 16,384 tokens: the memory-free hybrid and the
gated Llama, each trained with `--stream 4` and normally, plus the hybrid with two delta-rule slots per TONN layer.
`python experiments/stream_eval.py <runs> --windows 4` scores every held-out document at least 4 windows long, window by
window: streamed against every window alone. What would show the brain works:

- the streamed hybrid's later windows are clearly better than the same windows read alone, and the gain grows from
  window 1 to window 3 (the TONN state remembers more than one window back; the Llama's cache cannot);
- the gain over the streamed Llama is larger than the gain the two have at window 0;
- trained with `--stream`, the hybrid loses little when every window is read alone (eval_text).

## Next

- Weight-alter edits that persist across windows: the edits a slot writes into its MLP's weights are folded into a
  running low-rank change carried in the stream state, so what the model learns early in a text changes how it reads
  the rest.
- Streaming the board and the library: agents that keep their board across windows, and retrieval keyed on the
  carried state.
