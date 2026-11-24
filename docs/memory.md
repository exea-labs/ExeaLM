# Memory circuits: its own RAG (RG) and the agent board (AG)

Both are circuit types in TONN's menu, so they live inside the network: they read from the residual stream, write back
to it, and are trained by the same loss as everything else. Every layer starts with one of each, and the search can add
more but can never remove the last one.

|  | RG: retrieval | AG: agent board |
|---|---|---|
| Talks to | the **datastore**: chunks of training text | the **board**: messages from other agents |
| Access | read-only | read and write |
| Embeddings | made by the model's own chunk encoder | the model's own keys and values |
| When | one search per 64 tokens per RG slot | one post per 64 tokens per AG slot; reads at every token |
| Code | `exealm/model/retrieval.py`, `exealm/memory/datastore.py` | `exealm/model/board.py`, `exealm/memory/board.py` |

## RG: the model builds its own RAG engine

A normal RAG system has four parts built by different people: an embedding model, an index, a rule for what to search
for, and a prompt that pastes the results in. In ExeaLM all four are learned by the model:

1. **Embeddings.** `ChunkEncoder` turns a 64-token chunk into one 32-number key. Inside the chunk, one attention
   layer lets the tokens see each other, followed by an MLP. It reads the model's own token embeddings and is shared by
   every RG slot, so there is one index.
2. **What to search for.** The query for chunk c is the same encoder applied to chunk c - 1 (the text just read), plus
   the slot's own stream A scaled by a learned weight that starts at 0. Both sides go through the same encoder, so
   "find stored text like what I just read" works from the first step, and training sharpens it.
3. **Which results to trust.** The top-k results (k = 4) are scored against the query, and the softmax of those scores
   weights them. The sharpness starts at 1 (nearly even weights) so a bad early search can't drown out the read.
   Gradient flows through the scores to both the query and the encoder.
4. **What to take.** Every position in the next chunk reads the retrieved tokens with attention. Retrieved tokens go
   through the layer's own input projection, exactly as if they were in the context. Their attention keys look one
   token back, so "find the key, take what follows it" is a single step (an induction step).

How we got here (experiments/rag_lookup.py, results/README.md): the first design read the encoder's states through a
separate projection, made the query from the last position of the previous chunk, and gated the read by stream C. It
never learned to retrieve. Three things were each enough to block it:
- the query and the stored keys lived in unrelated spaces;
- the gate scrambled the retrieved value (a linear probe recovered it 68% of the time before the gate, 2.5% after);
- a zero-initialised matrix adding the encoder's states to retrieved tokens swamped their identity within 100 steps.

Rules that keep it honest:
- **Causal.** The query for chunk c is made from chunk c - 1 only, and positions in chunk c - 1 never see the result.
- **No looking up the answer.** The datastore never returns chunks from the document being read (`exclude=`). This
  matters: without it, the model would learn to copy the next tokens from the datastore. Near-duplicate documents
  leak the same way, which is one more reason deduplication matters.
- **Stale index, fresh gradients.** The stored keys only *find* candidates. The chosen chunks are re-encoded live so the
  encoder trains, and the whole index is re-encoded every `--refresh_every` steps (the approach REALM used).

To see what a checkpoint actually retrieves, `python -m exealm.eval.show_retrieval --ckpt runs/<run>/model.pt --data
<packed dir> --tokenizer tokenizer --prompt "..."` prints, for each chunk of the prompt, every RG slot's picks with
their weights and text. Random-looking picks mean retrieval hasn't learned much yet.

Open problems:
- **Scale.** Brute-force search is fine for a few million chunks. Billions of tokens need FAISS (IVF-PQ), sharded
  across GPUs, and re-encoding them costs real compute. How often is often enough?
- **What goes in the datastore?** All 100B training tokens is more than one index needs. A curated few billion (encyclopedias, textbooks,
  documentation, in Japanese and English) is the starting idea.
- **Key size.** Keys are 32 numbers (one slot's width). Is that enough to find the right chunk among millions?
- **Cost.** With retrieval on, the 0.1B model trains at 0.58 times its speed without it (results/README.md, section 5):
  every RG slot re-encodes and reads 4 retrieved chunks per chunk of text. Caching, sharing retrievals between layers,
  or fewer RG slots might be enough.
- **Context for retrieved tokens.** They are read as plain token embeddings. Giving them context (what they mean in their
  chunk) must not swamp their identity; a gated or slow-growing path is untested.
- **At inference,** the datastore can be swapped for a different one (new facts, private documents) without
  retraining. How well that works is an experiment nobody has run yet.

## AG: agents that talk

Once per 64-token chunk, every AG slot posts one message to its channel of the board:
- **value:** the chunk's average of stream A (what it says)
- **key:** the chunk's average of stream B (what it is about)
- **strength:** sigmoid(w · average of stream C + b), how much it wants the message read. Low strength means staying
  quiet.

At every position it reads the channel with attention (query = stream B at that position) over messages the other agents of its swarm
posted in **earlier** chunks, plus a "nothing" entry so it can ignore the board. The result is gated by stream C.

Where it runs:
- **Training:** the rows of a batch form **swarms** (`swarm_size = 4`), each with its own in-batch board, so posting
  and reading both get gradients. A batch holds many swarms at once, and swarms never hear each other. An agent may
  read its own past posts (`board_read_self`). What each swarm reads, and how swarms are used in post-training, is in
  [swarms.md](swarms.md).
- **Inference:** a persistent `AgentBoard` holds messages from agents that ran before or elsewhere. Posts are keyed by
  (agent, chunk), so re-reading the same text replaces messages instead of duplicating them. An agent never re-reads
  its own old posts from the persistent board, because it gets those from its own context.

Open problems:
- **What should each swarm read in pretraining?** See [swarms.md](swarms.md).
- **Readable messages.** Messages are vectors. A small decoder that turns a message into text would let people see
  what agents tell each other, and would let ExeaLM talk to other models.
- **Infrastructure.** `AgentBoard.save/load` is a file. Agents on different machines need a small board server
  (HTTP or Redis).
- **Safety.** A shared board is an input channel. Agents should treat other agents' messages as information, not as
  instructions. Post-training data should teach that.

### Agent identity

Set `{"board_identity_size": 4}` in a pretraining `--config` JSON file to learn author vectors for IDs 0–3. The
default size is zero. Each TONN layer learns one vector per ID and adds it to the values of that author's posts as
they are read. The vectors start at zero and learn through the board. The chunk visibility and attention masks
still determine which posts a reader can see.

In a batch, each swarm uses its own row indices. To assign explicit identities, pass a `(batch,)` integer
`agent_ids` tensor in `MemoryContext`. A standalone agent's in-batch posts have no author signal. A persistent
board uses each post's stored author ID, including when a single agent reads it. Set matching IDs for training and
inference. IDs outside the table have a zero signal and can still post and be read. Local and HTTP boards keep the
same stored key, value, strength, and author fields.

### Sampling datastore chunks

`Datastore.from_shards(shards, n_chunks, chunk_len, seed=0)` samples uniformly
with replacement over every valid window contained in one document. A document of
length L contributes `max(0, L - chunk_len + 1)` windows, so longer documents are
weighted by their eligible windows rather than chosen with equal probability.
The final window of the final document is included. Document IDs remain the row
indices in the shard offsets, including when earlier documents are too short.

Sampling uses cumulative valid-window counts and terminates without rejection
loops. Positive sizes are required; if no document is long enough it raises a
`ValueError` instead of waiting indefinitely. A fixed seed reproduces samples with
this implementation. The mapping from seeds to sampled windows differs from the
previous rejection sampler; reproducing a checkpoint created with that older
sampler requires the older code and data, since checkpoints retain retrieval keys
but do not include the sampled datastore tokens.
## Sharing an inference board over HTTP

Install `pip install -e '.[board]'`, then run:

```bash
python -m exealm.memory.board_server --host 127.0.0.1 --port 8000 --capacity 4096
```

Each agent uses the same URL and board ID, with its own globally unique nonnegative
agent ID:

```python
import os
from exealm.memory.board_http import RemoteAgentBoard

board = RemoteAgentBoard("http://127.0.0.1:8000", board_id="swarm-1",
                         token=os.environ.get("EXEALM_BOARD_TOKEN"))
# Pass board through MemoryContext(board=board, agent_ids=..., write_board=True).
```

`post`, `read` and `merge` follow the persistent `AgentBoard` interface. Only complete
chunks are posted; reposting an author's chunk replaces it and makes it newest.
Capacity is per channel. Reads preserve slot order, padding, author IDs and validity
masks, and return tensors on the requested device. Each read fetches current server
state. Network errors propagate to the caller. Merge accepts local or remote boards.

Run one server process/worker: it owns in-memory state and serializes each request.
Use a different board ID for each independent swarm, and compatible model weights
and slot layouts within a board. Separate agents may run concurrently, but their
multi-layer forward passes are not a single atomic transaction. Restarting the server
clears its boards. This service is for inference; messages are detached as with the
local persistent board, so gradients do not cross processes.

For other machines, set `EXEALM_BOARD_TOKEN` on the server and clients and bind to a
private interface (`--host ...`). Non-loopback CLI bindings require the token. Use a
trusted network or HTTPS reverse proxy: the built-in HTTP transport is unencrypted.
JSON tensor payloads avoid deserializing executable pickle data. The server validates
message widths and finite float32 values before applying a whole merge request.
## Optional CPU FAISS datastore

Install `pip install -e '.[faiss]'` and use the optional class explicitly:

```python
from exealm.memory.faiss import FaissDatastore

library = FaissDatastore(tokens, doc_ids, nlist=128, pq_m=8, nbits=8, nprobe=16)
library.refresh(model.encoder, model.emb, step=step)
# Use it as MemoryContext(datastore=library, ...).
```

The default `Datastore` remains exact. The optional index uses IVF-PQ and inner
product search on the encoder's keys; it does not normalize keys behind the caller's
back. `pq_m` must divide the key width. Training requires at least
`max(nlist, 2**nbits)` keys, including when a smaller `train_size` is specified.
For tiny tests, reduce `nlist` and `nbits`. Representative corpora generally need
many more training examples than that minimum; the default samples up to 50,000.
Every `refresh()` rebuilds the quantizers and index from the current encoder keys.
If replacing `keys` or `doc_ids` manually, call `rebuild_index()` before searching.
A failed rebuild invalidates the previous index.
The index returns up to `rerank * k` eligible candidates (`rerank=4` by default);
these are re-scored with the original float32 keys and the top k are returned.
This reduces PQ ranking error but cannot recover neighbors outside that candidate set.

Document exclusion uses FAISS's native ID selector **inside** the IVF scan, before
top-k selection. This avoids a fixed over-fetch factor: even 4k candidates can all
belong to the excluded document. Queries with the same excluded document are batched.
If a query has too few valid hits, `nprobe` doubles until it has enough or all lists
have been searched. This guarantees filling up to the number of eligible chunks,
without falling back to the exact PyTorch search. Missing entries have index zero
and a false validity mask; callers must respect the mask, just as with exact search.
Increasing probes cannot remove PQ quantization error, so approximate and exact
neighbors need not match. FAISS's own explanation and selector interface are in
[the index guide](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) and
[per-query parameters](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query).

This backend's index runs on CPU. Queries transfer to CPU and returned IDs/masks
transfer to the query device; live re-encoding of retrieved text still carries the
encoder gradients. The original float32 keys remain available for checkpointing,
so total datastore memory is **not** reduced to the compressed index size. Exclusion
also keeps a CPU grouping of chunk IDs by document. This is an optional search
backend, not a sharded or disk-backed billion-token library, and the pretraining CLI
still uses exact search. Compare it on the intended corpus before selecting it.

On macOS, some pip wheel combinations bundle two incompatible OpenMP runtimes.
Use compatible builds sharing one runtime (for example conda-forge `pytorch` and
`faiss-cpu`) if this occurs; do not bypass the runtime's duplicate-library check.

Run `python -m experiments.faiss_datastore --keys 1000000 --queries 256 --threads 4`
for recall@4 and median search latency against the exact backend, with and without
document exclusion. The script records index construction time and serialized size
separately. Synthetic timings do not establish downstream language-model quality.

## WA: a circuit that changes weights

`exealm/model/alter.py`. A WA slot does not write to the residual stream. Once per chunk it emits a rank-1 change to
a weight matrix, the output matrix of the gated MLP after its layer (or of a later layer, `wa_target`):

    change  =  strength x (u v^T)        u: what to write        v: which inputs it applies to

Both come from the slot's streams, averaged over the chunk, so the model learns what to write from the ordinary
loss. The change takes effect from the next chunk on and lasts for the rest of the pass. `wa_scope` says whose copy
of the matrix changes: `"self"`, `"others"` (the other agents of the swarm) or `"all"`. With `"others"`, an agent
edits its teammates' weights and learns how from their errors. Each agent of a swarm then runs with its own
version of that matrix.

It is opt-in: `get_config(..., pinned={"WA": 1}, menu=MENU + ("WA",))`. The default presets and old checkpoints are
unchanged. `experiments/swarm_relay.py --memory WA` runs the relay task with no board, so facts can only travel as
weight changes.

Where the key comes from matters. With `wa_key="learned"` (the default) v is a learned map of the slot's stream. On
the relay that passes about 52% of the facts: the values arrive, but the model does not learn to tie each to its
question. With `wa_key="hidden"` v is what the edited matrix itself receives at the fact in the writer's copy of the
layer (the MLP's hidden activation, or its normed input with `wa_matrix="in"`). A reader that meets the same token
produces nearly the same activation, so the key matches without being learned, as in ROME (Meng et al. 2022). On
the relay that passes 99.7 to 100% (6 runs, `results/funding/README.md` section 4). It has not been tried on text.

Limits. The changes live for one pass, like the in-batch board. They are fast weights (Schmidhuber 1992; Ba et al.
2016) written by a learned rule. Nothing is written to the stored parameters, so nothing carries over between
training steps or between conversations. Permanent self-editing is a different and riskier thing; it is not built.
