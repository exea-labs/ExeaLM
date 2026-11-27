# Proposal: a library the model reads as learned codes, not text

Status: ideas for approval, 27 November 2026. Nothing here has run. Sizes are arithmetic from the numbers in
`docs/library.md` and the 9B preset (`exealm/config.py`: width 4096, 42 layers, 41 of them attention). The cited
papers need checking before any of this goes into the paper.

## The question

Today the model reads the library as text. A search picks the top passage of each tier, and the model reads 2 x 384
tokens before the question. Could we store each passage as a vector state the model already understands, so the
library ships smaller and costs less to read?

Short answer: yes for reading cost, and yes for shipping if the stored form is discrete codes rather than vectors.
Ordinary embeddings make the library larger, not smaller. The detail is below.

## What the library costs now

| Part | Size | Source |
|---|---|---|
| Passage text, token ids (25.0B tokens, 36.8M human passages, about 680 tokens each) | 50 GB, 21 GB with zstd | docs/library.md |
| BM25 index, GPU (merged, 2.0 ms a query) | 25.0 GB | docs/library.md |
| BM25 index, CPU (Tantivy, extrapolated) | about 17 GB | docs/library.md |
| Reading at 9B: 2 passages x 384 tokens = 768 positions, about 2 x 9.3B FLOP each | about 14 TFLOP a question | arithmetic |

The search index is about as large as the text. Any plan that only shrinks the text halves the download at most.

## Three facts that shape the design

1. **Text is already the smallest form.** A 384-token passage is 768 bytes as token ids and about 320 with zstd.
   One 4096-wide vector in bf16 is 8,192 bytes, ten times the raw passage. Storing what the 9B computes when it
   reads a passage (its attention keys and values) is 258 MB a passage, 9.5 PB for the human tier.
2. **Vectors computed by the 9B are tied to that one set of weights.** Every retrain would mean reading all 25B
   tokens again: about 4.6e20 FLOP, about 320 H100-hours at 9B. A 0.1B encoder does the same in about 3.5.
3. **Compression loses exact detail.** Research and coding answers need exact detail: a function signature, a
   constant, a quotation for a citation. Published context compressors are close to lossless at about 4 times
   compression and lose facts as the ratio grows. One passage in 8 vectors is 48 times.

| Stored form of the human tier (36.8M passages) | Per passage | Total |
|---|---|---|
| Token ids, zstd (measured) | about 570 B | 21 GB |
| Arithmetic coding under the 0.1B model (about 0.96 bits per byte, assuming about 4 bytes a token) | about 330 B | about 12 GB |
| One 4096-wide vector, bf16 | 8 KB | 301 GB |
| 16 vectors of width 4096, bf16 | 128 KB | 4.8 TB |
| The 9B's keys and values for 384 tokens | 258 MB | 9.5 PB |
| **8 learned codes of 32 bytes (idea A)** | **256 B** | **9.4 GB** |
| **4 learned codes of 32 bytes** | **128 B** | **4.7 GB** |
| One 32-byte search code per passage (idea B) | 32 B | 1.2 GB |

Arithmetic coding is the smallest lossless form, but it decodes one token at a time through a model. That costs
generation speed for every passage read, and decoding must be bit-exact across GPUs. It is not worth it.

## Idea A: memory codes (recommended)

A passage becomes a few discrete codes, stored as bytes. The model holds the codebooks and reads a passage by looking
up its codes. This is your byte-compressed embedding: the stored form is bytes, and the model reads it directly.

- **Encoder.** A small frozen ExeaLM (0.1B or 0.5B) reads the passage, then k learned query positions (k = 4 to 16)
  collect what later text needs, as AutoCompressor summary vectors and gist tokens do (Chevalier et al., 2023; Mu et
  al., 2023).
- **Codes.** Each of the k vectors is quantized to 32 bytes: residual or product quantization with learned codebooks
  (Jégou et al., 2011). Quantization runs inside training with a straight-through gradient, so the reader learns to
  read the quantized form, not a clean vector it never sees at test time.
- **Reader.** A small trained projector maps each code to one input position of the big model. The big model reads k
  positions where it now reads 384 tokens. xRAG (Cheng et al., 2024) does this with one position a document.
- **Training.** This is the same truncated backpropagation through time ExeaLM's streams already use. The passage is
  an earlier window, and the loss on the question and answer flows back through the codes into the encoder and the
  projector. The RAG post-training items (section 11 of `results/expanse/README.md`) are the data.
- **Version independence.** The encoder is frozen and small, so the stored codes survive a retrain of the big model.
  Only the projector is retrained.

What it buys at 9B:

- **Shipping.** 4.7 to 9.4 GB for the human tier against 21 GB of text, and the codes double as the search index
  (idea B), which removes the 17 to 25 GB index. With the written tier's text, about 5 to 10 GB against 38 to 46 GB.
- **Reading.** 8 codes a passage means 16 passages cost 128 positions, against 768 for 2 passages now: 8 times the
  sources for a sixth of the reading. Generating a long answer costs more than reading the passages, so the larger
  gain is how many sources one answer can use, not the time saved.

What it risks: codes hold the gist, not exact strings. Hence the second level.

## Idea B: the codes are also the search index

Search in code space, as FAISS's IVF-PQ does: the question goes through the same encoder, and its code is compared
with every passage's first code through lookup tables. One 32-byte code per passage is 1.2 GB for 36.8M passages,
against 25.0 GB for the GPU BM25 index. Dense search can lose to BM25 on rare words and identifiers, which code
questions are full of, so the test measures both. A small BM25 index over identifiers alone stays as a fallback if
it is needed.

## Idea C: two levels, skim then read

The model reads the codes of many passages, then fetches the exact text of the one or two it needs. A special token
asks for the text, as a tool call would. The core edition ships codes and the written tier's text (about 0.4 GB). The
full edition adds the human tier's text (21 GB) for exact reads. This keeps code and citations exact while most of
the reading happens in codes.

## Ideas considered and set aside

- **Storing the 9B's keys and values** (precomputed KV caches, as TurboRAG does): fastest to read, but 258 MB a
  passage. Useful only as a cache of the most-read passages on the hosted service.
- **Storing ExeaLM's carried stream state** after each passage: the model already understands it. In the study's
  layout only one layer of 42 is TONN, and the attention layers carry the window's keys and values, so the state is
  either large or holds little. Worth measuring once the TONN layer earns its place (section 19 of the results).
- **Storing bytes instead of tokens:** about the same size after compression. It makes the stored text independent
  of the tokenizer, which matters if the 9B's tokenizer differs from the study's. Cheap to switch, so keep it as an
  option for the text level.
- **Putting the library into the weights** (memory layers, Berges et al., 2025): no retrieval at all, but no
  citations either. Training teacher text into a 0.1B model did not help (section 22). It stays on the later list.

## Tests, if approved (cheapest first)

| Test | What it answers | Compute |
|---|---|---|
| T0. zstd on token ids against UTF-8 bytes, on a 1 GB sample of the human tier | the text level's real size, tokens or bytes | CPU, minutes |
| T1. Memory codes at 0.1B on library v2's written tier: k = 4, 16, 64, with and without quantization; held-out questions read with nothing, the BM25 passage as text, and the codes | how much of the library's gain survives in codes (section 11: 1.645 bits per byte with nothing, 0.590 with the passage as text) | about 10 to 15 GPU-hours |
| T2. Search in code space against BM25: answer in the top passage, v2 held-out questions (BM25 written tier: 0.703) | whether the codes can replace the index | about 2 GPU-hours |
| T3. Skim then read: codes of 16 passages, then the text of the one the model picks | whether two levels beat two text passages at the same reading cost | about 5 GPU-hours |

Success for T1: codes with k = 16 recover at least 80% of the text passage's gain (0.80 bits per byte or lower on
held-out questions). If k = 64 is needed to get there, the codes cost more than zstd text and only the reading saving
remains. T1 and T2 together are about 15 GPU-hours. About 1,190 of the 2,000 remain, of which the 3B pair and the 0.5B
runs still need about 550.

## Prior work to check before writing this up

AutoCompressors (Chevalier et al., EMNLP 2023), gist tokens (Mu et al., NeurIPS 2023), ICAE (Ge et al., ICLR 2024),
xRAG (Cheng et al., NeurIPS 2024), CEPE (Yen et al., ACL 2024), RETRO (Borgeaud et al., ICML 2022), TurboRAG (2024),
product quantization (Jégou et al., TPAMI 2011), memory layers (Berges et al., ICML 2025), Byte Latent Transformer
(Pagnoni et al., 2024). What would be new: the codes as both the stored library and the search index, read through
ExeaLM's stream training, with a skim-then-read path for exact text.
