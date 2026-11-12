# Glossary

Every term the docs and the blog use, defined once. The same list, with Japanese, is at https://blog.exealabs.org/glossary/.

- **Agent board (AG)**: The circuit agents use to talk. After each chunk an agent posts a message (a value, a key and a strength); later it reads teammates' earlier posts with attention.
- **Architecture search**: During training, TONN measures how much each group of circuits is worth (how much the loss rises when that group is switched off) and redraws the weakest slots.
- **Bits per byte**: How surprised the model is by held-out text, averaged per byte of that text. Lower is better. Counting per byte instead of per token lets models with different tokenizers be compared fairly.
- **Bits per token**: The same measure counted per token. Used when every model in a comparison shares one tokenizer.
- **Chunk**: A fixed run of tokens (64 in the real model). Retrieval searches and the board posts once per chunk.
- **Circuit (slot)**: One small computation inside a TONN layer, such as a short convolution or a running sum. A layer has a fixed number of slots, and each slot holds one circuit type.
- **Convolution-heavy start**: The starting draw of circuits favours short convolutions (4 times the usual chance) and disfavours delays and gated units (a quarter). It fixed the hybrid's bad seed.
- **Equal compute**: Both models get the same training time on the same GPU, and each takes as many steps as it can. A faster model gets more steps. This is the fairest test of an architecture.
- **Equal steps**: Both models train for the same number of steps on the same data, however long each step takes. It compares what each architecture learns per example, not its speed.
- **Hit rate**: How often the search returns a chunk of the right article among its top 4 results.
- **Hybrid (2:1)**: ExeaLM's default layout: every third layer is a TONN layer and the other two are full attention layers. At 0.1B that is TONN at layers 0, 3, 6, 9 and 12.
- **Llama-style transformer**: Our baseline: a standard modern transformer (attention with rotary positions, gated MLP, RMS normalization) with the same width, depth and tokenizer as the model it is compared with.
- **Oracle search**: A diagnostic: the search is forced to look inside the right article. If retrieval helps with the oracle but not with the learned search, the search is the problem.
- **Parameters**: The number of learned weights. ExeaLM's planned size is 0.1 billion (101.1M with a 49k vocabulary, about 90M with the 32k tokenizer).
- **Prefix scan**: A way to compute a running sum over a sequence in parallel, in a logarithmic number of passes instead of one step at a time.
- **Rare-word keys**: A second part of each retrieval key: a fixed random code for every token in the chunk, weighted by how rare the token is. Chunks that share names and terms match from the first step.
- **Retrieval (RG)**: The circuit that searches a read-only library of text chunks, using embeddings the model learns itself, and reads the chunks it finds.
- **Seed**: The number that fixes a run's random choices: starting weights, starting circuits and data order. A result that holds on two seeds is less likely to be luck.
- **Streams A, B and C**: The three inputs every circuit reads, each a slice of the layer's input projection. A circuit may use one, two or all three.
- **Swarm**: Several copies of the model (agents) that run side by side and share one message board. In training, a batch holds many swarms.
- **Tokenizer**: Splits text into tokens, the units the model reads. ExeaLM uses byte-level BPE with 32,768 tokens, trained on English, Japanese and code.
- **TONN**: Exea Labs' architecture (pronounced "tone"). A TONN layer is a set of small circuits whose types are chosen during training by a search, so the network builds part of its own architecture from data.
