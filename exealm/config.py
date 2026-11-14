"""Model configuration.

Presets:
    "100m"        the model we plan to train: about 0.1B parameters (31M of them the embedding table), on about 100B
                  tokens; a hybrid (every third layer TONN, the others gated attention with QK-norm)
    "100m-tonn"   the same size, every layer TONN
    "small"       17M parameters (13M embedding table) with the same vocabulary, hybrid: for real-data experiments on
                  a laptop or a free Colab GPU
    "small-tonn"  the same size, every layer TONN (the base the experiment scripts start from)
    "tiny"        for tests and quick checks on synthetic data Every number here is a starting point, not a decision: changing them is one of the easiest ways to contribute
(see docs/ideas.md), as long as the change wins a fair comparison first (CONTRIBUTING.md).
"""
import json
from dataclasses import asdict, dataclass, field, replace

# Every circuit type a slot can hold (exealm/model/circuits.py). RG and AG are the memory circuits:
# RG reads from the read-only datastore, AG reads and writes the agent board.
MENU = ("D", "I", "GI", "M", "MI", "MIM", "EP", "G", "C", "RP", "SM", "OS", "RG", "AG")
LAGS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128)


@dataclass
class ModelConfig:
    vocab_size: int = 49152          # placeholder until the tokenizer is trained (exealm/data/tokenizer.py)
    d: int = 640                     # width of the residual stream
    n_layers: int = 14
    slots: int = 20                  # circuits per layer
    h: int = 32                      # channels per circuit (slots * h = d, so a layer costs what attention costs)
    ffn: int = -1                    # hidden width of the gated MLP after each layer; -1 = 8d/3, 0 = no MLP
    menu: tuple = MENU               # circuit types the search may use
    lags: tuple = LAGS               # time lags a circuit may use
    tie_embeddings: bool = True
    # Circuits every layer is guaranteed: each layer starts with this many and the search never goes below it.
    pinned: dict = field(default_factory=lambda: {"RG": 1, "AG": 1})
    init_weights: dict = None        # relative chance of each type in the starting circuits (unlisted: 1)
    attn_layers: tuple = ()          # layers that are a full multi-head attention block instead of a TONN layer
    delta_layers: tuple = ()         # of those (or of a Llama's layers), the ones that are Gated DeltaNet instead (Qwen3.5)
    attn_gate: bool = False          # gated attention: a sigmoid gate on each attention output channel (Qwen3-Next, 3.5);
                                     # on in the hybrid presets. False here so configs saved before it existed load as
                                     # they were trained
    qk_norm: bool = False            # RMSNorm on each head's queries and keys (Qwen3); on in the hybrid presets
    seed: int = 0
    # memory circuits
    chunk: int = 64                  # RG retrieves and AG posts once per chunk of this many tokens
    rag_topk: int = 4                # datastore chunks each retrieval returns
    rag_chunk_len: int = 64          # tokens per stored datastore chunk
    enc_dim: int = 256               # width of the model's own chunk encoder (its retrieval embedding model)
    rag_query: str = "last"          # RG query from the previous chunk's "last" position or its "mean"
    rag_null: bool = False           # RG's read may attend to "nothing" (a learned sink, as the board has), so it can
                                     # stay quiet on text where the retrieved chunks don't help
    rag_key_dim: int = 0             # size of retrieval keys (0: one circuit's width, h)
    rag_lexical: int = 0             # extra lexical part of retrieval keys: rare-word codes of this size (0: off)
    swarm_size: int = 4              # agents that share one board (a swarm); a training batch holds many swarms
    board_read_self: bool = True     # may an agent read its own earlier posts?
    board_chunk: int = 0             # AG posts once per this many tokens (0: every `chunk`, like RG)
    chunk_pool: str = "mean"         # how a chunk becomes one board post or one retrieval key: "mean" of its positions,
                                     # or "attn", a learned softmax over them that starts as the mean

    @property
    def ffn_width(self):
        return int(8 * self.d / 3) // 8 * 8 if self.ffn < 0 else self.ffn

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for k in ("menu", "lags", "attn_layers", "delta_layers"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


# The default ExeaLM is a hybrid: every third layer is a TONN layer and the others are full attention layers, and
# TONN's starting circuits favour short convolutions. Pure TONN lost to a same-size Llama on real tokens; the hybrid
# beats it at equal size and steps (13M, 27M, 48M) and, with this 2:1 layout, at equal compute at 48M too, on both
# seeds and every kind of text (results/README.md, sections 7 and 8).
# Its attention layers use two parts from Qwen3.5: gated attention and QK-norm. They improve the hybrid by 0.025 bits
# per byte at 13M (2 seeds) and the full 17M model with both memory skills by 0.026 (1 seed), and with them the hybrid
# leads a Qwen3.5-style model at 13M (results/README.md, sections 10 and 12).
HYBRID = dict(init_weights={"C": 4.0, "G": 0.25, "D": 0.25}, attn_gate=True, qk_norm=True)

PRESETS = {
    "100m": ModelConfig(attn_layers=tuple(i for i in range(14) if i % 3), **HYBRID),        # TONN at 0, 3, 6, 9, 12
    "100m-tonn": ModelConfig(),
    "small": ModelConfig(d=256, n_layers=6, slots=8, h=32, enc_dim=128, attn_layers=(1, 2, 4, 5), **HYBRID),
    "small-tonn": ModelConfig(d=256, n_layers=6, slots=8, h=32, enc_dim=128),
    "tiny": ModelConfig(vocab_size=256, d=64, n_layers=2, slots=8, h=8, chunk=8, rag_topk=2, rag_chunk_len=8,
                        enc_dim=32, swarm_size=2),
}


def get_config(name="100m", **overrides):
    return replace(PRESETS[name], **overrides)


def llama_baseline(cfg):
    """The config of the same-size Llama baseline for `cfg`: a plain Llama, without the Qwen3.5 parts the hybrid
    presets turn on. For a gated Llama, set attn_gate and qk_norm on the result."""
    return replace(cfg, attn_gate=False, qk_norm=False)
