"""The agent board: a read-write message store shared by several running copies of the model ("agents").

Every AG circuit is one channel of the board, named by (layer, slot). Once per chunk of text, an AG circuit posts one
message to its channel: a key (what the message is about), a value (what it says) and a write strength (how much the
agent wants it read). AG circuits of other agents read the channel with attention.

During pretraining no AgentBoard object is needed: the rows of a batch are grouped into swarms, each with its own in-batch
board, so posts get gradients (exealm/model/board.py). This class is the persistent board used at inference and in
multi-agent post-training, when agents run at different times or in different processes. Messages are vectors, not
text. Turning them into text a person can read is an open task (docs/memory.md).

Compare: the datastore (datastore.py) is read-only and holds training text; the board is read-write and holds what
agents chose to say.
"""
import torch


class AgentBoard:
    def __init__(self, capacity=4096):
        self.capacity = capacity                          # messages kept per channel (oldest dropped first)
        self.channels = {}                                # (layer, slot) -> {(author, chunk): (key, value, log_w)}
        self.clock = 0
        self._cache = {}                                  # read() results by (layer, slots, device), until the next post

    def post(self, layer, slots, key, val, log_w, authors, n_complete):
        """key, val (B, n, g, h); log_w (B, n, g); authors (B,) agent ids; only the first n_complete chunks are full.
        A message is identified by (author, chunk), so re-reading the same text replaces messages instead of
        duplicating them."""
        key, val, log_w = key.detach().float().cpu(), val.detach().float().cpu(), log_w.detach().float().cpu()
        for j, k in enumerate(slots):
            ch = self.channels.setdefault((layer, int(k)), {})
            for b, a in enumerate(authors.tolist()):
                for c in range(n_complete):
                    ch.pop((a, c), None)                  # re-insert at the end: dict order is posting order
                    ch[(a, c)] = (key[b, c, j], val[b, c, j], log_w[b, c, j])
            while len(ch) > self.capacity:
                ch.pop(next(iter(ch)))
        self.clock += 1
        self._cache.clear()

    def merge(self, other):
        """Add every message of another board (for example notes written by readers that worked separately)."""
        for ch, msgs in other.channels.items():
            self.channels.setdefault(ch, {}).update(msgs)
        self._cache.clear()

    def read(self, layer, slots, device):
        """Messages of the channels of one group of g slots, padded to the longest channel. Returns keys (g, E, h),
        values (g, E, h), log write strengths (g, E), authors (g, E) and a validity mask (g, E); None if empty."""
        ck = (layer, tuple(int(k) for k in slots), str(device))
        if ck in self._cache:                             # the board is read by every forward pass; build it once
            return self._cache[ck]
        chans = [self.channels.get((layer, int(k)), {}) for k in slots]
        E = max((len(c) for c in chans), default=0)
        if E == 0:
            self._cache[ck] = None
            return None
        h = next(iter(next(c for c in chans if c).values()))[0].shape[0]
        g = len(chans)
        K, V, W = torch.zeros(g, E, h), torch.zeros(g, E, h), torch.zeros(g, E)
        A, ok = torch.full((g, E), -1, dtype=torch.long), torch.zeros(g, E, dtype=torch.bool)
        for j, ch in enumerate(chans):
            if not ch:
                continue
            n = len(ch)
            keys, vals = list(ch.keys()), list(ch.values())
            K[j, :n] = torch.stack([v[0] for v in vals])
            V[j, :n] = torch.stack([v[1] for v in vals])
            W[j, :n] = torch.stack([v[2] for v in vals])
            A[j, :n] = torch.tensor([a for a, _ in keys])
            ok[j, :n] = True
        out = tuple(t.to(device) for t in (K, V, W, A, ok))
        self._cache[ck] = out
        return out

    def save(self, path):
        """TODO(infra): a small server (HTTP or Redis) so agents on different machines can share one board."""
        torch.save(dict(capacity=self.capacity, channels=self.channels, clock=self.clock), path)

    @classmethod
    def load(cls, path):
        st = torch.load(path)
        b = cls(st["capacity"])
        b.channels, b.clock = st["channels"], st["clock"]
        b._cache = {}
        return b
