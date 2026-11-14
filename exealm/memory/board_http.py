"""HTTP client for a shared inference board; tensor payloads use JSON, never pickle."""
import json
from urllib.request import Request, urlopen
from urllib.parse import quote

import torch

from .board import AgentBoard


def board_messages(board):
    return [dict(layer=layer, slot=slot, author=author, chunk=chunk,
                 key=key.tolist(), val=val.tolist(), log_w=float(log_w))
            for (layer, slot), channel in board.channels.items()
            for (author, chunk), (key, val, log_w) in channel.items()]


class RemoteAgentBoard:
    """Drop-in post/read/merge interface for AgentBoard, with no stale client read cache.

    Use a distinct board_id per swarm and distinct author IDs per agent. All agents
    must use compatible model weights and channel layouts. Requests have a finite
    timeout; transport errors propagate rather than silently using an empty board.
    """

    def __init__(self, url, board_id='default', token=None, timeout=30):
        self.url = url.rstrip('/') + '/boards/' + quote(board_id, safe='')
        self.token, self.timeout = token, timeout

    def _request(self, operation, data=None):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        payload = None if data is None else json.dumps(data, allow_nan=False).encode()
        req = Request(self.url + '/' + operation, data=payload, headers=headers)
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def post(self, layer, slots, key, val, log_w, authors, n_complete):
        # Reuse the local board's complete-chunk and duplicate-message semantics.
        batch, chunks, groups, width = key.shape
        if val.shape != key.shape or log_w.shape != key.shape[:-1]:
            raise ValueError('key, val and log_w shapes do not match')
        slots = [int(slot) for slot in slots]
        if len(slots) != groups or authors.shape != (batch,) or not 0 <= n_complete <= chunks:
            raise ValueError('slots, authors or n_complete do not match the posts')
        pending = AgentBoard(capacity=batch * n_complete)
        pending.post(layer, slots, key, val, log_w, authors, n_complete)
        self.merge(pending)

    def read(self, layer, slots, device):
        data = self._request('read', dict(layer=int(layer), slots=[int(s) for s in slots]))
        if data is None:
            return None
        return tuple(torch.tensor(t, dtype=dtype, device=device)
                     for t, dtype in zip(data, (torch.float32, torch.float32, torch.float32,
                                               torch.long, torch.bool)))

    def merge(self, other):
        messages = other._request('messages') if isinstance(other, RemoteAgentBoard) else board_messages(other)
        self._request('merge', {'messages': messages})
