"""Outside memory the RG and AG circuits use: the read-only datastore and the read-write agent board."""
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .board import AgentBoard
from .lexical import LexicalDatastore
from .datastore import Datastore


@dataclass
class MemoryContext:
    """Everything a forward pass needs besides the tokens.

    datastore:   read-only Datastore for RG circuits (None: RG slots output zeros)
    doc_ids:     (B, T) the document id of every position, so retrieval never returns the document being read
    swarm_size:  consecutive batch rows that form one swarm and share one in-batch board (AG circuits)
    board:       a persistent AgentBoard (inference, multi-agent post-training); None during pretraining
    agent_ids:   (B,) which agent each row is, for posting to and reading from the persistent board
    write_board: post this pass's messages to the persistent board
    board_prior: the first row of each swarm is an agent that has already finished reading: its posts are visible to
                 the other rows from their first token, and it hears no one (training for notes written ahead of time)
    """
    datastore: Optional[Datastore] = None
    doc_ids: Optional[torch.Tensor] = None
    swarm_size: int = 1
    board: Optional[AgentBoard] = None
    agent_ids: Optional[torch.Tensor] = None
    write_board: bool = False
    board_prior: bool = False
    encoder: Any = None              # set by the model: its ChunkEncoder, token embedding, the tokens being read,
    emb: Any = None                  # and a per-pass cache (the encoder's keys for the chunks being read)
    tokens: Any = None
    cache: Any = None


__all__ = ["AgentBoard", "Datastore", "LexicalDatastore", "MemoryContext"]
