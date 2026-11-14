"""Run one shared in-memory board service: python -m exealm.memory.board_server."""
import argparse
import os
import secrets
import threading

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .board import AgentBoard
from .board_http import board_messages


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    layer: int = Field(ge=0)
    slot: int = Field(ge=0)
    author: int = Field(ge=0, le=2**63 - 1)
    chunk: int = Field(ge=0)
    key: list[float] = Field(min_length=1)
    val: list[float] = Field(min_length=1)
    log_w: float

    @model_validator(mode='after')
    def matching_width(self):
        if len(self.key) != len(self.val):
            raise ValueError('key and value widths differ')
        return self


class Merge(BaseModel):
    messages: list[Message] = Field(max_length=100_000)


class Read(BaseModel):
    layer: int = Field(ge=0)
    slots: list[int] = Field(max_length=1024)


def create_app(capacity=4096, token=None):
    """One process owns all boards; use one uvicorn worker. Each request is atomic."""
    if capacity < 0:
        raise ValueError('capacity must be nonnegative')
    app = FastAPI()
    boards, widths = {}, {}
    lock = threading.RLock()

    def authorize(authorization: str | None = Header(default=None)):
        if token and not secrets.compare_digest(authorization or '', 'Bearer ' + token):
            raise HTTPException(401, 'Invalid board token')

    def get_board(board_id):
        return boards.setdefault(board_id, AgentBoard(capacity))

    @app.post('/boards/{board_id}/merge', dependencies=[Depends(authorize)])
    def merge(board_id: str, payload: Merge):
        with lock:
            width = widths.get(board_id)
            incoming = AgentBoard(capacity=len(payload.messages))
            # Validate the entire request before mutating the shared board.
            for msg in payload.messages:
                if width is None:
                    width = len(msg.key)
                if len(msg.key) != width:
                    raise HTTPException(422, 'All messages in a board must have the same width')
                key = torch.tensor(msg.key, dtype=torch.float32)
                val = torch.tensor(msg.val, dtype=torch.float32)
                log_w = torch.tensor(msg.log_w, dtype=torch.float32)
                if not all(torch.isfinite(t).all() for t in (key, val, log_w)):
                    raise HTTPException(422, 'Messages must be finite float32 values')
                channel = incoming.channels.setdefault((msg.layer, msg.slot), {})
                ident = (msg.author, msg.chunk)
                channel.pop(ident, None)
                channel[ident] = (key, val, log_w)
            get_board(board_id).merge(incoming)
            if width is not None:
                widths[board_id] = width
        return {'ok': True}

    @app.post('/boards/{board_id}/read', dependencies=[Depends(authorize)])
    def read(board_id: str, payload: Read):
        with lock:
            result = get_board(board_id).read(payload.layer, payload.slots, 'cpu')
            return None if result is None else [t.tolist() for t in result]

    @app.get('/boards/{board_id}/messages', dependencies=[Depends(authorize)])
    def messages(board_id: str):
        with lock:
            return board_messages(get_board(board_id))

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--capacity', type=int, default=4096)
    args = parser.parse_args()
    token = os.environ.get('EXEALM_BOARD_TOKEN')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not token:
        parser.error('set EXEALM_BOARD_TOKEN when listening beyond loopback')
    uvicorn.run(create_app(args.capacity, token), host=args.host, port=args.port, workers=1)


if __name__ == '__main__':
    main()
