from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class BroadcastHub:
    """Fan-out a single pipeline's output to N listener WebSockets.

    All listeners on a session receive identical data. TTS audio is
    encoded once (Opus) and the same binary frame is sent to every
    listener. Mute/unmute is handled client-side.
    """

    def __init__(self):
        self.listeners: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self.listeners)

    async def add(self, ws: WebSocket):
        async with self._lock:
            self.listeners.add(ws)
        logger.info("Listener added (total: %d)", len(self.listeners))

    async def remove(self, ws: WebSocket):
        async with self._lock:
            self.listeners.discard(ws)
        logger.info("Listener removed (total: %d)", len(self.listeners))

    async def broadcast_text(self, message: dict):
        """Send a JSON text message to all listeners."""
        payload = json.dumps(message)
        await self._send_to_all(payload, binary=False)

    async def broadcast_binary(self, data: bytes):
        """Send binary data (Opus audio) to all listeners."""
        await self._send_to_all(data, binary=True)

    async def _send_to_all(self, payload: str | bytes, binary: bool):
        async with self._lock:
            if not self.listeners:
                return

            dead: set[WebSocket] = set()
            tasks = [
                self._safe_send(ws, payload, binary, dead)
                for ws in self.listeners
            ]
            await asyncio.gather(*tasks)

            if dead:
                self.listeners -= dead
                logger.info(
                    "Removed %d dead listeners (remaining: %d)",
                    len(dead),
                    len(self.listeners),
                )

    @staticmethod
    async def _safe_send(
        ws: WebSocket,
        payload: str | bytes,
        binary: bool,
        dead: set[WebSocket],
    ):
        try:
            if binary:
                await ws.send_bytes(payload)
            else:
                await ws.send_text(payload)
        except Exception:
            dead.add(ws)
