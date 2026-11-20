"""Async event bus and WebSocket broadcaster for backend live mode (Phase 8).

Bridges internal synchronous and asynchronous components, broadcasting events
to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel

log = logging.getLogger(__name__)


class WebSocketBroadcaster:
    """Manages active WebSocket connections and broadcasts real-time telemetry."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        log.info("WebSocket connected (%d active)", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        log.info("WebSocket disconnected (%d active)", len(self.active_connections))

    async def broadcast(self, event_type: str, data: Any) -> None:
        """Broadcast a structured event to all active clients."""
        async with self._lock:
            connections = list(self.active_connections)

        if not connections:
            return

        if isinstance(data, BaseModel):
            payload = data.model_dump(mode="json")
        elif isinstance(data, dict):
            payload = data
        elif isinstance(data, list):
            payload = [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in data
            ]
        else:
            payload = str(data)

        message = json.dumps({"type": event_type, "data": payload}, default=str)
        dead: list[WebSocket] = []

        for conn in connections:
            try:
                await conn.send_text(message)
            except Exception:
                dead.append(conn)

        if dead:
            async with self._lock:
                for conn in dead:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)


broadcaster = WebSocketBroadcaster()
