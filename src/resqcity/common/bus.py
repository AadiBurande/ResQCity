"""Event bus: publish/subscribe of shared-schema messages between modules.

Phase 3 provides an in-process bus and a JSON-lines file bus (one line per message, so separate
processes such as ``vision.run`` and a later backend can exchange events without a broker).
Phase 8 adds Redis. Topics are plain strings, e.g. ``events.vision``.
"""

import json
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

Handler = Callable[[str, BaseModel], None]


class EventBus(Protocol):
    def publish(self, topic: str, message: BaseModel) -> None: ...


class InProcessBus:
    """Synchronous in-process bus; handlers run in the publisher's thread."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[topic].append(handler)

    def publish(self, topic: str, message: BaseModel) -> None:
        with self._lock:
            handlers = list(self._handlers.get(topic, []))
        for handler in handlers:
            handler(topic, message)


class JsonlBus:
    """Appends ``{"topic", "type", "payload"}`` lines to a file; readers tail the file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def publish(self, topic: str, message: BaseModel) -> None:
        line = json.dumps(
            {
                "topic": topic,
                "type": type(message).__name__,
                "payload": message.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    def read(self, topic: str | None = None) -> Iterator[dict[str, object]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if topic is None or record["topic"] == topic:
                    yield record


class FanOutBus:
    """Publishes every message to several buses (e.g. in-process + JSONL)."""

    def __init__(self, *buses: EventBus) -> None:
        self.buses = buses

    def publish(self, topic: str, message: BaseModel) -> None:
        for bus in self.buses:
            bus.publish(topic, message)
