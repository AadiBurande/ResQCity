"""Operator audit logging for emergency dispatch actions (Phase 8).

Maintains an immutable, append-only JSONL log of every human dispatcher action,
approval, and override in compliance with civic oversight and DPDP requirements.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from resqcity.common.config import PROJECT_ROOT

log = logging.getLogger(__name__)

AUDIT_DIR = PROJECT_ROOT / "results" / "audit"
AUDIT_LOG_FILE = AUDIT_DIR / "operator_actions.jsonl"


@dataclass
class AuditEntry:
    timestamp: str
    operator_id: str
    decision_id: str
    action: str
    reason: str = ""
    previous_units: dict[str, Any] | None = None
    final_units: dict[str, Any] | None = None
    client_ip: str | None = None


class AuditLogger:
    def __init__(self, log_path: Path = AUDIT_LOG_FILE) -> None:
        self.log_path = log_path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_action(
        self,
        action: str,
        decision_id: str,
        operator_id: str = "operator_1",
        reason: str = "",
        previous_units: dict[str, Any] | None = None,
        final_units: dict[str, Any] | None = None,
        client_ip: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.now(UTC).isoformat(),
            operator_id=operator_id,
            decision_id=decision_id,
            action=action,
            reason=reason,
            previous_units=previous_units,
            final_units=final_units,
            client_ip=client_ip,
        )
        line = json.dumps(asdict(entry)) + "\n"
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            log.error("Failed to write audit entry: %s", e)
        return entry

    def read_entries(self, limit: int = 100) -> list[AuditEntry]:
        if not self.log_path.exists():
            return []
        entries: list[AuditEntry] = []
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entries.append(AuditEntry(**data))
        except Exception as e:
            log.warning("Failed to read audit log: %s", e)
        return entries[-limit:]


audit_logger = AuditLogger()
