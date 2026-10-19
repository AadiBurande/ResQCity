"""Provenance sidecars for generated files (research-integrity rule 1).

Every result or generated data file gets ``<file>.meta.json`` recording which script, configs
(with SHA-256), seeds and code version produced it, and whether it is synthetic.
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from resqcity.common.config import PROJECT_ROOT

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "networkx", "osmnx", "pydantic")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict[str, Any]:
    """Current commit and whether tracked files have uncommitted changes (None if no git)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def meta_path(output: Path) -> Path:
    return Path(output).with_name(Path(output).name + ".meta.json")


def write_meta(
    output: Path,
    *,
    configs: list[Path],
    synthetic: bool,
    seeds: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<output>.meta.json`` next to ``output`` and return its path."""
    meta = {
        "file": _rel(output),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "synthetic": synthetic,
        "git": git_state(),
        "configs_sha256": {_rel(p): sha256_file(p) for p in configs},
        "seeds": seeds or {},
        "versions": _versions(),
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        **(extra or {}),
    }
    path = meta_path(output)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path
