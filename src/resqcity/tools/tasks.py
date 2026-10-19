"""Cross-platform task runner behind the Makefile.

GNU make is not available on every developer machine (notably Windows), so every Make target
delegates here and can also be run directly:

    uv run python -m resqcity.tools.tasks <target> [--scenario NAME] [--policy NAME]

Targets that belong to a later phase exit with status 3 and say so, rather than pretending to
succeed. A silent no-op ``reproduce`` would be a research-integrity hazard.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence

from resqcity.common.config import PROJECT_ROOT

NOT_IMPLEMENTED_EXIT = 3


def _run(*commands: Sequence[str]) -> int:
    env = {**os.environ, "PYTHONUTF8": "1"}  # Marathi/Hindi names on Windows consoles
    for cmd in commands:
        print("$", " ".join(cmd), flush=True)
        code = subprocess.call(list(cmd), cwd=PROJECT_ROOT, env=env)
        if code != 0:
            return code
    return 0


def _py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def setup(_: argparse.Namespace) -> int:
    code = _run(_py("resqcity.tools.gen_ts_types"))
    if code == 0 and (PROJECT_ROOT / ".git").exists() and shutil.which("git"):
        code = _run(_py("pre_commit", "install"))
    return code


def lint(_: argparse.Namespace) -> int:
    return _run(
        _py("ruff", "check", "."),
        _py("ruff", "format", "--check", "."),
        _py("resqcity.tools.gen_ts_types", "--check"),
    )


def fmt(_: argparse.Namespace) -> int:
    return _run(_py("ruff", "check", "--fix", "."), _py("ruff", "format", "."))


def test(_: argparse.Namespace) -> int:
    return _run(_py("pytest"))


def types(_: argparse.Namespace) -> int:
    return _run(_py("resqcity.tools.gen_ts_types"))


def data(_: argparse.Namespace) -> int:
    """Download raw inputs (network access), then build processed data offline."""
    return _run(_py("resqcity.geo.download"), _py("resqcity.geo.build"), _py("resqcity.geo.sanity"))


def _stub(phase: int, what: str) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        print(f"{what}: not implemented yet (phase {phase}).", file=sys.stderr)
        return NOT_IMPLEMENTED_EXIT

    return run


TASKS: dict[str, Callable[[argparse.Namespace], int]] = {
    "setup": setup,
    "lint": lint,
    "format": fmt,
    "test": test,
    "types": types,
    "data": data,
    "streams": _stub(3, "start simulated RTSP cameras"),
    "api": _stub(8, "start backend API"),
    "ui": _stub(9, "start dashboard"),
    "demo": _stub(9, "end-to-end live demo"),
    "sim": _stub(7, "run simulator"),
    "experiments": _stub(10, "run experiments"),
    "figures": _stub(10, "generate figures"),
    "reproduce": _stub(11, "regenerate every result and figure"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResQCity task runner")
    parser.add_argument("target", choices=sorted(TASKS))
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--policy", default="B0")
    args = parser.parse_args(argv)
    return TASKS[args.target](args)


if __name__ == "__main__":
    raise SystemExit(main())
