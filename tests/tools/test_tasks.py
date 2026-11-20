"""Tests for the cross-platform task runner."""

import pytest

from resqcity.common.config import PROJECT_ROOT
from resqcity.tools.tasks import NOT_IMPLEMENTED_EXIT, TASKS, main


def _makefile_targets() -> set[str]:
    """Targets declared .PHONY in the Makefile (parsed, so the two cannot drift apart)."""
    text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith(".PHONY:"))
    return set(line.removeprefix(".PHONY:").split())


MAKEFILE_TARGETS = _makefile_targets()


def test_every_make_target_has_a_task():
    assert set(TASKS) == MAKEFILE_TARGETS


def test_stub_helper_fails_loudly(capsys: pytest.CaptureFixture[str]):
    from resqcity.tools.tasks import _stub

    stub = _stub(99, "future target")
    import argparse

    assert stub(argparse.Namespace()) == NOT_IMPLEMENTED_EXIT
    assert "not implemented yet" in capsys.readouterr().err


def test_unknown_target_rejected():
    with pytest.raises(SystemExit):
        main(["deploy"])


def test_implemented_targets():
    """Phases 7, 8, 9, 10, 11 implemented these targets; they must no longer be stubs."""
    for target in ("sim", "experiments", "figures", "paper", "reproduce", "api", "ui", "demo"):
        assert target in TASKS
        assert TASKS[target].__doc__ is not None
        assert "not implemented" not in (TASKS[target].__doc__ or "")
