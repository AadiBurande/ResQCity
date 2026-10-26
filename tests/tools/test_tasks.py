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


@pytest.mark.parametrize("target", ["reproduce", "experiments", "sim"])
def test_unimplemented_targets_fail_loudly(target: str, capsys: pytest.CaptureFixture[str]):
    assert main([target]) == NOT_IMPLEMENTED_EXIT
    assert "not implemented yet" in capsys.readouterr().err


def test_unknown_target_rejected():
    with pytest.raises(SystemExit):
        main(["deploy"])
