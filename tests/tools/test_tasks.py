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


# "sim" moved off this list when phase 7 implemented it; the list is the remaining stubs
@pytest.mark.parametrize("target", ["reproduce", "experiments"])
def test_unimplemented_targets_fail_loudly(target: str, capsys: pytest.CaptureFixture[str]):
    assert main([target]) == NOT_IMPLEMENTED_EXIT
    assert "not implemented yet" in capsys.readouterr().err


def test_unknown_target_rejected():
    with pytest.raises(SystemExit):
        main(["deploy"])


def test_the_sim_target_is_implemented():
    """Phase 7 implemented it, so it must no longer be a stub. Guards against the target being
    silently reverted to one while the Makefile still advertises it."""
    from resqcity.tools.tasks import TASKS

    assert "sim" in TASKS
    assert TASKS["sim"].__doc__ is not None
    assert "not implemented" not in (TASKS["sim"].__doc__ or "")
