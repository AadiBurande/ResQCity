"""Tests for the cross-platform task runner."""

import pytest

from resqcity.tools.tasks import NOT_IMPLEMENTED_EXIT, TASKS, main

MAKEFILE_TARGETS = {
    "setup",
    "lint",
    "format",
    "test",
    "types",
    "data",
    "vision",
    "streams",
    "api",
    "ui",
    "demo",
    "sim",
    "experiments",
    "figures",
    "reproduce",
}


def test_every_make_target_has_a_task():
    assert set(TASKS) == MAKEFILE_TARGETS


@pytest.mark.parametrize("target", ["reproduce", "experiments", "sim"])
def test_unimplemented_targets_fail_loudly(target: str, capsys: pytest.CaptureFixture[str]):
    assert main([target]) == NOT_IMPLEMENTED_EXIT
    assert "not implemented yet" in capsys.readouterr().err


def test_unknown_target_rejected():
    with pytest.raises(SystemExit):
        main(["deploy"])
