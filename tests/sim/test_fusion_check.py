"""Synthetic-week check metrics (phase 4)."""

import pytest

from resqcity.sim.fusion_check import pair_scores


def test_pair_scores():
    groups = {"a": ["I1", "I1", None], "b": ["I1"], "c": [None, None], "d": ["I2", "I3"]}
    s = pair_scores(groups)
    # same-incident pairs: I1 has 3 items -> 3 pairs, 1 kept together (in "a")
    assert s["pair_recall"] == pytest.approx(1 / 3)
    # genuine pairs inside candidates: a: 1 (I1-I1) + 2 (I1-None); d: 1 (I2-I3) -> 1 of 4 correct
    assert s["pair_precision"] == pytest.approx(1 / 4)
    assert s["false_items_in_real_candidates"] == 1
