"""Named random streams (common random numbers)."""

import zlib

import numpy as np
import pytest

from common.rng import stream, stream_key


def test_same_inputs_same_stream():
    assert np.array_equal(stream(1, 0, "calls").random(5), stream(1, 0, "calls").random(5))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((1, 0, "calls"), (1, 0, "incidents")),
        ((1, 0, "calls"), (1, 1, "calls")),
        ((1, 0, "calls"), (2, 0, "calls")),
    ],
)
def test_different_inputs_different_streams(a, b):
    assert not np.array_equal(stream(*a).random(5), stream(*b).random(5))


def test_stream_key_is_stable_across_processes():
    assert stream_key("incidents") == zlib.crc32(b"incidents")


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        stream(-1, 0, "x")
