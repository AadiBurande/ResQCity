"""Window construction, labels, features and temperature calibration (no torch needed)."""

import itertools

import numpy as np
import pytest

from resqcity.vision.clip_classifier import (
    fit_temperature,
    make_windows,
    sigmoid,
    window_features,
    window_label,
)


def test_make_windows_cover_the_stream():
    times = np.arange(0, 10, 0.2)
    ws = make_windows(times, window_s=3.0, stride_s=1.0)
    assert ws[0].start_s == pytest.approx(0.0) and ws[0].end_s == pytest.approx(3.0)
    assert all(b.end_s - a.end_s == pytest.approx(1.0) for a, b in itertools.pairwise(ws))
    assert all(
        times[w.first] >= w.start_s - 1e-9 and times[w.last - 1] <= w.end_s + 1e-9 for w in ws
    )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [(0, 3, 0.0), (8, 11, 1.0), (12, 15, 1.0), (21, 24, None)],
)
def test_window_label(start, end, expected):
    from resqcity.vision.clip_classifier import Window

    assert window_label(Window(start, end, 0, 1), onset_s=10.0, offset_s=20.0) == expected


def test_window_label_unknown_onset():
    from resqcity.vision.clip_classifier import Window

    assert window_label(Window(0, 3, 0, 1), None, None) is None


def test_window_features_shape_and_motion():
    still = np.ones((15, 4))
    moving = np.vstack([np.zeros(4), np.ones(4)] * 7 + [np.zeros(4)])
    assert window_features(still).shape == (12,)
    assert window_features(still)[8:].sum() == 0  # no frame-to-frame change
    assert window_features(moving)[8:].sum() > 0


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20000)
    labels = (rng.random(20000) < sigmoid(true_logits)).astype(float)
    overconfident = 3.0 * true_logits
    assert fit_temperature(overconfident, labels) == pytest.approx(3.0, rel=0.1)
