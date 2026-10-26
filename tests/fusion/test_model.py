"""Evidence model and posterior (phase 4; docs/models/fusion.md)."""

import itertools
import math

import numpy as np
import pytest

from resqcity.common.schemas import EventType
from resqcity.fusion.config import DetectorModel, load_calls_config, load_fusion_config
from resqcity.fusion.model import Background, CameraObservation, Evidence, EvidenceModel

CALLS = load_calls_config()
FUSION = load_fusion_config()
DET = DetectorModel(
    p_detect=0.6,
    delay_s=3.0,
    false_rate_per_armed_h=55.0,
    cooldown_s=60.0,
    bin_edges=[0.3, 0.4, 0.5, 1.0],
    real_shares=[0.15, 0.17, 0.68],
    false_shares=[1 / 3, 1 / 3, 1 / 3],
    real_confidences=[0.6],
    false_confidences=[0.35],
)
BG = Background(
    node_share=np.full(100, 0.01),
    incident_rate_per_day=5.6,
    type_share={EventType.ACCIDENT: 0.43, EventType.FIRE_SMOKE: 0.54, EventType.OTHER: 0.03},
    false_call_rate_per_day=2.0,
)
MODEL = EvidenceModel(BG, DET, CALLS, FUSION)


def _ev(**kw: object) -> Evidence:
    base = {
        "region_mass": 0.01,
        "now_s": 1000.0,
        "first_evidence_s": 700.0,
        "onset_s": 697.0,
        "severity_class": "serious",
    }
    return Evidence(**{**base, **kw})


def test_prior_is_poisson_odds():
    w = FUSION.gate.time_window_s.value
    pi = 5.6 / 86400 * 0.01 * w
    assert MODEL.prior_log_odds(0.01) == pytest.approx(math.log(math.expm1(pi)))


def test_genuine_call_densities_match_monte_carlo():
    """G_0, G_1, G_2 against a brute-force simulation of the call process exactly as sim.calls
    generates it (pinned onset 0, observed at 600 s; densities from binned call times)."""
    rng = np.random.default_rng(0)
    rep, n_sims, now = CALLS.reporting, 2_000_000, 600.0
    reported = rng.random(n_sims) >= rep.p_unreported.value["serious"]
    first = rep.first_delay_median_s.value * np.exp(
        rep.first_delay_sigma.value * rng.standard_normal(n_sims)
    )
    extras = rng.poisson(rep.extra_calls_mean.value["serious"], n_sims)
    second = first + rng.exponential(rep.extra_gap_mean_s.value, n_sims)
    third = second + rng.exponential(rep.extra_gap_mean_s.value, n_sims)
    n_by_now = reported * (
        (first <= now).astype(int)
        + ((extras > 0) & (second <= now))
        + ((extras > 1) & (third <= now))
    )
    ev0 = _ev(onset_s=0.0, now_s=now, first_evidence_s=0.0)
    assert MODEL.genuine_densities(ev0)[0] == pytest.approx(np.mean(n_by_now == 0), rel=0.01)
    lo, hi = 150.0, 210.0  # first call in this bin, nothing else by now
    sel1 = (n_by_now == 1) & (first >= lo) & (first < hi)
    g1 = MODEL.genuine_densities(_ev(onset_s=0.0, now_s=now, call_times_s=[180.0]))[1]
    assert g1 == pytest.approx(np.mean(sel1) / (hi - lo), rel=0.05)
    lo2, hi2 = 60.0, 120.0  # second call 60-120 s after the first
    sel2 = (
        (n_by_now == 2)
        & (first >= lo)
        & (first < hi)
        & (second - first >= lo2)
        & (second - first < hi2)
    )
    g2 = MODEL.genuine_densities(_ev(onset_s=0.0, now_s=now, call_times_s=[180.0, 270.0]))[2]
    assert g2 == pytest.approx(np.mean(sel2) / ((hi - lo) * (hi2 - lo2)), rel=0.1)


def test_a_typically_timed_call_counts_more_when_a_camera_pinned_the_onset():
    call = 697.0 + CALLS.reporting.first_delay_median_s.value
    kw = {"call_times_s": [call], "now_s": call + 10, "first_evidence_s": 697.0}
    pinned = MODEL.calls_log_lr(_ev(onset_s=697.0, **kw))
    unknown = MODEL.calls_log_lr(_ev(onset_s=None, **{**kw, "first_evidence_s": call}))
    assert pinned > unknown


@pytest.mark.parametrize("onset", [697.0, None])
def test_more_calls_never_lower_the_posterior(onset):
    times = [800.0, 860.0, 900.0, 950.0, 990.0]
    lrs = [MODEL.calls_log_lr(_ev(call_times_s=times[:n], onset_s=onset)) for n in range(6)]
    assert all(b >= a for a, b in itertools.pairwise(lrs))


def test_camera_terms():
    seen = CameraObservation("C1", covered_mass=0.01, alarm_bin=2)
    low = CameraObservation("C1", covered_mass=0.01, alarm_bin=0)
    silent = CameraObservation("C1", covered_mass=0.01, alarm_bin=None)
    assert MODEL.camera_log_lr(seen, 0.01) > MODEL.camera_log_lr(low, 0.01) > 0.0
    assert MODEL.camera_log_lr(silent, 0.01) < 0.0
    # the silent term uses the detection probability AFTER self-masking (D-029), not p_detect
    assert MODEL.camera_log_lr(silent, 0.01) == pytest.approx(
        math.log(1 - 0.43 * DET.effective_p_detect)
    )
    assert MODEL.camera_log_lr(silent, 0.01) != pytest.approx(math.log(1 - 0.43 * DET.p_detect))


def test_effective_detection_is_p_detect_times_the_armed_fraction():
    assert DET.mean_armed_s == pytest.approx(3600.0 / DET.false_rate_per_armed_h)
    armed = DET.mean_armed_s / (DET.mean_armed_s + DET.cooldown_s)
    assert DET.armed_fraction == pytest.approx(armed)
    assert DET.effective_p_detect == pytest.approx(DET.p_detect * armed)
    assert DET.effective_p_detect < DET.p_detect


def test_an_alarm_is_weak_evidence_on_its_own():
    """With the measured inputs a single camera alarm multiplies the odds by at most ~1.1, even
    in the top confidence bin and when the camera covers the whole candidate region (D-029)."""
    for b in range(len(DET.bin_edges) - 1):
        lr = math.exp(MODEL.camera_log_lr(CameraObservation("C1", 0.01, b), 0.01))
        assert 1.0 < lr < 1.2, (b, lr)


def test_reviews_and_posterior():
    base, _ = MODEL.posterior(_ev(cameras=[CameraObservation("C1", 0.01, 2)]))
    confirmed, terms = MODEL.posterior(
        _ev(cameras=[CameraObservation("C1", 0.01, 2)], reviews=[True])
    )
    rejected, _ = MODEL.posterior(_ev(cameras=[CameraObservation("C1", 0.01, 2)], reviews=[False]))
    assert rejected < base < confirmed
    assert set(terms) == {"prior", "camera:C1", "calls", "review:0"}


def test_no_call_yet_lowers_the_posterior_over_time():
    early, _ = MODEL.posterior(_ev(now_s=710.0))
    late, _ = MODEL.posterior(_ev(now_s=1600.0))
    assert late < early


def test_the_emitted_alarm_likelihood_ratios_match_the_engine():
    """The LRs written to results/ (and quoted to phase 6) must be the ones the engine uses."""
    from resqcity.fusion.model import camera_alarm_likelihood_ratios

    emitted = camera_alarm_likelihood_ratios(DET, MODEL.window_s, MODEL.detectable_share)
    for b, lr in enumerate(emitted["alarmed_per_confidence_bin"]):
        engine = math.exp(MODEL.camera_log_lr(CameraObservation("C1", 0.01, b), 0.01))
        assert lr == pytest.approx(engine), b
    silent = math.exp(MODEL.camera_log_lr(CameraObservation("C1", 0.01, None), 0.01))
    assert emitted["silent"] == pytest.approx(silent)
    assert emitted["alarmed_range"] == [min(emitted["alarmed_per_confidence_bin"]),
                                        max(emitted["alarmed_per_confidence_bin"])]  # fmt: skip
