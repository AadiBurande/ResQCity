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
        "first_call_s": None,
        "n_calls": 0,
        "severity_class": "serious",
    }
    return Evidence(**{**base, **kw})


def test_prior_is_poisson_odds():
    w = FUSION.gate.time_window_s.value
    pi = 5.6 / 86400 * 0.01 * w
    assert MODEL.prior_log_odds(0.01) == pytest.approx(math.log(math.expm1(pi)))


# The model conditions on the observed first-call time; the simulation integrates over it, and
# the test plugs in the median first-call time: exact for n = 0, approximate for n >= 1
# (measured gaps: 0.4 %, 3 %, 15 %).
TOLERANCE = {0: 0.02, 1: 0.06, 2: 0.2}


def test_calls_likelihood_matches_monte_carlo():
    """P(n calls by now | H1) / P(n | H0), against a brute-force simulation of the call process
    exactly as sim.calls generates it (first-call delay, chained extra calls) plus background."""
    rng = np.random.default_rng(0)
    rep, w = CALLS.reporting, FUSION.gate.time_window_s.value
    onset, now, n_sims = 697.0, 1000.0, 400_000
    beta = 2.0 / 86400 * 0.01 * w
    reported = rng.random(n_sims) >= rep.p_unreported.value["serious"]
    first = onset + rep.first_delay_median_s.value * np.exp(
        rep.first_delay_sigma.value * rng.standard_normal(n_sims)
    )
    extras = rng.poisson(rep.extra_calls_mean.value["serious"], n_sims)
    seen = np.zeros(n_sims, int)
    arrival = first.copy()
    seen += reported & (first <= now)
    for k in range(int(extras.max())):
        arrival = arrival + rng.exponential(rep.extra_gap_mean_s.value, n_sims)
        seen += reported & (extras > k) & (arrival <= now)
    background = rng.poisson(beta, n_sims)
    total = seen + background
    for n in (0, 1, 2):
        sel = total == n
        p1 = float(np.mean(sel))
        p0 = beta**n * math.exp(-beta) / math.factorial(n)
        # condition on when the first observed call came: use the simulated median among sel
        first_obs = float(np.median(first[sel & (seen > 0)])) if n else None
        model = math.exp(MODEL.calls_log_lr(_ev(n_calls=n, first_call_s=first_obs, now_s=now)))
        assert model == pytest.approx(p1 / p0, rel=TOLERANCE[n]), n


@pytest.mark.parametrize("onset", [697.0, None])
def test_more_calls_never_lower_the_posterior(onset):
    lrs = [
        MODEL.calls_log_lr(_ev(n_calls=n, first_call_s=800.0 if n else None, onset_s=onset))
        for n in range(6)
    ]
    assert all(b >= a for a, b in itertools.pairwise(lrs))


def test_camera_terms():
    seen = CameraObservation("C1", covered_mass=0.01, alarm_bin=2)
    low = CameraObservation("C1", covered_mass=0.01, alarm_bin=0)
    silent = CameraObservation("C1", covered_mass=0.01, alarm_bin=None)
    assert MODEL.camera_log_lr(seen, 0.01) > MODEL.camera_log_lr(low, 0.01) > 0.0
    assert MODEL.camera_log_lr(silent, 0.01) < 0.0
    assert MODEL.camera_log_lr(silent, 0.01) == pytest.approx(math.log(1 - 0.43 / 1.0 * 0.6))


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
