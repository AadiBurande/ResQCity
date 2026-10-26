"""Brute-force check of the ASSEMBLED posterior (phase 4; promised in docs/plans/phase-4.md).

The other tests check single terms. This one simulates the generative process the model assumes
-- incidents Poisson in a region over a window, each reported after a lognormal delay with
chained extra calls, plus background false calls -- and asks whether the posterior the model
returns matches the share of those windows that really contained an incident. That is the
definition of a correct Bayesian posterior under its own assumptions, so it catches sign errors,
wrong normalisations and off-by-one errors in the call chain that term-level tests can miss.

It verifies the algebra, not the world: the assumptions here are the model's own (L19, L23).
"""

import math

import numpy as np
import pytest

from resqcity.common.schemas import EventType
from resqcity.fusion.config import DetectorModel, load_calls_config, load_fusion_config
from resqcity.fusion.model import Background, Evidence, EvidenceModel

CALLS = load_calls_config()
FUSION = load_fusion_config()
WINDOW = FUSION.gate.time_window_s.value
SEVERITY = "serious"
N_WINDOWS = 400_000
REGION_MASS = 0.02  # big enough that a fair share of windows contain an incident
INCIDENT_RATE_PER_DAY = 120.0  # likewise; the test is about the algebra, not Pune's rate
FALSE_CALL_RATE_PER_DAY = 40.0

DET = DetectorModel(
    p_detect=0.6,
    delay_s=3.0,
    false_rate_per_armed_h=55.0,
    cooldown_s=60.0,
    bin_edges=[0.3, 1.0],
    real_shares=[1.0],
    false_shares=[1.0],
    real_confidences=[0.6],
    false_confidences=[0.35],
)
BG = Background(
    node_share=np.full(50, 0.02),
    incident_rate_per_day=INCIDENT_RATE_PER_DAY,
    type_share={EventType.ACCIDENT: 1.0},
    false_call_rate_per_day=FALSE_CALL_RATE_PER_DAY,
)
MODEL = EvidenceModel(BG, DET, CALLS, FUSION)


def _simulate(rng: np.random.Generator, n: int) -> list[tuple[list[float], bool]]:
    """Windows of length WINDOW observed at their end: (call times seen, incident happened)."""
    rep = CALLS.reporting
    pi = INCIDENT_RATE_PER_DAY / 86400.0 * REGION_MASS * WINDOW
    beta = FALSE_CALL_RATE_PER_DAY / 86400.0 * REGION_MASS * WINDOW
    n_incidents = rng.poisson(pi, n)
    n_false = rng.poisson(beta, n)
    p_u = rep.p_unreported.value[SEVERITY]
    out = []
    for k in range(n):
        times = list(rng.uniform(0.0, WINDOW, n_false[k]))
        for _ in range(n_incidents[k]):
            onset = rng.uniform(0.0, WINDOW)
            if rng.random() < p_u:
                continue
            t = onset + rep.first_delay_median_s.value * math.exp(
                rep.first_delay_sigma.value * rng.standard_normal()
            )
            if t > WINDOW:
                continue
            times.append(t)
            for _ in range(rng.poisson(rep.extra_calls_mean.value[SEVERITY])):
                t = t + rng.exponential(rep.extra_gap_mean_s.value)
                if t > WINDOW:
                    break
                times.append(t)
        out.append((sorted(times), bool(n_incidents[k])))
    return out


def _posterior(times: list[float]) -> float:
    first = min(times) if times else WINDOW
    return MODEL.posterior(
        Evidence(
            region_mass=REGION_MASS,
            now_s=WINDOW,
            first_evidence_s=first,
            onset_s=None,
            severity_class=SEVERITY,
            call_times_s=times,
        )
    )[0]


@pytest.fixture(scope="module")
def windows() -> list[tuple[list[float], bool]]:
    return _simulate(np.random.default_rng(20261026), N_WINDOWS)


def test_posterior_matches_the_empirical_rate_by_call_count(windows):
    """The model's mean posterior must equal the share of windows that really had an incident.

    Exact for 0, 2 and 3 calls. The single-call bucket carries a known approximation, pinned by
    the next test: when the onset is not pinned by a camera the model spreads it uniformly over
    the window BEFORE the first call, which for an early call reaches back before the window
    began, so H1 gets onset positions the simulation never allows.
    """
    for n_calls, tol in ((0, 0.005), (2, 0.01), (3, 0.01)):
        sel = [(t, real) for t, real in windows if len(t) == n_calls]
        assert len(sel) > 500, n_calls
        empirical = float(np.mean([real for _, real in sel]))
        model = float(np.mean([_posterior(t) for t, _ in sel]))
        assert model == pytest.approx(empirical, abs=tol), (n_calls, empirical, model)


def test_the_unpinned_onset_prior_is_optimistic_only_for_early_evidence(windows):
    """Quantifies the approximation above (measured 2026-10-26, 400k windows): a lone call in the
    last third of the window is calibrated to ~0.01; one in the first third is optimistic by
    ~0.21. A camera alarm pins the onset and removes the approximation entirely."""
    one = [(t, real) for t, real in windows if len(t) == 1]
    bias = {}
    for name, lo, hi in (("early", 0.0, WINDOW / 3), ("late", 2 * WINDOW / 3, WINDOW)):
        sel = [(t, real) for t, real in one if lo <= t[0] < hi]
        assert len(sel) > 500, name
        empirical = float(np.mean([real for _, real in sel]))
        bias[name] = float(np.mean([_posterior(t) for t, _ in sel])) - empirical
    assert bias["late"] == pytest.approx(0.0, abs=0.03)
    assert 0.1 < bias["early"] < 0.3  # optimistic, and bounded
    assert bias["early"] > bias["late"]


def test_zero_calls_is_evidence_against_an_incident(windows):
    none = [(t, real) for t, real in windows if not t]
    some = [(t, real) for t, real in windows if len(t) == 1]
    assert np.mean([r for _, r in none]) < np.mean([r for _, r in some])
    assert _posterior([]) < _posterior([WINDOW - 60.0])


def test_the_posterior_orders_windows_correctly(windows):
    """Higher posterior must mean a genuinely higher chance of an incident (ranking check)."""
    scored = sorted(((_posterior(t), real) for t, real in windows), key=lambda x: x[0])
    half = len(scored) // 2
    low = float(np.mean([real for _, real in scored[:half]]))
    high = float(np.mean([real for _, real in scored[half:]]))
    assert high > low
