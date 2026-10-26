"""Evidence model and posterior of a candidate incident (formulation: docs/models/fusion.md).

Hypothesis H1 for a candidate: at least one real incident occurred inside the candidate's region
R (a set of road-graph nodes) during its window. H0: none did; all its evidence is background
(false camera alarms, false calls). The posterior is recomputed from ALL evidence of the
candidate whenever it changes, so it does not depend on the order in which evidence arrived.
Evidence kinds are conditionally independent given H (an explicit assumption):

- prior odds        e^pi - 1,  pi = incident rate x mass(R) x W  (Poisson)
- camera c alarmed  LR = 1 + q_c p_eff s_real(b) / (W mu_c s_false(b))
- camera c silent   LR = 1 - q_c p_eff             (covers part of R, no alarm in the window)
- calls             LR = L(call times | H1) / L(call times | H0)  (point-process densities:
                    genuine calls = first-call delay + chained gaps; background = Poisson)
- operator review   LR = sens / (1 - spec)  or  (1 - sens) / spec

with q_c = mass(coverage_c n R) / mass(R) x share of camera-detectable types, mu_c the long-run
false-alarm rate (per second), s the confidence-bin shares of real and false alarms, and p_eff the
detection probability AFTER self-masking (p_detect x armed fraction, D-029): a camera silenced by
its own last false alarm cannot alarm on a crash either.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from resqcity.common.schemas import EventType
from resqcity.fusion.config import CallsConfig, DetectorModel, FusionConfig

SECONDS_PER_DAY = 86400.0
DELAY_GRID = 41  # onset positions averaged over when the onset is unknown (call-only candidate)


@dataclass(frozen=True)
class Background:
    """What the control room knows about where and how often incidents and false calls occur.

    Built outside fusion (fusion may not read the simulator; D-028): ``node_share`` is the
    incident location distribution over node indices (sums to 1)."""

    node_share: np.ndarray
    incident_rate_per_day: float
    type_share: dict[EventType, float]
    false_call_rate_per_day: float

    def mass(self, nodes: np.ndarray) -> float:
        return float(self.node_share[nodes].sum()) if len(nodes) else 0.0


@dataclass(frozen=True)
class CameraObservation:
    camera_id: str
    covered_mass: float  # mass(coverage_c n R)
    alarm_bin: int | None  # confidence bin of the camera's first alarm in the window, or None


@dataclass
class Evidence:
    """Everything the posterior needs about one candidate (already associated)."""

    region_mass: float
    now_s: float
    first_evidence_s: float
    onset_s: float | None  # pinned by a camera alarm (alarm time - detector delay), else None
    severity_class: str
    call_times_s: list[float] = field(default_factory=list)
    cameras: list[CameraObservation] = field(default_factory=list)
    reviews: list[bool] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.call_times_s)


def _poisson(k: int, mean: float) -> float:
    if k < 0:
        return 0.0
    return math.exp(-mean + k * math.log(mean) - math.lgamma(k + 1)) if mean > 0 else float(k == 0)


def _poisson_tail(j: int, mean: float) -> float:
    """P(N >= j) for N ~ Poisson(mean)."""
    return max(0.0, 1.0 - sum(_poisson(i, mean) for i in range(j)))


def _lognormal_pdf(x: float, median: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    z = math.log(x / median) / sigma
    return math.exp(-0.5 * z * z) / (x * sigma * math.sqrt(2.0 * math.pi))


def _lognormal_cdf(x: float, median: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    return 0.5 * (1.0 + math.erf(math.log(x / median) / (sigma * math.sqrt(2.0))))


class EvidenceModel:
    def __init__(
        self,
        background: Background,
        detector: DetectorModel,
        calls: CallsConfig,
        cfg: FusionConfig,
    ) -> None:
        self.bg, self.det, self.calls, self.cfg = background, detector, calls, cfg
        self.window_s = cfg.gate.time_window_s.value
        detectable = cfg.camera.detected_types
        total = sum(background.type_share.values())
        self.detectable_share = sum(background.type_share.get(t, 0.0) for t in detectable) / total
        self.mu_per_s = detector.long_run_false_rate_per_h / 3600.0

    # -- terms ----------------------------------------------------------------------------------

    def prior_log_odds(self, region_mass: float) -> float:
        pi = self.bg.incident_rate_per_day / SECONDS_PER_DAY * region_mass * self.window_s
        return math.log(math.expm1(max(pi, 1e-300)))

    def camera_log_lr(self, obs: CameraObservation, region_mass: float) -> float:
        q = min(1.0, obs.covered_mass / region_mass) * self.detectable_share if region_mass else 0.0
        p_eff = self.det.effective_p_detect
        if obs.alarm_bin is None:
            return math.log(max(1.0 - q * p_eff, 1e-300))
        b = obs.alarm_bin
        real = q * p_eff * self.det.real_shares[b] / self.window_s
        return math.log1p(real / (self.mu_per_s * self.det.false_shares[b]))

    def _first_cdf(self, x: float) -> float:
        rep = self.calls.reporting
        return _lognormal_cdf(x, rep.first_delay_median_s.value, rep.first_delay_sigma.value)

    def _first_pdf(self, x: float) -> float:
        rep = self.calls.reporting
        return _lognormal_pdf(x, rep.first_delay_median_s.value, rep.first_delay_sigma.value)

    def _onsets(self, ev: Evidence) -> np.ndarray:
        """Onset times considered: pinned by a camera, else uniform over the window before the
        first evidence."""
        if ev.onset_s is not None:
            return np.array([ev.onset_s])
        return np.linspace(ev.first_evidence_s - self.window_s, ev.first_evidence_s, DELAY_GRID)

    def genuine_densities(self, ev: Evidence) -> list[float]:
        """G_k: density (s^-k) under H1 that exactly the first k calls are genuine and no further
        genuine call has come by now, for k = 0..n (G_0 is a probability).

        Genuine calls (sim.calls): unreported with p_u; else the first after a lognormal delay,
        then N ~ Poisson(m) extra calls, each one exponential gap (mean g) after the previous."""
        rep = self.calls.reporting
        p_u = rep.p_unreported.value[ev.severity_class]
        m = rep.extra_calls_mean.value[ev.severity_class]
        g = rep.extra_gap_mean_s.value
        t = sorted(ev.call_times_s)
        onsets = self._onsets(ev)
        no_call = [p_u + (1.0 - p_u) * (1.0 - self._first_cdf(ev.now_s - s)) for s in onsets]
        dens = [float(np.mean(no_call))]
        if not t:
            return dens
        first = float(np.mean([self._first_pdf(t[0] - s) for s in onsets]))
        chain = (1.0 - p_u) * first
        for k in range(1, len(t) + 1):
            if k > 1:  # k-th genuine call: an extra call exists and arrives after its gap
                q = _poisson_tail(k - 1, m) / max(_poisson_tail(k - 2, m), 1e-300)
                chain *= q * math.exp(-(t[k - 1] - t[k - 2]) / g) / g
            q_next = _poisson_tail(k, m) / max(_poisson_tail(k - 1, m), 1e-300)
            survive = 1.0 - q_next * (1.0 - math.exp(-(ev.now_s - t[k - 1]) / g))
            dens.append(chain * survive)
        return dens

    def calls_log_lr(self, ev: Evidence) -> float:
        """Calls are a point process: under H0 all are background (Poisson, rate b per s in R);
        under H1 the first k are genuine and the rest background, summed over k (the common
        exp(-b T) factor cancels)."""
        b = self.bg.false_call_rate_per_day / SECONDS_PER_DAY * ev.region_mass
        logs = [
            math.log(d) - k * math.log(b) for k, d in enumerate(self.genuine_densities(ev)) if d > 0
        ]
        if not logs:
            return math.log(1e-300)
        top = max(logs)
        return top + math.log(sum(math.exp(x - top) for x in logs))

    def review_log_lr(self, confirmed: bool) -> float:
        r = self.cfg.review
        sens, spec = r.sensitivity.value, r.specificity.value
        return math.log(sens / (1.0 - spec)) if confirmed else math.log((1.0 - sens) / spec)

    # -- posterior ------------------------------------------------------------------------------

    def log_odds_terms(self, ev: Evidence) -> dict[str, float]:
        terms = {"prior": self.prior_log_odds(ev.region_mass)}
        for obs in ev.cameras:
            terms[f"camera:{obs.camera_id}"] = self.camera_log_lr(obs, ev.region_mass)
        terms["calls"] = self.calls_log_lr(ev)
        for i, confirmed in enumerate(ev.reviews):
            terms[f"review:{i}"] = self.review_log_lr(confirmed)
        return terms

    def posterior(self, ev: Evidence) -> tuple[float, dict[str, float]]:
        terms = self.log_odds_terms(ev)
        z = sum(terms.values())
        floor = self.cfg.update.posterior_floor
        p = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
        return min(max(p, floor), 1.0 - floor), terms
