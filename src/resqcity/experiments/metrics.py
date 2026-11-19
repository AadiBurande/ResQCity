"""Per-incident and aggregate metrics for the experiment grid (phase 10).

Every metric is computed from the ``IncidentRow`` dataclass that ``sim.runner`` produces. The
primary outcome is ``time_to_last_required_arrival`` (seconds from onset); expected survival is
secondary and illustrative (see ``configs/survival.yaml``).

Warmup rows (``in_warmup=True``) and false alarms (``is_false_alarm=True``) are excluded from
response-time metrics but included in workload and false-alarm-cost metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from resqcity.common.schemas import Agency, EventType
from resqcity.sim.config import SurvivalConfig

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seconds_between(a: datetime | None, b: datetime | None) -> float | None:
    """Signed seconds from *a* to *b*, or None if either is missing."""
    if a is None or b is None:
        return None
    return (b - a).total_seconds()


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0 = perfect equality, 1 = max inequality)."""
    v = np.sort(values[np.isfinite(values)])
    n = len(v)
    if n < 2 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


# ---------------------------------------------------------------------------
# per-incident row → metric dict
# ---------------------------------------------------------------------------


@dataclass
class IncidentMetrics:
    """Metrics derived from one incident row."""

    incident_id: str
    policy: str
    replication: int
    ward_id: str | None = None
    incident_type: str | None = None
    is_false_alarm: bool = False
    in_warmup: bool = False

    # primary outcome
    time_to_last_required_arrival_s: float | None = None

    # secondary timing
    detection_delay_s: float | None = None
    detection_to_dispatch_s: float | None = None
    response_time_ambulance_s: float | None = None
    response_time_police_s: float | None = None
    response_time_fire_s: float | None = None
    first_to_last_gap_s: float | None = None

    # survival (secondary, illustrative)
    expected_survival: float | None = None

    # flags
    within_8min: bool | None = None
    within_15min: bool | None = None
    shortfall_units: int = 0


def compute_incident_metrics(
    row: dict,
    survival: SurvivalConfig | None = None,
) -> IncidentMetrics:
    """Turn one ``IncidentRow`` dict into computed metrics."""
    m = IncidentMetrics(
        incident_id=row["incident_id"],
        policy=row["policy"],
        replication=row["replication"],
        ward_id=row.get("ward_id"),
        incident_type=row.get("type"),
        is_false_alarm=row.get("is_false_alarm", False),
        in_warmup=row.get("in_warmup", False),
        shortfall_units=row.get("shortfall_units", 0),
    )

    onset = row.get("onset_ts")

    # detection delay: first evidence (camera or call) minus onset
    first_det = row.get("first_detection_ts")
    first_call = row.get("first_call_ts")
    first_evidence = None
    if first_det and first_call:
        first_evidence = min(first_det, first_call)
    elif first_det:
        first_evidence = first_det
    elif first_call:
        first_evidence = first_call
    m.detection_delay_s = _seconds_between(onset, first_evidence)

    # detection to dispatch
    decision = row.get("decision_ts")
    m.detection_to_dispatch_s = _seconds_between(first_evidence, decision)

    # per-agency response time (onset to arrival)
    for agency in Agency:
        arr = row.get(f"arrival_{agency.value}_ts")
        val = _seconds_between(onset, arr)
        setattr(m, f"response_time_{agency.value}_s", val)

    # primary outcome
    last_arr = row.get("last_required_arrival_ts")
    primary = _seconds_between(onset, last_arr)
    m.time_to_last_required_arrival_s = primary

    # first-to-last gap
    arrivals = []
    for agency in Agency:
        arr = row.get(f"arrival_{agency.value}_ts")
        if arr is not None:
            arrivals.append(arr)
    if len(arrivals) >= 2:
        m.first_to_last_gap_s = (max(arrivals) - min(arrivals)).total_seconds()

    # within-time thresholds
    if primary is not None:
        m.within_8min = primary <= 480.0
        m.within_15min = primary <= 900.0

    # expected survival (secondary)
    if survival is not None and primary is not None and m.incident_type:
        try:
            event_type = EventType(m.incident_type)
            m.expected_survival = survival.curve(event_type).at(primary)
        except (ValueError, KeyError):
            pass

    return m


# ---------------------------------------------------------------------------
# aggregate metrics across a (scenario, policy) cell
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    """Summary statistics across replications for one (scenario, policy) cell."""

    scenario: str
    policy: str
    n_replications: int = 0
    n_incidents: int = 0

    # primary: time to last required arrival
    primary_mean_s: float = float("nan")
    primary_median_s: float = float("nan")
    primary_p90_s: float = float("nan")
    primary_std_s: float = float("nan")

    # secondary timing
    detection_delay_mean_s: float = float("nan")
    detection_to_dispatch_mean_s: float = float("nan")
    response_ambulance_mean_s: float = float("nan")
    response_police_mean_s: float = float("nan")
    response_fire_mean_s: float = float("nan")
    first_to_last_gap_mean_s: float = float("nan")

    # thresholds
    pct_within_8min: float = float("nan")
    pct_within_15min: float = float("nan")

    # survival (secondary)
    expected_survival_mean: float = float("nan")

    # utilisation (averaged across replications)
    utilisation_ambulance: float = float("nan")
    utilisation_police: float = float("nan")
    utilisation_fire: float = float("nan")

    # false alarm cost
    false_alarm_incidents: int = 0

    # equity
    ward_gini: float = float("nan")
    ward_max_min_ratio: float = float("nan")

    # per-replication primary means (for paired testing)
    per_rep_primary_means: list[float] = field(default_factory=list)
    per_rep_ginis: list[float] = field(default_factory=list)


def aggregate_metrics(
    df: pd.DataFrame,
    scenario: str,
    policy: str,
    utilisations: list[dict[str, float]] | None = None,
    survival: SurvivalConfig | None = None,
) -> AggregateMetrics:
    """Compute aggregate metrics from the incident-level DataFrame for one (scenario, policy).

    Parameters
    ----------
    df : pd.DataFrame
        Incident-level metrics with columns matching ``IncidentMetrics`` fields.
        Must be pre-filtered to one scenario and one policy.
    scenario, policy : str
        Labels for the cell.
    utilisations : list[dict[str, float]] | None
        Per-replication utilisation dicts from ``Replication.utilisation``.
    survival : SurvivalConfig | None
        If given, compute expected survival as a secondary metric.
    """
    agg = AggregateMetrics(scenario=scenario, policy=policy)

    # exclude warmup and false alarms for response-time metrics
    real = df[(~df["in_warmup"]) & (~df["is_false_alarm"])].copy()
    agg.n_incidents = len(real)
    agg.n_replications = int(df["replication"].nunique())
    agg.false_alarm_incidents = int(df[df["is_false_alarm"]].shape[0])

    if real.empty:
        return agg

    # primary outcome
    primary = real["time_to_last_required_arrival_s"].dropna()
    if len(primary) > 0:
        agg.primary_mean_s = float(primary.mean())
        agg.primary_median_s = float(primary.median())
        agg.primary_p90_s = float(primary.quantile(0.9))
        agg.primary_std_s = float(primary.std())

    # secondary timing
    for col, attr in [
        ("detection_delay_s", "detection_delay_mean_s"),
        ("detection_to_dispatch_s", "detection_to_dispatch_mean_s"),
        ("response_time_ambulance_s", "response_ambulance_mean_s"),
        ("response_time_police_s", "response_police_mean_s"),
        ("response_time_fire_s", "response_fire_mean_s"),
        ("first_to_last_gap_s", "first_to_last_gap_mean_s"),
    ]:
        vals = real[col].dropna()
        if len(vals) > 0:
            setattr(agg, attr, float(vals.mean()))

    # within-time thresholds
    w8 = real["within_8min"].dropna()
    w15 = real["within_15min"].dropna()
    if len(w8) > 0:
        agg.pct_within_8min = float(w8.mean() * 100)
    if len(w15) > 0:
        agg.pct_within_15min = float(w15.mean() * 100)

    # expected survival
    surv = real["expected_survival"].dropna()
    if len(surv) > 0:
        agg.expected_survival_mean = float(surv.mean())

    # utilisation
    if utilisations:
        for agency in Agency:
            vals = [u.get(agency.value, 0.0) for u in utilisations]
            setattr(agg, f"utilisation_{agency.value}", float(np.mean(vals)))

    # per-replication primary means (for paired testing)
    per_rep = []
    per_rep_ginis = []
    for rep_id in sorted(real["replication"].unique()):
        rep_data = real[real["replication"] == rep_id]
        rep_primary = rep_data["time_to_last_required_arrival_s"].dropna()
        if len(rep_primary) > 0:
            per_rep.append(float(rep_primary.mean()))

        # ward-level p90 for Gini
        ward_p90 = (
            rep_data.dropna(subset=["time_to_last_required_arrival_s", "ward_id"])
            .groupby("ward_id")["time_to_last_required_arrival_s"]
            .quantile(0.9)
        )
        if len(ward_p90) >= 2:
            per_rep_ginis.append(_gini(ward_p90.values))

    agg.per_rep_primary_means = per_rep
    agg.per_rep_ginis = per_rep_ginis

    # overall ward equity
    ward_p90 = (
        real.dropna(subset=["time_to_last_required_arrival_s", "ward_id"])
        .groupby("ward_id")["time_to_last_required_arrival_s"]
        .quantile(0.9)
    )
    if len(ward_p90) >= 2:
        agg.ward_gini = _gini(ward_p90.values)
        finite = ward_p90.values[np.isfinite(ward_p90.values)]
        if len(finite) >= 2 and finite.min() > 0:
            agg.ward_max_min_ratio = float(finite.max() / finite.min())

    return agg
