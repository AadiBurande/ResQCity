"""Fusion engine: associates camera events, calls and reviews into candidate incidents and keeps
their posterior, severity and requirements up to date (phase 4; docs/models/fusion.md).

Association (gate): new evidence joins the candidate with the highest posterior among those
whose anchor node lies within ``base_distance_m + k_location_error * max(err_new, err_anchor)``
(road-graph distance; err = stated location error of a call, 0 for a camera) and whose latest
evidence is at most ``time_window_s`` old; otherwise it starts a new candidate. A candidate's
region is the union of its evidence regions (a camera's coverage, or the gate ball of a call);
its anchor is the most precise location seen (a camera junction beats a call).
Every change recomputes the posterior from all evidence (order-independent, fusion.model).
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from resqcity.common.requirements import RequirementsConfig, severity_class
from resqcity.common.schemas import (
    Agency,
    Call,
    Camera,
    Event,
    EventType,
    GeoPoint,
    Incident,
    IncidentStatus,
    Review,
    ServiceCounts,
    SourceKind,
    SourceRef,
)
from resqcity.fusion.config import CallsConfig, DetectorModel, FusionConfig
from resqcity.fusion.model import Background, CameraObservation, Evidence, EvidenceModel
from resqcity.fusion.triage import TriageResult
from resqcity.geo.network import RoadNetwork


@dataclass
class Candidate:
    id: str
    anchor: int  # node id
    anchor_error_m: float
    region: np.ndarray  # node indices
    created: datetime
    updated: datetime
    events: list[Event] = field(default_factory=list)
    calls: list[tuple[Call, TriageResult]] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)
    status: IncidentStatus = IncidentStatus.CANDIDATE
    posterior: float = 0.0
    terms: dict[str, float] = field(default_factory=dict)
    emitted: tuple[float, str, tuple[int, ...]] | None = None

    @property
    def first_evidence(self) -> datetime:
        times = [e.ts for e in self.events] + [c.ts for c, _ in self.calls]
        return min(times)


def _max_counts(counts: list[ServiceCounts]) -> ServiceCounts:
    return ServiceCounts(
        **{a.value: max((getattr(c, a.value) for c in counts), default=0) for a in Agency}
    )


class FusionEngine:
    def __init__(
        self,
        network: RoadNetwork,
        cameras: list[Camera],
        detector: DetectorModel,
        calls_cfg: CallsConfig,
        cfg: FusionConfig,
        requirements: RequirementsConfig,
        background: Background,
        coverage: dict[str, np.ndarray],
    ) -> None:
        self.network, self.cfg, self.requirements = network, cfg, requirements
        self.cameras = {c.id: c for c in cameras}
        self.coverage = coverage
        self.model = EvidenceModel(background, detector, calls_cfg, cfg)
        self.det, self.bg = detector, background
        self.candidates: dict[str, Candidate] = {}
        self._event_owner: dict[str, str] = {}
        self._ball_cache: dict[tuple[int, float], dict[int, float]] = {}
        self._n = 0

    # -- geometry -------------------------------------------------------------------------------

    def _ball(self, node: int, radius_m: float) -> dict[int, float]:
        """{node index: road-graph distance} within ``radius_m`` of ``node`` (cached)."""
        key = (node, round(radius_m, 1))
        if key not in self._ball_cache:
            d = self.network.shortest_lengths([node], limit_m=radius_m)[0]
            idx = np.flatnonzero(np.isfinite(d))
            self._ball_cache[key] = dict(zip(idx.tolist(), d[idx].tolist(), strict=True))
        return self._ball_cache[key]

    def _gate_m(self, err_m: float) -> float:
        g = self.cfg.gate
        return g.base_distance_m.value + g.k_location_error.value * err_m

    # -- association ----------------------------------------------------------------------------

    def _match(self, node: int, err_m: float, ts: datetime) -> Candidate | None:
        window = self.cfg.gate.time_window_s.value
        live = [
            c
            for c in self.candidates.values()
            if c.status == IncidentStatus.CANDIDATE and (ts - c.updated).total_seconds() <= window
        ]
        if not live:
            return None
        widest = max(err_m, max(c.anchor_error_m for c in live))
        ball = self._ball(node, self._gate_m(widest))
        hits = []
        for c in live:
            d = ball.get(self.network.index_of(c.anchor))
            if d is not None and d <= self._gate_m(max(err_m, c.anchor_error_m)):
                hits.append((-c.posterior, d, c.id, c))
        return min(hits)[3] if hits else None

    def _new(self, node: int, err_m: float, region: np.ndarray, ts: datetime) -> Candidate:
        self._n += 1
        cand = Candidate(f"INC-{self._n:06d}", node, err_m, region, ts, ts)
        self.candidates[cand.id] = cand
        return cand

    def _grow(
        self, cand: Candidate, node: int, err_m: float, region: np.ndarray, ts: datetime
    ) -> None:
        cand.region = np.union1d(cand.region, region)
        cand.updated = max(cand.updated, ts)
        if err_m < cand.anchor_error_m:
            cand.anchor, cand.anchor_error_m = node, err_m

    # -- evidence -------------------------------------------------------------------------------

    def on_event(self, event: Event) -> list[Incident]:
        cam = self.cameras[event.camera_id]
        region = self.coverage[cam.id]
        cand = self._match(cam.junction_node, 0.0, event.ts) or self._new(
            cam.junction_node, 0.0, region, event.ts
        )
        self._grow(cand, cam.junction_node, 0.0, region, event.ts)
        cand.events.append(event)
        self._event_owner[event.id] = cand.id
        return self._refresh(cand, event.ts, force=True)

    def on_call(self, call: Call, triage: TriageResult) -> list[Incident]:
        loc = call.reported_location
        node = int(self.network.nearest_nodes(loc.lat, loc.lon)[0][0])
        err = float(call.location_error_m)
        region = np.fromiter(self._ball(node, self._gate_m(err)), dtype=np.int64)
        cand = self._match(node, err, call.ts) or self._new(node, err, region, call.ts)
        self._grow(cand, node, err, region, call.ts)
        cand.calls.append((call, triage))
        return self._refresh(cand, call.ts, force=True)

    def on_review(self, review: Review) -> list[Incident]:
        owner = self._event_owner.get(review.event_id)
        if owner is None:
            return []
        cand = self.candidates[owner]
        cand.reviews.append(review)
        cand.updated = max(cand.updated, review.ts)
        return self._refresh(cand, review.ts, force=True)

    def tick(self, now: datetime) -> list[Incident]:
        """Time passes: posteriors drift (no call yet), stale candidates are dismissed."""
        out = []
        for cand in list(self.candidates.values()):
            if cand.status != IncidentStatus.CANDIDATE:
                continue
            if (now - cand.updated).total_seconds() > self.cfg.update.expiry_s:
                cand.status = IncidentStatus.DISMISSED
                out += self._refresh(cand, now, force=True)
            else:
                out += self._refresh(cand, now, force=False)
        return out

    # -- posterior and incident -----------------------------------------------------------------

    def _severity(self, cand: Candidate) -> tuple[float, ServiceCounts, EventType]:
        w = self.cfg.severity
        parts, weights = [], []
        if cand.events:
            parts.append(float(np.mean([e.severity_score for e in cand.events])))
            weights.append(w.camera_weight.value)
        if cand.calls:
            parts.append(float(np.mean([t.severity for _, t in cand.calls])))
            weights.append(w.call_weight.value)
        severity = float(np.average(parts, weights=weights))
        reqs = [e.required_services for e in cand.events] + [t.requirements for _, t in cand.calls]
        if cand.calls:
            votes = Counter()
            for _, t in cand.calls:
                votes[t.type] += t.confidence
            etype = votes.most_common(1)[0][0]
        else:
            etype = cand.events[0].type
        return severity, _max_counts(reqs), etype

    def evidence(self, cand: Candidate, now: datetime) -> Evidence:
        severity, _, _ = self._severity(cand)
        first_alarm: dict[str, Event] = {}
        for e in sorted(cand.events, key=lambda e: e.ts):
            first_alarm.setdefault(e.camera_id, e)
        onset = (
            min(e.ts for e in first_alarm.values()).timestamp() - self.det.delay_s
            if first_alarm
            else None
        )
        cams = []
        for cid, cover in self.coverage.items():
            inside = np.intersect1d(cover, cand.region, assume_unique=True)
            if not len(inside):
                continue
            alarm = first_alarm.get(cid)
            if alarm is None and (now - cand.first_evidence).total_seconds() < self.det.delay_s:
                continue  # too early to count a silent camera
            b = None if alarm is None else self.det.bin_of(alarm.confidence)
            cams.append(CameraObservation(cid, self.bg.mass(inside), b))
        calls = sorted(c.ts.timestamp() for c, _ in cand.calls)
        return Evidence(
            region_mass=max(self.bg.mass(cand.region), 1e-12),
            now_s=now.timestamp(),
            first_evidence_s=cand.first_evidence.timestamp(),
            onset_s=onset,
            severity_class=severity_class(severity, self.requirements.severity),
            call_times_s=calls,
            cameras=cams,
            reviews=[r.confirmed for r in cand.reviews],
        )

    def _refresh(self, cand: Candidate, now: datetime, force: bool) -> list[Incident]:
        cand.posterior, cand.terms = self.model.posterior(self.evidence(cand, now))
        severity, reqs, _ = self._severity(cand)
        key = (
            cand.posterior,
            severity_class(severity, self.requirements.severity),
            tuple(reqs.model_dump().values()),
        )
        last = cand.emitted
        moved = last is None or abs(key[0] - last[0]) >= self.cfg.update.posterior_epsilon
        if not (force or moved or key[1:] != last[1:]):
            return []
        cand.emitted = key
        return [self.incident(cand, now)]

    def incident(self, cand: Candidate, now: datetime) -> Incident:
        severity, reqs, etype = self._severity(cand)
        i = self.network.index_of(cand.anchor)
        nodes = self.network.nodes
        sources = [SourceRef(kind=SourceKind.CAMERA_EVENT, ref_id=e.id) for e in cand.events]
        sources += [SourceRef(kind=SourceKind.CALL, ref_id=c.id) for c, _ in cand.calls]
        sources += [SourceRef(kind=SourceKind.OPERATOR_REVIEW, ref_id=r.id) for r in cand.reviews]
        return Incident(
            id=cand.id,
            type=etype,
            location=GeoPoint(lat=float(nodes["lat"].iat[i]), lon=float(nodes["lon"].iat[i])),
            node=cand.anchor,
            ward_id=str(nodes["ward_id"].iat[i]) or "unknown",
            created_ts=cand.created,
            updated_ts=max(now, cand.created),
            sources=sources,
            posterior=cand.posterior,
            severity=severity,
            status=cand.status,
            requirements=reqs,
            evidence_log_odds={k: round(v, 4) for k, v in cand.terms.items()},
        )
