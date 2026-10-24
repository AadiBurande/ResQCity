"""Fusion engine: associates camera events, calls and reviews into candidate incidents and keeps
their posterior, severity and requirements up to date (phase 4; docs/models/fusion.md).

Association (gate): new evidence joins the candidate with the highest posterior among those
whose anchor lies within ``base_distance_m + k_location_error * sqrt(err_new^2 + err_anchor^2)``
(straight-line distance, because a caller's location error is a straight-line displacement;
err = stated location error of a call, 0 for a camera) and whose FIRST
evidence is at most ``time_window_s`` old (one incident per region and window: a camera's
recurring false alarms must not chain into one endless candidate); otherwise it starts a new
candidate. A candidate's
region is the union of its evidence regions (a camera's coverage, or the gate ball of a call);
its anchor is the most precise location seen (a camera junction beats a call). A camera's
coverage is a road-graph ball; a call's region is the straight-line disc of its gate radius.
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
    # running aggregates (kept incrementally; recomputing over long evidence lists is slow)
    first_evidence: datetime | None = None
    first_alarm: dict[str, Event] = field(default_factory=dict)
    camera_severity: list[float] = field(default_factory=lambda: [0.0, 0.0])  # sum, count
    call_severity: list[float] = field(default_factory=lambda: [0.0, 0.0])
    requirements: dict[str, int] = field(default_factory=lambda: dict.fromkeys(AGENCIES, 0))
    type_votes: Counter = field(default_factory=Counter)

    def _seen(self, ts: datetime) -> None:
        if self.first_evidence is None or ts < self.first_evidence:
            self.first_evidence = ts

    def _need(self, counts: ServiceCounts) -> None:
        for a in AGENCIES:
            self.requirements[a] = max(self.requirements[a], getattr(counts, a))

    def add_event(self, e: Event) -> None:
        self.events.append(e)
        self._seen(e.ts)
        first = self.first_alarm.get(e.camera_id)
        if first is None or e.ts < first.ts:
            self.first_alarm[e.camera_id] = e
        self.camera_severity[0] += e.severity_score
        self.camera_severity[1] += 1
        self._need(e.required_services)

    def add_call(self, call: Call, t: TriageResult) -> None:
        self.calls.append((call, t))
        self._seen(call.ts)
        self.call_severity[0] += t.severity
        self.call_severity[1] += 1
        self._need(t.requirements)
        self.type_votes[t.type] += t.confidence


AGENCIES = tuple(a.value for a in Agency)


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
        self._n = 0
        # indexes (a week has ~10^5 camera alarms): live candidates by anchor node index, the
        # widest anchor error among them, and the cameras that see each node
        self._live: dict[str, Candidate] = {}
        self._by_anchor: dict[int, set[str]] = {}
        self._max_err = 0.0
        self._cams_at: dict[int, list[str]] = {}
        for cid, nodes in coverage.items():
            for v in nodes.tolist():
                self._cams_at.setdefault(v, []).append(cid)
        self._lat = network.nodes["lat"].to_numpy(float)
        self._lon = network.nodes["lon"].to_numpy(float)
        self._ward = network.nodes["ward_id"].astype(str).to_numpy()

    # -- geometry -------------------------------------------------------------------------------

    def _disc(self, node: int, radius_m: float) -> dict[int, float]:
        """{node index: straight-line distance} within ``radius_m`` of ``node``."""
        x, y = self.network.xy[self.network.index_of(node)]
        idx, dist = self.network.nodes_within(x, y, radius_m)
        return dict(zip(idx.tolist(), dist.tolist(), strict=True))

    def _gate_m(self, err_a: float, err_b: float = 0.0) -> float:
        g = self.cfg.gate
        return g.base_distance_m.value + g.k_location_error.value * float(np.hypot(err_a, err_b))

    # -- association ----------------------------------------------------------------------------

    def _match(self, node: int, err_m: float, ts: datetime) -> Candidate | None:
        if not self._live:
            return None
        window = self.cfg.gate.time_window_s.value
        disc = self._disc(node, self._gate_m(err_m, self._max_err))
        hits = []
        for idx in disc.keys() & self._by_anchor.keys():
            for cid in self._by_anchor[idx]:
                c = self._live[cid]
                if abs((ts - c.first_evidence).total_seconds()) > window:
                    continue
                if disc[idx] <= self._gate_m(err_m, c.anchor_error_m):
                    hits.append((-c.posterior, disc[idx], c.id, c))
        return min(hits)[3] if hits else None

    def _index(self, cand: Candidate, add: bool) -> None:
        key = self.network.index_of(cand.anchor)
        if add:
            self._by_anchor.setdefault(key, set()).add(cand.id)
            self._max_err = max(self._max_err, cand.anchor_error_m)
        else:
            self._by_anchor[key].discard(cand.id)
            if not self._by_anchor[key]:
                del self._by_anchor[key]

    def _new(self, node: int, err_m: float, region: np.ndarray, ts: datetime) -> Candidate:
        self._n += 1
        cand = Candidate(f"INC-{self._n:06d}", node, err_m, region, ts, ts)
        self.candidates[cand.id] = self._live[cand.id] = cand
        self._index(cand, add=True)
        return cand

    def _grow(
        self, cand: Candidate, node: int, err_m: float, region: np.ndarray, ts: datetime
    ) -> None:
        cand.region = np.union1d(cand.region, region)
        cand.updated = max(cand.updated, ts)
        if err_m < cand.anchor_error_m:
            self._index(cand, add=False)
            cand.anchor, cand.anchor_error_m = node, err_m
            self._index(cand, add=True)

    # -- evidence -------------------------------------------------------------------------------

    def on_event(self, event: Event) -> list[Incident]:
        cam = self.cameras[event.camera_id]
        region = self.coverage[cam.id]
        cand = self._match(cam.junction_node, 0.0, event.ts) or self._new(
            cam.junction_node, 0.0, region, event.ts
        )
        self._grow(cand, cam.junction_node, 0.0, region, event.ts)
        cand.add_event(event)
        self._event_owner[event.id] = cand.id
        return self._refresh(cand, event.ts, force=True)

    def on_call(self, call: Call, triage: TriageResult) -> list[Incident]:
        loc = call.reported_location
        node = int(self.network.nearest_nodes(loc.lat, loc.lon)[0][0])
        err = float(call.location_error_m)
        region = np.sort(np.fromiter(self._disc(node, self._gate_m(err)), dtype=np.int64))
        cand = self._match(node, err, call.ts) or self._new(node, err, region, call.ts)
        self._grow(cand, node, err, region, call.ts)
        cand.add_call(call, triage)
        return self._refresh(cand, call.ts, force=True)

    def on_review(self, review: Review) -> list[Incident]:
        owner = self._event_owner.get(review.event_id)
        if owner is None:
            return []
        cand = self.candidates[owner]
        cand.reviews.append(review)
        cand.updated = max(cand.updated, review.ts)
        return self._refresh(cand, review.ts, force=True)

    def tick(self, now: datetime, refresh: bool = True) -> list[Incident]:
        """Time passes: stale candidates are dismissed and (``refresh``) posteriors drift as
        no call comes."""
        out = []
        for cand in list(self._live.values()):
            if (now - cand.updated).total_seconds() > self.cfg.update.expiry_s:
                cand.status = IncidentStatus.DISMISSED
                del self._live[cand.id]
                self._index(cand, add=False)
                out += self._refresh(cand, now, force=True)
            elif refresh:
                out += self._refresh(cand, now, force=False)
        return out

    # -- posterior and incident -----------------------------------------------------------------

    def _severity(self, cand: Candidate) -> tuple[float, ServiceCounts, EventType]:
        w = self.cfg.severity
        parts, weights = [], []
        for (total, n), weight in (
            (cand.camera_severity, w.camera_weight.value),
            (cand.call_severity, w.call_weight.value),
        ):
            if n:
                parts.append(total / n)
                weights.append(weight)
        severity = float(np.average(parts, weights=weights))
        etype = cand.type_votes.most_common(1)[0][0] if cand.type_votes else cand.events[0].type
        return severity, ServiceCounts(**cand.requirements), etype

    def evidence(self, cand: Candidate, now: datetime) -> Evidence:
        severity, _, _ = self._severity(cand)
        first_alarm = cand.first_alarm
        onset = (
            min(e.ts for e in first_alarm.values()).timestamp() - self.det.delay_s
            if first_alarm
            else None
        )
        cams = []
        seen = sorted({cid for v in cand.region.tolist() for cid in self._cams_at.get(v, ())})
        for cid in seen:
            inside = np.intersect1d(self.coverage[cid], cand.region, assume_unique=True)
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
        sources = [SourceRef(kind=SourceKind.CAMERA_EVENT, ref_id=e.id) for e in cand.events]
        sources += [SourceRef(kind=SourceKind.CALL, ref_id=c.id) for c, _ in cand.calls]
        sources += [SourceRef(kind=SourceKind.OPERATOR_REVIEW, ref_id=r.id) for r in cand.reviews]
        return Incident(
            id=cand.id,
            type=etype,
            location=GeoPoint(lat=float(self._lat[i]), lon=float(self._lon[i])),
            node=cand.anchor,
            ward_id=str(self._ward[i]) or "unknown",
            created_ts=cand.created,
            updated_ts=max(now, cand.created),
            sources=sources,
            posterior=cand.posterior,
            severity=severity,
            status=cand.status,
            requirements=reqs,
            evidence_log_odds={k: round(v, 4) for k, v in cand.terms.items()},
        )
