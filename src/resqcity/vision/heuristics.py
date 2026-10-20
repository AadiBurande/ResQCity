"""Stage 1 accident cues from tracks (explainable). Speeds are in box-heights per second, which
makes the thresholds roughly independent of camera distance and resolution.

Components (each in [0, 1]) over the last ``window_s`` seconds:
- deceleration: a moving vehicle loses >= ``decel_drop_frac`` of its speed within
  ``decel_window_s``.
- collision: two vehicle boxes overlap (IoU >= ``overlap_iou``) while at least one was moving,
  and both are (nearly) stopped ``stop_after_overlap_s`` later.
- orientation: a vehicle box's width/height ratio changes by >= ``aspect_flip_ratio``
  (overturn proxy).
- person_down: a person box wider than tall (>= ``person_down_aspect``), still for
  ``person_down_still_s``.
Heuristic score = weighted mean of the components (weights from config).
"""

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from resqcity.vision.config import HeuristicsConfig
from resqcity.vision.types import Box, Detections

SPEED_LAG_S = 0.4  # displacement measured over at least this interval (noise vs. responsiveness)
COMPONENTS = ("deceleration", "collision", "orientation", "person_down")


@dataclass
class HeuristicResult:
    t: float
    score: float
    components: dict[str, float]
    involved: set[int] = field(default_factory=set)  # vehicle track ids implicated
    overturned_ids: set[int] = field(default_factory=set)
    person_down_ids: set[int] = field(default_factory=set)


@dataclass
class _Track:
    cls: str
    history: deque[tuple[float, Box]] = field(default_factory=deque)


def _speed_series(history: deque[tuple[float, Box]]) -> list[tuple[float, float]]:
    """(t, speed in box-heights/s) using displacement over >= SPEED_LAG_S."""
    out: list[tuple[float, float]] = []
    items = list(history)
    j = 0
    for i, (t, box) in enumerate(items):
        while j < i and t - items[j + 1][0] >= SPEED_LAG_S:
            j += 1
        t0, box0 = items[j]
        dt = t - t0
        if j == i or dt <= 0:
            continue
        (cx0, cy0), (cx, cy) = box0.center, box.center
        h = max(1.0, (box0.h + box.h) / 2.0)
        out.append((t, float(np.hypot(cx - cx0, cy - cy0) / h / dt)))
    return out


def _speed_at(series: list[tuple[float, float]], t: float) -> float | None:
    """Speed at the last sample at or before t."""
    best = None
    for ts, v in series:
        if ts <= t:
            best = v
        else:
            break
    return best


class HeuristicScorer:
    def __init__(self, cfg: HeuristicsConfig, vehicle_classes: set[str], person_class: str) -> None:
        self.cfg = cfg
        self.vehicle_classes = vehicle_classes
        self.person_class = person_class
        self.tracks: dict[int, _Track] = defaultdict(lambda: _Track(cls=""))

    def reset(self) -> None:
        self.tracks.clear()

    def _prune(self, now: float) -> None:
        horizon = now - self.cfg.window_s - self.cfg.stop_after_overlap_s
        for tid in list(self.tracks):
            hist = self.tracks[tid].history
            while hist and hist[0][0] < horizon:
                hist.popleft()
            if not hist:
                del self.tracks[tid]

    def update(self, det: Detections) -> HeuristicResult:
        for obs in det.tracks:
            track = self.tracks[obs.track_id]
            track.cls = obs.cls
            track.history.append((det.t, obs.box))
        self._prune(det.t)
        return self.score(det.t)

    # -- components ---------------------------------------------------------------------------

    def _vehicles(self) -> dict[int, _Track]:
        c = self.cfg
        return {
            tid: tr
            for tid, tr in self.tracks.items()
            if tr.cls in self.vehicle_classes and len(tr.history) >= c.min_track_frames
        }

    def _deceleration(self, now: float, speeds: dict[int, list]) -> tuple[float, set[int]]:
        c = self.cfg
        best, ids = 0.0, set()
        for tid, series in speeds.items():
            recent = [(t, v) for t, v in series if t >= now - c.window_s]
            for (t0, v0), (t1, v1) in itertools.combinations(recent, 2):
                if v0 < c.moving_speed or t1 - t0 > c.decel_window_s:
                    continue
                drop = (v0 - v1) / v0
                score = float(np.clip(drop / c.decel_drop_frac, 0.0, 1.0))
                if score > best:
                    best = score
                if score >= 1.0:
                    ids.add(tid)
        return best, ids

    def _collision(
        self, now: float, vehicles: dict[int, _Track], speeds: dict[int, list]
    ) -> tuple[float, set[int]]:
        c = self.cfg
        best, ids = 0.0, set()
        for a, b in itertools.combinations(vehicles, 2):
            boxes_b = dict(vehicles[b].history)
            for t, box_a in vehicles[a].history:
                if t < now - c.window_s or t not in boxes_b:
                    continue
                if box_a.iou(boxes_b[t]) < c.overlap_iou:
                    continue
                va, vb = _speed_at(speeds[a], t), _speed_at(speeds[b], t)
                if va is None or vb is None or max(va, vb) < c.moving_speed:
                    continue  # overlapping but both slow: queue / parked, not a crash
                later = t + c.stop_after_overlap_s
                if now < later:
                    best = max(best, 0.5)  # contact seen, outcome not yet known
                    continue
                va2, vb2 = _speed_at(speeds[a], later), _speed_at(speeds[b], later)
                if va2 is not None and vb2 is not None and max(va2, vb2) <= c.stopped_speed:
                    best = 1.0
                    ids |= {a, b}
                else:
                    best = max(best, 0.5)
        return best, ids

    def _orientation(self, now: float, vehicles: dict[int, _Track]) -> tuple[float, set[int]]:
        c = self.cfg
        best, ids = 0.0, set()
        for tid, tr in vehicles.items():
            ratios = [b.w / b.h for t, b in tr.history if t >= now - c.window_s and b.h > 0]
            if len(ratios) < 2:
                continue
            change = max(ratios) / max(1e-6, min(ratios))
            score = float(np.clip((change - 1.0) / (c.aspect_flip_ratio - 1.0), 0.0, 1.0))
            best = max(best, score)
            if score >= 1.0:
                ids.add(tid)
        return best, ids

    def _person_down(self, now: float, speeds: dict[int, list]) -> tuple[float, set[int]]:
        c = self.cfg
        ids = set()
        for tid, tr in self.tracks.items():
            if tr.cls != self.person_class:
                continue
            lying = [(t, b) for t, b in tr.history if b.h > 0 and b.w / b.h >= c.person_down_aspect]
            if not lying or now - lying[0][0] < c.person_down_still_s:
                continue
            still = [v for t, v in speeds.get(tid, []) if t >= now - c.person_down_still_s]
            if still and max(still) <= c.stopped_speed:
                ids.add(tid)
        return (1.0 if ids else 0.0), ids

    def score(self, now: float) -> HeuristicResult:
        vehicles = self._vehicles()
        speeds = {tid: _speed_series(tr.history) for tid, tr in self.tracks.items()}
        decel, decel_ids = self._deceleration(now, {k: speeds[k] for k in vehicles})
        coll, coll_ids = self._collision(now, vehicles, speeds)
        orient, orient_ids = self._orientation(now, vehicles)
        person, person_ids = self._person_down(now, speeds)
        comps = {
            "deceleration": decel,
            "collision": coll,
            "orientation": orient,
            "person_down": person,
        }
        w = self.cfg.weights
        total = w.total()
        score = (
            w.deceleration * decel
            + w.collision * coll
            + w.orientation * orient
            + w.person_down * person
        ) / total
        return HeuristicResult(
            t=now,
            score=float(score),
            components=comps,
            involved=decel_ids | coll_ids | orient_ids,
            overturned_ids=orient_ids,
            person_down_ids=person_ids,
        )
