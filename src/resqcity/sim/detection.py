"""Statistical camera-event process: the measured detector as the simulator sees it (phase 4).

No neural network runs here (master plan, design principle 2). Per camera, a timeline mixes:
- false alarms: Poisson at ``false_rate_per_armed_h`` while armed, each followed by the cooldown
  (a share of cameras can trigger persistently more often; phase-10 factor);
- true detections: an incident of a camera-detectable type whose node lies within the coverage
  radius (road-graph distance) is detected with ``p_detect`` after ``delay_s``, unless the camera
  is in a cooldown at that moment (masked, as measured in D-027).
Confidences are drawn from the measured real / false alarm confidences (D-028, exploratory).
Severity cues seen by the camera are the true cues, each flipped with ``cue_flip_probability``.
Random streams: ``camera_false/<camera>`` and ``camera_detect/<incident>/<camera>`` (common random
numbers). The truth link (event -> incident) is returned separately.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from resqcity.common.requirements import RequirementsConfig, required_services, severity_score
from resqcity.common.rng import stream
from resqcity.common.schemas import Camera, Event, EventType, SeverityFeatures
from resqcity.fusion.config import DetectorModel, FusionConfig
from resqcity.geo.cameras import camera_coverage
from resqcity.geo.network import RoadNetwork
from resqcity.sim.groundtruth import GroundTruthIncident

FALSE_STREAM = "camera_false"
DETECT_STREAM = "camera_detect"
PERSISTENT_STREAM = "camera_persistent"


@dataclass(frozen=True)
class SimEvent:
    event: Event
    incident_id: str | None


class DetectionProcess:
    def __init__(
        self,
        network: RoadNetwork,
        cameras: list[Camera],
        detector: DetectorModel,
        cfg: FusionConfig,
        requirements: RequirementsConfig,
    ) -> None:
        self.network, self.cameras, self.det, self.cfg = network, cameras, detector, cfg
        self.requirements = requirements
        self.cover = camera_coverage(network, cameras, cfg.camera.coverage_radius_m.value)
        self._covering: dict[int, list[int]] = {}
        for k, cam in enumerate(cameras):
            for node in self.cover[cam.id]:
                self._covering.setdefault(int(node), []).append(k)

    def covering(self, node_id: int) -> list[int]:
        """Indices of the cameras that see this node."""
        return self._covering.get(self.network.index_of(node_id), [])

    # -- event construction ---------------------------------------------------------------------

    def _features(
        self, rng: np.random.Generator, true: SeverityFeatures | None
    ) -> SeverityFeatures:
        if true is None:
            return SeverityFeatures()
        flip = self.cfg.camera.cue_flip_probability.value
        return true.model_copy(
            update={
                cue: bool(getattr(true, cue)) != bool(rng.random() < flip)
                for cue in ("person_down", "overturned", "fire_visible")
            }
        )

    def _event(
        self,
        cam: Camera,
        t: datetime,
        confidence: float,
        features: SeverityFeatures,
        event_id: str,
    ) -> Event:
        etype = EventType.ACCIDENT
        return Event(
            id=event_id,
            camera_id=cam.id,
            ts=t,
            type=etype,
            confidence=confidence,
            severity_features=features,
            severity_score=severity_score(etype, features, self.requirements.severity),
            required_services=required_services(etype, features, self.requirements),
            confidence_components={"simulated": 1.0},
        )

    # -- generation -----------------------------------------------------------------------------

    def generate(
        self,
        incidents: list[GroundTruthIncident],
        start: datetime,
        hours: float,
        seed_base: int,
        replication: int,
    ) -> list[SimEvent]:
        end = start + timedelta(hours=hours)
        det = self.det
        detectable = set(self.cfg.camera.detected_types)
        # candidate true detections per camera: (time, incident index, incident)
        todo: dict[int, list[tuple[datetime, int, GroundTruthIncident]]] = {}
        for i, inc in enumerate(incidents):
            if inc.type not in detectable:
                continue
            for k in self.covering(inc.node):
                t = inc.onset_ts + timedelta(seconds=det.delay_s)
                todo.setdefault(k, []).append((t, i, inc))
        out: list[SimEvent] = []
        for k in range(len(self.cameras)):
            out += self._camera(k, todo.get(k, []), start, end, seed_base, replication)
        return sorted(out, key=lambda e: (e.event.ts, e.event.id))

    def _camera(
        self,
        k: int,
        todo: list[tuple[datetime, int, GroundTruthIncident]],
        start: datetime,
        end: datetime,
        seed_base: int,
        replication: int,
    ) -> list[SimEvent]:
        """Timeline of camera ``k``: false alarms while armed, true detections, cooldowns."""
        cam, det, cd = self.cameras[k], self.det, timedelta(seconds=self.det.cooldown_s)
        persistent = (
            stream(seed_base, replication, PERSISTENT_STREAM, k).random()
            < self.cfg.camera.persistent_share.value
        )
        mult = self.cfg.camera.persistent_rate_multiplier.value if persistent else 1.0
        mean_gap_s = 3600.0 / (det.false_rate_per_armed_h * mult)
        rng = stream(seed_base, replication, FALSE_STREAM, k)
        out: list[SimEvent] = []

        def gap() -> timedelta:
            return timedelta(seconds=float(rng.exponential(mean_gap_s)))

        def false_alarm(t: datetime) -> None:
            conf = float(rng.choice(det.false_confidences))
            eid = f"EV-F-{cam.id}-{replication:03d}-{len(out):06d}"
            out.append(SimEvent(self._event(cam, t, conf, SeverityFeatures(), eid), None))

        armed_from = start
        next_false = armed_from + gap()
        for t_det, i, inc in sorted(todo, key=lambda x: (x[0], x[1])):
            while next_false < min(t_det, end):
                false_alarm(next_false)
                armed_from = next_false + cd
                next_false = armed_from + gap()
            d_rng = stream(seed_base, replication, DETECT_STREAM, i, k)
            u, conf = d_rng.random(), float(d_rng.choice(det.real_confidences))
            features = self._features(d_rng, inc.severity_features)
            if t_det < armed_from or t_det >= end or u >= det.p_detect:
                continue  # masked by a cooldown, outside the horizon, or missed
            out.append(
                SimEvent(self._event(cam, t_det, conf, features, f"EV-T-{cam.id}-{inc.id}"), inc.id)
            )
            armed_from = t_det + cd
            if next_false < armed_from:
                next_false = armed_from + gap()
        while next_false < end:
            false_alarm(next_false)
            next_false = next_false + cd + gap()
        return out
