"""Per-camera live pipeline: frames -> tracks -> heuristics + classifier -> calibrated confidence ->
alarm -> ``common.schemas.Event`` on the bus. Fire/smoke runs alongside with its own persistence.

Only anonymised frames are ever handed to ``on_preview`` (privacy by design); raw frames stay in
memory inside this module for the duration of one window.
"""

import itertools
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np

from resqcity.common.bus import EventBus
from resqcity.common.requirements import RequirementsConfig
from resqcity.common.schemas import Camera, Event, EventType
from resqcity.vision.alarm import Alarm
from resqcity.vision.config import VisionConfig
from resqcity.vision.heuristics import COMPONENTS, HeuristicResult, HeuristicScorer
from resqcity.vision.privacy import Anonymizer
from resqcity.vision.scoring import Stacker
from resqcity.vision.severity import assess, severity_features
from resqcity.vision.types import Detections, FramePacket, Tracker

log = logging.getLogger(__name__)
TOPIC = "events.vision"
_counter = itertools.count(1)
COLORS = {"vehicle": (255, 178, 0), "person": (0, 158, 115), "alarm": (0, 0, 230)}


@dataclass
class WindowScore:
    t: float
    confidence: float
    components: dict[str, float]


class CameraPipeline:
    def __init__(
        self,
        camera: Camera,
        cfg: VisionConfig,
        tracker: Tracker,
        anonymizer: Anonymizer,
        rules: RequirementsConfig,
        bus: EventBus,
        classifier: object | None = None,  # ClipClassifier (torch); optional
        stacker: Stacker | None = None,
        fire: object | None = None,  # FireSmokeDetector; optional
        threshold: float | None = None,
        start_time: datetime | None = None,
        on_preview: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.camera = camera
        self.cfg = cfg
        self.tracker = tracker
        self.anonymizer = anonymizer
        self.rules = rules
        self.bus = bus
        self.classifier = classifier
        self.stacker = stacker
        self.fire = fire
        self.vehicle_classes = set(cfg.detector.vehicle_classes)
        self.scorer = HeuristicScorer(
            cfg.heuristics, self.vehicle_classes, cfg.detector.person_class
        )
        self.alarm = Alarm(cfg.alarm, threshold)
        self.fire_alarm = Alarm(cfg.alarm, threshold=cfg.fire_smoke.conf)
        self.start_time = start_time or datetime.now(UTC)
        self.on_preview = on_preview
        cc = cfg.clip_classifier
        maxlen = int(np.ceil(cc.window_s * cfg.ingest.sample_fps)) + 1
        self.feats: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self.heur: deque[tuple[float, HeuristicResult]] = deque(maxlen=maxlen)
        self.last_window_t = -np.inf
        self.events: list[Event] = []

    # -- scoring -------------------------------------------------------------------------------

    def _window_score(self, now: float) -> WindowScore:
        recent = [h for t, h in self.heur if t >= now - self.cfg.clip_classifier.window_s]
        heur_max = {k: max((h.components[k] for h in recent), default=0.0) for k in COMPONENTS}
        heur_score = max((h.score for h in recent), default=0.0)
        comps = {"heur_score": heur_score, **{f"c_{k}": v for k, v in heur_max.items()}}
        if self.classifier is not None and len(self.feats) >= 2:
            feats = np.stack([f for _, f in self.feats])
            p = self.classifier.proba_from_features(feats)  # type: ignore[attr-defined]
            comps["clf_logit"] = float(np.log(max(p, 1e-6) / max(1 - p, 1e-6)))
            comps["classifier_p"] = p
        if self.stacker is not None and "clf_logit" in comps:
            confidence = self.stacker.proba_row(comps)
        else:
            confidence = heur_score  # heuristics-only mode (no trained model available)
        return WindowScore(now, confidence, comps)

    # -- main loop step --------------------------------------------------------------------------

    def process(self, frame: FramePacket) -> list[Event]:
        det = self.tracker.update(frame)
        heur = self.scorer.update(det)
        self.heur.append((frame.t, heur))
        if self.classifier is not None:
            feat = self.classifier.encoder.encode([frame.image])[0]  # type: ignore[attr-defined]
            self.feats.append((frame.t, feat))
        emitted: list[Event] = []
        if frame.t - self.last_window_t >= self.cfg.clip_classifier.stride_s:
            self.last_window_t = frame.t
            ws = self._window_score(frame.t)
            if self.alarm.update(frame.t, ws.confidence):
                emitted.append(self._accident_event(frame, det, heur, ws))
        fire_seen = False
        if self.fire is not None:
            obs, persisted = self.fire.update(frame)  # type: ignore[attr-defined]
            fire_seen = obs.fire > 0
            conf = max(obs.fire, obs.smoke)
            if self.fire_alarm.update(frame.t, conf if persisted else 0.0):
                emitted.append(self._fire_event(frame, det, heur, conf, obs.fire, obs.smoke))
        for event in emitted:
            self.bus.publish(TOPIC, event)
            self.events.append(event)
            log.info("%s: %s event, confidence %.2f", self.camera.id, event.type, event.confidence)
        if self.on_preview is not None:
            self.on_preview(self.annotate(frame, det, emitted, fire_seen))
        return emitted

    # -- events -----------------------------------------------------------------------------------

    def _ts(self, t: float) -> datetime:
        return self.start_time + timedelta(seconds=t)

    def _event(
        self,
        event_type: EventType,
        frame: FramePacket,
        det: Detections,
        heur: HeuristicResult,
        confidence: float,
        components: dict[str, float],
        fire_visible: bool,
    ) -> Event:
        features = severity_features(det, heur, self.vehicle_classes, fire_visible)
        score, required = assess(event_type, features, self.rules)
        return Event(
            id=f"E-{self.camera.id}-{next(_counter):06d}",
            camera_id=self.camera.id,
            ts=self._ts(frame.t),
            type=event_type,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            severity_features=features,
            severity_score=score,
            required_services=required,
            clip_ref=None,
            confidence_components={k: round(float(v), 4) for k, v in components.items()},
        )

    def _accident_event(
        self, frame: FramePacket, det: Detections, heur: HeuristicResult, ws: WindowScore
    ) -> Event:
        return self._event(
            EventType.ACCIDENT, frame, det, heur, ws.confidence, ws.components, fire_visible=False
        )

    def _fire_event(
        self,
        frame: FramePacket,
        det: Detections,
        heur: HeuristicResult,
        conf: float,
        fire: float,
        smoke: float,
    ) -> Event:
        # NOTE: fire confidence is the detector's box confidence after persistence, not calibrated.
        return self._event(
            EventType.FIRE_SMOKE,
            frame,
            det,
            heur,
            conf,
            {"fire_box": fire, "smoke_box": smoke},
            fire_visible=fire > 0,
        )

    # -- preview ----------------------------------------------------------------------------------

    def annotate(
        self, frame: FramePacket, det: Detections, events: list[Event], fire_seen: bool
    ) -> np.ndarray:
        """Anonymised frame with boxes and status (the ONLY image that leaves the pipeline)."""
        image = self.anonymizer.apply(frame.image, det.tracks)
        for obs in det.tracks:
            color = (
                COLORS["person"] if obs.cls == self.cfg.detector.person_class else COLORS["vehicle"]
            )
            b = obs.box
            cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
            cv2.putText(
                image,
                f"{obs.cls} {obs.track_id}",
                (int(b.x1), max(12, int(b.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        status = f"{self.camera.id}  t={frame.t:6.1f}s"
        if events:
            status += "  EVENT: " + ", ".join(e.type.value for e in events)
        elif fire_seen:
            status += "  fire/smoke seen"
        cv2.putText(
            image,
            status,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS["alarm"] if events else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image
