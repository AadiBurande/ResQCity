"""Alarm state machine: turns a per-window confidence series into discrete Events.

An alarm fires when the calibrated confidence is >= ``threshold`` for ``consecutive_windows``
windows in a row; afterwards the camera is silent for ``cooldown_s`` (one incident, one Event;
fusion in phase 4 handles later updates).
"""

from dataclasses import dataclass

from resqcity.vision.config import AlarmConfig


@dataclass
class AlarmState:
    run: int = 0
    last_alarm_t: float | None = None


class Alarm:
    def __init__(self, cfg: AlarmConfig, threshold: float | None = None) -> None:
        self.cfg = cfg
        self.threshold = cfg.threshold if threshold is None else threshold
        self.state = AlarmState()

    def reset(self) -> None:
        self.state = AlarmState()

    def update(self, t: float, confidence: float) -> bool:
        """True exactly when an alarm fires at time ``t``."""
        s = self.state
        if s.last_alarm_t is not None and t - s.last_alarm_t < self.cfg.cooldown_s:
            return False
        s.run = s.run + 1 if confidence >= self.threshold else 0
        if s.run >= self.cfg.consecutive_windows:
            s.run = 0
            s.last_alarm_t = t
            return True
        return False


def first_alarm_time(
    times: list[float], confidences: list[float], cfg: AlarmConfig, threshold: float
) -> float | None:
    """Time of the first alarm for a whole series (used by the evaluation)."""
    alarm = Alarm(cfg, threshold)
    for t, c in zip(times, confidences, strict=True):
        if alarm.update(t, c):
            return t
    return None


def alarm_times(
    times: list[float], confidences: list[float], cfg: AlarmConfig, threshold: float
) -> list[float]:
    alarm = Alarm(cfg, threshold)
    return [t for t, c in zip(times, confidences, strict=True) if alarm.update(t, c)]
