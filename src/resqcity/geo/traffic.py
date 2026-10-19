"""Time-of-day, rain and vehicle-profile speed multipliers (all from ``configs/geo.yaml``)."""

import numpy as np

from resqcity.geo.config import GeoConfig

HOURS_PER_DAY = 24


class TrafficModel:
    """Speed multipliers indexed by road-class code (position in ``road_classes.priority``)."""

    def __init__(self, cfg: GeoConfig) -> None:
        self.classes: tuple[str, ...] = tuple(cfg.road_classes.priority)
        n = len(self.classes)
        # by_hour[h, c] = speed multiplier for class c at local hour h
        self.by_hour = np.ones((HOURS_PER_DAY, n))
        for period in cfg.traffic_periods:
            row = np.array([period.multipliers[c].value for c in self.classes])
            for hour in period.hours():
                self.by_hour[hour] = row
        self.base_speed_kph = np.array([cfg.base_speeds_kph[c].value for c in self.classes])
        self._rain_bands = [(b.min_mm_h, b.multiplier.value) for b in cfg.rain_speed_bands]
        self._profiles = {name: sv.value for name, sv in cfg.vehicle_profiles.items()}

    def class_code(self, road_class: str) -> int:
        return self.classes.index(road_class)

    def hour_multipliers(self, hour: int) -> np.ndarray:
        return self.by_hour[hour % HOURS_PER_DAY]

    def rain_multiplier(self, rain_mm_h: float) -> float:
        multiplier = 1.0
        for min_mm_h, value in self._rain_bands:
            if rain_mm_h >= min_mm_h:
                multiplier = value
        return multiplier

    def profile(self, name: str) -> float:
        try:
            return self._profiles[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._profiles))
            raise KeyError(f"unknown vehicle profile {name!r} (known: {known})") from exc

    @property
    def profiles(self) -> tuple[str, ...]:
        return tuple(self._profiles)
