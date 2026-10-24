"""Synthetic 112 calls from ground-truth incidents, plus false (prank / mistaken) calls.

SYNTHETIC by construction (phase 4, D-028). Every parameter is in ``configs/calls.yaml`` and is
UNVERIFIED. Common random numbers: each incident draws from its own streams
(``calls/<incident index>`` for the reporting process, ``call_text/<incident index>/<call no>`` for
the content of each call) and false calls from ``false_calls/<hour>``, so changing one parameter
does not reshuffle unrelated draws. The truth link (call -> incident) is returned separately;
fusion never sees it.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from resqcity.common.requirements import RequirementsConfig
from resqcity.common.rng import stream
from resqcity.common.schemas import Call, EventType, GeoPoint, Language, SeverityFeatures
from resqcity.fusion.config import CallsConfig, LanguageTemplates, PhraseSet, TemplatesConfig
from resqcity.geo.network import RoadNetwork
from resqcity.sim.groundtruth import GroundTruthIncident

CALLS_STREAM = "calls"
CALL_TEXT_STREAM = "call_text"
FALSE_CALLS_STREAM = "false_calls"
HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class SimCall:
    """A simulated call and its ground truth (``incident_id`` None = false call)."""

    call: Call
    incident_id: str | None
    true_error_m: float


def lognormal(rng: np.random.Generator, median: float, sigma: float) -> float:
    return float(median * math.exp(sigma * rng.standard_normal()))


def cue_flags(features: SeverityFeatures, multi_vehicle_min: int) -> dict[str, bool]:
    return {
        "person_down": features.person_down,
        "overturned": features.overturned,
        "fire_visible": features.fire_visible,
        "multi_vehicle": features.num_vehicles >= multi_vehicle_min,
    }


class CallSimulator:
    def __init__(
        self,
        network: RoadNetwork,
        cfg: CallsConfig,
        templates: TemplatesConfig,
        requirements: RequirementsConfig,
        node_share: np.ndarray,
        type_mix: dict[EventType, float],
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.templates = templates
        self.requirements = requirements
        self.node_share = np.asarray(node_share, float) / float(np.sum(node_share))
        types = list(type_mix)
        weights = np.asarray([type_mix[t] for t in types], float)
        self.types, self.type_p = types, weights / weights.sum()
        self.languages = list(cfg.language_mix.value)
        lang_w = np.asarray(list(cfg.language_mix.value.values()), float)
        self.language_p = lang_w / lang_w.sum()
        self._road_name = self._node_road_names()

    def _node_road_names(self) -> dict[int, str]:
        """Name of the most important named road at each node (callers name the main road)."""
        e = self.network.edges[["u", "name"]].assign(rank=self.network.class_code)
        e = e[e["name"].astype(str) != ""].sort_values(["rank"], kind="stable")
        first = e.drop_duplicates("u", keep="first")
        return dict(zip(first["u"].astype(int), first["name"].astype(str), strict=True))

    # -- text -----------------------------------------------------------------------------------

    def _text(
        self,
        rng: np.random.Generator,
        claimed: EventType,
        features: SeverityFeatures,
        cues_present: dict[str, bool],
        node: int,
    ) -> tuple[str, Language]:
        language = Language(self.languages[int(rng.choice(len(self.languages), p=self.language_p))])
        lang: LanguageTemplates = self.templates.languages[language]
        phrases: PhraseSet = (
            lang.held_out if rng.random() < self.templates.share_held_out else lang.main
        )
        road = self._road_name.get(node)
        loc = (
            lang.location[int(rng.integers(len(lang.location)))].format(road=road)
            if road
            else lang.location_unknown
        )
        words = [lang.vehicles.get(v, lang.vehicles["car"]) for v in features.vehicle_classes]
        v1 = words[0] if words else lang.vehicles["car"]
        v2 = words[1] if len(words) > 1 else None
        options = phrases.types[claimed]
        if v2 is None:  # prefer phrasings that do not name a second vehicle
            options = [o for o in options if "{v2}" not in o] or options
        template = options[int(rng.integers(len(options)))]
        parts = [template.format(loc=loc, v1=v1, v2=v2 or v1, n=features.num_vehicles)]
        cc = self.cfg.caller_cues
        for cue, present in cues_present.items():
            p = cc.p_mention_present.value if present else cc.p_mention_absent.value
            if rng.random() < p:
                cue_options = phrases.cues[cue]
                parts.append(
                    cue_options[int(rng.integers(len(cue_options)))].format(
                        v1=v1, n=max(features.num_vehicles, 3)
                    )
                )
        return " ".join(parts), language

    # -- locations ------------------------------------------------------------------------------

    def _reported(self, rng: np.random.Generator, node: int) -> tuple[GeoPoint, float]:
        mode = self.cfg.location.active
        r = lognormal(rng, mode.median_error_m.value, mode.sigma.value)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x, y = self.network.xy[self.network.index_of(node)]
        lat, lon = self.network.unproject(x + r * math.cos(theta), y + r * math.sin(theta))
        return GeoPoint(lat=lat, lon=lon), r

    def _call(
        self,
        rng: np.random.Generator,
        call_id: str,
        ts: datetime,
        claimed: EventType,
        features: SeverityFeatures,
        cues_present: dict[str, bool],
        node: int,
        incident_id: str | None,
    ) -> SimCall:
        text, language = self._text(rng, claimed, features, cues_present, node)
        location, err = self._reported(rng, node)
        call = Call(
            id=call_id,
            ts=ts,
            reported_location=location,
            location_error_m=self.cfg.location.active.median_error_m.value,
            caller_text=text,
            language=language,
        )
        return SimCall(call, incident_id, err)

    # -- genuine calls --------------------------------------------------------------------------

    def for_incident(
        self, inc: GroundTruthIncident, index: int, seed_base: int, replication: int
    ) -> list[SimCall]:
        rep = self.cfg.reporting
        cls = inc.severity_class
        rng = stream(seed_base, replication, CALLS_STREAM, index)
        u_report, z_delay = rng.random(), rng.standard_normal()
        n_extra = int(rng.poisson(rep.extra_calls_mean.value[cls]))
        if u_report < rep.p_unreported.value[cls]:
            return []
        delay = rep.first_delay_median_s.value * math.exp(rep.first_delay_sigma.value * z_delay)
        multi = self.requirements.severity.multi_vehicle_min
        cues = cue_flags(inc.severity_features, multi)
        out, t = [], inc.onset_ts + timedelta(seconds=delay)
        for j in range(1 + n_extra):
            text_rng = stream(seed_base, replication, CALL_TEXT_STREAM, index, j)
            if j > 0:
                t += timedelta(seconds=float(text_rng.exponential(rep.extra_gap_mean_s.value)))
            claimed = inc.type
            if text_rng.random() < self.cfg.caller_cues.p_type_confusion.value:
                others = [x for x in EventType if x != inc.type]
                claimed = others[int(text_rng.integers(len(others)))]
            out.append(
                self._call(
                    text_rng,
                    f"CALL-{inc.id}-{j}",
                    t,
                    claimed,
                    inc.severity_features,
                    cues,
                    inc.node,
                    inc.id,
                )
            )
        return out

    # -- false calls ----------------------------------------------------------------------------

    def false_calls(
        self, start: datetime, hours: int, seed_base: int, replication: int
    ) -> list[SimCall]:
        rate_h = self.cfg.false_calls.rate_per_day.value / 24.0
        out = []
        for h in range(hours):
            rng = stream(seed_base, replication, FALSE_CALLS_STREAM, h)
            count = int(rng.poisson(rate_h))
            for j in range(count):
                t = start + h * HOUR + timedelta(seconds=float(rng.uniform(0, 3600)))
                node = int(
                    self.network.node_ids[rng.choice(len(self.node_share), p=self.node_share)]
                )
                claimed = self.types[int(rng.choice(len(self.types), p=self.type_p))]
                features = SeverityFeatures()
                none = dict.fromkeys(cue_flags(features, 1), False)
                out.append(
                    self._call(
                        rng,
                        f"CALL-F-{replication:03d}-{h:05d}-{j}",
                        t,
                        claimed,
                        features,
                        none,
                        node,
                        None,
                    )
                )
        return out

    def generate(
        self,
        incidents: list[GroundTruthIncident],
        start: datetime,
        hours: int,
        seed_base: int,
        replication: int,
    ) -> list[SimCall]:
        """All calls of a replication, sorted by time (ties by id)."""
        calls = [
            c
            for k, inc in enumerate(incidents)
            for c in self.for_incident(inc, k, seed_base, replication)
        ]
        calls += self.false_calls(start, hours, seed_base, replication)
        return sorted(calls, key=lambda c: (c.call.ts, c.call.id))
