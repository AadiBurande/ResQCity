"""Background knowledge handed to fusion: where and how often incidents and false calls occur.

Built from the same configs the generator uses (hotspot weights, base rate, type mix) and the
call process (false-call rate). This is what a control room could know from history; it holds
no ground truth of a particular replication (D-028).
"""

from resqcity.fusion.config import CallsConfig
from resqcity.fusion.model import Background
from resqcity.geo.network import RoadNetwork
from resqcity.sim.generator import IncidentsConfig, hotspot_weights


def build_background(
    network: RoadNetwork,
    incidents: IncidentsConfig,
    calls: CallsConfig,
    rate_multiplier: float = 1.0,
) -> Background:
    mix = incidents.type_mix.value
    total = sum(mix.values())
    return Background(
        node_share=hotspot_weights(network, incidents.hotspot),
        incident_rate_per_day=incidents.base_rate_per_day.value * rate_multiplier,
        type_share={k: v / total for k, v in mix.items()},
        false_call_rate_per_day=calls.false_calls.rate_per_day.value,
    )
