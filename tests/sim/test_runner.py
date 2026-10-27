"""The runner: common random numbers, the warm-up, and the log (phase 7 slices 2+6).

These need the real processed network, because the camera sites, stations and hospitals are real
OSM nodes, so they are skipped when ``data/processed`` has not been built. The cheap structural
checks that do not need it are at the top.
"""

import pytest

from resqcity.common.config import load_base_config
from resqcity.geo.config import load_geo_config
from resqcity.geo.network import RoadNetwork
from resqcity.sim.config import SimConfig, load_sim_config
from resqcity.sim.runner import IncidentRow, Week, run_replication, to_frame

BASE, GEO = load_base_config(), load_geo_config()
PROCESSED = BASE.paths.resolve("data_processed")
SIM = load_sim_config()


# -- structural, no data needed ---------------------------------------------------------------


def test_the_log_row_covers_every_configured_column() -> None:
    """configs/sim.yaml lists the columns the metrics need; the row must actually carry them."""
    fields = set(IncidentRow("x", "B0", 0).__dict__)
    missing = set(SIM.logs.incident_columns) - fields
    assert not missing, f"IncidentRow is missing configured columns: {sorted(missing)}"


def test_the_frame_is_empty_rather_than_broken_with_no_rows() -> None:
    assert to_frame([]).empty


# -- the real thing ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not (PROCESSED / GEO.network.edges_file).exists(),
    reason="processed data not built (make data)",
)


@pytest.fixture(scope="module")
def network() -> RoadNetwork:
    if not (PROCESSED / GEO.network.edges_file).exists():
        pytest.skip("processed data not built (make data)")
    return RoadNetwork.load(PROCESSED, GEO)


def _short(days: int = 1, warmup_hours: float = 2.0) -> SimConfig:
    raw = SIM.model_dump(mode="json")
    raw["run"]["horizon_days"] = days
    raw["run"]["warmup_hours"] = warmup_hours
    return SimConfig.model_validate(raw)


@pytestmark_real
def test_two_policies_see_the_identical_week(network: RoadNetwork) -> None:
    """Common random numbers, asserted rather than claimed.

    If the evidence differed between policies, every policy comparison in phase 10 would be
    confounded by a different week rather than measuring the policy.
    """
    a = Week(network, "normal", BASE.random.seed_base, 0)
    b = Week(network, "normal", BASE.random.seed_base, 0)
    assert a.digest() == b.digest()
    assert [i.id for i in a.incidents] == [i.id for i in b.incidents]
    # and a different replication is a genuinely different week
    c = Week(network, "normal", BASE.random.seed_base, 1)
    assert c.digest() != a.digest()


@pytestmark_real
@pytest.mark.slow
def test_a_short_replication_produces_a_usable_log(
    network: RoadNetwork, monkeypatch: pytest.MonkeyPatch
) -> None:
    from resqcity.sim import runner

    short = _short()
    monkeypatch.setattr(runner, "load_sim_config", lambda *a, **k: short)
    rep = run_replication(network, "normal", "B0", 0, BASE.random.seed_base)
    assert rep.rows, "a day of the normal scenario should produce at least one candidate"
    frame = to_frame([rep])
    assert set(SIM.logs.incident_columns) <= set(frame.columns)
    # a served incident must have its last-required arrival after its onset
    served = frame.dropna(subset=["onset_ts", "last_required_arrival_ts"])
    if not served.empty:
        assert (served["last_required_arrival_ts"] >= served["onset_ts"]).all()
    # the warm-up flag is set for the incidents inside it, and they are a minority of a day
    assert frame["in_warmup"].sum() < len(frame)


@pytestmark_real
@pytest.mark.slow
def test_the_background_workload_lifts_utilisation_off_the_floor(
    network: RoadNetwork, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-031/L30 measured 0.3 % utilisation without this process. The point of slice 4 is that
    the simulated city is no longer capacity-free, so the run must show it."""
    from resqcity.sim import runner

    monkeypatch.setattr(runner, "load_sim_config", lambda *a, **k: _short())
    rep = run_replication(network, "normal", "B0", 0, BASE.random.seed_base)
    assert rep.utilisation["police"] > 0.1, rep.utilisation
    assert rep.background["enabled"] is True
    assert rep.background["by_agency"]["police"]["jobs_started"] > 0


@pytestmark_real
def test_camera_coverage_is_compared_in_the_same_numbering(network: RoadNetwork) -> None:
    """`camera_coverage` returns internal node INDICES; incident nodes are OSM node IDS.

    Comparing the two silently reported that no incident was ever within camera range while the
    detector was detecting some, which made the `was_camera_covered` column of the incident log
    wrong. The sets must be in the same numbering and must actually overlap the graph.
    """
    week = Week(network, "normal", BASE.random.seed_base, 0)
    node_ids = set(network.nodes["node_id"].astype(int))
    assert week.covered_nodes, "the 40 camera sites must cover some nodes"
    assert week.covered_nodes <= node_ids, "covered_nodes must be OSM ids, not indices"
    # a camera's own junction is by definition within its coverage radius
    assert {int(c.junction_node) for c in week.cameras} <= week.covered_nodes


@pytestmark_real
def test_camera_coverage_of_incidents_matches_the_phase_6_measurement(
    network: RoadNetwork,
) -> None:
    """D-031 measured the 40 phase-2 sites covering 3.95 % of risk-weighted demand. Incidents are
    drawn proportionally to that same demand, so the share of incidents falling in camera range
    must be of that order -- a loose band, because a week holds only ~36 incidents."""
    seen = total = 0
    for rep in range(4):
        week = Week(network, "normal", BASE.random.seed_base, rep)
        total += len(week.incidents)
        seen += sum(int(i.node) in week.covered_nodes for i in week.incidents)
    assert total > 100
    assert 0.0 < seen / total < 0.15, f"{seen}/{total} incidents in camera range"
