"""configs/sim.yaml and configs/survival.yaml: the guards that keep phase 7 honest.

Three of these tests exist because of findings from earlier phases rather than from style:
the background workload is required by D-031/L30, the survival metric must stay secondary
(master plan section 6 item 8), and every assumption must be listed by `find_unverified`
(research integrity rule 3).
"""

import pytest
from pydantic import ValidationError

from resqcity.common.config import find_unverified
from resqcity.common.schemas import Agency, EventType
from resqcity.sim.config import (
    SimConfig,
    SurvivalConfig,
    load_sim_config,
    load_survival_config,
)

SIM = load_sim_config()
SURVIVAL = load_survival_config()


def test_every_assumed_simulator_parameter_is_flagged() -> None:
    assert set(find_unverified(SIM)) == {
        "on_scene.median_s",
        "on_scene.false_alarm_s",
        "transport.p_transport",
        "transport.handover_median_s",
        "second_wave.assessment_s",
        "background.target_utilisation",
    }


def test_every_survival_parameter_is_flagged() -> None:
    assert set(find_unverified(SURVIVAL)) == {
        "curves.accident",
        "curves.fire_smoke",
        "curves.crowd",
        "curves.other",
        "sensitivity.tau_multipliers",
        "sensitivity.flat_curve_ablation",
    }


# -- the background workload is not optional (D-031 item 3, L30) ----------------------------


def test_background_workload_is_on_with_a_target_for_every_agency() -> None:
    assert SIM.background.enabled
    assert set(SIM.background.target_utilisation.value) == set(Agency)
    assert all(0.0 < v < 1.0 for v in SIM.background.target_utilisation.value.values())


def test_the_police_target_matches_the_verified_call_volume() -> None:
    """0.412 is derived, not chosen: 771 Dial-112 calls a day at 1800 s over 39 units.

    The figure is the project's only verified volume (docs/data-sources.md), and D-031 item 3
    uses the same arithmetic, so a silent change here would contradict that decision record.
    """
    offered = 771.0 * 1800.0 / 86400.0
    assert SIM.background.target_utilisation.value[Agency.POLICE] == pytest.approx(
        offered / 39, abs=5e-4
    )


def test_switching_the_background_off_must_be_explicit() -> None:
    """The capacity-blind ablation is allowed, but it has to say so: disabling the process while
    leaving a non-zero target would quietly restore the 0.3 %-utilisation model of L30."""
    raw = SIM.model_dump(mode="json")
    raw["background"]["enabled"] = False
    with pytest.raises(ValidationError, match="must say so explicitly"):
        SimConfig.model_validate(raw)
    raw["background"]["target_utilisation"]["value"] = dict.fromkeys(
        raw["background"]["target_utilisation"]["value"], 0.0
    )
    assert SimConfig.model_validate(raw).background.enabled is False  # the honest ablation


def test_an_unclearable_target_is_rejected() -> None:
    raw = SIM.model_dump(mode="json")
    raw["background"]["target_utilisation"]["value"]["police"] = 1.0
    with pytest.raises(ValidationError, match="never clears"):
        SimConfig.model_validate(raw)


# -- survival stays a secondary outcome -----------------------------------------------------


def test_the_primary_outcome_cannot_be_switched_to_survival() -> None:
    raw = SURVIVAL.model_dump(mode="json")
    raw["primary_outcome"] = "expected_survival"
    with pytest.raises(ValidationError, match="secondary outcome"):
        SurvivalConfig.model_validate(raw)


def test_a_survival_curve_may_not_claim_a_citation() -> None:
    """The project could not confirm a per-minute survival-vs-time curve for road trauma, so a
    curve carrying a source would let a conclusion rest on a number nobody verified."""
    raw = SURVIVAL.model_dump(mode="json")
    raw["curves"]["accident"]["source"] = "Valenzuela et al. 1997"
    with pytest.raises(ValidationError, match="UNVERIFIED or ILLUSTRATIVE"):
        SurvivalConfig.model_validate(raw)


def test_the_cited_sources_say_what_they_are_for() -> None:
    """Both entries are about why the curves are illustrative, not sources for their values."""
    by_id = {s.id: s for s in SURVIVAL.sources}
    assert "inconsistent" in by_id["bedard_2020"].supports
    assert by_id["bedard_2020"].doi == "10.1186/s12245-020-00324-7"
    assert by_id["valenzuela_1997"].supports.startswith("NOTHING IN THIS FILE")


# -- the curve itself -------------------------------------------------------------------------


@pytest.mark.parametrize("event_type", list(EventType))
def test_every_curve_is_non_increasing_and_bounded(event_type: EventType) -> None:
    curve = SURVIVAL.curve(event_type)
    values = [curve.at(t) for t in (0.0, 60.0, 600.0, 3600.0, 86400.0)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(curve.s0)
    assert values[-1] >= curve.s_floor
    assert all(0.0 <= v <= 1.0 for v in values)


def test_a_curve_that_rises_with_time_is_rejected() -> None:
    raw = SURVIVAL.model_dump(mode="json")
    raw["curves"]["accident"]["value"]["s_floor"] = 0.99
    with pytest.raises(ValidationError, match="non-increasing"):
        SurvivalConfig.model_validate(raw)


def test_negative_time_since_onset_is_an_error_not_a_number() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        SURVIVAL.curve(EventType.ACCIDENT).at(-1.0)


def test_the_sensitivity_sweep_includes_the_configured_curve() -> None:
    assert 1.0 in SURVIVAL.sensitivity.tau_multipliers.value
    raw = SURVIVAL.model_dump(mode="json")
    raw["sensitivity"]["tau_multipliers"]["value"] = [0.5, 2.0]
    with pytest.raises(ValidationError, match=r"must include 1.0"):
        SurvivalConfig.model_validate(raw)


# -- the run itself ---------------------------------------------------------------------------


def test_the_run_leaves_something_to_measure_after_the_warm_up() -> None:
    assert SIM.run.warmup_s < SIM.run.horizon_s
    assert SIM.run.replications >= 30  # master plan: >= 30 replications
    raw = SIM.model_dump(mode="json")
    raw["run"]["warmup_hours"] = raw["run"]["horizon_days"] * 24
    with pytest.raises(ValidationError, match="leave some of the horizon"):
        SimConfig.model_validate(raw)


def test_unknown_policies_are_rejected() -> None:
    raw = SIM.model_dump(mode="json")
    raw["run"]["policies"] = ["B0", "B9"]
    with pytest.raises(ValidationError, match="unknown policies"):
        SimConfig.model_validate(raw)


def test_the_incident_log_keeps_the_columns_the_metrics_need() -> None:
    assert "last_required_arrival_ts" in SIM.logs.incident_columns
    raw = SIM.model_dump(mode="json")
    raw["logs"]["incident_columns"] = ["incident_id", "onset_ts"]
    with pytest.raises(ValidationError, match="must include"):
        SimConfig.model_validate(raw)
