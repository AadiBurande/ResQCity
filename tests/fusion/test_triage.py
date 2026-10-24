"""Call triage: rule baseline in en/hi/mr and the offline-LLM answer parser (phase 4)."""

import os

import pytest

from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import Call, EventType, GeoPoint, Language
from resqcity.fusion.config import load_triage_config
from resqcity.fusion.triage import LocalLLMTriager, RuleTriager, make_triager, triaged
from tests.conftest import at

CFG = load_triage_config()
REQ = load_requirements_config()
RULES = RuleTriager(CFG, REQ)


def _call(text: str, language: Language) -> Call:
    return Call(
        id="CALL-T",
        ts=at(0, 10),
        reported_location=GeoPoint(lat=18.5, lon=73.85),
        location_error_m=300.0,
        caller_text=text,
        language=language,
    )


@pytest.mark.parametrize(
    ("text", "language", "etype", "person_down", "fire", "vehicles"),
    [
        (
            "A car and a truck have collided near Laxmi Road. "
            "Someone is injured and lying on the road.",
            Language.ENGLISH, EventType.ACCIDENT, True, False, ["car", "truck"],
        ),
        ("There is a fire on Tilak Road.", Language.ENGLISH, EventType.FIRE_SMOKE, False, True, []),
        (
            "लक्ष्मी रोड के पास एक कार और एक बाइक की टक्कर हो गई है। गाड़ी में आग लग गई है।",
            Language.HINDI, EventType.ACCIDENT, False, True, ["car", "motorcycle"],
        ),
        ("टिळक रोड वर आग लागली आहे.", Language.MARATHI, EventType.FIRE_SMOKE, False, True, []),
        (
            "चौकाजवळ अपघात झाला आहे. एक माणूस जखमी होऊन रस्त्यावर पडला आहे.",
            Language.MARATHI, EventType.ACCIDENT, True, False, [],
        ),
        (
            "मोठी गर्दी जमली आहे आणि लोक भांडत आहेत.",
            Language.MARATHI, EventType.CROWD, False, False, [],
        ),
    ],
)  # fmt: skip
def test_rule_triage_examples(text, language, etype, person_down, fire, vehicles):
    r = RULES.triage(_call(text, language))
    assert r.type == etype and r.method == "rules"
    assert r.features.person_down == person_down
    assert r.features.fire_visible == fire
    assert sorted(r.features.vehicle_classes) == sorted(vehicles)
    assert r.confidence == CFG.rules.confidence_matched.value
    if etype == EventType.ACCIDENT:
        assert r.requirements.police >= 1 and r.features.num_vehicles >= 1


def test_vehicle_count_and_word_boundaries():
    r = RULES.triage(
        _call("Road accident near the junction. 4 vehicles are involved.", Language.ENGLISH)
    )
    assert r.features.num_vehicles == 4
    r = RULES.triage(_call("There is a fire, someone scared the cargo men.", Language.ENGLISH))
    assert r.features.vehicle_classes == []  # "car" inside "scared"/"cargo" is not a car


def test_fallback_when_nothing_matches():
    r = RULES.triage(_call("Hello? Is anyone there?", Language.ENGLISH))
    assert r.type == CFG.rules.fallback_type
    assert r.confidence == CFG.rules.confidence_fallback.value


def test_triaged_fills_the_call_fields():
    call = _call("There has been an accident near Laxmi Road.", Language.ENGLISH)
    out = triaged(call, RULES.triage(call))
    assert out.triaged_type == EventType.ACCIDENT and out.triage_confidence == 0.8
    assert out.triaged_severity is not None


def test_llm_is_off_by_default():
    assert not CFG.llm.enabled
    assert isinstance(make_triager(CFG, REQ, use_llm=True), RuleTriager)


def test_llm_answer_parser_without_a_model():
    llm = LocalLLMTriager(CFG, REQ)  # nothing is loaded until the first call
    good = 'Sure. {"type": "accident", "person_down": true, "overturned": false, ' + (
        '"fire_visible": false, "num_vehicles": 2, "vehicles": ["car", "bus", "tank"], '
        '"confidence": 0.9}'
    )
    etype, features, conf = llm.parse(good)
    assert etype == EventType.ACCIDENT and features.person_down and conf == 0.9
    assert features.vehicle_classes == ["car", "bus"]  # unknown classes dropped
    assert llm.parse("I am not sure.") is None
    assert llm.parse('{"type": "earthquake"}') is None


@pytest.mark.skipif(
    os.environ.get("RESQCITY_LLM_TEST") != "1",
    reason="needs the cached local model and a GPU; set RESQCITY_LLM_TEST=1",
)
def test_local_llm_triage_runs():
    llm = LocalLLMTriager(CFG, REQ)
    r = llm.triage(_call("Two cars crashed on Laxmi Road and a man is bleeding.", Language.ENGLISH))
    assert r.method in {"llm", "llm_fallback_rules"}
