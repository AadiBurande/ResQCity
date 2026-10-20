"""Event bus implementations."""

from resqcity.common.bus import FanOutBus, InProcessBus, JsonlBus
from resqcity.common.schemas import GeoPoint


def test_in_process_and_jsonl_fan_out(tmp_path):
    got = []
    mem = InProcessBus()
    mem.subscribe("t", lambda topic, msg: got.append((topic, msg)))
    file_bus = JsonlBus(tmp_path / "bus.jsonl")
    FanOutBus(mem, file_bus).publish("t", GeoPoint(lat=1.0, lon=2.0))
    assert got == [("t", GeoPoint(lat=1.0, lon=2.0))]
    assert list(file_bus.read("t")) == [
        {"topic": "t", "type": "GeoPoint", "payload": {"lat": 1.0, "lon": 2.0}}
    ]
    assert list(file_bus.read("other")) == []
