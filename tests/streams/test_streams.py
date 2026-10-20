"""Stream publisher command lines and config (no ffmpeg/docker needed)."""

from pathlib import Path

from resqcity.streams.config import load_streams_config
from resqcity.streams.loop_cameras import build_command


def test_config_has_at_least_four_cameras():
    cfg = load_streams_config()
    assert len(cfg.cameras) >= 4
    assert cfg.url("C01") == "rtsp://127.0.0.1:8554/C01"
    assert all(c.camera in cfg.cameras for c in cfg.scenario)


def test_build_command_loops_normal_and_plays_accident_once():
    cfg = load_streams_config()
    loop = build_command("ffmpeg", Path("a.mp4"), "rtsp://h/C01", cfg, loop=True)
    once = build_command("ffmpeg", Path("b.mp4"), "rtsp://h/C01", cfg, loop=False)
    assert loop[loop.index("-stream_loop") + 1] == "-1"
    assert once[once.index("-stream_loop") + 1] == "0"
    assert "-re" in loop and loop[-1] == "rtsp://h/C01"
    assert loop.index("-stream_loop") < loop.index("-i")  # input option must precede -i


def _switch(camera, clip, kind, started, duration, onset=None, offset=None):
    return {
        "camera": camera,
        "clip": clip,
        "kind": kind,
        "started_at_utc": started.isoformat(),
        "clip_duration_s": duration,
        "looped": kind == "normal",
        "accident_onset_s_in_clip": onset,
        "accident_offset_s_in_clip": offset,
    }


def test_scene_cuts_include_switches_and_loop_restarts():
    from datetime import UTC, datetime, timedelta

    from resqcity.streams.acceptance import scene_cuts

    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    switches = [
        _switch("C01", "n.mp4", "normal", t0, 25.0),
        _switch("C01", "a.mp4", "accident", t0 + timedelta(seconds=60), 12.0, 8.0, 10.0),
        _switch("C02", "n.mp4", "normal", t0, 25.0),
    ]
    cuts = scene_cuts(switches, "C01", t0 + timedelta(seconds=90))
    assert [(c - t0).total_seconds() for c in cuts] == [0.0, 25.0, 50.0, 60.0]


def test_score_counts_accident_events_only_and_ends_window_at_clip_end(tmp_path, monkeypatch):
    import json
    from datetime import UTC, datetime, timedelta

    from resqcity.streams import acceptance

    t0 = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    log, events = tmp_path / "scenario.jsonl", tmp_path / "events.jsonl"
    switches = [
        _switch("C01", "n.mp4", "normal", t0, 100.0),
        _switch("C01", "a.mp4", "accident", t0 + timedelta(seconds=40), 12.0, 8.0, 10.0),
        _switch("C01", "n2.mp4", "normal", t0 + timedelta(seconds=52), 100.0),
    ]
    log.write_text("".join(json.dumps(s) + "\n" for s in switches), encoding="utf-8")

    def event(sec, kind="accident"):
        ts = (t0 + timedelta(seconds=sec)).isoformat()
        return {"payload": {"camera_id": "C01", "ts": ts, "type": kind}}

    # fire event inside the window, accident alarm after the clip ended (cut-induced), and a
    # far-away accident false alarm
    rows = [event(49, "fire_smoke"), event(56), event(80)]
    events.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(acceptance, "SCENARIO_LOG", log)
    monkeypatch.setattr(acceptance, "EVENTS", events)
    monkeypatch.setattr(acceptance, "detector_provenance", lambda: {})
    report = acceptance.score(t0, t0 + timedelta(seconds=120), ["C01"], cut_s=8.0)
    cam = report["cameras"][0]
    assert cam["other_events"] == {"fire_smoke": 1}
    assert cam["detected"] is False  # fire is not an accident detection; 56 s is after clip end
    assert cam["events_after_clip_end_within_offline_window"] == 1
    assert cam["false_alarms"] == 2 and cam["false_alarms_near_scene_cut"] == 1


def test_vision_privacy_knows_the_simulated_cameras():
    from resqcity.vision.config import load_vision_config

    cfg = load_streams_config()
    privacy = load_vision_config().privacy
    assert privacy.unblurred_streams.urls() == {cfg.url(c) for c in cfg.cameras}
