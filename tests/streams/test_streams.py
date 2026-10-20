"""Stream publisher command lines and config (no ffmpeg/docker needed)."""

from pathlib import Path

from resqcity.streams.config import load_streams_config
from resqcity.streams.loop_cameras import build_command


def test_config_has_at_least_four_cameras():
    cfg = load_streams_config()
    assert len(cfg.cameras) >= 4
    assert cfg.url("C01") == "rtsp://localhost:8554/C01"
    assert all(c.camera in cfg.cameras for c in cfg.scenario)


def test_build_command_loops_normal_and_plays_accident_once():
    cfg = load_streams_config()
    loop = build_command("ffmpeg", Path("a.mp4"), "rtsp://h/C01", cfg, loop=True)
    once = build_command("ffmpeg", Path("b.mp4"), "rtsp://h/C01", cfg, loop=False)
    assert loop[loop.index("-stream_loop") + 1] == "-1"
    assert once[once.index("-stream_loop") + 1] == "0"
    assert "-re" in loop and loop[-1] == "rtsp://h/C01"
    assert loop.index("-stream_loop") < loop.index("-i")  # input option must precede -i
