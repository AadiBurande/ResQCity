"""Typed model for ``configs/streams.yaml`` and helpers shared by the stream tools."""

import os
import shutil
from pathlib import Path

from pydantic import Field

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT, ConfigModel, load_model

# winget puts ffmpeg here on Windows; used only if ffmpeg is not on PATH (e.g. an old shell)
WINGET_LINKS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"


class Transcode(ConfigModel):
    codec: str
    preset: str
    tune: str
    gop: int = Field(gt=0)


class Cue(ConfigModel):
    camera: str
    at_s: float = Field(ge=0)


class StreamsConfig(ConfigModel):
    rtsp_base: str
    ffmpeg: str
    video_dir: Path
    cameras: list[str] = Field(min_length=1)
    clips_per_camera: dict[str, int]
    transcode: Transcode
    scenario: list[Cue] = Field(default_factory=list)

    def camera_dir(self, camera_id: str) -> Path:
        return PROJECT_ROOT / self.video_dir / camera_id

    def url(self, camera_id: str) -> str:
        return f"{self.rtsp_base.rstrip('/')}/{camera_id}"


def load_streams_config(path: Path | None = None) -> StreamsConfig:
    return load_model(StreamsConfig, path or CONFIG_DIR / "streams.yaml")


def find_ffmpeg(name: str) -> str:
    """ffmpeg on PATH, else a winget install (whose bin dir only reaches PATH in new shells)."""
    found = shutil.which(name)
    if found:
        return found
    packages = WINGET_LINKS.parent / "Packages"
    candidates = sorted(packages.glob(f"*FFmpeg*/**/bin/{name}.exe")) if packages.exists() else []
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError(f"{name} not found on PATH (install: winget install Gyan.FFmpeg)")
