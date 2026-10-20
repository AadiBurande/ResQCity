"""Download the public vision datasets (no account or sign-up needed), within a size cap.

    uv run python -m resqcity.vision.datasets.download [--only ucf_crime cadp dfire]

- UCF-Crime: only the ``RoadAccidents`` videos and a seeded subset of normal test videos are
  extracted from the large zips, using HTTP range reads (no full-zip download), plus the official
  split and temporal-annotation files.
- CADP: annotation JSON + the extracted-frames archive (Google Drive).
- D-Fire: parquet shards of the Hugging Face mirror (converted to YOLO format by ``prepare``).

Everything lands in ``data/raw/vision/`` (gitignored). Licences: see docs/data-sources.md.
A manifest with sizes and SHA-256 per file is written to ``data/raw/vision/manifest.json``.
"""

import argparse
import hashlib
import io
import json
import logging
import random
import shutil
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

from resqcity.common.config import load_base_config
from resqcity.vision.config import VisionConfig, load_vision_config

log = logging.getLogger(__name__)
USER_AGENT = "ResQCity research prototype (university project)"
CHUNK = 1 << 20
GB = 1e9


class SizeCapExceededError(RuntimeError):
    pass


class Budget:
    """Tracks bytes written against the configured cap (existing files count too)."""

    def __init__(self, root: Path, cap_gb: float) -> None:
        self.cap = cap_gb * GB
        self.used = (
            sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
        )

    def reserve(self, n_bytes: int, what: str) -> None:
        if self.used + n_bytes > self.cap:
            raise SizeCapExceededError(
                f"{what}: {n_bytes / GB:.2f} GB would exceed the {self.cap / GB:.1f} GB cap "
                f"(already {self.used / GB:.2f} GB)"
            )
        self.used += n_bytes


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range requests (for zipfile)."""

    def __init__(self, url: str) -> None:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.size = int(resp.headers["Content-Length"])
            self.url = resp.geturl()  # follow the redirect to the CDN once
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        n = min(len(buffer), self.size - self.pos)
        if n <= 0:
            return 0
        headers = {"Range": f"bytes={self.pos}-{self.pos + n - 1}", "User-Agent": USER_AGENT}
        req = urllib.request.Request(self.url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=8 * CHUNK))


def fetch(url: str, dest: Path, budget: Budget | None = None) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        size = int(resp.headers.get("Content-Length") or 0)
        if budget is not None:
            budget.reserve(size, dest.name)
        with tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh, CHUNK)
    tmp.replace(dest)
    return dest


LOCAL_HEADER_LEN = 30  # fixed part of a zip local file header


PARALLEL_MEMBERS = 8  # the CDN serves ~1 MB/s per range request; members are fetched in parallel


def _fetch_member(url: str, info: zipfile.ZipInfo, dest: Path) -> None:
    """One member: read its local header, stream its compressed bytes, inflate, check CRC."""
    headers = {
        "Range": f"bytes={info.header_offset}-{info.header_offset + LOCAL_HEADER_LEN - 1}",
        "User-Agent": USER_AGENT,
    }
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as resp:
        header = resp.read()
    start = (
        info.header_offset
        + LOCAL_HEADER_LEN
        + int.from_bytes(header[26:28], "little")
        + int.from_bytes(header[28:30], "little")
    )
    if info.compress_type == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    elif info.compress_type == zipfile.ZIP_STORED:
        inflater = None
    else:
        raise OSError(f"{info.filename}: unsupported compression {info.compress_type}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    crc = 0
    headers = {"Range": f"bytes={start}-{start + info.compress_size - 1}", "User-Agent": USER_AGENT}
    with (
        urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as resp,
        tmp.open("wb") as fh,
    ):
        for block in iter(lambda: resp.read(CHUNK), b""):
            data = inflater.decompress(block) if inflater else block
            crc = zlib.crc32(data, crc)
            fh.write(data)
        if inflater:
            tail = inflater.flush()
            crc = zlib.crc32(tail, crc)
            fh.write(tail)
    if crc & 0xFFFFFFFF != info.CRC:
        tmp.unlink()
        raise OSError(f"CRC mismatch for {info.filename}")
    tmp.replace(dest)


def extract_members(zf: zipfile.ZipFile, names: list[str], out_dir: Path, budget: Budget) -> int:
    """Extract members of a remote zip, several at a time (each = one streamed range request)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_dir.mkdir(parents=True, exist_ok=True)
    url = zf.fp.raw.url  # type: ignore[union-attr]  (BufferedReader over HttpRangeFile)
    todo = []
    for name in names:
        dest = out_dir / Path(name).name
        if not dest.exists():
            info = zf.getinfo(name)
            budget.reserve(info.file_size, name)
            todo.append((info, dest))
    with ThreadPoolExecutor(max_workers=PARALLEL_MEMBERS) as pool:
        futures = {pool.submit(_fetch_member, url, info, dest): (info, dest) for info, dest in todo}
        for fut in as_completed(futures):
            info, dest = futures[fut]
            fut.result()
            log.info("  %s (%.1f MB)", dest.name, info.file_size / 1e6)
    return len(todo)


# ---------------------------------------------------------------------------------------------


def download_ucf_crime(cfg: VisionConfig, root: Path, budget: Budget) -> None:
    c = cfg.datasets.ucf_crime
    out = root / "ucf_crime"
    for rel in (
        c.temporal_annotations,
        "UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Train.txt",
        "UCF_Crimes-Train-Test-Split/Anomaly_Detection_splits/Anomaly_Test.txt",
    ):
        fetch(f"{c.base_url}/{rel}", out / Path(rel).name, budget)

    log.info("UCF-Crime: %s/* from %s (range reads)", c.accident_prefix, c.accident_zip)
    zf = remote_zip(f"{c.base_url}/{c.accident_zip}")
    accident = sorted(
        n for n in zf.namelist() if f"/{c.accident_prefix}/" in f"/{n}" and n.endswith(".mp4")
    )
    extract_members(zf, accident, out / c.accident_prefix, budget)

    log.info("UCF-Crime: %d normal test videos from %s", c.normal_max_videos, c.normal_zip)
    zf = remote_zip(f"{c.base_url}/{c.normal_zip}")
    normal = sorted(n for n in zf.namelist() if n.endswith(".mp4"))
    rng = random.Random(cfg.datasets.split.seed)
    chosen = sorted(rng.sample(normal, min(c.normal_max_videos, len(normal))))
    extract_members(zf, chosen, out / "Normal", budget)


def download_cadp(cfg: VisionConfig, root: Path, budget: Budget) -> None:
    import gdown

    c = cfg.datasets.cadp
    out = root / "cadp"
    fetch(c.annotations_url, out / "annotations_CADP.json", budget)
    archive = out / "cadp_frames.zip"
    if not archive.exists() and not (out / "frames").exists():
        log.info("CADP: extracted frames (Google Drive)")
        gdown.download(id=c.frames_gdrive_id, output=str(archive), quiet=False)
        budget.reserve(archive.stat().st_size, archive.name)
    if archive.exists() and not (out / "frames").exists():
        with zipfile.ZipFile(archive) as zf:
            budget.reserve(sum(i.file_size for i in zf.infolist()), "CADP frames (extracted)")
            zf.extractall(out / "frames")
        archive.unlink()  # keep only the extracted frames (halves disk use)


def download_dfire(cfg: VisionConfig, root: Path, budget: Budget) -> None:
    repo = cfg.datasets.dfire.hf_repo
    api = f"https://huggingface.co/api/datasets/{repo}/tree/main/data"
    req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        files = [f for f in json.load(resp) if f["path"].endswith(".parquet")]
    for f in files:
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{f['path']}"
        fetch(url, root / "dfire" / Path(f["path"]).name, budget)


DOWNLOADERS = {"ucf_crime": download_ucf_crime, "cadp": download_cadp, "dfire": download_dfire}


def write_manifest(root: Path) -> Path:
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".part"):
            if p.suffix.lower() in {".jpg", ".png"}:  # thousands of frames: count, do not hash
                continue
            digest = hashlib.sha256()
            with p.open("rb") as fh:
                for block in iter(lambda fh=fh: fh.read(CHUNK), b""):
                    digest.update(block)
            files[p.relative_to(root).as_posix()] = {
                "bytes": p.stat().st_size,
                "sha256": digest.hexdigest(),
            }
    frames = sum(1 for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    manifest = {
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_gb": round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / GB, 3),
        "image_files_not_hashed": frames,
        "files": files,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download public vision datasets")
    parser.add_argument("--only", nargs="*", choices=sorted(DOWNLOADERS), default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base, cfg = load_base_config(), load_vision_config()
    root = base.paths.resolve("data_raw") / "vision"
    root.mkdir(parents=True, exist_ok=True)
    budget = Budget(root, cfg.datasets.cap_gb)
    failures = []
    for name in args.only or list(DOWNLOADERS):
        try:
            DOWNLOADERS[name](cfg, root, budget)
        except Exception as exc:  # keep going: one unavailable source must not block the others
            log.error("%s failed: %s: %s", name, type(exc).__name__, exc)
            failures.append(name)
    log.info(
        "manifest: %s (%.2f GB used of %.1f GB cap)",
        write_manifest(root),
        budget.used / GB,
        cfg.datasets.cap_gb,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
