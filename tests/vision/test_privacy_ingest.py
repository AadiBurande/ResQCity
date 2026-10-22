"""Privacy filters and frame ingest on synthetic images/videos."""

import cv2
import numpy as np
import pytest

from resqcity.vision.ingest import file_frames, is_live
from resqcity.vision.privacy import Anonymizer, blur_region, plate_band
from resqcity.vision.types import Box, TrackObs


class FixedFace:
    def detect(self, image):
        return [Box(10, 10, 60, 60)]  # pretend a face here


def noisy(h=240, w=320, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def detail(region: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_blur_removes_detail_only_inside_region():
    img = noisy()
    outside_before = img[150:, 200:].copy()
    blur_region(img, Box(20, 20, 120, 120), 0.25)
    assert detail(img[20:120, 20:120]) < 0.05 * detail(noisy()[20:120, 20:120])
    assert np.array_equal(img[150:, 200:], outside_before)


def test_plate_band_is_lower_part_of_vehicle():
    assert plate_band(Box(0, 100, 50, 200), (0.55, 1.0)) == Box(0, 155, 50, 200)
    assert plate_band(Box(0, 100, 50, 200), (0.6, 1.0), (0.1, 0.9)) == Box(5, 160, 45, 200)


TRACKS = [
    TrackObs(1, "car", 0.9, Box(150, 100, 300, 200)),
    TrackObs(2, "person", 0.9, Box(80, 120, 120, 230)),
]


def _anon(vcfg, mode):
    cfg = vcfg.privacy.model_copy(update={"mode": mode})
    return Anonymizer(
        cfg, vcfg.detector.vehicle_classes, vcfg.detector.person_class, faces=FixedFace()
    )


def test_faces_plates_mode_keeps_people_and_vehicle_body_clear(vcfg):
    img = noisy()
    out = _anon(vcfg, "faces_plates").apply(img, TRACKS)
    assert detail(out[10:60, 10:60]) < 0.05 * detail(img[10:60, 10:60])  # face blurred
    assert detail(out[165:200, 170:280]) < 0.05 * detail(img[165:200, 170:280])  # plate region
    assert np.array_equal(out[100:150, 200:300], img[100:150, 200:300])  # vehicle body clear
    assert np.array_equal(out[120:230, 85:115], img[120:230, 85:115])  # person clear
    assert not np.shares_memory(out, img)  # original untouched


def test_full_mode_also_blurs_people(vcfg):
    img = noisy()
    out = _anon(vcfg, "full").apply(img, TRACKS)
    assert detail(out[120:230, 80:120]) < 0.05 * detail(img[120:230, 80:120])


def test_off_mode_returns_an_unmodified_copy(vcfg):
    img = noisy()
    anon = _anon(vcfg, "off")
    out = anon.apply(img, TRACKS)
    assert np.array_equal(out, img) and not np.shares_memory(out, img)
    assert anon.regions(img, TRACKS) == {"faces": [], "plates": [], "persons": []}


@pytest.mark.parametrize(
    ("src", "live"),
    [("rtsp://x/C01", True), ("http://10.0.0.5:8080/video", True), ("clip.mp4", False)],
)
def test_is_live(src, live):
    assert is_live(src) is live


def test_file_frames_sampling_and_timestamps(tmp_path, vcfg):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (320, 240))
    for i in range(100):  # 4 s at 25 fps
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.putText(frame, str(i), (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    got = list(file_frames(path, vcfg.ingest))  # 5 fps sampling -> every 5th frame
    assert len(got) == 20
    assert [f.t for f in got[:3]] == pytest.approx([0.0, 0.2, 0.4])
    assert got[0].image.shape == (240, 320, 3)  # narrower than resize_width: unchanged
    pts = list(file_frames(path, vcfg.ingest, use_pts=True))  # constant frame rate: same times
    assert [f.t for f in pts] == pytest.approx([f.t for f in got], abs=1e-3)


def test_missing_file_raises(tmp_path, vcfg):
    with pytest.raises(OSError):
        list(file_frames(tmp_path / "nope.mp4", vcfg.ingest))


@pytest.mark.parametrize(
    ("source", "allowed"),
    [
        ("data/videos/C01/accident/x.mp4", True),
        ("data/raw/vision/ucf_crime/RoadAccidents/x.mp4", True),
        ("rtsp://127.0.0.1:8554/C01", True),
        ("http://192.168.1.20:8080/video", False),  # phone IP camera: real people
        ("rtsp://10.0.0.5/stream", False),
        ("rtsp://127.0.0.1:8554/relay", False),  # a relay that is not a simulated camera
        ("recordings/street.mp4", False),  # a local file outside the public-dataset folders
    ],
)
def test_unblurred_only_for_public_dataset_sources(source, allowed, vcfg):
    from resqcity.vision.privacy import unblurred_allowed

    assert unblurred_allowed(source, vcfg.privacy) is allowed


def test_pipeline_refuses_unblurred_mode_for_other_sources(vcfg):
    from resqcity.vision.privacy import require_source_allowed

    off = vcfg.privacy.model_copy(update={"mode": "off"})
    require_source_allowed(vcfg.privacy, None)  # default mode needs no source check
    require_source_allowed(off, "rtsp://127.0.0.1:8554/C01")
    for bad in (None, "http://192.168.1.20:8080/video"):
        with pytest.raises(PermissionError):
            require_source_allowed(off, bad)
