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


def test_anonymizer_covers_faces_plates_and_persons(vcfg):
    anon = Anonymizer(
        vcfg.privacy, vcfg.detector.vehicle_classes, vcfg.detector.person_class, faces=FixedFace()
    )
    img = noisy()
    tracks = [
        TrackObs(1, "car", 0.9, Box(150, 100, 300, 200)),
        TrackObs(2, "person", 0.9, Box(80, 120, 120, 230)),
    ]
    out = anon.apply(img, tracks)
    assert detail(out[10:60, 10:60]) < 0.05 * detail(img[10:60, 10:60])  # face
    assert detail(out[160:200, 150:300]) < 0.05 * detail(img[160:200, 150:300])  # plate band
    assert detail(out[120:230, 80:120]) < 0.05 * detail(img[120:230, 80:120])  # person
    assert np.array_equal(out[100:150, 200:300], img[100:150, 200:300])  # upper vehicle body kept
    assert not np.shares_memory(out, img)  # original untouched


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


def test_missing_file_raises(tmp_path, vcfg):
    with pytest.raises(OSError):
        list(file_frames(tmp_path / "nope.mp4", vcfg.ingest))
