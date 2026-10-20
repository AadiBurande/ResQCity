"""Dataset tooling: splits, UCF temporal annotations, D-Fire class inference, download budget."""

import pandas as pd
import pytest

from resqcity.vision.config import SplitConfig
from resqcity.vision.datasets.annotations import assign_splits, parse_ucf_temporal
from resqcity.vision.datasets.download import Budget, SizeCapExceededError
from resqcity.vision.datasets.prepare_dfire import infer_names


def test_splits_are_by_video_stratified_and_seeded():
    df = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(40)],
            "source": ["ucf"] * 20 + ["cadp"] * 20,
            "label": (["accident"] * 10 + ["normal"] * 10) * 2,
            "has_onset": [True] * 5 + [False] * 15 + [True] * 20,
        }
    )
    cfg = SplitConfig(train=0.6, val=0.2, test=0.2, seed=7)
    a, b = assign_splits(df, cfg), assign_splits(df, cfg)
    assert a.equals(b)
    for _, g in df.assign(split=a).groupby(["source", "label", "has_onset"]):
        if len(g) >= 3:
            assert {"val", "test"} <= set(g["split"])
    assert len(a) == len(df)  # exactly one split per video


def test_parse_ucf_temporal(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(
        "RoadAccidents001_x264.mp4  RoadAccidents  210  300  -1  -1\n"
        "Normal_Videos_003_x264.mp4  Normal  -1  -1  -1  -1\n",
        encoding="utf-8",
    )
    t = parse_ucf_temporal(p)
    assert t["RoadAccidents001_x264.mp4"] == (210, 300, -1, -1)
    assert t["Normal_Videos_003_x264.mp4"][0] == -1


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({0: 11865, 1: 14692}, {1: "fire", 0: "smoke"}),
        ({0: 14700, 1: 11860}, {0: "fire", 1: "smoke"}),
    ],
)
def test_dfire_names_inferred_from_readme_totals(counts, expected):
    assert infer_names(counts) == expected


def test_dfire_names_mismatch_fails():
    with pytest.raises(ValueError):
        infer_names({0: 100, 1: 200})


def test_budget_cap(tmp_path):
    (tmp_path / "existing.bin").write_bytes(b"x" * 1000)
    budget = Budget(tmp_path, cap_gb=2000 / 1e9)
    budget.reserve(900, "small")
    with pytest.raises(SizeCapExceededError):
        budget.reserve(200, "too big")


def test_onset_videos_only_in_evaluation_splits():
    df = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(30)],
            "source": ["ucf"] * 30,
            "label": ["accident"] * 20 + ["normal"] * 10,
            "has_onset": [True] * 9 + [False] * 21,
        }
    )
    split = assign_splits(df, SplitConfig(train=0.6, val=0.2, test=0.2, seed=1))
    onset = split[:9]
    assert set(onset) == {"val", "test"} and (onset == "test").sum() == 5
