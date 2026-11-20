"""Tests for resqcity.paper.build."""

from __future__ import annotations

import json
from pathlib import Path

from resqcity.paper.build import (
    build_paper,
    load_equity_tests,
    load_paired_tests,
    load_summaries,
    main,
)


def test_load_summaries():
    summaries = load_summaries()
    assert isinstance(summaries, dict)
    assert "normal" in summaries
    norm = summaries["normal"]
    for pol in ("B0", "B1", "B2", "B3", "FULL"):
        assert pol in norm
        assert "primary_mean_s" in norm[pol]


def test_load_paired_tests():
    tests = load_paired_tests()
    assert isinstance(tests, dict)
    assert "normal" in tests
    assert len(tests["normal"]) >= 1
    first = tests["normal"][0]
    assert "policy_a" in first
    assert "policy_b" in first
    assert "p_value" in first


def test_load_equity_tests():
    equity = load_equity_tests()
    assert isinstance(equity, dict)
    assert "normal" in equity
    assert len(equity["normal"]) >= 1
    assert "delta" in equity["normal"][0]


def test_build_paper(tmp_path: Path):
    files = build_paper(tmp_path)
    assert len(files) == 6

    # Verify each file exists and is populated
    s1 = tmp_path / "01_introduction.md"
    assert s1.exists()
    content1 = s1.read_text(encoding="utf-8")
    assert "Section 1: Introduction" in content1
    assert "RQ1" in content1
    assert "[CITE:" in content1

    s2 = tmp_path / "02_methods.md"
    assert s2.exists()
    content2 = s2.read_text(encoding="utf-8")
    assert "Section 2: Methods" in content2
    assert "Erlang-C" in content2

    s3 = tmp_path / "03_results.md"
    assert s3.exists()
    content3 = s3.read_text(encoding="utf-8")
    assert "Section 3: Results" in content3
    assert "Table 1" in content3
    assert "B0" in content3

    s4 = tmp_path / "04_discussion.md"
    assert s4.exists()
    content4 = s4.read_text(encoding="utf-8")
    assert "Section 4: Discussion" in content4

    s_figs = tmp_path / "figures_list.md"
    assert s_figs.exists()
    content_figs = s_figs.read_text(encoding="utf-8")
    assert "Figure 1" in content_figs
    assert "results/figures/primary_comparison.png" in content_figs

    paper_md = tmp_path / "paper.md"
    assert paper_md.exists()
    content_paper = paper_md.read_text(encoding="utf-8")
    assert "ResQCity: Confidence-Aware" in content_paper
    assert "Abstract" in content_paper
    assert "Keywords:" in content_paper
    assert "References" in content_paper

    # Verify meta provenance sidecar
    meta_file = tmp_path / "paper.md.meta.json"
    assert meta_file.exists()
    meta_json = json.loads(meta_file.read_text(encoding="utf-8"))
    assert "generated_at_utc" in meta_json


def test_main_cli(tmp_path: Path):
    exit_code = main(["--out-dir", str(tmp_path)])
    assert exit_code == 0
    assert (tmp_path / "paper.md").exists()
