"""Provenance sidecars."""

import json

from resqcity.common.config import CONFIG_DIR
from resqcity.common.provenance import meta_path, sha256_file, write_meta


def test_write_meta_records_configs_seeds_and_synthetic_flag(tmp_path):
    out = tmp_path / "result.csv"
    out.write_text("a,b\n1,2\n", encoding="utf-8")
    path = write_meta(
        out,
        configs=[CONFIG_DIR / "base.yaml"],
        synthetic=True,
        seeds={"seed_base": 1, "replication": 0},
        extra={"note": "test"},
    )
    assert path == meta_path(out) == tmp_path / "result.csv.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["synthetic"] is True
    assert meta["seeds"] == {"seed_base": 1, "replication": 0}
    assert meta["configs_sha256"] == {"configs/base.yaml": sha256_file(CONFIG_DIR / "base.yaml")}
    assert meta["note"] == "test"
    assert {"commit", "dirty"} <= set(meta["git"])
    assert "numpy" in meta["versions"]
