"""Cross-platform task runner behind the Makefile.

GNU make is not available on every developer machine (notably Windows), so every Make target
delegates here and can also be run directly:

    uv run python -m resqcity.tools.tasks <target> [--scenario NAME] [--policy NAME]

Targets that belong to a later phase exit with status 3 and say so, rather than pretending to
succeed. A silent no-op ``reproduce`` would be a research-integrity hazard.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence

from resqcity.common.config import PROJECT_ROOT

NOT_IMPLEMENTED_EXIT = 3


def _run(*commands: Sequence[str]) -> int:
    env = {**os.environ, "PYTHONUTF8": "1"}  # Marathi/Hindi names on Windows consoles
    for cmd in commands:
        print("$", " ".join(cmd), flush=True)
        code = subprocess.call(list(cmd), cwd=PROJECT_ROOT, env=env)
        if code != 0:
            return code
    return 0


def _py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


def setup(_: argparse.Namespace) -> int:
    code = _run(_py("resqcity.tools.gen_ts_types"))
    if code == 0 and (PROJECT_ROOT / ".git").exists() and shutil.which("git"):
        code = _run(_py("pre_commit", "install"))
    return code


def lint(_: argparse.Namespace) -> int:
    return _run(
        _py("ruff", "check", "."),
        _py("ruff", "format", "--check", "."),
        _py("resqcity.tools.gen_ts_types", "--check"),
    )


def fmt(_: argparse.Namespace) -> int:
    return _run(_py("ruff", "check", "--fix", "."), _py("ruff", "format", "."))


def test(_: argparse.Namespace) -> int:
    return _run(_py("pytest"))


def types(_: argparse.Namespace) -> int:
    return _run(_py("resqcity.tools.gen_ts_types"))


def data(_: argparse.Namespace) -> int:
    """Download raw inputs (network access), build processed data offline, then calibrate:
    MoRTH extract (needs the MoRTH 2022 PDF in data/raw/morth/), base rate, Google-Maps speeds."""
    return _run(
        _py("resqcity.geo.download"),
        _py("resqcity.geo.build"),
        _py("resqcity.sim.morth_extract"),
        _py("resqcity.sim.base_rate"),
        _py("resqcity.geo.calibrate_speeds"),
        _py("resqcity.geo.sanity"),
    )


def fusion(args: argparse.Namespace) -> int:
    """Phase 4: triage check on the caller templates, scripted fusion replays (acceptance) and
    the synthetic-week model check. Minutes; no GPU (the optional LLM triager stays off)."""
    return _run(
        _py("resqcity.fusion.eval_triage"),
        _py("resqcity.sim.gate_coverage"),
        _py("resqcity.sim.fusion_replay", "--all"),
        _py("resqcity.sim.fusion_check", "--reps", str(args.reps), "--scenario", args.scenario),
    )


def streams(_: argparse.Namespace) -> int:
    """Start MediaMTX (Docker) and publish simulated cameras with scenario cues (Ctrl-C stops)."""
    return _run(_py("resqcity.streams.loop_cameras", "--start-server"))


CLIP_SETTINGS = ("s0_t0", "s03_t1", "s1_t1")


V5_RUNS = tuple(
    f"v5_{bb}_{lab}_{data}"
    for bb in ("mnet", "sig", "vjepa")
    for lab in ("mil", "int")
    for data in ("ucf", "all")
)


def vision(_: argparse.Namespace) -> int:
    """Vision pipeline: data (ACCIDENT needs a Kaggle token), annotations, UCA intervals,
    evaluation v4 (cross-fitting), evaluation v5 (phase 3b: extraction, 12 candidates, selection
    stages A, C, B, one final run on the ACCIDENT holdout), fire/smoke detector, demo clips.
    GPU recommended (hours); every step resumes from its cache."""
    train = []
    for s in CLIP_SETTINGS:
        for fit, img_run, vid_run in (
            ("val", f"clip_{s}", f"video_{s}_fitval"),
            ("test", f"clip_{s}_fittest", f"video_{s}_fittest"),
        ):
            train.append(
                _py(
                    "resqcity.vision.train_clip",
                    "--run-id",
                    img_run,
                    "--overlay",
                    f"configs/vision_sweeps/clip_{s}.yaml",
                    "--fit-split",
                    fit,
                )
            )
            train.append(
                _py(
                    "resqcity.vision.train_clip",
                    "--run-id",
                    vid_run,
                    "--overlay",
                    f"configs/vision_sweeps/video_{s}.yaml",
                    "--fit-split",
                    fit,
                )
            )
    v5 = [
        _py(
            "resqcity.vision.train_clip",
            "--run-id",
            r,
            "--overlay",
            f"configs/vision_sweeps/{r}.yaml",
            "--no-holdout",
        )
        for r in V5_RUNS
    ]
    return _run(
        _py("resqcity.vision.datasets.download"),
        _py("resqcity.vision.datasets.annotations"),
        _py("resqcity.vision.datasets.uca"),
        *train,
        _py("resqcity.vision.eval.crossfit"),
        _py("resqcity.vision.extract"),
        *v5,
        *(_py("resqcity.vision.eval.select_v5", "--stage", s) for s in ("a", "c", "b")),
        _py("resqcity.vision.eval.final_v5"),
        _py("resqcity.vision.datasets.prepare_dfire"),
        _py("resqcity.vision.train_fire", "--run-id", "fire_main"),
        _py("resqcity.streams.prepare_videos"),
    )


def _stub(phase: int, what: str) -> Callable[[argparse.Namespace], int]:
    def run(args: argparse.Namespace) -> int:
        print(f"{what}: not implemented yet (phase {phase}).", file=sys.stderr)
        return NOT_IMPLEMENTED_EXIT

    return run


TASKS: dict[str, Callable[[argparse.Namespace], int]] = {
    "setup": setup,
    "lint": lint,
    "format": fmt,
    "test": test,
    "types": types,
    "data": data,
    "streams": streams,
    "vision": vision,
    "fusion": fusion,
    "api": _stub(8, "start backend API"),
    "ui": _stub(9, "start dashboard"),
    "demo": _stub(9, "end-to-end live demo"),
    "sim": _stub(7, "run simulator"),
    "experiments": _stub(10, "run experiments"),
    "figures": _stub(10, "generate figures"),
    "reproduce": _stub(11, "regenerate every result and figure"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResQCity task runner")
    parser.add_argument("target", choices=sorted(TASKS))
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--policy", default="B0")
    parser.add_argument("--reps", type=int, default=30, help="fusion: week-check replications")
    args = parser.parse_args(argv)
    return TASKS[args.target](args)


if __name__ == "__main__":
    raise SystemExit(main())
