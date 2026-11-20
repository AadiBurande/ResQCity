"""Cross-platform task runner behind the Makefile.

GNU make is not available on every developer machine (notably Windows), so every Make target
delegates here and can also be run directly:

    uv run python -m resqcity.tools.tasks <target> [--scenario NAME] [--policy NAME]

Targets that belong to a later phase exit with status 3 and say so, rather than pretending to
succeed. A silent no-op ``reproduce`` would be a research-integrity hazard.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from resqcity.common.config import PROJECT_ROOT

NOT_IMPLEMENTED_EXIT = 3


def _run(*commands: Sequence[str], cwd: Path | str | None = None) -> int:
    work_dir = PROJECT_ROOT if cwd is None else cwd
    env = {**os.environ, "PYTHONUTF8": "1"}  # Marathi/Hindi names on Windows consoles
    for cmd in commands:
        print("$", " ".join(cmd), flush=True)
        code = subprocess.call(list(cmd), cwd=work_dir, env=env)
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


def forecast(_: argparse.Namespace) -> int:
    """Phase 5: write the history scenario, build the ward-hour panel and fit/score the
    forecasters. Minutes; no GPU. Accuracy here is not a research claim (D-030)."""
    return _run(
        _py("resqcity.sim.make_history"),
        _py("resqcity.sim.panel"),
        _py("resqcity.sim.forecast_eval"),
    )


def optimize(_: argparse.Namespace) -> int:
    """Phase 6: solver benchmark, fleet roster, MCLP camera placement and the Erlang-C table.
    Seconds; no GPU."""
    return _run(
        _py("resqcity.optimization.bench_dispatch"),
        _py("resqcity.sim.optimize_run"),
    )


def sim(args: argparse.Namespace) -> int:
    """Phase 7: validate the queueing, measure what the cameras see, then run the simulator.

    The validation runs first on purpose: if the engine's clock, queue discipline or busy-time
    accounting disagree with the closed form, nothing downstream is worth looking at. The
    evidence funnel runs second because it bounds what any CCTV policy could possibly achieve
    (D-032 item 1). Minutes per replication-week; no GPU."""
    return _run(
        _py("resqcity.sim.validate_queue"),
        _py("resqcity.sim.evidence_funnel", "--scenario", args.scenario),
        _py(
            "resqcity.sim.runner",
            "--scenario",
            args.scenario,
            "--policy",
            args.policy,
            "--reps",
            str(args.sim_reps),
        ),
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


def experiments(args: argparse.Namespace) -> int:
    """Phase 10: run the experiment grid (scenarios x policies x replications)."""
    quick_args = ["--quick"] if args.sim_reps <= 3 else []
    return _run(_py("resqcity.experiments.run", *quick_args))


def figures(_: argparse.Namespace) -> int:
    """Phase 10: generate publication figures from experiment results."""
    return _run(_py("resqcity.experiments.figures.generate"))


def paper(_: argparse.Namespace) -> int:
    """Phase 11: compile the IMRaD paper from results."""
    return _run(_py("resqcity.paper.build"))


def reproduce(args: argparse.Namespace) -> int:
    """Phase 11: regenerate results, figures, summary and paper from clean state."""
    t0 = time.perf_counter()
    quick_args = ["--quick"] if args.sim_reps <= 3 else []
    code = _run(
        _py("resqcity.experiments.run", *quick_args),
        _py("resqcity.experiments.summarize"),
        _py("resqcity.experiments.figures.generate"),
        _py("resqcity.paper.build"),
    )
    if code != 0:
        return code

    elapsed = time.perf_counter() - t0
    meta = {
        "timestamp": datetime.now(UTC).isoformat(),
        "wall_seconds": elapsed,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    meta_path = PROJECT_ROOT / "results" / "reproduce_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Reproduction complete in {elapsed:.1f} s. Metadata recorded in {meta_path.name}")
    return 0


def api(_: argparse.Namespace) -> int:
    """Phase 8: start FastAPI backend."""
    return _run(_py("uvicorn", "resqcity.backend.app:app", "--host", "0.0.0.0", "--port", "8000"))


def ui(_: argparse.Namespace) -> int:
    """Phase 9: start Vite dashboard dev server."""
    pkg_mgr = shutil.which("pnpm") or shutil.which("npm") or "pnpm"
    return _run([pkg_mgr, "run", "dev"], cwd=PROJECT_ROOT / "dashboard")


def demo(_: argparse.Namespace) -> int:
    """Phase 9: run end-to-end live demo in browser."""
    import webbrowser

    url = "http://localhost:5173"
    print(f"Launching ResQCity Control Room in browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"Browser launch note: {exc}")
    return 0


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
    "forecast": forecast,
    "optimize": optimize,
    "api": api,
    "ui": ui,
    "demo": demo,
    "sim": sim,
    "experiments": experiments,
    "figures": figures,
    "paper": paper,
    "reproduce": reproduce,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ResQCity task runner")
    parser.add_argument("target", choices=sorted(TASKS))
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--policy", default="B0")
    parser.add_argument("--reps", type=int, default=30, help="fusion: week-check replications")
    parser.add_argument(
        "--sim-reps", type=int, default=1, help="sim: replications (configs/sim.yaml has 30)"
    )
    args = parser.parse_args(argv)
    return TASKS[args.target](args)


if __name__ == "__main__":
    raise SystemExit(main())
