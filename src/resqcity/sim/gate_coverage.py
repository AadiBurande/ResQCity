"""Coverage of the association gate under the assumed call location-error model (phase 4).

    uv run python -m resqcity.sim.gate_coverage

For each candidate ``k_location_error`` this reports the share of same-incident evidence pairs
that fall inside the gate ``base_distance_m + k * sqrt(err_a^2 + err_b^2)``, under the error model
of ``configs/calls.yaml`` (a lognormal distance in a uniformly random direction). Two cases:
call-call (both displaced) and call-camera (one displaced, the camera exact).

Honest status (D-029): this criterion was written AFTER a synthetic week showed the original gate
splitting same-incident calls, and it is evaluated on the same UNVERIFIED error model that
generates those calls, so it is in-sample by construction. It fixes a metric bug (the gate used
road-graph distance, which is always >= the straight-line displacement a caller's error actually
is), and it is not tuned on any outcome; it is not independent evidence that the gate is right.
Output: ``results/fusion/gate_k_synthetic.json``.
"""

import argparse
import json
import logging

import numpy as np

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.fusion.config import load_calls_config, load_fusion_config

OUT = PROJECT_ROOT / "results" / "fusion" / "gate_k_synthetic.json"
K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TARGET = 0.95
DRAWS = 4_000_000
SEED = 20261026
log = logging.getLogger(__name__)


def displacements(rng: np.random.Generator, median: float, sigma: float, n: int) -> np.ndarray:
    """(n, 2) offsets: a lognormal distance in a uniformly random direction."""
    r = median * np.exp(sigma * rng.standard_normal(n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def coverage(base_m: float, median: float, sigma: float, n: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    a, b = displacements(rng, median, sigma, n), displacements(rng, median, sigma, n)
    sep_call = np.hypot(*(a - b).T)  # both reports displaced from the same true point
    sep_cam = np.hypot(*a.T)  # one report displaced, the camera exact
    gate_call = base_m + np.asarray(K_GRID)[:, None] * np.hypot(median, median)
    gate_cam = base_m + np.asarray(K_GRID)[:, None] * median
    return {
        "draws": n,
        "k_grid": K_GRID,
        "call_call": [float(np.mean(sep_call <= g)) for g in gate_call],
        "call_camera": [float(np.mean(sep_cam <= g)) for g in gate_cam],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gate, calls = load_fusion_config().gate, load_calls_config()
    mode = calls.location.active
    cov = coverage(
        gate.base_distance_m.value,
        mode.median_error_m.value,
        mode.sigma.value,
        args.draws,
        args.seed,
    )
    worst = [min(c, d) for c, d in zip(cov["call_call"], cov["call_camera"], strict=True)]
    passing = [k for k, w in zip(K_GRID, worst, strict=True) if w >= TARGET]
    report = {
        "status": "synthetic: the assumed call error model of configs/calls.yaml, which is "
        "UNVERIFIED. In-sample by construction (D-029) - not independent evidence.",
        "base_distance_m": gate.base_distance_m.value,
        "location_mode": calls.location.mode,
        "median_error_m": mode.median_error_m.value,
        "sigma": mode.sigma.value,
        "target_coverage": TARGET,
        "selected_k": gate.k_location_error.value,
        "smallest_k_reaching_target": passing[0] if passing else None,
        **cov,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    write_meta(
        OUT,
        configs=[CONFIG_DIR / "fusion.yaml", CONFIG_DIR / "calls.yaml"],
        synthetic=True,
        seeds={"draws": args.seed},
    )
    for k, cc, cm in zip(K_GRID, cov["call_call"], cov["call_camera"], strict=True):
        log.info("k %.1f: call-call %.4f, call-camera %.4f", k, cc, cm)
    log.info("selected k=%.1f; smallest reaching %.0f%% is %s", report["selected_k"],
             TARGET * 100, report["smallest_k_reaching_target"])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
