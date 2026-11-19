"""Publication-quality figures for the Phase 10 experiment results.

    uv run python -m resqcity.experiments.figures.generate

Generates colourblind-safe PDF + PNG figures from ``results/experiments/``. Each function
produces one figure and saves it to ``results/figures/``.

Style: matplotlib with the ``tableau-colorblind10`` palette, 300 DPI PNGs, vector PDFs.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resqcity.common.config import PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.sim.config import POLICIES

log = logging.getLogger(__name__)

EXP_DIR = PROJECT_ROOT / "results" / "experiments"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
SCENARIOS = ("normal", "monsoon", "ganeshotsav")

# colourblind-safe palette (Wong 2011 / tableau-colorblind10)
COLORS = {
    "B0": "#006BA4",
    "B1": "#FF800E",
    "B2": "#ABABAB",
    "B3": "#595959",
    "FULL": "#C85200",
}
POLICY_ORDER = list(POLICIES)


def _setup() -> None:
    """Configure matplotlib for publication."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.figsize": (7, 4.5),
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as PDF + PNG with provenance."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        write_meta(path, configs=[], synthetic=True)
    plt.close(fig)
    log.info("Saved %s", name)


def _load_summaries(scenario: str) -> dict[str, dict]:
    """Load all policy summaries for a scenario."""
    out = {}
    for policy in POLICY_ORDER:
        path = EXP_DIR / f"summary_{scenario}_{policy}.json"
        if path.exists():
            out[policy] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# Figure 1: Primary outcome comparison (bar chart)
# ---------------------------------------------------------------------------


def fig_primary_comparison() -> None:
    """Bar chart of primary outcome (mean ± std) across policies, per scenario."""
    _setup()
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(12, 5), sharey=True)
    if len(SCENARIOS) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, SCENARIOS, strict=False):
        summaries = _load_summaries(scenario)
        policies = [p for p in POLICY_ORDER if p in summaries]
        means = [summaries[p].get("primary_mean_s") or 0 for p in policies]
        colors = [COLORS.get(p, "#999999") for p in policies]

        bars = ax.bar(policies, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(scenario.title())
        ax.set_ylabel("Mean time to last required arrival (s)" if ax == axes[0] else "")
        ax.set_xlabel("Policy")

        # value labels
        for bar, val in zip(bars, means, strict=False):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("RQ2: Time to Last Required Arrival by Policy", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "primary_comparison")


# ---------------------------------------------------------------------------
# Figure 2: Response time CDF (per scenario)
# ---------------------------------------------------------------------------


def fig_response_cdf() -> None:
    """Empirical CDF of primary outcome for each policy, normal scenario."""
    _setup()
    scenario = "normal"
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in POLICY_ORDER:
        path = EXP_DIR / f"metrics_{scenario}_{policy}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        real = df[(~df["in_warmup"]) & (~df["is_false_alarm"])]
        vals = real["time_to_last_required_arrival_s"].dropna().sort_values()
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals.values / 60, cdf, label=policy, color=COLORS.get(policy), linewidth=1.5)

    ax.axvline(8, color="red", linestyle="--", alpha=0.5, label="8 min target")
    ax.axvline(15, color="orange", linestyle="--", alpha=0.5, label="15 min target")
    ax.set_xlabel("Time to last required arrival (minutes)")
    ax.set_ylabel("Cumulative fraction of incidents")
    ax.set_title("Response Time CDF — Normal Scenario")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 30)
    fig.tight_layout()
    _save(fig, "response_cdf_normal")


# ---------------------------------------------------------------------------
# Figure 3: Equity — ward-level p90 distribution
# ---------------------------------------------------------------------------


def fig_ward_equity() -> None:
    """Box plot of ward-level p90 response times for B0 vs FULL, normal scenario."""
    _setup()
    scenario = "normal"
    fig, ax = plt.subplots(figsize=(8, 5))

    ward_data = {}
    for policy in ("B0", "FULL"):
        path = EXP_DIR / f"metrics_{scenario}_{policy}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        real = df[(~df["in_warmup"]) & (~df["is_false_alarm"])]
        ward_p90 = (
            real.dropna(subset=["time_to_last_required_arrival_s", "ward_id"])
            .groupby("ward_id")["time_to_last_required_arrival_s"]
            .quantile(0.9)
        )
        ward_data[policy] = ward_p90.values / 60  # minutes

    if ward_data:
        labels = list(ward_data.keys())
        data = [ward_data[k] for k in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
        for patch, policy in zip(bp["boxes"], labels, strict=False):
            patch.set_facecolor(COLORS.get(policy, "#999999"))
            patch.set_alpha(0.7)

    ax.set_ylabel("Ward-level p90 response time (minutes)")
    ax.set_title("RQ4: Ward Equity — B0 vs FULL (Normal Scenario)")
    fig.tight_layout()
    _save(fig, "ward_equity_normal")


# ---------------------------------------------------------------------------
# Figure 4: Utilisation by agency and policy
# ---------------------------------------------------------------------------


def fig_utilisation() -> None:
    """Grouped bar chart of fleet utilisation by agency across policies."""
    _setup()
    scenario = "normal"
    summaries = _load_summaries(scenario)
    policies = [p for p in POLICY_ORDER if p in summaries]
    agencies = ["ambulance", "police", "fire"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(policies))
    width = 0.25

    for i, agency in enumerate(agencies):
        vals = []
        for p in policies:
            util = summaries[p].get("utilisation", {})
            vals.append((util.get(agency) or 0) * 100)
        ax.bar(x + i * width, vals, width, label=agency.title(), alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Utilisation (%)")
    ax.set_title("Fleet Utilisation by Agency and Policy (Normal Scenario)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "utilisation_normal")


# ---------------------------------------------------------------------------
# Figure 5: Detection pathway breakdown
# ---------------------------------------------------------------------------


def fig_detection_breakdown() -> None:
    """Stacked bar of detection delay vs dispatch delay for each policy."""
    _setup()
    scenario = "normal"
    summaries = _load_summaries(scenario)
    policies = [p for p in POLICY_ORDER if p in summaries]

    fig, ax = plt.subplots(figsize=(8, 5))

    det_delays = [(summaries[p].get("detection_delay_mean_s") or 0) / 60 for p in policies]
    disp_delays = [(summaries[p].get("detection_to_dispatch_mean_s") or 0) / 60 for p in policies]

    x = np.arange(len(policies))
    ax.bar(x, det_delays, label="Detection delay", color="#006BA4", alpha=0.85)
    ax.bar(x, disp_delays, bottom=det_delays, label="Dispatch delay", color="#FF800E", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Time (minutes)")
    ax.set_title("RQ1: Detection + Dispatch Delay Breakdown (Normal Scenario)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "detection_breakdown_normal")


# ---------------------------------------------------------------------------
# generate all
# ---------------------------------------------------------------------------


def generate_all() -> None:
    """Generate all publication figures."""
    fig_primary_comparison()
    fig_response_cdf()
    fig_ward_equity()
    fig_utilisation()
    fig_detection_breakdown()
    log.info("All figures generated in %s", FIG_DIR)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
