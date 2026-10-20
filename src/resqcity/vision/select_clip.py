"""Pick the clip-classifier run by its VALIDATION within-video AUROC (never the test split).

    uv run python -m resqcity.vision.select_clip clip_s0_t0 clip_s03_t1 clip_s1_t1

Writes ``results/vision/clip_selection.csv`` (one row per run: best epoch and val AUROCs) and
prints the selected run id.
"""

import argparse

import pandas as pd

from resqcity.common.config import CONFIG_DIR, load_base_config
from resqcity.common.provenance import write_meta

CRITERION = "val_auroc_within"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    results = load_base_config().paths.resolve("results") / "vision"
    rows = []
    for run in args.runs:
        log = pd.read_csv(results / run / "train_log.csv")
        best = log.loc[log[CRITERION].idxmax()]
        rows.append(
            {
                "run_id": run,
                "best_epoch": int(best["epoch"]),
                CRITERION: float(best[CRITERION]),
                "val_auroc_all": float(best["val_auroc_all"]),
            }
        )
    table = pd.DataFrame(rows).sort_values(CRITERION, ascending=False, ignore_index=True)
    table["selected"] = table.index == 0
    out = results / "clip_selection.csv"
    table.to_csv(out, index=False, lineterminator="\n")
    write_meta(
        out,
        configs=[
            CONFIG_DIR / "vision.yaml",
            *sorted((CONFIG_DIR / "vision_sweeps").glob("*.yaml")),
        ],
        synthetic=False,
        extra={"criterion": CRITERION, "split": "val"},
    )
    print(table.to_string(index=False))
    print(table.iloc[0]["run_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
