"""Triage accuracy on the synthetic caller templates (a tool check, NOT a research result).

    uv run python -m resqcity.fusion.eval_triage [--llm]

Every type template of configs/call_templates.yaml is combined with no cue and with each cue,
in every language and in both phrasing sets (main = what the rule lexicon was written from,
held_out = unseen paraphrases). Reports type accuracy and cue recall / false-positive rate per
language and set, for the rule triager and (with --llm) the offline LLM triager.
Output: results/fusion/triage_eval_synthetic.json (synthetic).
"""

import argparse
import json
import logging
import time
from collections import defaultdict

from resqcity.common.config import CONFIG_DIR, PROJECT_ROOT
from resqcity.common.provenance import write_meta
from resqcity.common.requirements import load_requirements_config
from resqcity.common.schemas import Call, EventType, GeoPoint, Language
from resqcity.fusion.config import CUES, load_templates, load_triage_config
from resqcity.fusion.triage import CallTriager, LocalLLMTriager, RuleTriager

ROAD = "Laxmi Road"
N_VEHICLES = 3
OUT = PROJECT_ROOT / "results" / "fusion" / "triage_eval_synthetic.json"
TS = "2026-02-03T10:00:00+05:30"


def items() -> list[dict]:
    """(text, language, set, true type, true cue or None)."""
    tpl = load_templates()
    out = []
    for language, lang in tpl.languages.items():
        v1, v2 = lang.vehicles["car"], lang.vehicles["truck"]
        loc = lang.location[0].format(road=ROAD)
        for set_name, phrases in (("main", lang.main), ("held_out", lang.held_out)):
            for etype, templates in phrases.types.items():
                for template in templates:
                    base = template.format(loc=loc, v1=v1, v2=v2, n=N_VEHICLES)
                    for cue in (None, *CUES):
                        text = base
                        if cue is not None:
                            text += " " + phrases.cues[cue][0].format(v1=v1, n=N_VEHICLES)
                        out.append(
                            {"text": text, "language": language, "set": set_name,
                             "type": etype, "cue": cue}
                        )  # fmt: skip
    return out


def cue_present(result_features: object, cue: str) -> bool:
    if cue == "multi_vehicle":
        return result_features.num_vehicles >= N_VEHICLES
    return bool(getattr(result_features, cue))


def evaluate(triager: CallTriager, data: list[dict]) -> dict:
    stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    started = time.perf_counter()
    for k, it in enumerate(data):
        call = Call(
            id=f"CALL-E-{k:05d}",
            ts=TS,
            reported_location=GeoPoint(lat=18.5, lon=73.85),
            location_error_m=0.0,
            caller_text=it["text"],
            language=it["language"],
        )
        r = triager.triage(call)
        s = stats[(it["language"].value, it["set"])]
        s["n"] += 1
        s["type_correct"] += r.type == it["type"]
        if it["cue"] is not None and not (
            it["cue"] == "fire_visible" and it["type"] == EventType.FIRE_SMOKE
        ):
            s["cue_n"] += 1
            s["cue_found"] += cue_present(r.features, it["cue"])
        if it["cue"] is None and it["type"] == EventType.ACCIDENT:
            s["clean_n"] += 1
            s["clean_false_cue"] += any(
                cue_present(r.features, c) for c in ("person_down", "overturned", "fire_visible")
            )
    out = {}
    for (language, set_name), s in sorted(stats.items()):
        out[f"{language}/{set_name}"] = {
            "items": int(s["n"]),
            "type_accuracy": s["type_correct"] / s["n"],
            "cue_recall": s["cue_found"] / s["cue_n"] if s["cue_n"] else None,
            "false_cue_rate_on_clean_accidents": (
                s["clean_false_cue"] / s["clean_n"] if s["clean_n"] else None
            ),
        }
    out["seconds_per_call"] = (time.perf_counter() - started) / len(data)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="also evaluate the offline LLM")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg, req = load_triage_config(), load_requirements_config()
    data = items()
    report: dict[str, object] = {
        "status": "synthetic tool check (templates and lexicon by the same authors); not a "
        "research result",
        "items": len(data),
        "languages": sorted(lang.value for lang in Language),
        "rules": evaluate(RuleTriager(cfg, req), data),
    }
    if args.llm:
        llm = LocalLLMTriager(cfg, req)
        report["llm"] = {"model": cfg.llm.model, "revision": cfg.llm.revision}
        report["llm"] |= evaluate(llm, data)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    configs = [CONFIG_DIR / "call_templates.yaml", CONFIG_DIR / "triage.yaml"]
    write_meta(OUT, configs=configs, synthetic=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
