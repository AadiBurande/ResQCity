"""Call triage: caller text -> incident type, severity cues, severity and confidence (phase 4).

``RuleTriager`` (keyword lexicon per language, configs/triage.yaml) is the baseline used by the
simulator and the experiments. ``LocalLLMTriager`` is optional and OFFLINE (D-028): a model from
the local Hugging Face cache, pinned revision, 4-bit, deterministic decoding; the caller's text
never leaves the machine. It is off by default and never needed by tests or experiments; if its
answer cannot be parsed it falls back to the rules and says so.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from resqcity.common.requirements import (
    RequirementsConfig,
    required_services,
    severity_class,
    severity_score,
)
from resqcity.common.schemas import Call, EventType, Language, ServiceCounts, SeverityFeatures
from resqcity.fusion.config import RuleLexicon, TriageConfig

log = logging.getLogger(__name__)
VEHICLE_CLASSES = ("car", "motorcycle", "auto_rickshaw", "truck", "bus")
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class TriageResult:
    type: EventType
    features: SeverityFeatures
    severity: float
    severity_class: str
    requirements: ServiceCounts
    confidence: float
    method: str  # "rules", "llm" or "llm_fallback_rules"


class CallTriager(Protocol):
    def triage(self, call: Call) -> TriageResult: ...


def _result(
    etype: EventType,
    features: SeverityFeatures,
    confidence: float,
    method: str,
    requirements: RequirementsConfig,
) -> TriageResult:
    score = severity_score(etype, features, requirements.severity)
    return TriageResult(
        type=etype,
        features=features,
        severity=score,
        severity_class=severity_class(score, requirements.severity),
        requirements=required_services(etype, features, requirements),
        confidence=confidence,
        method=method,
    )


def triaged(call: Call, result: TriageResult) -> Call:
    """The call with its triage fields filled in."""
    return call.model_copy(
        update={
            "triaged_type": result.type,
            "triaged_severity": result.severity,
            "triage_confidence": result.confidence,
        }
    )


class RuleTriager:
    def __init__(self, cfg: TriageConfig, requirements: RequirementsConfig) -> None:
        self.cfg, self.requirements = cfg.rules, requirements

    @staticmethod
    def _found(text: str, word: str, language: Language) -> list[str]:
        if language == Language.ENGLISH:  # word boundaries work for Latin script only
            return re.findall(rf"\b{re.escape(word)}\b", text)
        return [word] * text.count(word)

    def _cues(self, text: str, lex: RuleLexicon, language: Language) -> dict[str, bool]:
        return {
            cue: any(self._found(text, w, language) for w in words)
            for cue, words in lex.cues.items()
        }

    def triage(self, call: Call) -> TriageResult:
        language = call.language
        lex = self.cfg.languages[language]
        text = call.caller_text.lower() if language == Language.ENGLISH else call.caller_text
        etype = next(
            (
                t
                for t in self.cfg.type_priority
                if any(self._found(text, w, language) for w in lex.types.get(t, []))
            ),
            None,
        )
        confidence = self.cfg.confidence_matched.value
        if etype is None:
            etype, confidence = self.cfg.fallback_type, self.cfg.confidence_fallback.value
        vehicles = [
            cls for word, cls in lex.vehicles.items() for _ in self._found(text, word, language)
        ]
        count = re.search(lex.vehicle_count_pattern, text)
        n = max(int(count.group(1)) if count else 0, len(vehicles))
        if etype == EventType.ACCIDENT:
            n = max(n, 1)
        cues = self._cues(text, lex, language)
        features = SeverityFeatures(
            vehicle_classes=vehicles,
            num_vehicles=n,
            person_down=cues.get("person_down", False),
            overturned=cues.get("overturned", False),
            fire_visible=cues.get("fire_visible", False) or etype == EventType.FIRE_SMOKE,
        )
        return _result(etype, features, confidence, "rules", self.requirements)


class LocalLLMTriager:
    """Offline LLM triage (optional). Loads the model on first use."""

    def __init__(
        self, cfg: TriageConfig, requirements: RequirementsConfig, device: str = "cuda:0"
    ) -> None:
        self.cfg, self.requirements, self.device = cfg, requirements, device
        self.fallback = RuleTriager(cfg, requirements)
        self._model = None
        self._processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        llm = self.cfg.llm
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=llm.quantization,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._processor = AutoProcessor.from_pretrained(
            llm.model,
            revision=llm.revision,
            local_files_only=True,  # offline only (D-028)
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            llm.model,
            revision=llm.revision,
            dtype=torch.float16,
            quantization_config=quant,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()

    def answer(self, text: str) -> str:
        import torch

        if self._model is None:
            self._load()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{self.cfg.llm.prompt}\n{text}"}],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.device)
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs, max_new_tokens=self.cfg.llm.max_new_tokens, do_sample=False
            )
        new = ids[0, inputs["input_ids"].shape[1] :]
        return self._processor.tokenizer.decode(new, skip_special_tokens=True).strip()

    def parse(self, raw: str) -> tuple[EventType, SeverityFeatures, float] | None:
        match = JSON_OBJECT.search(raw)
        if not match:
            return None
        try:
            d = json.loads(match.group(0))
            etype = EventType(str(d["type"]))
            vehicles = [str(v) for v in d.get("vehicles", []) if str(v) in VEHICLE_CLASSES]
            features = SeverityFeatures(
                vehicle_classes=vehicles,
                num_vehicles=max(int(d.get("num_vehicles", 0)), len(vehicles)),
                person_down=bool(d.get("person_down", False)),
                overturned=bool(d.get("overturned", False)),
                fire_visible=bool(d.get("fire_visible", False)),
            )
            confidence = min(max(float(d.get("confidence", 0.5)), 0.0), 1.0)
        except (KeyError, TypeError, ValueError):
            return None
        return etype, features, confidence

    def triage(self, call: Call) -> TriageResult:
        parsed = self.parse(self.answer(call.caller_text))
        if parsed is None:
            log.warning("LLM triage answer for %s could not be parsed; using rules", call.id)
            r = self.fallback.triage(call)
            return TriageResult(**{**r.__dict__, "method": "llm_fallback_rules"})
        etype, features, confidence = parsed
        return _result(etype, features, confidence, "llm", self.requirements)


def make_triager(
    cfg: TriageConfig, requirements: RequirementsConfig, use_llm: bool = False
) -> CallTriager:
    """Rules unless the LLM is explicitly requested AND enabled in configs/triage.yaml."""
    if use_llm and cfg.llm.enabled:
        return LocalLLMTriager(cfg, requirements)
    return RuleTriager(cfg, requirements)
