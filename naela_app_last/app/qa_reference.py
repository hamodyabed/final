"""Authoritative question / acceptable-answer reference for Gemini scoring.

This module loads :file:`data/extracted_questions_answers.json` and exposes a
clean lookup keyed by the NCP question identifier the rest of the app already
uses (e.g. ``ncp_u1_setting_literal_correct``).

The JSON file uses ``stop_number`` 1..7 and a ``concept`` label per stop, with
``primary_question`` + ``acceptable_answers`` and optional
``follow_up_question`` + ``follow_up_acceptable_answers``. Each NCP key in
``extracted_keys.json`` matches one of those two slots. The mapping below was
extracted by inspection of both files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# NCP key  →  (stop_number, "primary" | "followup")
# ---------------------------------------------------------------------------
NCP_KEY_TO_QA: Dict[str, tuple[int, str]] = {
    # Stop 1 — Setting
    "ncp_u1_setting_literal_correct":           (1, "primary"),
    "ncp_u1_setting_description_response":      (1, "followup"),
    # Stop 2 — Initiating Event
    "ncp_u2_initiating_event_literal_correct":  (2, "primary"),
    "ncp_u2_initiating_event_causal_correct":   (2, "followup"),
    # Stop 3 — Internal Response (emotion)
    "ncp_u3_emotion_identification_correct":    (3, "primary"),
    "ncp_u3_emotion_explanation_correct":       (3, "followup"),
    # Stop 4 — Plan
    "ncp_u4_plan_inference_correct":            (4, "primary"),
    "ncp_u4_intention_response_correct":        (4, "followup"),
    # Stop 5 — Attempt
    "ncp_u5_attempt_evaluation_correct":        (5, "primary"),
    "ncp_u5_attempt_causal_correct":            (5, "followup"),
    # Stop 6 — Climax + Inference
    "ncp_u6_identity_inference_correct":        (6, "primary"),
    "ncp_u6_evidence_based_inference_score":    (6, "followup"),
    # Stop 7 — Consequence + Resolution
    "ncp_u7_consequence_action_correct":        (7, "primary"),
    "ncp_u7_resolution_speech_correct":         (7, "followup"),
}


@dataclass
class QAReference:
    """One canonical question + acceptable-answer list for an NCP item."""

    key: str
    stop_number: int
    concept: str
    slot: str  # "primary" or "followup"
    question_text: str
    acceptable_answers: List[str] = field(default_factory=list)
    # Optional alternate phrasing of the same question used in the video.
    alternative_question: str = ""
    purpose: str = ""


# ---------------------------------------------------------------------------
# Lazy-loaded cache so we hit the file system exactly once per process.
# ---------------------------------------------------------------------------
_CACHE: Optional[Dict[str, QAReference]] = None


def _load_raw(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_index(raw: dict) -> Dict[str, QAReference]:
    """Build {ncp_key: QAReference} from the raw JSON dict."""
    by_stop: Dict[int, dict] = {
        item["stop_number"]: item for item in raw.get("questions_and_answers", [])
    }
    index: Dict[str, QAReference] = {}
    for key, (stop_number, slot) in NCP_KEY_TO_QA.items():
        item = by_stop.get(stop_number)
        if item is None:
            continue
        if slot == "primary":
            question = item.get("primary_question", "")
            answers = list(item.get("acceptable_answers", []) or [])
        else:
            question = item.get("follow_up_question", "")
            answers = list(item.get("follow_up_acceptable_answers", []) or [])
            # Some stops (e.g. stop 1) describe the follow-up as
            # "alternative_question_in_video" without a dedicated list of
            # follow-up answers. Reuse the primary acceptable_answers as
            # the reference set so scoring isn't completely empty.
            if not question and item.get("alternative_question_in_video"):
                question = item["alternative_question_in_video"]
            if not answers:
                answers = list(item.get("acceptable_answers", []) or [])
        index[key] = QAReference(
            key=key,
            stop_number=stop_number,
            concept=item.get("concept", ""),
            slot=slot,
            question_text=question,
            acceptable_answers=answers,
            alternative_question=item.get("alternative_question_in_video", "") if slot == "primary" else "",
            purpose=item.get("purpose", "") if slot == "primary" else "",
        )
    return index


def load_qa_reference(json_path: Path) -> Dict[str, QAReference]:
    """Return (and cache) the full ``{ncp_key: QAReference}`` mapping."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not json_path.is_file():
        _CACHE = {}
        return _CACHE
    raw = _load_raw(json_path)
    _CACHE = _build_index(raw)
    return _CACHE


def get_qa(ncp_key: str, json_path: Optional[Path] = None) -> Optional[QAReference]:
    """Look up the question + acceptable answers for one NCP key."""
    if json_path is None:
        # Default to data/extracted_questions_answers.json relative to project.
        json_path = Path(__file__).resolve().parent.parent / "data" / "extracted_questions_answers.json"
    return load_qa_reference(json_path).get(ncp_key)


if __name__ == "__main__":  # pragma: no cover — manual sanity check
    ref_path = Path(__file__).resolve().parent.parent / "data" / "extracted_questions_answers.json"
    idx = load_qa_reference(ref_path)
    for k, ref in idx.items():
        print(f"{k}  stop={ref.stop_number}  slot={ref.slot}")
        print(f"  Q: {ref.question_text}")
        print(f"  A: {ref.acceptable_answers}")
        print()
