"""Scoring helpers for the Naela narrative-assessment app.

This module centralises:

* the **scale** that applies to every JSON key in ``extracted_keys.json``
  (binary 0/1 vs qualitative 0/1/2),
* the **mapping from NCP question keys to macrostructure score keys**
  (so an answer to ``ncp_u1_setting_literal_correct`` contributes to
  ``setting_score``),
* the **aggregation logic** that turns the per-question scores entered by
  the examiner into the values written into the Excel columns
  (``setting_score``, ``characters_score`` ..., ``macro_total_score``).

The general coding rule (provided by the user):

    قاعدة الترميز العامة:
        تُرمّز البنود الثنائية كالتالي: 0 = غير موجود أو غير صحيح،
        1 = موجود أو صحيح.
        وتُرمّز البنود النوعية أو المركبة كالتالي: 0 = غائب،
        1 = جزئي أو محدود، 2 = واضح أو مكتمل.

The scale for each key follows that rule:

* keys ending with ``_correct``, ``_accuracy``, ``_response``  → binary 0/1
* keys ending with ``_score``                                  → qualitative 0/1/2
* totals (``*_total_score`` / ``macro_*``)                      → numeric sum
* everything else (counts, ratios, free text)                  → left blank /
  filled later by the analyst
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------
SCALE_BINARY = "binary"          # 0 / 1
SCALE_QUALITATIVE = "qualitative"  # 0 / 1 / 2
SCALE_NUMERIC = "numeric"         # aggregated total (read-only)
SCALE_NONE = "none"               # not scored by the examiner


def scale_for_key(key: str) -> str:
    """Return the scoring scale that applies to a JSON key."""
    k = key.lower()
    # Aggregated totals are computed, not entered.
    if k.endswith("_total_score") or k == "macro_total_score" or "total_score" in k:
        return SCALE_NUMERIC
    # Qualitative 0/1/2 — macrostructure + composite scores.
    if k.endswith("_score"):
        return SCALE_QUALITATIVE
    # Binary 0/1 — comprehension / response accuracy items.
    if (
        k.endswith("_correct")
        or k.endswith("_accuracy")
        or k.endswith("_response")
        or k.endswith("_speech_correct")
    ):
        return SCALE_BINARY
    return SCALE_NONE


def allowed_values(scale: str) -> List[int]:
    """Return the allowed integer values for a scale."""
    if scale == SCALE_BINARY:
        return [0, 1]
    if scale == SCALE_QUALITATIVE:
        return [0, 1, 2]
    return []


def scale_label_ar(scale: str) -> str:
    """Short Arabic label shown beside the scoring buttons."""
    if scale == SCALE_BINARY:
        return "ثنائي: 0 = غير صحيح، 1 = صحيح"
    if scale == SCALE_QUALITATIVE:
        return "نوعي: 0 = غائب، 1 = جزئي، 2 = واضح/مكتمل"
    if scale == SCALE_NUMERIC:
        return "مجموع رقمي (يُحسب آليًا)"
    return "—"


# ---------------------------------------------------------------------------
# NCP question key  →  macrostructure score key
# ---------------------------------------------------------------------------
# The macrostructure score columns (Section 2) are not asked as questions to
# the child, but the NCP comprehension questions (Section 3) probe exactly the
# same story elements. We aggregate the relevant NCP answers into each
# ``*_score`` column using the maximum value across the linked questions
# (a child who answers either the literal *or* the follow-up correctly is
# credited for that element; both correct = full credit on 0/1/2 scale).
NCP_TO_MACRO: Dict[str, str] = {
    # Setting
    "ncp_u1_setting_literal_correct": "setting_score",
    "ncp_u1_setting_description_response": "setting_score",
    # Characters – the setting follow-up also names characters.
    "ncp_u1_setting_description_response": "characters_score",  # noqa: F601
    # Initiating event
    "ncp_u2_initiating_event_literal_correct": "initiating_event_score",
    "ncp_u2_initiating_event_causal_correct": "initiating_event_score",
    # Internal response (emotion)
    "ncp_u3_emotion_identification_correct": "internal_response_score",
    "ncp_u3_emotion_explanation_correct": "internal_response_score",
    # Plan
    "ncp_u4_plan_inference_correct": "plan_score",
    "ncp_u4_intention_response_correct": "plan_score",
    # Attempt
    "ncp_u5_attempt_evaluation_correct": "attempt_score",
    "ncp_u5_attempt_causal_correct": "attempt_score",
    # Turning point / identity inference
    "ncp_u6_identity_inference_correct": "turning_point_inference_score",
    "ncp_u6_evidence_based_inference_score": "turning_point_inference_score",
    # Consequence
    "ncp_u7_consequence_action_correct": "consequence_score",
    # Resolution
    "ncp_u7_resolution_speech_correct": "resolution_score",
}

# Because ``ncp_u1_setting_description_response`` contributes to TWO macro
# columns (setting + characters), we store the mapping as a list-of-targets.
def _build_multi_map() -> Dict[str, List[str]]:
    multi: Dict[str, List[str]] = {}
    # Repeat the literal mapping so both targets are captured.
    pairs = [
        ("ncp_u1_setting_literal_correct", "setting_score"),
        ("ncp_u1_setting_description_response", "setting_score"),
        ("ncp_u1_setting_description_response", "characters_score"),
        ("ncp_u2_initiating_event_literal_correct", "initiating_event_score"),
        ("ncp_u2_initiating_event_causal_correct", "initiating_event_score"),
        ("ncp_u3_emotion_identification_correct", "internal_response_score"),
        ("ncp_u3_emotion_explanation_correct", "internal_response_score"),
        ("ncp_u3_emotion_identification_correct", "emotion_identification_accuracy"),
        ("ncp_u3_emotion_explanation_correct", "emotion_explanation_accuracy"),
        ("ncp_u4_plan_inference_correct", "plan_score"),
        ("ncp_u4_intention_response_correct", "plan_score"),
        ("ncp_u5_attempt_evaluation_correct", "attempt_score"),
        ("ncp_u5_attempt_causal_correct", "attempt_score"),
        ("ncp_u6_identity_inference_correct", "turning_point_inference_score"),
        ("ncp_u6_evidence_based_inference_score", "turning_point_inference_score"),
        ("ncp_u7_consequence_action_correct", "consequence_score"),
        ("ncp_u7_resolution_speech_correct", "resolution_score"),
    ]
    for src, dst in pairs:
        multi.setdefault(src, []).append(dst)
    return multi


NCP_TO_MACRO_MULTI: Dict[str, List[str]] = _build_multi_map()


# Macrostructure score columns that are computed by summing other columns.
MACRO_SCORE_COLUMNS: List[str] = [
    "setting_score",
    "characters_score",
    "initiating_event_score",
    "internal_response_score",
    "plan_score",
    "attempt_score",
    "turning_point_inference_score",
    "consequence_score",
    "resolution_score",
    "coherence_score",
    "conclusion_or_meta_ending_score",
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_answers_to_columns(
    answers: Dict[str, dict],
) -> Dict[str, object]:
    """Turn the per-question scores into the values written to Excel.

    Parameters
    ----------
    answers:
        ``{question_key: {"score": int | None, ...}}`` as produced by the
        recording flow in :mod:`app.main_window`.

    Returns
    -------
    A dict ``{column_key: value}`` ready to pass to
    :func:`app.excel_export.append_session`. Includes:

    * Every NCP key that has a numeric ``score`` (written verbatim).
    * Every macrostructure ``*_score`` column derived from the related NCP
      answers (max of contributing scores; a binary 1 is treated as 2
      on the 0/1/2 scale to mean "fully present").
    * ``macro_total_score`` = sum of the macrostructure score columns.
    """
    out: Dict[str, object] = {}

    # 1) NCP / direct keys — write each numeric score straight through.
    macro_buckets: Dict[str, List[int]] = {col: [] for col in MACRO_SCORE_COLUMNS}
    macro_buckets["emotion_identification_accuracy"] = []
    macro_buckets["emotion_explanation_accuracy"] = []

    for q_key, payload in answers.items():
        score = payload.get("score") if isinstance(payload, dict) else None
        if score is None:
            continue
        # Always write the per-question column.
        out[q_key] = score

        # Fan out into the macrostructure columns it contributes to.
        targets = NCP_TO_MACRO_MULTI.get(q_key, [])
        for target in targets:
            target_scale = scale_for_key(target)
            # Promote binary 1 → 2 so it can saturate a 0/1/2 column.
            if target_scale == SCALE_QUALITATIVE and scale_for_key(q_key) == SCALE_BINARY:
                contrib = 2 if score == 1 else 0
            else:
                contrib = int(score)
            macro_buckets.setdefault(target, []).append(contrib)

    # 2) Macrostructure score columns — max of contributing NCP answers.
    for col, values in macro_buckets.items():
        if not values:
            continue
        if scale_for_key(col) == SCALE_BINARY:
            out[col] = 1 if max(values) >= 1 else 0
        else:
            out[col] = max(values)

    # 3) macro_total_score = sum of macrostructure score columns we filled in.
    macro_values = [
        int(out[c]) for c in MACRO_SCORE_COLUMNS if isinstance(out.get(c), int)
    ]
    if macro_values:
        out["macro_total_score"] = sum(macro_values)

    return out


def coerce_score(value: Optional[object], scale: str) -> Optional[int]:
    """Validate that ``value`` is a legal score for the given ``scale``."""
    if value is None or value == "":
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue in allowed_values(scale):
        return ivalue
    return None
