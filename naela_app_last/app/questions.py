"""
Builds the list of questions asked to the child during the story flow.

The Narrative Comprehension Protocol (NCP) section of `extracted_keys.json` is
the natural source of the questions the child should answer while watching the
video / looking at the story. Each entry contains both the literal question
("سؤال الفهم") and the follow-up question ("سؤال المتابعة").

We parse those out so the UI can present them one at a time and record an
audio answer for each.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any


# Markers used inside the Arabic descriptions in extracted_keys.json
_PRIMARY_MARKER = "سؤال الفهم:"
_FOLLOWUP_MARKER = "سؤال المتابعة:"


@dataclass
class Question:
    """A single question the child must answer."""

    key: str                 # JSON key, e.g. "ncp_u1_setting_literal_correct"
    section: str             # JSON section title (Arabic)
    order: int               # 1-based order in the flow
    kind: str                # "primary" | "followup" | "macro"
    text: str                # The Arabic question text shown to the child
    full_description: str    # The original JSON value (kept for the Excel header)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_question_text(description: str) -> str:
    """Pull the question sentence out of an NCP description.

    NCP values look like:
        "سؤال الفهم:  من في الصورة؟يقيس فهم الطفل ..."
        "سؤال المتابعة:  اوصف لي ... كيف شكلهم؟يقيس قدرة ..."
    We want everything from the marker up to (and including) the first '؟'.
    """
    for marker in (_PRIMARY_MARKER, _FOLLOWUP_MARKER):
        if description.startswith(marker):
            tail = description[len(marker):].strip()
            # Take up to the first Arabic question mark (؟) or ASCII '?'.
            match = re.search(r"[؟?]", tail)
            if match:
                return tail[: match.end()].strip()
            return tail.split("يقيس", 1)[0].strip()
    # Macrostructure prompt – use the description as the displayed prompt.
    return description.strip()


def _kind_for_key(key: str, description: str) -> str:
    if description.startswith(_PRIMARY_MARKER):
        return "primary"
    if description.startswith(_FOLLOWUP_MARKER):
        return "followup"
    return "macro"


def load_questions(json_path: Path) -> List[Question]:
    """Load every NCP question (in order) from the keys file.

    Only the NCP section is interactive (it contains literal questions asked
    during the story). The macrostructure section is intentionally excluded
    from the interactive flow because those are scored from the child's free
    retelling, not from a direct question.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    questions: List[Question] = []
    order = 0
    for section_name, section in data.items():
        # The NCP section is the only one whose values are framed as questions.
        if "Narrative Comprehension Protocol" not in section.get("topic", ""):
            continue
        for key, value in section.items():
            if key == "topic" or not isinstance(value, str):
                continue
            if not (value.startswith(_PRIMARY_MARKER) or value.startswith(_FOLLOWUP_MARKER)):
                continue
            order += 1
            questions.append(
                Question(
                    key=key,
                    section=section_name,
                    order=order,
                    kind=_kind_for_key(key, value),
                    text=_extract_question_text(value),
                    full_description=value,
                )
            )
    return questions


def load_all_keys(json_path: Path) -> List[Dict[str, str]]:
    """Flatten the whole JSON into a list of {section, key, description}.

    Used by the Excel exporter to build one column per JSON key.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    rows: List[Dict[str, str]] = []
    for section_name, section in data.items():
        topic = section.get("topic", "")
        for key, value in section.items():
            if key == "topic":
                continue
            rows.append(
                {
                    "section": section_name,
                    "topic": topic,
                    "key": key,
                    "description": value if isinstance(value, str) else str(value),
                }
            )
    return rows


if __name__ == "__main__":
    # Quick manual sanity check.
    here = Path(__file__).resolve().parent.parent
    qs = load_questions(here / "data" / "extracted_keys.json")
    for q in qs:
        print(f"{q.order:2d} [{q.kind:8s}] {q.key}: {q.text}")
