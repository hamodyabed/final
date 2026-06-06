"""Gemini-based automatic scorer for child audio answers.

The scorer takes a recorded WAV file (the child's spoken answer) plus the
question metadata and asks Google Gemini to assign a 0/1 or 0/1/2 score
following the project's coding rule:

    قاعدة الترميز العامة:
        تُرمّز البنود الثنائية كالتالي: 0 = غير موجود أو غير صحيح،
        1 = موجود أو صحيح.
        وتُرمّز البنود النوعية أو المركبة كالتالي: 0 = غائب،
        1 = جزئي أو محدود، 2 = واضح أو مكتمل.

The model is asked to return JSON ``{"score": <int>, "transcript": <str>,
"justification": <str>}`` so we can store the rationale next to the audio for
auditing.

Configuration
-------------
* ``GEMINI_API_KEY`` — required env var with your Google AI Studio key.
* ``GEMINI_MODEL`` — optional override; defaults to ``gemini-2.5-flash``.

The SDK is the official ``google-genai`` package
(``pip install google-genai``).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .scoring import (
    SCALE_BINARY,
    SCALE_QUALITATIVE,
    allowed_values,
    scale_for_key,
)


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-2.5-flash-lite"
# Free-tier daily quotas (RPD):
#   gemini-2.5-flash-lite  → 1000 RPD  ← we use this
#   gemini-2.5-flash       →   20 RPD  ← falling back to this would just
#                                        fail again immediately on heavy days.
# Therefore we deliberately do NOT chain a fallback model on the free tier.
# Examiners who need higher throughput should enable billing (paid tier
# removes the cap entirely).
FALLBACK_MODEL = ""


GENERAL_RULE = (
    "قاعدة الترميز العامة: تُرمّز البنود الثنائية كالتالي: "
    "0 = غير موجود أو غير صحيح، 1 = موجود أو صحيح. "
    "وتُرمّز البنود النوعية أو المركبة كالتالي: "
    "0 = غائب، 1 = جزئي أو محدود، 2 = واضح أو مكتمل. "
    "أي اختلاف في سلم الترميز يجب أن يُذكر صراحة داخل البند."
)


class GeminiScorerError(RuntimeError):
    """Raised when the Gemini scorer cannot be initialised or called."""


# Errors / HTTP statuses that are worth retrying with exponential backoff.
# Source: Google AI Studio docs — 429, 500, 502, 503, 504 are all transient.
_TRANSIENT_STATUSES = (429, 500, 502, 503, 504)
_TRANSIENT_TAGS = (
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a temporary Gemini API hiccup worth retrying."""
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _TRANSIENT_STATUSES:
        return True
    msg = str(exc).upper()
    return any(tag in msg for tag in _TRANSIENT_TAGS) or any(
        f" {s} " in f" {msg} " for s in (str(s) for s in _TRANSIENT_STATUSES)
    )


def _is_daily_quota_error(exc: BaseException) -> bool:
    """True for the 'free-tier requests per day exhausted' error specifically.

    Gemini's free tier enforces a low daily cap per model (e.g. 20 RPD for
    ``gemini-2.5-flash``). When that cap is hit we should *not* keep retrying –
    we should immediately switch to a fallback model with a higher cap.
    """
    msg = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in msg
        and "per day" in msg.lower()
        or "FreeTier" in msg
        or "free_tier_requests" in msg
    )


def _call_with_retry(fn, *, max_attempts: int = 4, base_delay: float = 1.5):
    """Run ``fn()`` with exponential backoff for transient Gemini errors.

    Re-raises the last exception once all attempts are exhausted. Daily-quota
    errors bypass retry and are raised immediately so callers can fall back to
    a different model.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_daily_quota_error(exc):
                # No point retrying – the daily cap won't reset for hours.
                raise
            if not _is_transient_error(exc) or attempt == max_attempts:
                raise
            # Backoff: 1.5s, 3s, 6s + jitter so multiple threads don't sync up.
            sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Transient Gemini error on attempt %d/%d (%s). "
                "Retrying in %.1fs…",
                attempt, max_attempts, exc, sleep_for,
            )
            time.sleep(sleep_for)
    # Should be unreachable, but keep mypy happy.
    if last_exc is not None:
        raise last_exc
    return None


@dataclass
class ScoreResult:
    """Result returned by :meth:`GeminiScorer.score_answer`."""

    score: Optional[int]
    transcript: str = ""
    justification: str = ""
    raw: str = ""


# Arabic markers that introduce the model-answer exemplars in
# data/extracted_keys.json. We extract the text after these markers so we
# can spotlight it for Gemini.
_EXPECTED_ANSWER_MARKERS = (
    "الإجابة الصحيحة:",
    "الإجابة المتوقعة:",
    "الجواب الصحيح:",
    "الإجابات الصحيحة:",
)
# Markers that signal the end of the expected-answer paragraph (next section
# of the rubric, like a note or the next question intro).
_EXPECTED_END_MARKERS = (
    "ملاحظة:",
    "ملاحظة للمبرمج",
    "ملاحظة:",
    "سؤال المتابعة:",
    "سؤال الفهم:",
    "يقيس",
)


def _extract_expected_answers(rubric_description: str) -> str:
    """Pull the «الإجابة الصحيحة / المتوقعة» block out of a rubric description.

    Returns the cleaned exemplar text, or an empty string if none of the
    markers appear (in which case the full description is fed to Gemini
    anyway, just without the extra spotlight).
    """
    if not rubric_description:
        return ""
    text = rubric_description
    # Find the earliest marker that appears.
    start = -1
    used_marker = ""
    for marker in _EXPECTED_ANSWER_MARKERS:
        pos = text.find(marker)
        if pos != -1 and (start == -1 or pos < start):
            start = pos
            used_marker = marker
    if start == -1:
        return ""
    body = text[start + len(used_marker):]
    # Trim at the next section marker if any.
    end_positions = [body.find(m) for m in _EXPECTED_END_MARKERS]
    end_positions = [p for p in end_positions if p != -1]
    if end_positions:
        body = body[: min(end_positions)]
    return body.strip(" .،\n")


def _extract_transcript_field(raw_text: str) -> str:
    """Robustly pull the ``transcript`` value out of Gemini's reply.

    Gemini occasionally returns malformed JSON (e.g. two attempts concatenated
    like ``{"transcript": "X"}\\nX"}``). We try, in order:

    1. ``json.loads`` on the whole reply.
    2. ``json.loads`` on the *first* balanced ``{ … }`` object.
    3. A direct regex extraction of ``"transcript": "…"`` so a usable string
       is returned even when the model's JSON is broken.

    Returns the trimmed transcript string, or ``""`` if nothing usable is found.
    """
    if not raw_text:
        return ""
    text = raw_text.strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 1) Try the whole thing.
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "transcript" in payload:
            return str(payload["transcript"]).strip()
    except json.JSONDecodeError:
        pass

    # 2) Try the first balanced JSON object (non-greedy, but balanced-ish).
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and "transcript" in payload:
                return str(payload["transcript"]).strip()
        except json.JSONDecodeError:
            pass

    # 3) Direct regex on the transcript field. This handles the case where
    #    the model returned malformed JSON like '{"transcript": "X"}\nX"}'.
    direct = re.search(
        r'"transcript"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text, flags=re.DOTALL,
    )
    if direct:
        # Decode JSON escapes inside the captured string.
        try:
            return json.loads(f'"{direct.group(1)}"').strip()
        except json.JSONDecodeError:
            return direct.group(1).strip()

    return ""


class GeminiScorer:
    """Thin wrapper around ``google.genai`` for audio-to-score grading."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not api_key:
            raise GeminiScorerError(
                "GEMINI_API_KEY غير معرّف. أضف مفتاح Gemini إلى متغيرات "
                "البيئة GEMINI_API_KEY قبل تشغيل التطبيق."
            )

        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:  # pragma: no cover - import error path
            raise GeminiScorerError(
                "حزمة google-genai غير مثبتة. شغّل:\n"
                "    pip install google-genai"
            ) from exc

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        # If the primary model gets daily-quota-exhausted we permanently
        # switch to FALLBACK_MODEL for the rest of the process lifetime so
        # subsequent calls don't waste a round-trip first.
        self._models_to_try = [self._model_name]
        if FALLBACK_MODEL and FALLBACK_MODEL != self._model_name:
            self._models_to_try.append(FALLBACK_MODEL)

    # ------------------------------------------------------------------
    def _generate(self, *, contents, config):
        """Call ``generate_content`` with retry + automatic model fallback.

        Tries each model in :attr:`_models_to_try` until one succeeds. When a
        model is daily-quota-exhausted it's dropped from the list so future
        calls go straight to the fallback.
        """
        last_exc: Optional[BaseException] = None
        for model in list(self._models_to_try):
            try:
                return _call_with_retry(
                    lambda m=model: self._client.models.generate_content(
                        model=m, contents=contents, config=config,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_daily_quota_error(exc) and len(self._models_to_try) > 1:
                    logger.warning(
                        "Daily quota reached for %s – falling back to next model.",
                        model,
                    )
                    self._models_to_try.remove(model)
                    continue
                raise
        if last_exc is not None:
            raise last_exc

    # ------------------------------------------------------------------
    def score_answer(
        self,
        audio_path: Path,
        question_key: str,
        question_text: str,
        rubric_description: str,
    ) -> ScoreResult:
        """Send one recorded answer to Gemini and parse the returned JSON."""
        audio_path = Path(audio_path)
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            return ScoreResult(score=None, justification="ملف الصوت غير موجود أو فارغ.")

        scale = scale_for_key(question_key)
        if scale not in (SCALE_BINARY, SCALE_QUALITATIVE):
            # Nothing to score for non-scored columns.
            return ScoreResult(score=None)

        allowed = allowed_values(scale)
        scale_instructions = (
            "هذا البند ثنائي: الإجابة المسموح بها هي 0 (غير صحيح/غير موجود) أو 1 (صحيح/موجود)."
            if scale == SCALE_BINARY
            else "هذا البند نوعي: الإجابة المسموح بها هي 0 (غائب) أو 1 (جزئي/محدود) أو 2 (واضح/مكتمل)."
        )

        prompt = self._build_prompt(
            question_key=question_key,
            question_text=question_text,
            rubric_description=rubric_description,
            scale_instructions=scale_instructions,
            allowed=allowed,
        )

        mime = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
        audio_bytes = audio_path.read_bytes()
        audio_part = self._types.Part.from_bytes(data=audio_bytes, mime_type=mime)

        try:
            response = self._generate(
                contents=[prompt, audio_part],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    # Plenty of headroom so long transcripts + justifications
                    # are never truncated mid-sentence.
                    max_output_tokens=4096,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini call failed for %s", question_key)
            raise GeminiScorerError(
                f"فشل الاتصال بـ Gemini أثناء تقييم «{question_key}»: {exc}"
            ) from exc

        text = (getattr(response, "text", "") or "").strip()
        return self._parse_response(text, allowed)

    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        question_key: str,
        question_text: str,
        rubric_description: str,
        scale_instructions: str,
        allowed: list,
    ) -> str:
        """Build the Arabic prompt that asks Gemini to grade the audio.

        Acts as an expert clinical-language analyst / researcher who:
        * transcribes the child's answer verbatim,
        * compares it to the structured ``acceptable_answers`` from
          ``data/extracted_questions_answers.json`` (preferred), falling
          back to the «الإجابة الصحيحة» exemplars from
          ``extracted_keys.json`` when the key isn't in the reference,
        * applies the project's general 0/1 or 0/1/2 coding rule,
        * returns a structured JSON with explicit reasoning + score.
        """
        from .qa_reference import get_qa  # lazy import keeps the test suite quick

        allowed_str = "، ".join(str(v) for v in allowed)

        # ---- Reference of acceptable answers ----------------------------
        qa = get_qa(question_key)
        concept_line = ""
        if qa is not None and qa.acceptable_answers:
            bullets = "\n".join(f"  - {a}" for a in qa.acceptable_answers)
            expected_block = (
                "الإجابات المقبولة (مرجع رسمي من بروتوكول التقييم):\n"
                f"{bullets}\n"
            )
            # Prefer the curated question phrasing if available.
            if qa.question_text:
                question_text = qa.question_text
            if qa.concept:
                concept_line = f"المفهوم السردي (Concept): {qa.concept}\n"
        else:
            expected_answers = _extract_expected_answers(rubric_description)
            expected_block = (
                f"الإجابات النموذجية المتوقعة (مستخرجة من بطاقة البند):\n"
                f"{expected_answers}\n"
                if expected_answers
                else "(لا توجد إجابات نموذجية صريحة — استند إلى وصف البند بالكامل.)\n"
            )

        return (
            "أنت باحث ومحلِّل لغوي إكلينيكي متخصص في تقييم عينات السرد "
            "القصصي لأطفال يتحدثون العربية الفلسطينية المحكية، وضمن بروتوكول "
            "Naela لتقييم الفهم السردي (NCP) والبنية الكلية (Macrostructure). "
            "سأرسل لك ملفًا صوتيًا فيه إجابة طفل على سؤال محدد. عليك أن تتصرف "
            "كمقيّم محايد، صارم في تطبيق المعايير، ولا تكافئ الطفل على إجابات "
            "غير وثيقة الصلة بالسؤال.\n\n"

            "## السؤال الموجَّه للطفل\n"
            f"{question_text}\n"
            f"{concept_line}\n"

            "## بطاقة البند (الوصف الكامل من ملف المعايير)\n"
            f"{rubric_description}\n\n"

            f"## ما يُتوقع أن يقوله الطفل\n{expected_block}\n"

            "## قاعدة الترميز العامة\n"
            f"{GENERAL_RULE}\n"
            f"{scale_instructions}\n"
            f"الدرجات المسموح بها لهذا البند: **{allowed_str}** فقط — لا تستخدم أي قيمة أخرى.\n\n"

            "## خطوات التحليل المطلوبة (نفّذها داخليًا قبل اختيار الدرجة)\n"
            "1. فرّغ كلام الطفل **حرفيًا وكاملًا** (transcript): من أول كلمة "
            "إلى آخر كلمة، دون اختصار ودون حذف الكلمات الأخيرة حتى لو كانت "
            "غير واضحة أو متكررة. لا تُجرِ تصحيحًا لغويًا ولا إعرابًا.\n"
            "2. قارن مفردات وأفكار التفريغ بـ **«الإجابات النموذجية المتوقعة»** "
            "أعلاه: حدّد المفاهيم/الكلمات المفتاحية التي ظهرت، وتلك التي غابت.\n"
            "3. قيّم درجة المطابقة وفق قاعدة الترميز:\n"
            "   • إذا كان البند ثنائيًا (0/1): 1 فقط عند ظهور الفكرة "
            "الرئيسية بوضوح وإن اختلفت الصياغة؛ 0 إذا كانت الإجابة غائبة أو "
            "غير ذات صلة أو خاطئة.\n"
            "   • إذا كان البند نوعيًا (0/1/2): 2 = ذكر واضح ومكتمل للفكرة "
            "كما في بطاقة البند؛ 1 = ذكر جزئي أو محدود أو غامض؛ 0 = غائب "
            "تمامًا أو لا علاقة له بالسؤال.\n"
            "4. كن متسامحًا مع لكنة الطفل ومع الأخطاء الصرفية الصغيرة "
            "(مثلًا: «بدّو» = «يريد»، «راح» = «ذهب»). لا تخصم نقاطًا "
            "بسبب الصياغة العامية ما دامت الفكرة المطلوبة موجودة.\n"
            "5. اكتب تبريرًا قصيرًا بالعربية يربط بين ما قاله الطفل فعلًا "
            "وبين المعايير، ويذكر الكلمات المفتاحية التي أثّرت في القرار.\n\n"

            "## شكل الإخراج المطلوب (JSON فقط، بدون أي نص آخر)\n"
            "{\n"
            "  \"transcript\": \"<نص حرفي كامل لما قاله الطفل>\",\n"
            "  \"matched_keywords\": [\"<كلمة/فكرة من إجابة الطفل تطابق بطاقة البند>\", ...],\n"
            "  \"missing_keywords\": [\"<كلمة/فكرة كانت متوقعة ولم تظهر>\", ...],\n"
            "  \"justification\": \"<تبرير عربي مختصر يربط الإجابة بالمعايير>\",\n"
            f"  \"score\": <عدد صحيح من {allowed_str}>\n"
            "}\n\n"

            "## حالات خاصة\n"
            "- إذا لم تسمع الطفل أو لم يُجب: score = 0، transcript = \"\"، "
            "وتبرير يوضح أنّ الإجابة لم تُسجَّل.\n"
            "- إذا أجاب الطفل بـ «لا أعرف» أو ما يماثله: score = 0 مع تبرير.\n"
            "- إذا كانت الإجابة قريبة من الصحيحة لكنها غامضة: "
            "اختر 1 على المقياس النوعي (لا 2)."
        )

    # ------------------------------------------------------------------
    def transcribe(self, audio_path: Path) -> str:
        """Return only the Arabic transcript of a recorded answer.

        Cheaper than full scoring – used to populate the live "answers table"
        in the UI right after a recording finishes. Returns an empty string
        if the model output is unparseable or the file is missing. Any
        underlying error is logged to stderr for diagnosis.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            logger.warning("transcribe: audio file missing or empty: %s", audio_path)
            return ""

        mime = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
        audio_bytes = audio_path.read_bytes()
        audio_part = self._types.Part.from_bytes(data=audio_bytes, mime_type=mime)

        prompt = (
            "هذا ملف صوتي لطفل يتحدث باللهجة الفلسطينية يجيب على سؤال. "
            "فرّغ كلامه حرفيًا وكاملًا إلى نص عربي، من أول كلمة إلى آخر كلمة، "
            "دون اختصار ودون تلخيص ودون حذف الكلمات الأخيرة. "
            "حافظ على الكلمات كما نطقها الطفل تمامًا، حتى لو كانت غير واضحة أو متكررة، "
            "ولا تقم بأي تصحيح لغوي أو إعراب. "
            "إذا لم تسمع أي كلام، أعد سلسلة فارغة. "
            "أرجع الإجابة بصيغة JSON صالحة فقط: {\"transcript\": \"النص الكامل هنا\"}"
        )
        try:
            response = self._generate(
                contents=[prompt, audio_part],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                    # Allow plenty of room so long answers aren't truncated.
                    max_output_tokens=4096,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gemini transcription failed for %s: %s: %s",
                audio_path.name, type(exc).__name__, exc,
            )
            # Re-raise so callers (e.g. _TranscribeThread) can surface a
            # diagnostic to the user instead of silently showing an empty box.
            raise GeminiScorerError(f"{type(exc).__name__}: {exc}") from exc

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return ""
        return _extract_transcript_field(text)

    # ------------------------------------------------------------------
    def _parse_response(self, text: str, allowed: list) -> ScoreResult:
        """Parse the model's JSON output into a :class:`ScoreResult`."""
        if not text:
            return ScoreResult(score=None, raw=text, justification="استجابة فارغة من Gemini.")

        # Strip optional ```json fences just in case.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find a JSON object inside the text as a fallback.
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return ScoreResult(
                    score=None,
                    raw=text,
                    justification="تعذّر تحويل ردّ Gemini إلى JSON.",
                )
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return ScoreResult(
                    score=None,
                    raw=text,
                    justification="JSON غير صالح في ردّ Gemini.",
                )

        score_value = payload.get("score")
        try:
            score_int = int(score_value)
        except (TypeError, ValueError):
            score_int = None

        if score_int is not None and score_int not in allowed:
            # Clamp into the allowed range to avoid writing junk.
            score_int = max(allowed[0], min(allowed[-1], score_int))

        # Compose a richer justification that also exposes the matched /
        # missing keywords the analyst can audit later.
        justification = str(payload.get("justification", "")).strip()
        matched = payload.get("matched_keywords") or []
        missing = payload.get("missing_keywords") or []
        if isinstance(matched, list) and matched:
            justification += "\nمفاتيح ظهرت: " + "، ".join(str(x) for x in matched)
        if isinstance(missing, list) and missing:
            justification += "\nمفاتيح غابت: " + "، ".join(str(x) for x in missing)

        return ScoreResult(
            score=score_int,
            transcript=str(payload.get("transcript", "")).strip(),
            justification=justification.strip(),
            raw=text,
        )


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------
def score_session_answers(
    answers: Dict[str, dict],
    questions_by_key: Dict[str, "object"],
    rubric_by_key: Dict[str, str],
    scorer: Optional[GeminiScorer] = None,
) -> Dict[str, ScoreResult]:
    """Score every recorded answer in ``answers`` using Gemini.

    Parameters
    ----------
    answers:
        ``{key: {"audio_path": str, "question_text": str, ...}}`` from
        :class:`app.main_window.MainWindow`.
    questions_by_key:
        Maps question key to its ``Question`` dataclass (for the prompt text).
    rubric_by_key:
        Maps every JSON key to its description (the raw value from
        ``extracted_keys.json``).
    scorer:
        Optional pre-built :class:`GeminiScorer`. A fresh one is created
        from environment variables when not supplied.
    """
    scorer = scorer or GeminiScorer()
    out: Dict[str, ScoreResult] = {}
    for key, payload in answers.items():
        audio = payload.get("audio_path", "")
        if not audio:
            continue
        question = questions_by_key.get(key)
        question_text = (
            getattr(question, "text", None) or payload.get("question_text", "")
        )
        rubric = rubric_by_key.get(key, "")
        try:
            result = scorer.score_answer(
                audio_path=Path(audio),
                question_key=key,
                question_text=question_text,
                rubric_description=rubric,
            )
        except GeminiScorerError as exc:
            logger.warning("Failed to score %s: %s", key, exc)
            result = ScoreResult(score=None, justification=str(exc))
        out[key] = result
    return out
