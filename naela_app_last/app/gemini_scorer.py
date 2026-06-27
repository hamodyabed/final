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
    SCALE_NONE,
    SCALE_NUMERIC,
    SCALE_QUALITATIVE,
    allowed_values,
    scale_for_key,
)


logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gemini-2.5-flash-lite"
# When the primary model returns 503 (overloaded / high demand), we try a
# different model rather than failing. ``gemini-2.5-flash`` is usually less
# congested than the lite tier during demand spikes. Override via env var
# ``GEMINI_FALLBACK_MODEL`` (set to "" to disable the fallback entirely).
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")


GENERAL_RULE = (
    "قاعدة الترميز العامة: تُرمّز البنود الثنائية كالتالي: "
    "0 = غير موجود أو غير صحيح، 1 = موجود أو صحيح. "
    "وتُرمّز البنود النوعية أو المركبة كالتالي: "
    "0 = غائب، 1 = جزئي أو محدود، 2 = واضح أو مكتمل. "
    "أي اختلاف في سلم الترميز يجب أن يُذكر صراحة داخل البند."
)


# Map file extensions to MIME types Gemini actually accepts. Python's
# ``mimetypes`` returns values like ``audio/mp4a-latm`` for ``.m4a`` which
# Gemini rejects, so we override the common audio/video formats explicitly.
_MIME_BY_EXT = {
    # video
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".m4v": "video/mp4",
    ".avi": "video/x-msvideo",
    # audio
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}


def _media_mime_for(path: Path) -> str:
    """Return a Gemini-supported MIME type for a media file.

    Prefers an explicit override (so ``.m4a`` → ``audio/mp4`` etc.), then
    falls back to ``mimetypes.guess_type``, then to ``video/mp4``.
    """
    ext = Path(path).suffix.lower()
    if ext in _MIME_BY_EXT:
        return _MIME_BY_EXT[ext]
    return mimetypes.guess_type(str(path))[0] or "video/mp4"


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


def _call_with_retry(fn, *, max_attempts: int = 6, base_delay: float = 1.5):
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
                more_models_available = model != self._models_to_try[-1]
                # Daily-quota exhaustion: permanently drop this model.
                if _is_daily_quota_error(exc) and len(self._models_to_try) > 1:
                    logger.warning(
                        "Daily quota reached for %s – falling back to next model.",
                        model,
                    )
                    self._models_to_try.remove(model)
                    continue
                # Persistent overload (503) after retries: try the next model
                # (don't drop it permanently — it may recover for later calls).
                if _is_transient_error(exc) and more_models_available:
                    logger.warning(
                        "%s still overloaded after retries – trying next model.",
                        model,
                    )
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


    # ------------------------------------------------------------------
    def score_full_video(
        self,
        video_path: Path,
        questions: list,
        rubric_by_key: Dict[str, str],
    ) -> Dict[str, "ScoreResult"]:
        """Grade an ENTIRE recorded session from a single video file.

        The video typically contains three overlapping sound sources:
        the examiner / teacher asking the questions, the narrated story
        video playing in the background, and the child's spoken answers.
        Gemini is multimodal, so we send the whole clip once and instruct
        it to **listen only to the child** (ignoring the teacher's voice
        and the story-video narration), match each of the child's answers
        to the matching NCP question, validate correctness against the
        acceptable-answer reference, and return a score per question key.

        Parameters
        ----------
        video_path:
            Path to the uploaded video (mp4 / mov / webm / mkv …).
        questions:
            Ordered list of :class:`app.questions.Question` to grade.
        rubric_by_key:
            ``{key: description}`` from ``extracted_keys.json`` used to
            give Gemini the full coding card for each item.

        Returns
        -------
        ``{question_key: ScoreResult}`` — one entry per gradable question.
        Questions Gemini did not return are filled with ``score=None``.
        """
        video_path = Path(video_path)
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise GeminiScorerError("ملف الفيديو غير موجود أو فارغ.")

        # Build the gradable list (skip non-scored columns).
        gradable = [
            q for q in questions
            if scale_for_key(q.key) in (SCALE_BINARY, SCALE_QUALITATIVE)
        ]
        if not gradable:
            return {}

        prompt = self._build_video_prompt(gradable, rubric_by_key)

        mime = _media_mime_for(video_path)
        video_bytes = video_path.read_bytes()
        video_part = self._types.Part.from_bytes(data=video_bytes, mime_type=mime)

        try:
            response = self._generate(
                contents=[prompt, video_part],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    # A whole-session transcript + per-question reasoning can
                    # be long, so give the model generous headroom.
                    max_output_tokens=8192,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini full-video scoring failed")
            raise GeminiScorerError(
                f"فشل الاتصال بـ Gemini أثناء تحليل الفيديو: {exc}"
            ) from exc

        text = (getattr(response, "text", "") or "").strip()
        return self._parse_video_response(text, gradable)

    # ------------------------------------------------------------------
    def _build_video_prompt(
        self,
        questions: list,
        rubric_by_key: Dict[str, str],
    ) -> str:
        """Build the Arabic prompt for grading a full-session video.

        Lists every gradable question with its allowed scale + acceptable
        answers and asks Gemini to return one JSON object keyed by
        ``question_key``.
        """
        from .qa_reference import get_qa  # lazy import keeps tests quick

        blocks: list = []
        for q in questions:
            scale = scale_for_key(q.key)
            allowed = allowed_values(scale)
            allowed_str = "، ".join(str(v) for v in allowed)
            scale_hint = (
                "ثنائي (0/1)" if scale == SCALE_BINARY
                else "نوعي (0/1/2)"
            )

            qa = get_qa(q.key)
            question_text = getattr(q, "text", "") or ""
            expected = ""
            if qa is not None and qa.acceptable_answers:
                if qa.question_text:
                    question_text = qa.question_text
                expected = "؛ ".join(qa.acceptable_answers)
            else:
                expected = _extract_expected_answers(
                    rubric_by_key.get(q.key, "")
                )

            blocks.append(
                f"### المعرّف (question_key): {q.key}\n"
                f"- السؤال: {question_text}\n"
                f"- سلم الترميز: {scale_hint} — القيم المسموح بها: {allowed_str}\n"
                f"- الإجابات المقبولة المرجعية: "
                f"{expected or '(استند إلى وصف البند)'}\n"
            )

        questions_block = "\n".join(blocks)

        return (
            "أنت باحث ومحلِّل لغوي إكلينيكي متخصص في تقييم عينات السرد "
            "القصصي لأطفال يتحدثون العربية الفلسطينية المحكية، ضمن بروتوكول "
            "Naela لتقييم الفهم السردي (NCP).\n\n"

            "## طبيعة المُدخل\n"
            "سأرسل لك **مقطع فيديو واحدًا** لجلسة كاملة. يحتوي الفيديو على "
            "ثلاثة مصادر صوتية متداخلة:\n"
            "1. صوت **المعلّم/الفاحص** وهو يطرح الأسئلة على الطفل.\n"
            "2. صوت **فيديو القصة** المسرود في الخلفية.\n"
            "3. صوت **الطفل** وهو يجيب على الأسئلة.\n\n"

            "## المطلوب منك بدقة\n"
            "- اعزل وركّز **فقط على صوت الطفل**، وتجاهل تمامًا صوت المعلّم "
            "وصوت فيديو القصة. لا تَعُدَّ كلام المعلّم أو سرد الفيديو إجابةً "
            "للطفل أبدًا.\n"
            "- استعن بالصورة (حركة شفاه الطفل، تفاعله) إن لزم لتمييز صوته.\n"
            "- لكل سؤال في القائمة أدناه: ابحث عن إجابة الطفل المقابلة في "
            "الفيديو، وفرّغها حرفيًا، ثم قارنها بالإجابات المقبولة المرجعية، "
            "ثم امنحها درجة ضمن سلم الترميز المسموح به لذلك البند فقط.\n"
            "- إذا لم يُجب الطفل عن سؤال ما، أو لم تجد إجابته في الفيديو: "
            "ضع score = 0 و transcript = \"\" مع تبرير يوضح غياب الإجابة.\n"
            "- كن متسامحًا مع لكنة الطفل والأخطاء العامية البسيطة ما دامت "
            "الفكرة المطلوبة موجودة، لكن لا تكافئ إجابات غير ذات صلة.\n\n"

            f"## قاعدة الترميز العامة\n{GENERAL_RULE}\n\n"

            "## قائمة الأسئلة المطلوب تقييمها\n"
            f"{questions_block}\n"

            "## شكل الإخراج المطلوب (JSON فقط، بدون أي نص آخر)\n"
            "أعد كائن JSON واحدًا مفتاحه «results» وقيمته مصفوفة، كل عنصر فيها "
            "يخص سؤالًا واحدًا بالشكل التالي:\n"
            "{\n"
            "  \"results\": [\n"
            "    {\n"
            "      \"question_key\": \"<المعرّف كما ورد أعلاه>\",\n"
            "      \"transcript\": \"<نص حرفي لما قاله الطفل لهذا السؤال>\",\n"
            "      \"justification\": \"<تبرير عربي مختصر يربط الإجابة بالمعايير>\",\n"
            "      \"score\": <عدد صحيح ضمن القيم المسموح بها لهذا البند>\n"
            "    }\n"
            "    // ... عنصر واحد لكل سؤال في القائمة\n"
            "  ]\n"
            "}\n"
            "لا تستخدم أي قيمة درجة خارج المسموح به لكل بند، ولا تُضِف أي نص "
            "خارج كائن JSON."
        )

    # ------------------------------------------------------------------
    def _parse_video_response(
        self,
        text: str,
        questions: list,
    ) -> Dict[str, "ScoreResult"]:
        """Parse the full-video JSON into ``{key: ScoreResult}``."""
        results: Dict[str, ScoreResult] = {}
        # Pre-fill every gradable question with a "not found" placeholder so
        # the caller always gets a complete mapping.
        for q in questions:
            results[q.key] = ScoreResult(
                score=None,
                justification="لم تُعِد Gemini نتيجة لهذا البند.",
            )

        if not text:
            return results

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        payload = None
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
        if not isinstance(payload, dict):
            return results

        items = payload.get("results")
        if not isinstance(items, list):
            return results

        allowed_by_key = {
            q.key: allowed_values(scale_for_key(q.key)) for q in questions
        }
        for item in items:
            if not isinstance(item, dict):
                continue
            key = item.get("question_key")
            if key not in results:
                continue
            allowed = allowed_by_key.get(key, [0, 1])
            score_value = item.get("score")
            try:
                score_int = int(score_value)
            except (TypeError, ValueError):
                score_int = None
            if score_int is not None and score_int not in allowed:
                score_int = max(allowed[0], min(allowed[-1], score_int))
            results[key] = ScoreResult(
                score=score_int,
                transcript=str(item.get("transcript", "")).strip(),
                justification=str(item.get("justification", "")).strip(),
                raw=text,
            )
        return results

    # ------------------------------------------------------------------
    def analyze_full_video(
        self,
        video_path: Path,
        all_keys: list,
    ) -> Dict[str, object]:
        """Run a COMPLETE linguistic analysis of a session video.

        Unlike :meth:`score_full_video` (which only grades the NCP
        comprehension questions), this sends Gemini **every** column from
        ``extracted_keys.json`` with its Arabic description and asks the
        model to:

        1. transcribe the child's full narrative (child voice only),
        2. compute a value for **every** column — counts, ratios,
           0/1 or 0/1/2 scores, totals, and short text/list fields —
        3. return one JSON object ``{key: value}`` covering all columns.

        Returns ``{key: value, "_transcript": str}`` ready to merge into
        the Excel row. Values are best-effort estimates produced by the
        model; numeric columns come back as numbers, score columns as
        integers in range, list/text columns as strings.
        """
        video_path = Path(video_path)
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise GeminiScorerError("ملف الجلسة غير موجود أو فارغ.")

        prompt = self._build_full_analysis_prompt(all_keys)

        mime = _media_mime_for(video_path)
        media_bytes = video_path.read_bytes()
        media_part = self._types.Part.from_bytes(data=media_bytes, mime_type=mime)

        try:
            response = self._generate(
                contents=[prompt, media_part],
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    # Full 78-column analysis + transcript needs lots of room.
                    max_output_tokens=16384,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini full-analysis failed")
            raise GeminiScorerError(
                f"فشل الاتصال بـ Gemini أثناء التحليل الكامل للفيديو: {exc}"
            ) from exc

        text = (getattr(response, "text", "") or "").strip()
        return self._parse_full_analysis(text, all_keys)

    # ------------------------------------------------------------------
    def _build_full_analysis_prompt(self, all_keys: list) -> str:
        """Build the Arabic prompt listing every column to be filled."""
        lines: list = []
        for row in all_keys:
            key = row.get("key", "")
            if not key:
                continue
            scale = scale_for_key(key)
            if scale == SCALE_BINARY:
                kind = "عدد صحيح 0 أو 1"
            elif scale == SCALE_QUALITATIVE:
                kind = "عدد صحيح 0 أو 1 أو 2"
            elif scale == SCALE_NUMERIC:
                kind = "عدد (مجموع/نسبة)"
            else:
                kind = "عدد أو نسبة أو نص قصير حسب الوصف"
            desc = (row.get("description", "") or "").replace("\n", " ")[:200]
            lines.append(f"- {key} [{kind}]: {desc}")
        keys_block = "\n".join(lines)

        return (
            "أنت باحث ومحلِّل لغوي إكلينيكي خبير في تقييم عينات السرد القصصي "
            "لأطفال يتحدثون العربية الفلسطينية المحكية، ضمن بروتوكول Naela.\n\n"

            "## المُدخل\n"
            "سأرسل لك **مقطعًا واحدًا** (فيديو أو صوت) لجلسة كاملة فيها: صوت "
            "المعلّم وهو يسأل، وصوت فيديو القصة في الخلفية، وصوت الطفل وهو "
            "يجيب ويسرد. ركّز **فقط على كلام الطفل** وتجاهل صوت المعلّم وصوت "
            "فيديو القصة.\n\n"

            "## المهمة\n"
            "1. فرّغ كلام الطفل **حرفيًا وكاملًا** (كل ما قاله طوال الجلسة).\n"
            "2. حلّل هذا التفريغ تحليلًا لغويًا شاملًا، واحسب قيمة **كل** عمود "
            "من الأعمدة المذكورة أدناه بالاعتماد على كلام الطفل فقط.\n"
            "3. التزم بنوع القيمة المحدد بين الأقواس لكل عمود:\n"
            "   • أعمدة الدرجات (0/1 أو 0/1/2): أعطِ عددًا صحيحًا ضمن المدى.\n"
            "   • أعمدة العدّ (counts) والنِّسب (ratios/percentages): أعطِ عددًا.\n"
            "   • أعمدة القوائم أو الأمثلة النصية: أعطِ نصًا عربيًا قصيرًا "
            "(كلمات مفصولة بفواصل)، وإن لم يوجد فاترك \"\".\n"
            "4. **لا تترك أي عمود فارغًا**: إن تعذّر الحساب فاجعل القيمة 0 "
            "للأعمدة الرقمية و \"\" للأعمدة النصية. قدّر بأفضل ما يمكن بدل "
            "الترك فارغًا.\n"
            "5. للنِّسب المئوية أعطِ رقمًا بين 0 و100. للنِّسب مثل TTR أعطِ "
            "رقمًا عشريًا بين 0 و1.\n\n"

            f"## قاعدة الترميز العامة للأعمدة الدرجية\n{GENERAL_RULE}\n\n"

            "## قائمة الأعمدة المطلوب ملؤها (المعرّف ثم النوع ثم الوصف)\n"
            f"{keys_block}\n\n"

            "## شكل الإخراج المطلوب (JSON فقط، بدون أي نص آخر)\n"
            "أعد كائن JSON واحدًا يحتوي:\n"
            "{\n"
            "  \"transcript\": \"<التفريغ الحرفي الكامل لكلام الطفل>\",\n"
            "  \"columns\": {\n"
            "     \"<key1>\": <value1>,\n"
            "     \"<key2>\": <value2>\n"
            "     // عنصر واحد لكل عمود من القائمة أعلاه، بلا استثناء\n"
            "  }\n"
            "}\n"
            "تأكد أن مفتاح \"columns\" يحتوي **كل** المعرّفات المذكورة في "
            "القائمة أعلاه دون نقصان."
        )

    # ------------------------------------------------------------------
    def _parse_full_analysis(
        self,
        text: str,
        all_keys: list,
    ) -> Dict[str, object]:
        """Parse the full-analysis JSON into ``{key: value}`` + transcript."""
        out: Dict[str, object] = {}
        if not text:
            return out

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        payload = None
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None
        if not isinstance(payload, dict):
            return out

        transcript = str(payload.get("transcript", "")).strip()
        columns = payload.get("columns")
        if not isinstance(columns, dict):
            # Some models flatten it — treat the whole object as columns.
            columns = {
                k: v for k, v in payload.items() if k != "transcript"
            }

        valid_keys = {row.get("key") for row in all_keys}
        for key, value in columns.items():
            if key not in valid_keys:
                continue
            scale = scale_for_key(key)
            if scale in (SCALE_BINARY, SCALE_QUALITATIVE):
                allowed = allowed_values(scale)
                try:
                    ivalue = int(value)
                except (TypeError, ValueError):
                    # e.g. "1" or "غير متوفر" → coerce / skip
                    try:
                        ivalue = int(float(value))
                    except (TypeError, ValueError):
                        continue
                ivalue = max(allowed[0], min(allowed[-1], ivalue))
                out[key] = ivalue
            else:
                # numeric or text: store as-is (number stays number, text stays text)
                out[key] = value

        if transcript:
            out["_transcript"] = transcript
        return out


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


def score_video_to_answers(
    video_path: Path,
    questions: list,
    rubric_by_key: Dict[str, str],
    scorer: Optional[GeminiScorer] = None,
) -> Dict[str, dict]:
    """Score a whole session video and return an ``answers``-shaped dict.

    Convenience wrapper around :meth:`GeminiScorer.score_full_video` that
    produces ``{question_key: {"score", "transcript", "justification",
    "question_order", "question_text"}}`` — exactly the shape consumed by
    :func:`app.scoring.aggregate_answers_to_columns` so the result can be
    written straight to one Excel row.

    Only questions Gemini actually scored (``score is not None``) are
    included, so unanswered items fall back to the column default.
    """
    scorer = scorer or GeminiScorer()
    score_results = scorer.score_full_video(
        video_path=Path(video_path),
        questions=questions,
        rubric_by_key=rubric_by_key,
    )
    by_key = {q.key: q for q in questions}
    answers: Dict[str, dict] = {}
    for key, result in score_results.items():
        if result.score is None:
            continue
        q = by_key.get(key)
        answers[key] = {
            "question_order": getattr(q, "order", 0),
            "question_text": getattr(q, "text", ""),
            "score": result.score,
            "transcript": result.transcript,
            "justification": result.justification,
            "scored_from": "video",
        }
    return answers


def analyze_video_to_columns(
    video_path: Path,
    questions: list,
    all_keys: list,
    rubric_by_key: Dict[str, str],
    scorer: Optional[GeminiScorer] = None,
) -> Dict[str, object]:
    """Produce a value for **every** Excel column from one session video.

    Combines two passes so the full row is filled:

    1. :meth:`GeminiScorer.score_full_video` — precise per-question NCP
       scoring, aggregated into the macrostructure ``*_score`` columns and
       ``macro_total_score`` via
       :func:`app.scoring.aggregate_answers_to_columns`.
    2. :meth:`GeminiScorer.analyze_full_video` — a complete linguistic
       analysis that fills every remaining column (microstructure counts,
       ratios, IPSyn / theory-of-mind scores, verb measures, language-mix
       counts, text/list fields, …).

    The precise question-derived values from pass 1 take precedence over the
    estimates from pass 2 for the columns they cover.

    Returns ``{column_key: value}`` ready to pass to
    :func:`app.excel_export.append_session`.
    """
    from .scoring import aggregate_answers_to_columns

    scorer = scorer or GeminiScorer()

    # Pass 2 first (full analysis → baseline for every column).
    try:
        full_cols = scorer.analyze_full_video(
            video_path=Path(video_path), all_keys=all_keys,
        )
    except GeminiScorerError as exc:
        logger.warning("Full analysis failed, continuing with QA scores: %s", exc)
        full_cols = {}
    full_cols.pop("_transcript", None)

    # Pass 1 (accurate NCP question scores) overrides the estimates.
    answers = score_video_to_answers(
        video_path=Path(video_path),
        questions=questions,
        rubric_by_key=rubric_by_key,
        scorer=scorer,
    )
    qa_cols = aggregate_answers_to_columns(answers)

    merged: Dict[str, object] = {}
    merged.update(full_cols)
    merged.update(qa_cols)  # precise values win
    return merged
