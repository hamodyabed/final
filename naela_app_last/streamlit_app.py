"""Streamlit web version of the Naela narrative-assessment app.

Run locally:

    streamlit run streamlit_app.py

Deploy to Streamlit Cloud:

    1. Push the repo to GitHub.
    2. On https://share.streamlit.io, point a new app at this file.
    3. Add the following secret in the app settings:
           GEMINI_API_KEY = "AIza..."        (Google AI Studio key)
           GEMINI_MODEL   = "gemini-2.5-flash-lite"

The browser asks the child's iPad / phone for microphone permission; recorded
WAV/WebM blobs are sent to Google Gemini for transcription + scoring, and the
results for every child in the session are appended to one shared Excel
workbook which is offered as a download at the end of the batch.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# Load .env so local dev works without exporting variables manually.
import app  # noqa: F401 — triggers _load_dotenv_once()

# Push Streamlit Cloud "secrets" into os.environ so the existing modules pick
# them up transparently.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    # Running outside Streamlit Cloud and no .streamlit/secrets.toml — fine.
    pass

from app.excel_export import append_session, create_workbook, make_session_meta
from app.gemini_scorer import (
    GeminiScorer,
    GeminiScorerError,
    analyze_video_to_columns,
    score_video_to_answers,
)
from app.questions import Question, load_all_keys, load_questions
from app.scoring import aggregate_answers_to_columns


# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_JSON = DATA_DIR / "extracted_keys.json"
DEFAULT_VIDEO = DATA_DIR / "video.mp4"
DEFAULT_PDF = DATA_DIR / "story.pdf"

# A public URL of the story video — preferred over a local file on the web
# (no GitHub size limit, no Streamlit Cloud disk pressure). Supports:
#   * YouTube (https://youtu.be/… or https://www.youtube.com/watch?v=…)
#   * Vimeo
#   * Direct .mp4 / .webm URLs
#   * Google Drive share links (https://drive.google.com/file/d/<ID>/view…)
# Falls back to ``DEFAULT_VIDEO`` if the env var is empty.
VIDEO_URL = (os.environ.get("VIDEO_URL") or "").strip()

# Google Drive account whose Drive should open when the examiner clicks
# "افتح Google Drive" in the upload tab. Falls back to the default email
# recipient so a single account configuration drives the whole app.
GOOGLE_DRIVE_EMAIL = (
    os.environ.get("GOOGLE_DRIVE_EMAIL")
    or os.environ.get("DEFAULT_EMAIL_RECIPIENT")
    or ""
).strip()


def _google_drive_account_url(email: str) -> str:
    """Return a Drive URL that opens for ``email`` via the account chooser.

    Google's ``AccountChooser`` honours the ``Email`` hint and the
    ``authuser`` parameter so a user already signed into multiple Google
    accounts lands in the right Drive. If no email is configured we fall
    back to the plain Drive home URL.
    """
    base = "https://drive.google.com/drive/u/0/my-drive"
    if not email:
        return base
    from urllib.parse import quote
    return (
        "https://accounts.google.com/AccountChooser"
        f"?Email={quote(email)}"
        f"&continue={quote(base)}"
    )


def _google_drive_file_id(url: str) -> Optional[str]:
    """Extract the file ID from any Google Drive share/view/download URL.

    Returns ``None`` if the URL doesn't look like a Drive link.
    Handles the three common shapes:
        https://drive.google.com/file/d/<ID>/view?usp=sharing
        https://drive.google.com/open?id=<ID>
        https://drive.google.com/uc?export=download&id=<ID>
    """
    if not url or "drive.google.com" not in url:
        return None
    import re
    m = re.search(r"/file/d/([\w-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([\w-]+)", url)
    if m:
        return m.group(1)
    return None


def _download_drive_audio(url: str) -> tuple[Optional[bytes], str, Optional[str]]:
    """Download the audio bytes behind a Google Drive share link.

    Returns a ``(data, filename, error)`` tuple. ``data`` is ``None`` when the
    download fails (the file must be shared as "anyone with the link"). Handles
    Drive's large-file virus-scan confirmation page transparently.
    """
    file_id = _google_drive_file_id(url)
    if not file_id:
        return None, "", "هذا الرابط لا يبدو رابط Google Drive صالحًا."
    try:
        import requests
    except ImportError:
        return None, "", "حزمة requests غير مثبتة."

    base = "https://drive.google.com/uc?export=download"
    try:
        session = requests.Session()
        resp = session.get(base, params={"id": file_id}, stream=True, timeout=60)
        # Large files return an HTML interstitial with a confirm token.
        token = None
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break
        if token:
            resp = session.get(
                base, params={"id": file_id, "confirm": token},
                stream=True, timeout=120,
            )
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            return None, "", (
                "تعذّر تنزيل الملف. تأكد أن مشاركة الملف على "
                "«أي شخص لديه الرابط» (Anyone with the link)."
            )
        data = resp.content
        if not data:
            return None, "", "الملف فارغ أو غير قابل للتنزيل."
        # Try to recover a sensible filename / extension from the headers.
        filename = f"{file_id}.wav"
        disp = resp.headers.get("Content-Disposition") or ""
        m = re.search(r'filename="?([^";]+)"?', disp)
        if m:
            filename = m.group(1)
        return data, filename, None
    except Exception as exc:  # noqa: BLE001
        return None, "", f"خطأ أثناء التنزيل: {exc}"

# Streamlit Cloud filesystem is ephemeral — store sessions in a writable temp
# folder. Local runs still keep their own ``output/sessions/`` for convenience.
SESSIONS_DIR = Path(
    os.environ.get("NAELA_SESSIONS_DIR")
    or (PROJECT_ROOT / "output" / "sessions")
)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ("نمو طبيعي (TD)", "طيف توحد (ASD)", "اضطراب لغة نمائي (DLD)", "أخرى")
GENDERS = ("ذكر", "أنثى", "غير محدد")


# --------------------------------------------------------------------------
# Page config + RTL CSS
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Naela – تقييم السرد القصصي للأطفال",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"] { direction: rtl; text-align: right; }
        section[data-testid="stSidebar"] { direction: rtl; }
        .stButton > button { font-weight: bold; }
        .answer-row { padding: 8px; border-radius: 8px; background: #f7f3e9; margin-bottom: 6px; }
        .question-banner {
            background: #2c3e50; color: white; padding: 12px;
            border-radius: 8px; font-size: 18px; font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Session state bootstrap
# --------------------------------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    # Single-video analysis is the default (and only) entry point: the
    # examiner uploads ONE session video and the app fills every Excel
    # column automatically.
    ss.setdefault("phase", "video_intake")  # video_intake | …
    ss.setdefault("child_info", None)  # dict
    ss.setdefault("session_id", "")
    ss.setdefault("session_dir", None)  # Path
    ss.setdefault("questions", load_questions(DEFAULT_JSON))
    ss.setdefault("current_idx", 0)
    ss.setdefault("answers", {})  # {key: {audio_bytes, transcript, score, ...}}
    ss.setdefault("scorer", None)  # cached GeminiScorer
    # --- Batch-level state ---------------------------------------------
    # One Excel workbook accumulates ALL children assessed in this browser
    # session. The path is computed once on first save (using a batch_id
    # that's stable across reruns).
    ss.setdefault("batch_id",
                  datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    ss.setdefault("batch_dir", SESSIONS_DIR / f"batch_{ss['batch_id']}")
    Path(ss["batch_dir"]).mkdir(parents=True, exist_ok=True)
    ss.setdefault("batch_excel_path", Path(ss["batch_dir"]) / "narrative_coding.xlsx")
    ss.setdefault("completed_children", [])  # list of dicts: name, session_id, row
    ss.setdefault("rubric_by_key", {
        row["key"]: row["description"] for row in load_all_keys(DEFAULT_JSON)
    })


_init_state()
ss = st.session_state


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _safe_filename(name: str) -> str:
    cleaned = "".join(c for c in (name or "").strip() if c.isalnum() or c in "_-")
    return cleaned or "child"


def _ensure_scorer() -> Optional[GeminiScorer]:
    if ss.scorer is not None:
        return ss.scorer
    try:
        ss.scorer = GeminiScorer()
        return ss.scorer
    except GeminiScorerError as exc:
        st.warning(f"⚠️ تعذّر تهيئة Gemini: {exc}")
        return None


def _save_audio_to_session(key: str, audio_bytes: bytes, suffix: str = ".wav") -> Path:
    """Persist a recording on disk under the active session folder."""
    if ss.session_dir is None:
        # Defensive fallback – should never happen if the flow is followed.
        ss.session_dir = SESSIONS_DIR / f"unknown_{uuid.uuid4().hex[:6]}"
        ss.session_dir.mkdir(parents=True, exist_ok=True)
    q = next((q for q in ss.questions if q.key == key), None)
    order = q.order if q else 0
    fname = f"q{order:02d}_{key}{suffix}"
    path = Path(ss.session_dir) / fname
    path.write_bytes(audio_bytes)
    return path


def _ingest_answer_audio(q, audio_bytes: bytes, suffix: str = ".wav") -> None:
    """Common pipeline for both live recordings and uploaded audio files.

    Saves the bytes to disk, registers the answer in session_state, runs the
    live transcription (so the examiner sees Arabic text immediately),
    persists an updated session.json manifest so the child can be resumed
    later, then triggers a Streamlit rerun to refresh the UI.
    """
    if not audio_bytes:
        return
    path = _save_audio_to_session(q.key, audio_bytes, suffix=suffix)
    ss.answers[q.key] = {
        "question_order": q.order,
        "question_text": q.text,
        "audio_path": str(path),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    # Live transcription so the table shows the text within seconds.
    txt, err = _transcribe(path, q.key)
    if txt:
        ss.answers[q.key]["transcript"] = txt
    elif err:
        ss.answers[q.key]["transcription_error"] = err
    # Auto-save the in-progress manifest so the child can be resumed even if
    # the browser tab is closed mid-session.
    _save_in_progress()
    st.rerun()


# --------------------------------------------------------------------------
# Persisting / restoring in-progress sessions
# --------------------------------------------------------------------------
SESSION_MANIFEST_NAME = "session.json"


def _save_in_progress() -> None:
    """Write a ``session.json`` describing the current child + answers.

    The file lives at ``<session_dir>/session.json`` and lets the examiner
    reopen the app later and resume from the last answered question.
    """
    if ss.session_dir is None or ss.child_info is None:
        return
    manifest = {
        "status": "in_progress",
        "batch_id": ss.get("batch_id", ""),
        "session_id": ss.session_id,
        "child": ss.child_info,
        "current_idx": ss.current_idx,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        # Drop non-serialisable bits (e.g. Path objects) from the answers.
        "answers": {
            k: {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool, type(None)))}
            for k, v in ss.answers.items()
        },
    }
    out = Path(ss.session_dir) / SESSION_MANIFEST_NAME
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_in_progress_sessions() -> List[dict]:
    """Scan ``output/sessions/`` for session.json files with status=in_progress.

    Returns a list of dicts sorted by `saved_at` descending. Each dict has
    keys: ``session_dir``, ``session_id``, ``child_name``, ``saved_at``,
    ``answered``, ``total``, ``batch_id``.
    """
    results: List[dict] = []
    if not SESSIONS_DIR.exists():
        return results
    total_questions = len(ss.questions)
    for manifest_path in SESSIONS_DIR.rglob(SESSION_MANIFEST_NAME):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") != "in_progress":
            continue
        child = data.get("child") or {}
        results.append({
            "session_dir": manifest_path.parent,
            "session_id": data.get("session_id", manifest_path.parent.name),
            "child_name": child.get("name", "(بدون اسم)"),
            "child_age": child.get("age", ""),
            "child_group": child.get("group", ""),
            "saved_at": data.get("saved_at", ""),
            "answered": len([
                v for v in (data.get("answers") or {}).values()
                if v.get("audio_path")
            ]),
            "total": total_questions,
            "batch_id": data.get("batch_id", ""),
        })
    results.sort(key=lambda r: r["saved_at"], reverse=True)
    return results


def _resume_session(session_dir: Path) -> bool:
    """Load a paused session back into Streamlit session_state.

    Returns True on success, False if the file is missing or malformed.
    """
    manifest_path = session_dir / SESSION_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    ss.child_info = data.get("child") or {}
    ss.session_id = data.get("session_id", session_dir.name)
    ss.session_dir = session_dir
    ss.current_idx = int(data.get("current_idx", 0) or 0)
    ss.answers = dict(data.get("answers") or {})
    # If the paused session was part of a batch, rejoin that batch so the
    # Excel append at finish targets the right shared workbook.
    batch_id = data.get("batch_id")
    if batch_id:
        ss.batch_id = batch_id
        ss.batch_dir = SESSIONS_DIR / f"batch_{batch_id}"
        ss.batch_dir.mkdir(parents=True, exist_ok=True)
        ss.batch_excel_path = Path(ss.batch_dir) / "narrative_coding.xlsx"
    ss.phase = "questions"
    ss._child_saved = False
    return True


def _mark_session_finished(session_dir: Path) -> None:
    """Update session.json so it no longer appears in the resume list."""
    manifest_path = Path(session_dir) / SESSION_MANIFEST_NAME
    if not manifest_path.is_file():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    data["status"] = "finished"
    data["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _transcribe(path: Path, key: str) -> tuple[str, str]:
    """Return (transcript, error) for a recorded answer."""
    scorer = _ensure_scorer()
    if scorer is None:
        return "", "Gemini غير مُهيّأ."
    try:
        text = scorer.transcribe(path)
        return text, "" if text else "ردّ فارغ من Gemini."
    except GeminiScorerError as exc:
        return "", str(exc)


def _score_all() -> tuple[bool, list[str]]:
    """Score every recorded answer with Gemini. Returns (any_success, errors)."""
    scorer = _ensure_scorer()
    if scorer is None:
        return False, ["Gemini غير مُهيّأ — لم يتم تقييم الإجابات."]
    errors: list[str] = []
    successes = 0
    questions_by_key = {q.key: q for q in ss.questions}
    bar = st.progress(0.0, text="جارٍ التقييم التلقائي…")
    recorded = [(k, v) for k, v in ss.answers.items() if v.get("audio_path")]
    for idx, (key, payload) in enumerate(recorded, start=1):
        bar.progress(idx / max(1, len(recorded)), text=f"({idx}/{len(recorded)}) {key}")
        try:
            result = scorer.score_answer(
                audio_path=Path(payload["audio_path"]),
                question_key=key,
                question_text=questions_by_key[key].text
                if key in questions_by_key
                else payload.get("question_text", ""),
                rubric_description=ss.rubric_by_key.get(key, ""),
            )
        except GeminiScorerError as exc:
            errors.append(f"{key}: {exc}")
            continue
        payload["score"] = result.score
        if result.transcript:
            payload["transcript"] = result.transcript
        payload["scoring_justification"] = result.justification
        payload["scored_by"] = "gemini"
        payload["scored_at"] = datetime.now().isoformat(timespec="seconds")
        if result.score is not None:
            successes += 1
    bar.empty()
    return successes > 0, errors


def _append_current_child_to_batch() -> int:
    """Append the current child's row to the shared batch workbook.

    Creates the workbook on first child. Returns the 1-based row that was
    written, so it can be shown in the UI / "completed children" list.
    """
    out_path: Path = ss.batch_excel_path
    if not out_path.exists():
        create_workbook(DEFAULT_JSON, out_path)
    column_values = aggregate_answers_to_columns(ss.answers)
    child = ss.child_info
    meta = make_session_meta(
        session_id=ss.session_id,
        child_name=child["name"],
        child_age=str(child["age"]),
        audio_dir=ss.session_dir,
        notes=child.get("notes", ""),
        child_gender=child.get("gender", ""),
        child_group=child.get("group", ""),
    )
    return append_session(DEFAULT_JSON, out_path, meta, answers=column_values)


# --------------------------------------------------------------------------
# Phase: intake
# --------------------------------------------------------------------------
def render_intake() -> None:
    st.title("👶 بدء جلسة لطفل جديد")
    st.caption("Naela – تقييم السرد القصصي للأطفال")

    # ---- Quick entry to the single-video analysis flow --------------------
    st.info(
        "🎥 **جديد:** يمكنك الآن رفع **فيديو واحد** للجلسة كاملة، وسيقوم "
        "التطبيق باستخراج صوت الطفل فقط، وتحليل إجاباته، وتعبئة كل أعمدة "
        "ملف Excel تلقائيًا — بدون تسجيل كل سؤال على حدة."
    )
    if st.button(
        "🎬 تحليل فيديو واحد للجلسة الكاملة",
        use_container_width=True,
        type="primary",
    ):
        ss.phase = "video_intake"
        st.rerun()
    st.divider()

    # ---- Resume an existing in-progress session ---------------------------
    pending = _list_in_progress_sessions()
    if pending:
        with st.expander(
            f"📂 استئناف جلسة سابقة ({len(pending)} متوقفة)",
            expanded=False,
        ):
            st.caption(
                "هذه جلسات لم تكتمل بعد. اختر طفلًا للمتابعة من حيث توقفت."
            )
            for s in pending:
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"**{s['child_name']}**"
                        f" — {s['answered']}/{s['total']} إجابات"
                        f" — تم التوقف في: {s['saved_at']}"
                        f"<br><span style='color:#666;font-size:11px'>{s['session_id']}</span>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button(
                        "▶️ متابعة",
                        key=f"resume_{s['session_id']}",
                        use_container_width=True,
                    ):
                        if _resume_session(Path(s["session_dir"])):
                            st.rerun()
                        else:
                            st.error("تعذّر تحميل الجلسة. الملف مفقود أو تالف.")
            st.divider()

    with st.form("intake", clear_on_submit=False):
        name = st.text_input("اسم الطفل *", placeholder="مثال: عبد الرحمن")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("العمر (سنوات)", min_value=2, max_value=17, value=6)
            gender = st.selectbox("الجنس", GENDERS, index=0)
        with col2:
            group = st.selectbox("المجموعة", GROUPS, index=0)
        notes = st.text_area(
            "ملاحظات (مدرسة، تشخيص سابق، لغة سائدة...)", height=80
        )

        submitted = st.form_submit_button("✅ ابدأ الجلسة", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("اسم الطفل مطلوب لبدء الجلسة.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        token = _safe_filename(name)
        ss.session_id = f"{stamp}_{token}"
        ss.session_dir = SESSIONS_DIR / ss.session_id
        ss.session_dir.mkdir(parents=True, exist_ok=True)
        ss.child_info = {
            "name": name.strip(),
            "age": int(age),
            "gender": gender,
            "group": group,
            "notes": notes.strip(),
        }
        ss.current_idx = 0
        ss.answers = {}
        ss.phase = "questions"
        # Persist the manifest immediately so the child shows up in the resume
        # list even if the examiner closes the tab before recording anything.
        _save_in_progress()
        st.rerun()


# --------------------------------------------------------------------------
# Persistent media panel (rendered next to the questions)
# --------------------------------------------------------------------------
def render_media_panel() -> None:
    """Video + PDF tabs that stay visible while the examiner asks questions."""
    tab_video, tab_pdf = st.tabs(["🎬 الفيديو", "📖 القصة المصورة"])
    with tab_video:
        # Prefer a remote URL (YouTube / Vimeo / direct MP4 / Google Drive)
        # so we don't have to commit the heavy video file to the repo.
        # Google Drive needs an iframe to its /preview endpoint because
        # st.video can't follow Drive's redirect / virus-scan page for
        # files >25 MB.
        if VIDEO_URL:
            drive_id = _google_drive_file_id(VIDEO_URL)
            if drive_id:
                import streamlit.components.v1 as components
                components.iframe(
                    f"https://drive.google.com/file/d/{drive_id}/preview",
                    height=380,
                )
            else:
                st.video(VIDEO_URL)
        elif DEFAULT_VIDEO.exists():
            st.video(str(DEFAULT_VIDEO))
        else:
            st.info(
                "لم يتم تحديد رابط الفيديو. "
                "أضف VIDEO_URL إلى ملف .env "
                "(يدعم: YouTube / Vimeo / رابط MP4 مباشر / Google Drive) "
                "أو ضع الملف data/video.mp4 محليًا."
            )
    with tab_pdf:
        if DEFAULT_PDF.exists():
            with DEFAULT_PDF.open("rb") as f:
                st.download_button(
                    "⬇️ تحميل ملف القصة (PDF)",
                    data=f,
                    file_name="story.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.caption("افتح الملف على جهازك لعرض الصور للطفل.")
        else:
            st.info("ملف القصة غير موجود (data/story.pdf).")


# --------------------------------------------------------------------------
# Phase: questions (media + question recorder side-by-side)
# --------------------------------------------------------------------------
def render_questions() -> None:
    child = ss.child_info
    questions: List[Question] = ss.questions
    if not questions:
        st.error("لم يتم تحميل أي سؤال من data/extracted_keys.json.")
        return

    idx = ss.current_idx
    q = questions[idx]

    # ----- Sidebar: child banner + jump-to-question list ------------------
    with st.sidebar:
        n_done = len(ss.completed_children)
        st.caption(
            f"👨‍👩‍👧‍👦 الأطفال المنتهية جلساتهم في هذه الجلسة: **{n_done}**"
        )
        st.markdown(
            f"<div class='question-banner'>الطفل #{n_done + 1}: {child['name']}<br>"
            f"العمر: {child['age']} • {child['gender']} • {child['group']}</div>",
            unsafe_allow_html=True,
        )
        st.progress((idx + 1) / len(questions), text=f"{idx + 1} / {len(questions)}")
        if n_done > 0 and st.button(
            "📊 إنهاء الجلسة الكاملة وإرسال Excel",
            use_container_width=True, type="secondary",
        ):
            ss.phase = "batch_finished"
            st.rerun()
        st.divider()
        st.caption("📋 الأسئلة")
        for i, qq in enumerate(questions):
            done = qq.key in ss.answers and ss.answers[qq.key].get("audio_path")
            marker = "✅" if done else ("👉" if i == idx else "⬜")
            label = f"{marker} {qq.order}. {qq.text[:32]}"
            if st.button(label, key=f"jump_{i}", use_container_width=True):
                ss.current_idx = i
                st.rerun()

    # ----- Main pane: media on the LEFT, question/recorder on the RIGHT --
    # On a phone the two columns stack automatically (Streamlit reflows).
    # On a tablet/desktop they sit side-by-side so the video stays visible
    # while the child answers each question.
    media_col, qa_col = st.columns([1, 1], gap="large")

    with media_col:
        render_media_panel()

    with qa_col:
        st.markdown(
            f"<div class='question-banner'>سؤال {q.order} من {len(questions)}<br>{q.text}</div>",
            unsafe_allow_html=True,
        )
        st.caption({
            "primary": "نوع السؤال: سؤال الفهم",
            "followup": "نوع السؤال: سؤال المتابعة",
            "macro": "نوع السؤال: بند سردي",
        }.get(q.kind, q.kind))

        existing = ss.answers.get(q.key, {})
        existing_audio_path = existing.get("audio_path")

        if existing_audio_path and Path(existing_audio_path).exists():
            st.audio(str(existing_audio_path))
            cols = st.columns(2)
            if cols[0].button("🔁 إعادة الإجابة", key=f"reset_{q.key}",
                              use_container_width=True):
                try:
                    Path(existing_audio_path).unlink(missing_ok=True)
                except OSError:
                    pass
                ss.answers.pop(q.key, None)
                st.rerun()
            if cols[1].button("🔄 إعادة التفريغ", key=f"retx_{q.key}",
                              use_container_width=True):
                txt, err = _transcribe(Path(existing_audio_path), q.key)
                if txt:
                    existing["transcript"] = txt
                    existing.pop("transcription_error", None)
                else:
                    existing["transcription_error"] = err
                st.rerun()
            transcript = (existing.get("transcript") or "").strip()
            if transcript:
                st.success(f"📝 {transcript}")
            elif existing.get("transcription_error"):
                st.error(f"⚠️ {existing['transcription_error']}")
        else:
            st.info(
                "اختر «🎙️ تسجيل مباشر» للتسجيل من الميكروفون، "
                "أو «📁 رفع ملف» لإرفاق ملف صوتي مسجَّل مسبقًا."
            )

        # Two ways to provide audio: live recording (browser mic) OR an
        # uploaded file that was captured separately (WAV/MP3/M4A/OGG/WEBM).
        tab_rec, tab_upload = st.tabs(["🎙️ تسجيل مباشر", "📁 رفع ملف صوتي"])

        with tab_rec:
            rec = st.audio_input(
                "🎙️ سجّل الإجابة هنا",
                key=f"rec_{q.key}_{idx}",
                help="اضغط لبدء التسجيل، ثم اضغط مرة أخرى للإيقاف.",
            )
            if rec is not None and not existing_audio_path:
                _ingest_answer_audio(
                    q=q, audio_bytes=rec.getvalue(), suffix=".wav",
                )

        with tab_upload:
            # --- 1) Open Google Drive for the configured email account -----
            drive_url = _google_drive_account_url(GOOGLE_DRIVE_EMAIL)
            if GOOGLE_DRIVE_EMAIL:
                st.caption(f"📂 حساب Google Drive: **{GOOGLE_DRIVE_EMAIL}**")
            st.link_button(
                "📂 افتح Google Drive",
                drive_url,
                use_container_width=True,
                help="يفتح Google Drive للحساب المحدد في تبويب جديد لاختيار "
                     "الملف الصوتي.",
            )

            # --- 2) Import directly from a Drive share link ---------------
            drive_link = st.text_input(
                "أو الصق رابط مشاركة Google Drive للملف الصوتي",
                key=f"drvlink_{q.key}_{idx}",
                placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
                help="انسخ رابط المشاركة من Drive (مع ضبط الإذن على «أي شخص "
                     "لديه الرابط») ثم الصقه هنا.",
            )
            if st.button(
                "⬇️ استورد من Google Drive",
                key=f"drvimp_{q.key}_{idx}",
                use_container_width=True,
                disabled=not drive_link.strip() or bool(existing_audio_path),
            ):
                with st.spinner("جارٍ التنزيل من Google Drive…"):
                    data, fname, err = _download_drive_audio(drive_link.strip())
                if err:
                    st.error(err)
                elif data:
                    ext = Path(fname).suffix.lower() or ".wav"
                    if ext not in (".wav", ".mp3", ".m4a", ".ogg", ".webm"):
                        ext = ".wav"
                    _ingest_answer_audio(q=q, audio_bytes=data, suffix=ext)

            st.divider()

            # --- 3) Manual file picker (still available) -------------------
            uploaded = st.file_uploader(
                "أو اختر ملفًا صوتيًا من جهازك (WAV / MP3 / M4A / OGG / WEBM)",
                type=["wav", "mp3", "m4a", "ogg", "webm"],
                key=f"upl_{q.key}_{idx}",
                accept_multiple_files=False,
                help="مفيد إذا تم تسجيل صوت الطفل من تطبيق آخر "
                     "(مثل WhatsApp أو Voice Memos) أو بعد تنزيله من Drive.",
            )
            if uploaded is not None and not existing_audio_path:
                # Preserve the original extension when possible so Gemini gets
                # the right MIME type. Default to .wav for safety.
                ext = Path(uploaded.name).suffix.lower() or ".wav"
                if ext not in (".wav", ".mp3", ".m4a", ".ogg", ".webm"):
                    ext = ".wav"
                _ingest_answer_audio(
                    q=q, audio_bytes=uploaded.getvalue(), suffix=ext,
                )

        st.divider()

        # Navigation
        c_prev, c_next = st.columns(2)
        if c_prev.button("← السابق", disabled=idx == 0, use_container_width=True):
            ss.current_idx -= 1
            st.rerun()
        if c_next.button(
            "التالي →", disabled=idx >= len(questions) - 1,
            use_container_width=True,
        ):
            ss.current_idx += 1
            st.rerun()
        if st.button(
            "💾 إنهاء جلسة هذا الطفل وحساب الدرجات",
            type="primary", use_container_width=True,
            disabled=not ss.answers,
        ):
            ss.phase = "child_finished"
            st.rerun()
        if st.button(
            "⏸ احفظ مؤقتًا وأكمل لاحقًا",
            use_container_width=True,
            help="تُحفظ الإجابات الحالية في القرص ويظهر الطفل في قائمة "
                 "«📂 استئناف جلسة سابقة» على الصفحة الرئيسية.",
        ):
            _save_in_progress()
            st.success(
                f"✅ تم الحفظ مؤقتًا. ستجد «{ss.child_info['name']}» "
                "في قائمة الجلسات المتوقفة."
            )
            # Clear the per-child state but keep the batch intact so the
            # examiner can start another child right away.
            ss.child_info = None
            ss.session_id = ""
            ss.session_dir = None
            ss.current_idx = 0
            ss.answers = {}
            ss.phase = "intake"
            st.rerun()

    # ----- Below both columns: live answers table -------------------------
    st.divider()
    st.subheader("📝 إجابات الطفل المسجّلة")
    recorded = [
        (qq, ss.answers[qq.key])
        for qq in questions
        if qq.key in ss.answers and ss.answers[qq.key].get("audio_path")
    ]
    if not recorded:
        st.caption("لا توجد إجابات مسجّلة بعد.")
        return
    for qq, payload in recorded:
        with st.expander(f"{qq.order}: {qq.text}", expanded=False):
            transcript = (payload.get("transcript") or "").strip()
            err = (payload.get("transcription_error") or "").strip()
            if transcript:
                st.markdown(f"**النص:** {transcript}")
            elif err:
                st.error(err)
            else:
                st.caption("… جارٍ التفريغ النصي")
            audio = payload.get("audio_path", "")
            if audio and Path(audio).exists():
                st.audio(str(audio))


# --------------------------------------------------------------------------
# Phase: child_finished (score → append row → choose next action)
# --------------------------------------------------------------------------
def _reset_for_next_child() -> None:
    """Clear per-child state but KEEP the shared batch workbook intact."""
    ss.child_info = None
    ss.session_id = ""
    ss.session_dir = None
    ss.current_idx = 0
    ss.answers = {}


def render_child_finished() -> None:
    """After each child: score with Gemini, append to the shared Excel,
    then ask the examiner whether to start the next child or finalise the
    whole batch (download + email)."""
    child = ss.child_info
    if child is None:
        # Shouldn't happen but guard anyway.
        ss.phase = "intake"
        st.rerun()
        return

    # Only score+append if we haven't already done it for this child in this
    # run. ``_child_saved`` is set after a successful append so reruns of the
    # same phase (e.g. when the examiner clicks a button below) don't re-call
    # Gemini or duplicate the Excel row.
    if not ss.get("_child_saved"):
        st.title(f"💾 حفظ جلسة {child['name']}…")
        st.info("جارٍ تقييم الإجابات تلقائيًا عبر Gemini وإلحاق صفّ جديد بملف Excel المشترك…")
        ok, errors = _score_all()
        if not ok:
            st.warning("لم يتم تقييم أي إجابة تلقائيًا. سيتم إلحاق الصف بدون درجات.")
        if errors:
            with st.expander("تفاصيل أخطاء التقييم"):
                for e in errors[:20]:
                    st.code(e)
        try:
            row = _append_current_child_to_batch()
        except Exception as exc:  # noqa: BLE001
            st.error(f"تعذّر إلحاق الصف بملف Excel: {exc}")
            return
        # Flip the on-disk manifest so this child no longer appears in the
        # "resume" list — the row is now in the shared Excel.
        if ss.session_dir is not None:
            _mark_session_finished(ss.session_dir)
        ss.completed_children.append({
            "name": child["name"],
            "session_id": ss.session_id,
            "row": row,
        })
        ss._child_saved = True
        st.rerun()
        return

    # Successful save — show the summary + next-action buttons.
    st.title(f"✅ تم حفظ جلسة {child['name']}")
    last = ss.completed_children[-1]
    st.success(
        f"تم إلحاق صف جديد في ملف Excel المشترك (الصف رقم {last['row']}). "
        f"عدد الأطفال المكتملين حتى الآن: **{len(ss.completed_children)}**."
    )

    aggregated = aggregate_answers_to_columns(ss.answers)
    score_cols = {k: v for k, v in aggregated.items() if isinstance(v, int)}
    if score_cols:
        with st.expander("📊 الدرجات المحسوبة لهذا الطفل"):
            st.table({"العمود": list(score_cols.keys()),
                      "القيمة": list(score_cols.values())})

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("👶 ابدأ طفلًا جديدًا", type="primary", use_container_width=True):
        _reset_for_next_child()
        ss._child_saved = False
        ss.phase = "intake"
        st.rerun()
    if c2.button(
        "📊 إنهاء الجلسة الكاملة وإرسال Excel",
        use_container_width=True,
    ):
        _reset_for_next_child()
        ss._child_saved = False
        ss.phase = "batch_finished"
        st.rerun()


# --------------------------------------------------------------------------
# Phase: batch_finished (final Excel for ALL children → download + email)
# --------------------------------------------------------------------------
def render_batch_finished() -> None:
    st.title("📊 إنهاء الجلسة الكاملة")
    out_path: Path = ss.batch_excel_path
    if not ss.completed_children or not out_path.exists():
        st.warning(
            "لا يوجد أي طفل مكتمل في هذه الجلسة بعد. "
            "ارجع إلى صفحة الأسئلة وأكمل جلسة طفل واحد على الأقل."
        )
        if st.button("↩️ العودة", use_container_width=True):
            ss.phase = "intake" if ss.child_info is None else "questions"
            st.rerun()
        return

    st.success(
        f"اكتملت جلسات **{len(ss.completed_children)} طفل**. الملف جاهز للتحميل والإرسال."
    )

    # Completed children list
    with st.expander("👨‍👩‍👧‍👦 الأطفال المسجَّلة في هذا الملف", expanded=True):
        rows = [
            {"#": i + 1, "اسم الطفل": c["name"],
             "صف Excel": c["row"], "معرف الجلسة": c["session_id"]}
            for i, c in enumerate(ss.completed_children)
        ]
        st.table(rows)

    # Download
    st.subheader("⬇️ تحميل الملف")
    with out_path.open("rb") as f:
        st.download_button(
            "تحميل narrative_coding.xlsx",
            data=f,
            file_name=f"narrative_coding_batch_{ss.batch_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("👶 إضافة طفل آخر", type="primary", use_container_width=True):
        ss.phase = "intake"
        st.rerun()
    if c2.button(
        "🆕 ابدأ دفعة جديدة (ملف Excel جديد)",
        use_container_width=True,
    ):
        # Reset EVERYTHING — fresh batch id + new workbook.
        for k in (
            "child_info", "session_id", "session_dir", "current_idx", "answers",
            "completed_children", "batch_id", "batch_dir", "batch_excel_path",
            "_child_saved",
        ):
            ss.pop(k, None)
        ss.phase = "intake"
        st.rerun()


# --------------------------------------------------------------------------
# Phase: single-video analysis
# --------------------------------------------------------------------------
# The session recording may be a video OR an audio-only file. Gemini scores
# both, so accept the common formats for each.
VIDEO_TYPES = [
    # video
    "mp4", "mov", "webm", "mkv", "m4v", "avi",
    # audio
    "m4a", "mp3", "wav", "ogg", "aac", "flac",
]


def render_video_intake() -> None:
    """Upload ONE session video → extract only the child's answers →
    score every question → write a single Excel row → offer the download."""
    st.title("🎬 تحليل فيديو الجلسة الكاملة")
    st.caption(
        "ارفع ملفًا واحدًا للجلسة (فيديو أو صوت). سيستمع التطبيق لصوت "
        "الطفل فقط (متجاهلًا صوت المعلّم وصوت فيديو القصة)، ويحلّل "
        "إجاباته، ويملأ كل أعمدة ملف Excel دفعةً واحدة."
    )

    with st.form("video_intake", clear_on_submit=False):
        name = st.text_input("اسم الطفل *", placeholder="مثال: عبد الرحمن")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("العمر (سنوات)", min_value=2, max_value=17, value=6)
            gender = st.selectbox("الجنس", GENDERS, index=0)
        with col2:
            group = st.selectbox("المجموعة", GROUPS, index=0)
        notes = st.text_area(
            "ملاحظات (مدرسة، تشخيص سابق، لغة سائدة...)", height=80
        )
        uploaded = st.file_uploader(
            "🎥 ملف الجلسة (فيديو أو صوت) *",
            type=VIDEO_TYPES,
            accept_multiple_files=False,
            help="ملف واحد (فيديو mp4/mov… أو صوت m4a/mp3/wav…) يحتوي "
                 "على إجابات الطفل على كل الأسئلة.",
        )
        submitted = st.form_submit_button(
            "🚀 حلّل الملف واملأ Excel", use_container_width=True, type="primary"
        )

    if not submitted:
        return
    if not name.strip():
        st.error("اسم الطفل مطلوب.")
        return
    if uploaded is None:
        st.error("الرجاء رفع ملف الجلسة (فيديو أو صوت).")
        return

    scorer = _ensure_scorer()
    if scorer is None:
        st.error("تعذّر تهيئة Gemini. تأكد من ضبط GEMINI_API_KEY.")
        return

    # --- Persist the uploaded session file (video OR audio) ---------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    token = _safe_filename(name)
    session_id = f"{stamp}_{token}"
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded.name).suffix.lower() or ".mp4"
    video_path = session_dir / f"session_media{ext}"
    video_path.write_bytes(uploaded.getvalue())

    child_info = {
        "name": name.strip(),
        "age": int(age),
        "gender": gender,
        "group": group,
        "notes": notes.strip(),
    }

    # --- Full analysis of the whole video (fills EVERY column) -------------
    all_keys = load_all_keys(DEFAULT_JSON)
    with st.spinner(
        "⏳ جارٍ تحليل الفيديو بالكامل: استخراج كلام الطفل، تقييم الإجابات، "
        "وحساب كل أعمدة ملف Excel..."
    ):
        try:
            column_values = analyze_video_to_columns(
                video_path=video_path,
                questions=ss.questions,
                all_keys=all_keys,
                rubric_by_key=ss.rubric_by_key,
                scorer=scorer,
            )
        except GeminiScorerError as exc:
            st.error(f"فشل تحليل الفيديو: {exc}")
            return

    if not column_values:
        st.warning(
            "لم يتمكن التطبيق من استخراج أي بيانات للطفل من هذا الملف. "
            "تأكد من وضوح صوت الطفل في التسجيل."
        )
        return

    # --- Write ONE row with every column ----------------------------------
    out_path: Path = ss.batch_excel_path
    if not out_path.exists():
        create_workbook(DEFAULT_JSON, out_path)
    meta = make_session_meta(
        session_id=session_id,
        child_name=child_info["name"],
        child_age=str(child_info["age"]),
        audio_dir=session_dir,
        notes=child_info.get("notes", ""),
        child_gender=child_info.get("gender", ""),
        child_group=child_info.get("group", ""),
    )
    row = append_session(DEFAULT_JSON, out_path, meta, answers=column_values)

    filled = sum(1 for v in column_values.values() if v not in (None, ""))
    st.success(
        f"✅ تم تحليل الفيديو وتعبئة **{filled} عمودًا** في صف Excel واحد "
        f"للطفل «{child_info['name']}» (الصف رقم {row})."
    )

    # --- Show every filled column -----------------------------------------
    st.subheader("📋 كل الأعمدة المستخرجة من التحليل")
    desc_by_key = {k["key"]: k.get("description", "") for k in all_keys}
    table_rows = [
        {
            "العمود": key,
            "القيمة": column_values.get(key),
            "الوصف": (desc_by_key.get(key, "") or "")[:60],
        }
        for key in (k["key"] for k in all_keys)
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # --- Offer the workbook for download ----------------------------------
    with out_path.open("rb") as f:
        st.download_button(
            "⬇️ تحميل ملف Excel",
            data=f,
            file_name="narrative_coding.xlsx",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet",
            use_container_width=True,
        )

    if st.button("🎬 تحليل فيديو لطفل آخر", use_container_width=True):
        st.rerun()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
# Single-video analysis is the only supported flow now: upload ONE video and
# the app fills every Excel column. The legacy per-question recording phases
# (intake / questions / child_finished / batch_finished) are no longer wired
# into the router.
PHASES = {
    "video_intake": render_video_intake,
}
PHASES.get(ss.phase, render_video_intake)()
