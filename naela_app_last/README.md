# Naela – Children's Narrative Assessment App

A small desktop application (Python + PyQt5) that lets a clinician show the
Naela story video or PDF to a child, walk through the Narrative Comprehension
Protocol (NCP) questions one at a time, **record the child's voice answer per
question**, and save everything – including **one new row per child** in a
master Excel workbook whose columns mirror the JSON keys in
[`data/extracted_keys.json`](data/extracted_keys.json:1).

The app supports **assessing many children in the same session** without
restarting: every time you finish a child, click **"👶 طفل جديد"** and the
intake dialog pops up again for the next child. All children land in the same
Excel file, one row each.

---

## 1. What is inside

```
.
├── data/
│   ├── extracted_keys.json   # Coding scheme (Arabic) – every key = one Excel column
│   ├── instructions.docx
│   ├── story.pdf             # The story shown to the child
│   └── video.mp4             # The narrated video shown to the child
├── app/
│   ├── main_window.py        # PyQt5 main window (video / story toggle + questions)
│   ├── pdf_view.py           # Simple PDF viewer (PyMuPDF)
│   ├── recorder.py           # Microphone recorder (sounddevice → WAV)
│   ├── questions.py          # Extracts NCP questions from the JSON
│   └── excel_export.py       # Builds / appends to the master Excel workbook
├── output/
│   ├── narrative_coding.xlsx           # master workbook (created on first save)
│   └── sessions/<session_id>/          # per-session audio + JSON manifest
├── requirements.txt
├── run_app.py                # python run_app.py
└── generate_excel.py         # python generate_excel.py [out.xlsx]
```

---

## 2. Install

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> macOS users will be asked for **microphone permission** the first time the
> app records – allow it in *System Settings → Privacy & Security → Microphone*.

---

## 3. Run the application

```bash
python run_app.py
# or, equivalently
python -m app
```

You will see a window with:

| Area | What it does |
|------|--------------|
| **Child intake dialog** | Pops up automatically. Enter the child's name, age, gender, group (TD / ASD / DLD / other) and notes, then click **«ابدأ الجلسة»**. |
| **Current child banner** | Top-right shows the active child (#1, #2, …), age, gender, group. |
| **👶 طفل جديد** | Toolbar button – save the current child (if needed) and re-open the intake dialog for the next child. |
| **📂 افتح Excel** | Opens the master workbook in the OS default app so you can review live. |
| **🎬 عرض الفيديو** | Plays [`data/video.mp4`](data/video.mp4:1) with play / pause / stop + seek slider. |
| **📖 عرض القصة** | Renders [`data/story.pdf`](data/story.pdf:1) page by page (previous / next). |
| Question panel | Walks through the NCP questions in order. Each question shows the Arabic prompt that should be asked to the child. |
| **🎙️ سجّل الإجابة** | Starts recording the child's answer to **this** question. |
| **⏹ أوقف التسجيل** | Stops & saves the WAV file (one per question). |
| **💾 إنهاء جلسة هذا الطفل** | Saves the session manifest and appends **one new row** to `output/narrative_coding.xlsx`. |
| Completed-children list | Bottom-right shows every child finished in this run with their Excel row number. |

### Multi-child flow at a glance

```
[Intake child A] → record 14 answers → 💾 إنهاء → row 7 written
[Intake child B] → record 14 answers → 💾 إنهاء → row 8 written
[Intake child C] → record 14 answers → 💾 إنهاء → row 9 written
                       …                              …
```

After the last question is recorded the app automatically asks whether you
want to save the current child and start the next one.

Each session is stored in `output/sessions/<timestamp>_<child_name>/`:

```
output/sessions/20260101_140530_ab12cd/
├── q01_ncp_u1_setting_literal_correct.wav
├── q02_ncp_u1_setting_description_response.wav
├── …
└── session.json     # metadata + path of every audio answer
```

---

## 4. Generate / regenerate the Excel template only

If you just need the workbook without running the GUI:

```bash
python generate_excel.py
# → output/narrative_coding_template.xlsx
```

### Workbook layout

The workbook has **one column per JSON key** taken from `extracted_keys.json`.
The first rows of the sheet form a self-documenting header:

| Row | Content |
|-----|---------|
| 1 | General coding rule (the Arabic text supplied in the spec). |
| 2 | Section name (e.g. *ثانيًا: قياسات البنية الكلية للسرد*). |
| 3 | Topic in English (e.g. *Narrative Macrostructure Measures*). |
| 4 | Column id = the JSON key (e.g. `setting_score`). |
| 5 | The full Arabic description copied verbatim from the JSON. |
| 6 | The scoring scale inferred from the key name (see table below). |
| 7… | One row per session – filled by the analyst. |

#### Scale inferred per column

| Suffix in JSON key                                         | Scale shown in row 6 |
|------------------------------------------------------------|----------------------|
| `_score` (non-total)                                       | **Qualitative 0 / 1 / 2** (`غائب / جزئي / واضح`) |
| `_correct`, `_accuracy`, `_speech_correct`, `_response`    | **Binary 0 / 1** |
| `_count`, `_tokens`, `_raw`, `TNW`, `NDW`                  | Integer count |
| `_percentage`, `_ratio`, `_density`, `TTR_*`, `MLU_*`      | Float / percentage |
| `_list`, `_examples_optional`, `justification_type`        | Free text |
| `*_total_score`, `macro_*_score`                           | Computed total |

This matches the general rule:

> قاعدة الترميز العامة: تُرمّز البنود الثنائية كالتالي: 0 = غير موجود أو غير صحيح، 1 = موجود أو صحيح.
> وتُرمّز البنود النوعية أو المركبة كالتالي: 0 = غائب، 1 = جزئي أو محدود، 2 = واضح أو مكتمل.
> أي اختلاف في سلم الترميز يجب أن يُذكر صراحة داخل البند.

The sheet is rendered right-to-left so Arabic reads naturally, and the
metadata + header rows are frozen so the analyst always sees the column
labels while scrolling.

---

## 5. Notes for the analyst

* Coding is now **auto-filled by Gemini** when you press *إنهاء جلسة هذا الطفل*.
  The desktop app sends each recorded WAV to Google Gemini and writes the
  returned 0/1 or 0/1/2 score into the matching Excel column. Macrostructure
  columns (`setting_score`, `plan_score`, …) and `macro_total_score` are
  computed by aggregating the related NCP answers.
* Each new session **appends** a new row to the same workbook, so all
  sessions are coded side by side.
* The `audio_dir` column on each row points to the session folder, so you can
  always trace a coded row back to the original recordings.

---

## 6. 🌐 Mobile / web version (Streamlit)

The same scoring engine is exposed as a **mobile-friendly web app** so a
clinician can run a session from a phone or tablet:

```bash
.venv/bin/streamlit run streamlit_app.py
```

Open the URL Streamlit prints (e.g. `http://localhost:8501`) on the iPad /
phone — the browser will ask for microphone permission, then walk through:

```
[Child intake form]  →  [Video / PDF tabs]  →  [Question 1 of N + 🎙️ recorder]
                                              ⋮
[Question N]  →  [💾 Finish & score]  →  [Download Excel] + [📧 Email Excel]
```

Every child in the same browser session is appended as a new row to **one
shared `narrative_coding.xlsx`**. After the last child you press
*"📊 إنهاء الجلسة الكاملة"* and the workbook is offered as a single
download button.

### 6.1 Configure secrets

For local dev, put your keys in a [`.env`](.env) file at the project root
(already auto-loaded by [`app/__init__.py`](app/__init__.py:1)):

```env
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.5-flash-lite
# Unlisted YouTube URL of the story video — see §6.2.
VIDEO_URL=https://youtu.be/XXXXXXXXXXX
```

For **Streamlit Cloud**, paste the same key/value pairs into the app's
*Secrets* panel (see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)).

### 6.2 Where to host the story video (FREE)

The narrative video [`data/video.mp4`](data/video.mp4) is **545 MB** — too big
for GitHub (100 MB per-file cap) and unsuitable for Streamlit Cloud's small
ephemeral disk. The web app reads `VIDEO_URL` instead so the video is streamed
from a free third-party host.

**Recommended: YouTube as Unlisted**

1. Go to https://studio.youtube.com → **CREATE → Upload video**.
2. Drop your `data/video.mp4`.
3. While it uploads, set:
   * **Title** anything (e.g. *Naela story*).
   * **Made for kids: No** (avoids COPPA restrictions).
   * Skip ad / monetisation / classification screens.
4. On the **Visibility** screen pick **Unlisted** (NOT *Private* — anyone with
   the link can watch but it won't appear in search).
5. After processing finishes, copy the share URL (something like
   `https://youtu.be/abcdEFGH123`).
6. Paste it into [`.env`](.env) (`VIDEO_URL=https://youtu.be/abcdEFGH123`) or
   into your Streamlit Cloud secrets, then restart the app.

The video file [`data/video.mp4`](data/video.mp4) stays on your laptop for the
PyQt5 desktop app, but [`.gitignore`](.gitignore) excludes it from commits, so
GitHub never sees it.

### 6.3 Deploy to Streamlit Cloud (free, public URL)

1. Push this repo to a GitHub repository (the [`.gitignore`](.gitignore)
   already excludes `.env` and `.streamlit/secrets.toml`).
2. Go to https://share.streamlit.io → **New app**.
3. Repository: your fork. **Main file path**: `streamlit_app.py`. Python
   version: 3.11+.
4. Click **Advanced settings → Secrets** and paste the contents of
   `.streamlit/secrets.toml.example` (with real values).
5. Deploy. Streamlit Cloud installs from [`requirements.txt`](requirements.txt)
   and gives you a public `*.streamlit.app` URL you can open on any phone.

> **Note on mic permissions:** browsers only allow microphone access on
> `https://` URLs and on `http://localhost`. Streamlit Cloud always serves
> HTTPS, so the mic widget works out of the box on phones.

### 6.4 What's reused vs. new

| Module | Used by desktop? | Used by Streamlit? |
|---|---|---|
| [`app/questions.py`](app/questions.py:1) | ✅ | ✅ |
| [`app/scoring.py`](app/scoring.py:1) | ✅ | ✅ |
| [`app/gemini_scorer.py`](app/gemini_scorer.py:1) | ✅ | ✅ |
| [`app/excel_export.py`](app/excel_export.py:1) | ✅ | ✅ |
| [`app/emailer.py`](app/emailer.py:1) | — | ✅ |
| [`app/main_window.py`](app/main_window.py:1) (PyQt5) | ✅ | — |
| [`app/recorder.py`](app/recorder.py:1) (sounddevice) | ✅ | — (browser uses `st.audio_input`) |
| [`streamlit_app.py`](streamlit_app.py:1) | — | ✅ |
