"""Main PyQt5 window for the Naela narrative-assessment app.

Multi-child workflow
--------------------
The app is designed to assess **many children** in one sitting. The clinician:

1. Sees a child-intake dialog asking for the next child's name, age, gender,
   group and notes (see :class:`ChildIntakeDialog`).
2. Watches the story (video or PDF) with the child while walking through the
   NCP questions one at a time.
3. Records the child's voice answer per question (one WAV file per question).
4. Saves the session, which:
       * writes ``session.json`` to the child's audio folder,
       * appends a **new row** to the shared Excel workbook
         (``output/narrative_coding.xlsx``).
5. Immediately gets prompted to start the next child – which resets the
   question index, clears recorded answers, creates a new session folder, and
   re-opens the intake dialog.

A "completed children" list on the side panel shows everyone assessed so far
in this run, so the clinician can see progress at a glance.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Make sure Gemini diagnostics (transcription failures etc.) actually appear
# in the terminal, instead of being swallowed by the default WARNING handler.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .child_dialog import ChildInfo, ChildIntakeDialog, build_session_id
from .excel_export import append_session, create_workbook, make_session_meta
from .gemini_scorer import GeminiScorer, GeminiScorerError
from .pdf_view import PdfViewer
from .questions import Question, load_questions
from .recorder import AudioRecorder
from .scoring import aggregate_answers_to_columns


# Project paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SESSIONS_DIR = OUTPUT_DIR / "sessions"
DEFAULT_JSON = DATA_DIR / "extracted_keys.json"
DEFAULT_VIDEO = DATA_DIR / "video.mp4"
DEFAULT_PDF = DATA_DIR / "story.pdf"
DEFAULT_WORKBOOK = OUTPUT_DIR / "narrative_coding.xlsx"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Naela – تقييم السرد القصصي للأطفال")
        self.resize(1320, 820)
        self.setLayoutDirection(Qt.RightToLeft)

        # Static state -------------------------------------------------
        self._questions: List[Question] = load_questions(DEFAULT_JSON)
        self._recorder = AudioRecorder()
        self._completed_children: List[dict] = []

        # Per-child (will be reset by _reset_for_new_child) ------------
        self._child: Optional[ChildInfo] = None
        self._current_idx: int = 0
        self._session_id: str = ""
        self._session_dir: Path = SESSIONS_DIR  # placeholder until first child
        self._answers: dict[str, dict] = {}
        # Cached Gemini scorer + background-thread bookkeeping for transcription.
        self._gemini_scorer = None  # type: Optional[GeminiScorer]
        self._transcribe_threads: List["_TranscribeThread"] = []

        # Build UI ------------------------------------------------------
        self._build_ui()
        self._load_media()

        # Ask for the first child as soon as the window is shown.
        # (Use a 0-ms timer so the window can paint first.)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._prompt_for_new_child)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ============== LEFT: media panel ============================
        media_panel = QVBoxLayout()

        toggle_row = QHBoxLayout()
        self._btn_show_video = QPushButton("🎬 عرض الفيديو")
        self._btn_show_story = QPushButton("📖 عرض القصة")
        for btn in (self._btn_show_video, self._btn_show_story):
            btn.setCheckable(True)
            btn.setMinimumHeight(40)
            btn.setFont(QFont("", 12, QFont.Bold))
        self._btn_show_video.setChecked(True)
        self._btn_show_video.clicked.connect(lambda: self._show_media("video"))
        self._btn_show_story.clicked.connect(lambda: self._show_media("story"))
        toggle_row.addWidget(self._btn_show_video)
        toggle_row.addWidget(self._btn_show_story)
        media_panel.addLayout(toggle_row)

        self._media_stack = QStackedWidget()

        # Video page.
        video_page = QWidget()
        video_layout = QVBoxLayout(video_page)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumSize(QSize(640, 360))
        video_layout.addWidget(self._video_widget, 1)

        self._media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self._media_player.setVideoOutput(self._video_widget)
        self._media_player.positionChanged.connect(self._on_position_changed)
        self._media_player.durationChanged.connect(self._on_duration_changed)

        controls = QHBoxLayout()
        self._btn_play = QPushButton("▶ تشغيل")
        self._btn_pause = QPushButton("⏸ إيقاف مؤقت")
        self._btn_stop = QPushButton("⏹ إيقاف")
        self._btn_play.clicked.connect(self._media_player.play)
        self._btn_pause.clicked.connect(self._media_player.pause)
        self._btn_stop.clicked.connect(self._media_player.stop)
        for b in (self._btn_play, self._btn_pause, self._btn_stop):
            controls.addWidget(b)
        video_layout.addLayout(controls)

        self._position_slider = QSlider(Qt.Horizontal)
        self._position_slider.setRange(0, 0)
        self._position_slider.sliderMoved.connect(self._media_player.setPosition)
        video_layout.addWidget(self._position_slider)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self._time_label)

        # PDF page.
        self._pdf_view = PdfViewer()

        self._media_stack.addWidget(video_page)
        self._media_stack.addWidget(self._pdf_view)
        media_panel.addWidget(self._media_stack, 1)

        root.addLayout(media_panel, 3)

        # ============== RIGHT: side panel ============================
        side = QVBoxLayout()

        # ---- Current child banner ---------------------------------
        self._child_banner = QLabel()
        self._child_banner.setWordWrap(True)
        self._child_banner.setStyleSheet(
            "background:#2c3e50; color:white; padding:10px;"
            " border-radius:8px; font-size:13px;"
        )
        self._child_banner.setMinimumHeight(70)
        side.addWidget(self._child_banner)

        # ---- "New child" toolbar -----------------------------------
        toolbar = QHBoxLayout()
        self._btn_new_child = QPushButton("👶 طفل جديد")
        self._btn_new_child.setMinimumHeight(36)
        self._btn_new_child.setStyleSheet(
            "background:#3498db; color:white; font-weight:bold;"
        )
        self._btn_new_child.clicked.connect(self._on_new_child_clicked)
        toolbar.addWidget(self._btn_new_child)

        self._btn_open_excel = QPushButton("📂 افتح Excel")
        self._btn_open_excel.setMinimumHeight(36)
        self._btn_open_excel.clicked.connect(self._open_workbook_externally)
        toolbar.addWidget(self._btn_open_excel)
        side.addLayout(toolbar)

        side.addWidget(self._make_separator())

        # ---- Question panel ---------------------------------------
        self._progress_label = QLabel()
        self._progress_label.setFont(QFont("", 11, QFont.Bold))
        side.addWidget(self._progress_label)

        self._question_text = QLabel()
        self._question_text.setWordWrap(True)
        self._question_text.setFont(QFont("", 14, QFont.Bold))
        self._question_text.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self._question_text.setStyleSheet(
            "background:#f7f3e9; color:#222; padding:12px; border-radius:8px;"
        )
        self._question_text.setMinimumHeight(110)
        side.addWidget(self._question_text)

        self._question_kind_label = QLabel()
        self._question_kind_label.setAlignment(Qt.AlignRight)
        side.addWidget(self._question_kind_label)

        rec_row = QHBoxLayout()
        self._btn_record = QPushButton("🎙️ سجّل الإجابة")
        self._btn_record.setMinimumHeight(46)
        self._btn_record.setStyleSheet(
            "background:#c0392b; color:white; font-weight:bold; font-size:14px;"
        )
        self._btn_record.clicked.connect(self._start_recording)
        rec_row.addWidget(self._btn_record)

        self._btn_stop_record = QPushButton("⏹ أوقف التسجيل")
        self._btn_stop_record.setMinimumHeight(46)
        self._btn_stop_record.setEnabled(False)
        self._btn_stop_record.setStyleSheet(
            "background:#2c3e50; color:white; font-weight:bold; font-size:14px;"
        )
        self._btn_stop_record.clicked.connect(self._stop_recording)
        rec_row.addWidget(self._btn_stop_record)

        self._btn_reset_answer = QPushButton("🔁 إعادة الإجابة")
        self._btn_reset_answer.setMinimumHeight(46)
        self._btn_reset_answer.setToolTip(
            "احذف الإجابة الحالية للسؤال وسجّل من جديد إذا تم الضغط بالخطأ."
        )
        self._btn_reset_answer.setStyleSheet(
            "background:#e67e22; color:white; font-weight:bold; font-size:14px;"
        )
        self._btn_reset_answer.clicked.connect(self._reset_current_answer)
        rec_row.addWidget(self._btn_reset_answer)
        side.addLayout(rec_row)

        self._recording_status = QLabel("جاهز للتسجيل.")
        self._recording_status.setAlignment(Qt.AlignCenter)
        side.addWidget(self._recording_status)

        # Informational note: scoring is automatic via Gemini after finish.
        self._scoring_hint = QLabel(
            "🤖 سيتم تحليل التسجيلات وحساب الدرجات تلقائيًا عند إنهاء الجلسة "
            "(باستخدام Gemini)."
        )
        self._scoring_hint.setWordWrap(True)
        self._scoring_hint.setStyleSheet(
            "color:#555; font-size:11px; padding:4px;"
            " background:#fdf6e3; border:1px solid #e0d9c4; border-radius:4px;"
        )
        side.addWidget(self._scoring_hint)

        side.addWidget(QLabel("ملاحظات الفاحص لهذا السؤال (اختياري):"))
        self._notes_edit = QTextEdit()
        self._notes_edit.setMaximumHeight(60)
        side.addWidget(self._notes_edit)

        nav_row = QHBoxLayout()
        self._btn_prev_q = QPushButton("← السابق")
        self._btn_next_q = QPushButton("التالي →")
        self._btn_prev_q.clicked.connect(self._go_prev)
        self._btn_next_q.clicked.connect(self._go_next)
        nav_row.addWidget(self._btn_prev_q)
        nav_row.addWidget(self._btn_next_q)
        side.addLayout(nav_row)

        self._btn_finish = QPushButton("💾 إنهاء جلسة هذا الطفل")
        self._btn_finish.setMinimumHeight(44)
        self._btn_finish.setStyleSheet(
            "background:#27ae60; color:white; font-weight:bold; font-size:14px;"
        )
        self._btn_finish.clicked.connect(self._finish_current_child)
        side.addWidget(self._btn_finish)

        side.addWidget(self._make_separator())

        # ---- Answers table (Question | Transcript | Play) ----------
        answers_header = QHBoxLayout()
        answers_header.addWidget(QLabel("📝 إجابات الطفل المسجّلة:"))
        self._btn_retranscribe = QPushButton("🔄 إعادة التفريغ النصي")
        self._btn_retranscribe.setToolTip(
            "أعد إرسال جميع التسجيلات إلى Gemini للحصول على النص."
        )
        self._btn_retranscribe.clicked.connect(self._retranscribe_all_answers)
        answers_header.addWidget(self._btn_retranscribe)
        side.addLayout(answers_header)

        self._answers_table = QTableWidget(0, 3)
        self._answers_table.setHorizontalHeaderLabels(
            ["السؤال", "إجابة الطفل (نص)", "تشغيل"]
        )
        self._answers_table.verticalHeader().setVisible(False)
        self._answers_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._answers_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._answers_table.setAlternatingRowColors(True)
        self._answers_table.setWordWrap(True)
        header = self._answers_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._answers_table.setMinimumHeight(180)
        side.addWidget(self._answers_table, 1)

        # Dedicated media player for answer playback (separate from story video).
        self._answer_player = QMediaPlayer(self)

        side.addWidget(self._make_separator())

        # ---- Completed children list ------------------------------
        side.addWidget(QLabel("👨‍👩‍👧‍👦 الأطفال المنتهية جلساتهم في هذه الجلسة:"))
        self._completed_list = QListWidget()
        self._completed_list.setMaximumHeight(120)
        side.addWidget(self._completed_list)

        side.addStretch()

        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setMinimumWidth(420)
        root.addWidget(side_container, 2)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            f"ملف النتائج: {DEFAULT_WORKBOOK} — كل طفل = صف جديد."
        )

        # Initial enable/disable until a child is loaded.
        self._set_session_widgets_enabled(False)

    @staticmethod
    def _make_separator() -> QWidget:
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#bbb;")
        return sep

    def _set_session_widgets_enabled(self, enabled: bool) -> None:
        """Enable/disable widgets that need an active child session."""
        for w in (
            self._btn_record,
            self._btn_reset_answer,
            self._btn_prev_q,
            self._btn_next_q,
            self._btn_finish,
            self._notes_edit,
        ):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Media handling
    # ------------------------------------------------------------------
    def _load_media(self) -> None:
        if DEFAULT_VIDEO.exists():
            self._media_player.setMedia(QMediaContent(QUrl.fromLocalFile(str(DEFAULT_VIDEO))))
        else:
            QMessageBox.warning(
                self,
                "ملف الفيديو غير موجود",
                f"تعذّر العثور على ملف الفيديو:\n{DEFAULT_VIDEO}",
            )
        if DEFAULT_PDF.exists():
            self._pdf_view.load(DEFAULT_PDF)

    def _show_media(self, which: str) -> None:
        if which == "video":
            self._media_stack.setCurrentIndex(0)
            self._btn_show_video.setChecked(True)
            self._btn_show_story.setChecked(False)
        else:
            self._media_player.pause()
            self._media_stack.setCurrentIndex(1)
            self._btn_show_video.setChecked(False)
            self._btn_show_story.setChecked(True)

    def _on_position_changed(self, position: int) -> None:
        self._position_slider.setValue(position)
        self._time_label.setText(
            f"{_fmt_ms(position)} / {_fmt_ms(self._media_player.duration())}"
        )

    def _on_duration_changed(self, duration: int) -> None:
        self._position_slider.setRange(0, duration)

    # ------------------------------------------------------------------
    # Multi-child workflow
    # ------------------------------------------------------------------
    def _on_new_child_clicked(self) -> None:
        """Toolbar handler – warn if a session is in progress, then start fresh."""
        if self._child is not None and self._answers and not self._is_current_finished():
            reply = QMessageBox.question(
                self,
                "جلسة قيد التنفيذ",
                "يوجد جلسة لطفل لم تكتمل بعد. هل تريد حفظها أولاً قبل بدء طفل جديد؟",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Save:
                if not self._finish_current_child():
                    return
        self._prompt_for_new_child()

    def _prompt_for_new_child(self) -> None:
        """Show the intake dialog and start the next child's session."""
        if self._recorder.is_recording:
            self._stop_recording()

        next_index = len(self._completed_children) + 1
        info = ChildIntakeDialog.get_child_info(self, default_index=next_index)
        if info is None:
            # User cancelled. If we have no active child at all, the UI stays
            # disabled until they press "👶 طفل جديد" again.
            if self._child is None:
                self._child_banner.setText(
                    "لم يتم بدء أي جلسة بعد. اضغط «👶 طفل جديد» للبدء."
                )
            return

        self._start_session_for(info)

    def _start_session_for(self, info: ChildInfo) -> None:
        """Reset all per-child state and prepare the UI for the new child."""
        self._child = info
        self._session_id = build_session_id(info)
        self._session_dir = SESSIONS_DIR / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._answers = {}
        self._current_idx = 0
        self._notes_edit.clear()
        # Stop any lingering playback and reset the answers table.
        self._answer_player.stop()
        self._refresh_answers_table()

        self._set_session_widgets_enabled(True)
        self._update_child_banner()
        self._update_question_panel()
        self.statusBar().showMessage(
            f"جلسة جديدة • {info.name} • معرف الجلسة: {self._session_id}"
        )

        # Auto-rewind & switch to video for the new child.
        self._media_player.stop()
        self._show_media("video")

    def _update_child_banner(self) -> None:
        if self._child is None:
            self._child_banner.setText(
                "لم يتم بدء أي جلسة بعد. اضغط «👶 طفل جديد» للبدء."
            )
            return
        idx = len(self._completed_children) + 1
        c = self._child
        self._child_banner.setText(
            f"<b>الطفل #{idx}: {c.name}</b><br>"
            f"العمر: {c.age} • الجنس: {c.gender} • المجموعة: {c.group}"
        )

    # ------------------------------------------------------------------
    # Question flow
    # ------------------------------------------------------------------
    def _update_question_panel(self) -> None:
        if self._child is None or not self._questions:
            self._progress_label.setText("—")
            self._question_text.setText("ابدأ جلسة لطفل لرؤية الأسئلة.")
            self._question_kind_label.setText("")
            self._recording_status.setText("")
            return

        q = self._questions[self._current_idx]
        self._progress_label.setText(
            f"السؤال {self._current_idx + 1} من {len(self._questions)} — {q.key}"
        )
        self._question_text.setText(q.text)
        kind_ar = {
            "primary": "سؤال الفهم",
            "followup": "سؤال المتابعة",
            "macro": "بند سردي",
        }.get(q.kind, q.kind)
        self._question_kind_label.setText(f"نوع السؤال: {kind_ar}")

        if q.key in self._answers:
            audio = self._answers[q.key].get("audio_path", "")
            if audio:
                self._recording_status.setText(f"✓ تم التسجيل: {Path(audio).name}")
                self._recording_status.setStyleSheet("color:#27ae60; font-weight:bold;")
            else:
                self._recording_status.setText("جاهز للتسجيل.")
                self._recording_status.setStyleSheet("")
        else:
            self._recording_status.setText("جاهز للتسجيل.")
            self._recording_status.setStyleSheet("")

        self._btn_prev_q.setEnabled(self._current_idx > 0)
        self._btn_next_q.setEnabled(self._current_idx < len(self._questions) - 1)

    def _go_prev(self) -> None:
        if self._recorder.is_recording:
            self._stop_recording()
        if self._current_idx > 0:
            self._current_idx -= 1
            self._update_question_panel()

    def _go_next(self) -> None:
        if self._recorder.is_recording:
            self._stop_recording()
        if self._current_idx < len(self._questions) - 1:
            self._current_idx += 1
            self._update_question_panel()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _reset_current_answer(self) -> None:
        """Discard the recorded answer for the currently displayed question.

        Used when the examiner clicked the wrong question / mis-recorded and
        wants to start over for the same question. Stops any active recording,
        deletes the WAV file from disk, removes the entry from ``self._answers``,
        and refreshes the UI status.
        """
        if self._child is None or not self._questions:
            return
        q = self._questions[self._current_idx]

        # Stop a live recording without saving it as the "answer".
        if self._recorder.is_recording:
            try:
                self._recorder.stop()
            except Exception:  # noqa: BLE001
                pass
            self._btn_record.setEnabled(True)
            self._btn_stop_record.setEnabled(False)

        existing = self._answers.get(q.key)
        if not existing:
            # Nothing recorded yet – just reset the UI to "ready".
            self._notes_edit.clear()
            self._recording_status.setText("جاهز للتسجيل.")
            self._recording_status.setStyleSheet("")
            return

        # Confirm before deleting (especially if the audio file exists).
        audio_path = existing.get("audio_path", "")
        reply = QMessageBox.question(
            self,
            "إعادة الإجابة",
            (
                f"هل تريد حذف الإجابة الحالية للسؤال «{q.key}» وإعادة التسجيل؟"
                + (f"\n\nسيتم حذف الملف:\n{Path(audio_path).name}" if audio_path else "")
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Delete the wav file from the session folder.
        if audio_path:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                # Non-fatal – the entry will be cleared either way.
                pass

        # Drop the entry entirely so re-recording behaves as if it was never asked.
        self._answers.pop(q.key, None)
        self._notes_edit.clear()
        self._recording_status.setText("تمت إعادة الإجابة. جاهز للتسجيل من جديد.")
        self._recording_status.setStyleSheet("color:#e67e22; font-weight:bold;")
        self._auto_save_manifest()
        self._refresh_answers_table()

    def _start_recording(self) -> None:
        if self._child is None:
            QMessageBox.information(
                self, "ابدأ جلسة", "الرجاء بدء جلسة لطفل قبل التسجيل."
            )
            return
        if self._recorder.is_recording:
            return
        q = self._questions[self._current_idx]
        filename = f"q{q.order:02d}_{q.key}.wav"
        out_path = self._session_dir / filename
        try:
            self._recorder.start(out_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "خطأ في التسجيل", str(exc))
            return
        self._btn_record.setEnabled(False)
        self._btn_stop_record.setEnabled(True)
        self._recording_status.setText("● جاري التسجيل…")
        self._recording_status.setStyleSheet("color:#c0392b; font-weight:bold;")

    def _stop_recording(self) -> None:
        if not self._recorder.is_recording:
            return
        try:
            saved = self._recorder.stop()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "خطأ في حفظ التسجيل", str(exc))
            saved = None
        self._btn_record.setEnabled(True)
        self._btn_stop_record.setEnabled(False)
        if saved is None:
            self._recording_status.setText("تعذّر حفظ التسجيل.")
            self._recording_status.setStyleSheet("color:#c0392b;")
            return
        q = self._questions[self._current_idx]
        self._answers[q.key] = {
            "question_order": q.order,
            "question_text": q.text,
            "audio_path": str(saved),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "examiner_notes": self._notes_edit.toPlainText().strip(),
        }
        self._recording_status.setText(f"✓ تم الحفظ: {saved.name}")
        self._recording_status.setStyleSheet("color:#27ae60; font-weight:bold;")
        self._auto_save_manifest()
        # Add an immediate "transcribing…" row and try to transcribe in the
        # background so the examiner sees the text appear shortly after recording.
        self._refresh_answers_table()
        self._transcribe_answer_async(q.key)

        if self._current_idx == len(self._questions) - 1:
            self._maybe_auto_finish()

    def _maybe_auto_finish(self) -> None:
        if self._is_current_finished():
            reply = QMessageBox.question(
                self,
                "اكتملت الأسئلة",
                "تم تسجيل إجابات لجميع الأسئلة. هل تريد حفظ الجلسة وبدء طفل جديد؟",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                if self._finish_current_child():
                    self._prompt_for_new_child()

    def _is_current_finished(self) -> bool:
        return bool(self._questions) and all(
            q.key in self._answers for q in self._questions
        )

    # ------------------------------------------------------------------
    # Save / open Excel
    # ------------------------------------------------------------------
    def _auto_save_manifest(self) -> None:
        if self._child is None:
            return
        manifest = {
            "session_id": self._session_id,
            "child": {
                "name": self._child.name,
                "age": self._child.age,
                "gender": self._child.gender,
                "group": self._child.group,
                "notes": self._child.notes,
            },
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "answers": self._answers,
        }
        (self._session_dir / "session.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _finish_current_child(self) -> bool:
        """Persist current child's row to Excel. Returns True on success."""
        if self._child is None:
            return False
        if self._recorder.is_recording:
            self._stop_recording()
        if not self._answers:
            QMessageBox.warning(
                self,
                "لا توجد إجابات",
                "لم يتم تسجيل أي إجابة لهذا الطفل بعد.",
            )
            return False

        self._auto_save_manifest()

        # Auto-score every recorded answer with Gemini, then aggregate into
        # macrostructure columns + macro_total_score.
        scored_ok = self._auto_score_answers_with_gemini()
        if not scored_ok:
            # User chose to abort instead of saving with missing scores.
            return False
        column_values = aggregate_answers_to_columns(self._answers)

        meta = make_session_meta(
            session_id=self._session_id,
            child_name=self._child.name,
            child_age=str(self._child.age),
            audio_dir=self._session_dir,
            notes=self._child.notes,
            child_gender=self._child.gender,
            child_group=self._child.group,
        )
        try:
            DEFAULT_WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
            row = append_session(
                json_path=DEFAULT_JSON,
                workbook_path=DEFAULT_WORKBOOK,
                session_meta=meta,
                answers=column_values,
            )
        except PermissionError:
            QMessageBox.critical(
                self,
                "تعذّر حفظ Excel",
                (
                    "الملف مفتوح في برنامج آخر. الرجاء إغلاق ملف Excel ثم "
                    "إعادة المحاولة:\n" + str(DEFAULT_WORKBOOK)
                ),
            )
            return False
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "خطأ في حفظ Excel", str(exc))
            return False

        # Track in the completed list.
        record = {
            "child_name": self._child.name,
            "session_id": self._session_id,
            "row": row,
            "answers_count": len(self._answers),
            "total_questions": len(self._questions),
        }
        self._completed_children.append(record)
        item = QListWidgetItem(
            f"#{len(self._completed_children)} • {self._child.name} "
            f"• {len(self._answers)}/{len(self._questions)} إجابات • صف {row}"
        )
        item.setToolTip(self._session_id)
        self._completed_list.addItem(item)

        QMessageBox.information(
            self,
            "تم الحفظ",
            (
                f"تم حفظ جلسة الطفل «{self._child.name}» بنجاح.\n\n"
                f"• الصف في Excel: {row}\n"
                f"• مجلد التسجيلات:\n{self._session_dir}\n\n"
                f"يمكنك الآن الضغط على «👶 طفل جديد» لبدء طفل آخر."
            ),
        )

        # Clear current child state so the UI shows it's idle.
        self._child = None
        self._answers = {}
        self._session_id = ""
        self._notes_edit.clear()
        self._set_session_widgets_enabled(False)
        self._update_child_banner()
        self._update_question_panel()
        self.statusBar().showMessage(
            f"تم حفظ {len(self._completed_children)} طفل في {DEFAULT_WORKBOOK.name}."
        )
        return True

    # ------------------------------------------------------------------
    # Gemini auto-scoring
    # ------------------------------------------------------------------
    def _auto_score_answers_with_gemini(self) -> bool:
        """Score every recorded answer via Gemini and store the result in-place.

        Each entry in ``self._answers`` gains the keys ``score``, ``transcript``,
        and ``scoring_justification``. Returns ``True`` if the run should
        continue (even if some answers failed to score), ``False`` if the user
        aborted because the API key is missing or all calls failed.
        """
        # Build the rubric map once.
        from .questions import load_all_keys
        rubric_by_key = {
            row["key"]: row["description"] for row in load_all_keys(DEFAULT_JSON)
        }
        questions_by_key = {q.key: q for q in self._questions}

        # Try to construct the scorer up-front so we fail fast on missing key.
        try:
            scorer = GeminiScorer()
        except GeminiScorerError as exc:
            reply = QMessageBox.warning(
                self,
                "تعذّر التقييم التلقائي (Gemini)",
                f"{exc}\n\nهل تريد حفظ الجلسة دون تقييم تلقائي؟",
                QMessageBox.Save | QMessageBox.Cancel,
            )
            return reply == QMessageBox.Save

        # Show a non-blocking progress dialog while we hit the API.
        recorded = [
            (k, v) for k, v in self._answers.items() if v.get("audio_path")
        ]
        if not recorded:
            return True  # nothing to score

        progress = QProgressDialog(
            "جارٍ تقييم إجابات الطفل تلقائيًا عبر Gemini…",
            "إلغاء",
            0,
            len(recorded),
            self,
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        successes = 0
        failures: list[str] = []
        for idx, (key, payload) in enumerate(recorded, start=1):
            if progress.wasCanceled():
                break
            progress.setLabelText(
                f"({idx}/{len(recorded)}) تقييم: {key}"
            )
            QApplication.processEvents()
            try:
                result = scorer.score_answer(
                    audio_path=Path(payload["audio_path"]),
                    question_key=key,
                    question_text=(
                        questions_by_key[key].text
                        if key in questions_by_key
                        else payload.get("question_text", "")
                    ),
                    rubric_description=rubric_by_key.get(key, ""),
                )
            except GeminiScorerError as exc:
                failures.append(f"{key}: {exc}")
                progress.setValue(idx)
                continue

            # Persist the result on the answer entry.
            payload["score"] = result.score
            payload["transcript"] = result.transcript
            payload["scoring_justification"] = result.justification
            payload["scored_by"] = "gemini"
            payload["scored_at"] = datetime.now().isoformat(timespec="seconds")
            if result.score is not None:
                successes += 1
            progress.setValue(idx)

        progress.close()
        self._auto_save_manifest()
        self._refresh_answers_table()

        if successes == 0 and failures:
            reply = QMessageBox.warning(
                self,
                "تعذّر التقييم التلقائي",
                (
                    "لم يتمكن Gemini من تقييم أي إجابة:\n\n"
                    + "\n".join(failures[:5])
                    + "\n\nهل تريد حفظ الجلسة بدون درجات؟"
                ),
                QMessageBox.Save | QMessageBox.Cancel,
            )
            return reply == QMessageBox.Save

        if failures:
            QMessageBox.information(
                self,
                "بعض الإجابات لم تُقيَّم",
                "تم تقييم معظم الإجابات. الإجابات التي تعذّر تقييمها:\n"
                + "\n".join(failures[:10]),
            )
        return True

    # ------------------------------------------------------------------
    # Answers table (Question | Transcript | Play)
    # ------------------------------------------------------------------
    def _refresh_answers_table(self) -> None:
        """Rebuild the answers table from ``self._answers``."""
        table = self._answers_table
        table.setRowCount(0)
        if not self._questions:
            return

        for q in self._questions:
            payload = self._answers.get(q.key)
            if not payload or not payload.get("audio_path"):
                continue  # only show questions that actually got an answer
            row = table.rowCount()
            table.insertRow(row)

            # Column 0 — Arabic question text prefixed with its order number.
            q_item = QTableWidgetItem(f"{q.order}: {q.text}")
            q_item.setToolTip(q.full_description)
            q_item.setTextAlignment(Qt.AlignTop | Qt.AlignRight)
            table.setItem(row, 0, q_item)

            # Column 1 — transcript, loading placeholder, or last error.
            transcript = (payload.get("transcript") or "").strip()
            error = (payload.get("transcription_error") or "").strip()
            if transcript:
                text_cell = QTableWidgetItem(transcript)
                tooltip = transcript
            elif error:
                # Show a short, user-readable hint with the full error in tooltip.
                text_cell = QTableWidgetItem(
                    "⚠️ تعذّر التفريغ — اضغط «🔄 إعادة التفريغ النصي»."
                )
                text_cell.setForeground(QBrush(QColor("#c0392b")))
                tooltip = f"خطأ Gemini:\n{error}"
            else:
                text_cell = QTableWidgetItem("… جارٍ التفريغ النصي")
                text_cell.setForeground(QBrush(QColor("#7f8c8d")))
                tooltip = "اضغط «🔄 إعادة التفريغ النصي» لإعادة المحاولة."
            text_cell.setTextAlignment(Qt.AlignTop | Qt.AlignRight)
            text_cell.setToolTip(tooltip)
            table.setItem(row, 1, text_cell)

            # Column 2 — play button.
            audio_path = payload.get("audio_path", "")
            btn = QPushButton("▶ تشغيل")
            btn.setStyleSheet(
                "background:#27ae60; color:white; font-weight:bold; padding:6px;"
            )
            btn.clicked.connect(lambda _checked, p=audio_path: self._play_answer_audio(p))
            table.setCellWidget(row, 2, btn)

        table.resizeRowsToContents()

    def _play_answer_audio(self, audio_path: str) -> None:
        """Play (or stop) the WAV at ``audio_path`` through the answer player."""
        if not audio_path:
            return
        path = Path(audio_path)
        if not path.exists():
            QMessageBox.warning(
                self, "الملف غير موجود",
                f"تعذّر العثور على ملف الصوت:\n{audio_path}",
            )
            return
        # Stop any in-progress playback first, then play the new one.
        self._answer_player.stop()
        self._answer_player.setMedia(QMediaContent(QUrl.fromLocalFile(str(path))))
        self._answer_player.play()

    def _transcribe_answer_async(self, question_key: str) -> None:
        """Run Gemini transcription for one question and update the table.

        We use a short-lived background ``QThread`` so the UI doesn't freeze
        during the network call.
        """
        payload = self._answers.get(question_key)
        if not payload or not payload.get("audio_path"):
            return
        audio_path = payload["audio_path"]

        # Build the scorer lazily; if no API key, skip silently and leave the
        # placeholder text in the table.
        try:
            scorer = self._get_or_create_scorer()
        except GeminiScorerError:
            return

        thread = _TranscribeThread(scorer, audio_path, question_key, parent=self)
        thread.finished_with_text.connect(self._on_transcription_ready)
        # Keep a reference so the thread isn't garbage-collected mid-run.
        self._transcribe_threads.append(thread)
        thread.start()

    def _on_transcription_ready(
        self, question_key: str, transcript: str, error: str = ""
    ) -> None:
        """Slot called from a worker thread when a transcription finishes."""
        payload = self._answers.get(question_key)
        if payload is not None:
            if transcript:
                payload["transcript"] = transcript
                payload.pop("transcription_error", None)
            elif error:
                payload["transcription_error"] = error
                # Don't overwrite a previously valid transcript on a retry.
                payload.setdefault("transcript", "")
            self._auto_save_manifest()
        self._refresh_answers_table()
        # Reap finished threads.
        self._transcribe_threads = [
            t for t in self._transcribe_threads if t.isRunning()
        ]

    def _retranscribe_all_answers(self) -> None:
        """Force a fresh Gemini transcription for every recorded answer."""
        if self._child is None or not self._answers:
            return
        try:
            self._get_or_create_scorer()
        except GeminiScorerError as exc:
            QMessageBox.warning(self, "تعذّر الاتصال بـ Gemini", str(exc))
            return
        # Clear all transcripts so the placeholder shows again.
        for payload in self._answers.values():
            if payload.get("audio_path"):
                payload.pop("transcript", None)
        self._refresh_answers_table()
        for key, payload in self._answers.items():
            if payload.get("audio_path"):
                self._transcribe_answer_async(key)

    def _get_or_create_scorer(self) -> "GeminiScorer":
        """Cache a single :class:`GeminiScorer` instance per session run."""
        if self._gemini_scorer is None:
            self._gemini_scorer = GeminiScorer()
        return self._gemini_scorer

    def _open_workbook_externally(self) -> None:
        if not DEFAULT_WORKBOOK.exists():
            QMessageBox.information(
                self,
                "لا يوجد ملف بعد",
                "لم يتم حفظ أي جلسة بعد. سيتم إنشاء الملف بعد أول حفظ.",
            )
            return
        import subprocess
        import sys

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(DEFAULT_WORKBOOK)])
            elif sys.platform.startswith("win"):
                import os
                os.startfile(str(DEFAULT_WORKBOOK))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(DEFAULT_WORKBOOK)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "تعذّر الفتح", str(exc))

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: D401
        if self._recorder.is_recording:
            self._stop_recording()
        if self._child is not None and self._answers and not self._is_current_finished():
            reply = QMessageBox.question(
                self,
                "هناك جلسة لم تُحفظ",
                "يوجد جلسة طفل لم تُحفظ. هل تريد حفظها قبل الإغلاق؟",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Save:
                self._finish_current_child()
        event.accept()


class _TranscribeThread(QThread):
    """One-shot QThread that runs ``GeminiScorer.transcribe`` off the GUI thread.

    Emits ``finished_with_text(question_key, transcript, error)`` when done.
    On success ``error`` is empty; on failure ``transcript`` is empty and
    ``error`` contains a short diagnostic so the UI can show it in red.
    """

    finished_with_text = pyqtSignal(str, str, str)

    def __init__(
        self,
        scorer: "GeminiScorer",
        audio_path: str,
        question_key: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._scorer = scorer
        self._audio_path = audio_path
        self._question_key = question_key

    def run(self) -> None:  # noqa: D401
        text = ""
        error = ""
        try:
            text = self._scorer.transcribe(Path(self._audio_path))
            if not text:
                error = "ردّ فارغ من Gemini (لم يُلتقط أي كلام)."
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        self.finished_with_text.emit(self._question_key, text, error)


def _fmt_ms(ms: int) -> str:
    if ms <= 0:
        return "00:00"
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def run() -> int:
    """Entry point used by `python -m app` and `python run_app.py`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_WORKBOOK.exists():
        try:
            create_workbook(DEFAULT_JSON, DEFAULT_WORKBOOK)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not pre-create workbook: {exc}")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec_()
