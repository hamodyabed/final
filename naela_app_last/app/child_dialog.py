"""Modal dialog asking for a new child's identification before a session starts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)


@dataclass
class ChildInfo:
    """Plain data carrier returned by :class:`ChildIntakeDialog`."""

    name: str
    age: int
    gender: str        # ذكر / أنثى / غير محدد
    group: str         # TD (نمو طبيعي) / ASD / DLD / أخرى
    notes: str

    def safe_filename_token(self) -> str:
        """Slugify the child's name for use inside a folder name."""
        cleaned = "".join(c for c in self.name.strip() if c.isalnum() or c in "_-")
        return cleaned or "child"


class ChildIntakeDialog(QDialog):
    """Collect basic information about the next child to be assessed."""

    GROUPS = ("نمو طبيعي (TD)", "طيف توحد (ASD)", "اضطراب لغة نمائي (DLD)", "أخرى")
    GENDERS = ("ذكر", "أنثى", "غير محدد")

    def __init__(self, parent=None, default_index: int = 1) -> None:
        super().__init__(parent)
        self.setWindowTitle("بيانات الطفل")
        self.setLayoutDirection(Qt.RightToLeft)
        self.setModal(True)
        self.setMinimumWidth(420)

        self._result: Optional[ChildInfo] = None

        intro = QLabel(
            f"الطفل رقم {default_index}. الرجاء إدخال البيانات قبل بدء الجلسة."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#333; padding:4px;")

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("مثال: عبد الرحمن")

        self._age_spin = QSpinBox()
        self._age_spin.setRange(2, 17)
        self._age_spin.setValue(6)
        self._age_spin.setSuffix(" سنة")

        self._gender_combo = QComboBox()
        self._gender_combo.addItems(self.GENDERS)

        self._group_combo = QComboBox()
        self._group_combo.addItems(self.GROUPS)

        self._notes_edit = QTextEdit()
        self._notes_edit.setPlaceholderText(
            "أي ملاحظات (مدرسة، تشخيص سابق، لغة سائدة في البيت...)"
        )
        self._notes_edit.setMaximumHeight(80)

        form = QFormLayout()
        form.addRow("اسم الطفل *:", self._name_edit)
        form.addRow("العمر:", self._age_spin)
        form.addRow("الجنس:", self._gender_combo)
        form.addRow("المجموعة:", self._group_combo)
        form.addRow("ملاحظات:", self._notes_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.button(QDialogButtonBox.Ok).setText("ابدأ الجلسة")
        buttons.button(QDialogButtonBox.Cancel).setText("إلغاء")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self._name_edit.setFocus()

    # ------------------------------------------------------------------
    def _on_accept(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "حقل مطلوب", "اسم الطفل مطلوب لبدء الجلسة.")
            self._name_edit.setFocus()
            return
        self._result = ChildInfo(
            name=name,
            age=self._age_spin.value(),
            gender=self._gender_combo.currentText(),
            group=self._group_combo.currentText(),
            notes=self._notes_edit.toPlainText().strip(),
        )
        self.accept()

    # ------------------------------------------------------------------
    @staticmethod
    def get_child_info(parent=None, default_index: int = 1) -> Optional[ChildInfo]:
        """Convenience helper: show the dialog and return the result or ``None``."""
        dlg = ChildIntakeDialog(parent=parent, default_index=default_index)
        if dlg.exec_() == QDialog.Accepted:
            return dlg._result
        return None


def build_session_id(child: ChildInfo) -> str:
    """Build a unique session id that embeds the child's name slug."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{child.safe_filename_token()}"
