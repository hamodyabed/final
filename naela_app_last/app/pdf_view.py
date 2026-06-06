"""A simple PyQt5 widget that renders a PDF page-by-page with PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class PdfViewer(QWidget):
    """Render a PDF and expose previous/next navigation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._doc: Optional[fitz.Document] = None
        self._page_index = 0
        self._zoom = 1.4

        self._image_label = QLabel("لم يتم تحميل ملف القصة بعد.")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background: #222; color: #ddd;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._image_label)

        self._prev_btn = QPushButton("⏮ السابق")
        self._next_btn = QPushButton("التالي ⏭")
        self._page_label = QLabel("0 / 0")
        self._page_label.setAlignment(Qt.AlignCenter)

        self._prev_btn.clicked.connect(self.prev_page)
        self._next_btn.clicked.connect(self.next_page)

        nav = QHBoxLayout()
        nav.addWidget(self._prev_btn)
        nav.addWidget(self._page_label, 1)
        nav.addWidget(self._next_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addLayout(nav)

    # ------------------------------------------------------------------
    def load(self, pdf_path: Path) -> None:
        if not pdf_path.exists():
            self._image_label.setText(f"الملف غير موجود:\n{pdf_path}")
            return
        if self._doc is not None:
            self._doc.close()
        self._doc = fitz.open(str(pdf_path))
        self._page_index = 0
        self._render()

    def next_page(self) -> None:
        if self._doc and self._page_index < self._doc.page_count - 1:
            self._page_index += 1
            self._render()

    def prev_page(self) -> None:
        if self._doc and self._page_index > 0:
            self._page_index -= 1
            self._render()

    # ------------------------------------------------------------------
    def _render(self) -> None:
        if self._doc is None:
            return
        page = self._doc.load_page(self._page_index)
        matrix = fitz.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = QImage(
            pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888
        )
        self._image_label.setPixmap(QPixmap.fromImage(image))
        self._image_label.setText("")
        self._page_label.setText(f"{self._page_index + 1} / {self._doc.page_count}")
