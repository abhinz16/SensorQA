from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget


class FileDropZone(QFrame):
    """Drag-and-drop area for selecting CSV files."""
    file_selected = Signal(str)
    browse_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the file drop zone.
        
        Args:
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(parent)
        self.setProperty("role", "dropZone")
        self.setProperty("dragActive", False)
        self.setAcceptDrops(True)
        self.setMinimumHeight(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        glyph = QLabel("+")
        glyph.setProperty("role", "accent")
        glyph.setStyleSheet("font-size: 38px; font-weight: 600;")
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Drop a CSV here")
        title.setProperty("role", "sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QLabel("or choose one from your computer")
        text.setProperty("role", "muted")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        button = QPushButton("Choose file")
        button.setProperty("role", "secondaryButton")
        button.clicked.connect(self.browse_requested.emit)

        layout.addWidget(glyph)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addSpacing(8)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Run drag enter event.
        
        Args:
            event: Qt event passed to the handler.
        
        Returns:
            None.
        """
        urls = event.mimeData().urls()
        if any(Path(url.toLocalFile()).suffix.lower() == ".csv" for url in urls):
            event.acceptProposedAction()
            self._set_drag_active(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        """Run drag leave event.
        
        Args:
            event: Qt event passed to the handler.
        
        Returns:
            None.
        """
        self._set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Run drop event.
        
        Args:
            event: Qt event passed to the handler.
        
        Returns:
            None.
        """
        self._set_drag_active(False)
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".csv":
                self.file_selected.emit(str(path))
                event.acceptProposedAction()
                return
        event.ignore()

    def _set_drag_active(self, active: bool) -> None:
        """Update the drop-zone style while a file is being dragged.
        
        Args:
            active: Active used by this function.
        
        Returns:
            None.
        """
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
