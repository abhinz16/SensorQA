from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget


class EmptyState(QFrame):
    """Reusable empty-state panel."""
    action_requested = Signal()

    def __init__(
        self,
        title: str,
        message: str,
        *,
        action_text: str | None = None,
        glyph: str = "◇",
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the empty state.
        
        Args:
            title: Title shown to the user.
            message: Message text.
            action_text: Action text used by this function.
            glyph: Glyph used by this function.
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(parent)
        self.setProperty("role", "card")
        self.setMinimumHeight(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        g = QLabel(glyph)
        g.setProperty("role", "accent")
        g.setStyleSheet("font-size: 46px;")
        g.setAlignment(Qt.AlignmentFlag.AlignCenter)

        t = QLabel(title)
        t.setProperty("role", "sectionTitle")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)

        m = QLabel(message)
        m.setProperty("role", "muted")
        m.setWordWrap(True)
        m.setMaximumWidth(560)
        m.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)
        layout.addWidget(g)
        layout.addWidget(t)
        layout.addWidget(m)

        if action_text:
            button = QPushButton(action_text)
            button.setProperty("role", "primaryButton")
            button.clicked.connect(self.action_requested.emit)
            layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)
