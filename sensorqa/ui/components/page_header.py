from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PageHeader(QWidget):
    """Reusable title and subtitle block for desktop pages."""
    def __init__(
        self,
        eyebrow: str,
        title: str,
        subtitle: str,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the page header.
        
        Args:
            eyebrow: Eyebrow used by this function.
            title: Title shown to the user.
            subtitle: Subtitle used by this function.
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        e = QLabel(eyebrow)
        e.setProperty("role", "eyebrow")
        t = QLabel(title)
        t.setProperty("role", "pageTitle")
        s = QLabel(subtitle)
        s.setProperty("role", "pageSubtitle")
        s.setWordWrap(True)
        s.setMaximumWidth(860)

        layout.addWidget(e)
        layout.addWidget(t)
        layout.addWidget(s)
