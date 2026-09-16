from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget


class StatusBadge(QLabel):
    """Small badge used to show a status value."""
    def __init__(
        self,
        text: str,
        status: str = "not_evaluated",
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the status badge.
        
        Args:
            text: Text to process.
            status: Status value.
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(text, parent)
        self.setProperty("role", "statusBadge")
        self.set_status(status, text)

    def set_status(self, status: str, text: str | None = None) -> None:
        """Update the badge text and styling.
        
        Args:
            status: Status value.
            text: Text to process.
        
        Returns:
            None.
        """
        self.setProperty("status", status)
        if text is not None:
            self.setText(text)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
