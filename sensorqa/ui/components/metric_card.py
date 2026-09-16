from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget


class MetricCard(QFrame):
    """Reusable card for a label, value, and short note."""
    def __init__(
        self,
        label: str,
        value: str,
        detail: str = "",
        *,
        accent: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the metric card.
        
        Args:
            label: Label shown to the user.
            value: Value to process.
            detail: Detail used by this function.
            accent: Accent used by this function.
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(parent)
        self.setProperty("role", "accentCard" if accent else "card")
        self.setMinimumHeight(124)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(5)

        self.label_widget = QLabel(label)
        self.label_widget.setProperty("role", "metricLabel")
        self.value_widget = QLabel(value)
        self.value_widget.setProperty("role", "metricValue")
        self.detail_widget = QLabel(detail)
        self.detail_widget.setProperty("role", "muted")
        self.detail_widget.setWordWrap(True)

        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
        layout.addWidget(self.detail_widget)
        layout.addStretch(1)

    def set_value(self, value: str, detail: str | None = None) -> None:
        """Update the value shown by a metric card.
        
        Args:
            value: Value to process.
            detail: Detail used by this function.
        
        Returns:
            None.
        """
        self.value_widget.setText(value)
        if detail is not None:
            self.detail_widget.setText(detail)
