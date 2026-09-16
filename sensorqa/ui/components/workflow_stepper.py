from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget


class WorkflowStepper(QWidget):
    """Step indicator used by the New Analysis workflow."""
    def __init__(self, steps: list[str], parent: QWidget | None = None) -> None:
        """Initialize the workflow stepper.
        
        Args:
            steps: Steps used by this function.
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(parent)
        self.steps = list(steps)
        self._items: list[tuple[QLabel, QLabel]] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for index, name in enumerate(self.steps):
            item = QFrame()
            item.setProperty("role", "step")
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(8, 4, 8, 4)
            item_layout.setSpacing(7)

            circle = QLabel(str(index + 1))
            circle.setProperty("role", "stepCircle")
            circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            circle.setFixedSize(25, 25)

            label = QLabel(name)
            label.setProperty("role", "muted")

            item_layout.addWidget(circle)
            item_layout.addWidget(label)
            layout.addWidget(item)
            self._items.append((circle, label))

        layout.addStretch(1)
        self.set_current(0)

    def set_current(self, current: int) -> None:
        """Update the current workflow step.
        
        Args:
            current: Current used by this function.
        
        Returns:
            None.
        """
        for index, (circle, label) in enumerate(self._items):
            if index < current:
                state = "complete"
                circle.setText("✓")
            elif index == current:
                state = "active"
                circle.setText(str(index + 1))
            else:
                state = "pending"
                circle.setText(str(index + 1))

            circle.setProperty("state", state)
            label.setProperty("active", index == current)
            for widget in (circle, label):
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
