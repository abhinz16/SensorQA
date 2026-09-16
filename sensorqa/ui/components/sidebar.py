from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class Sidebar(QWidget):
    navigate_requested = Signal(str)

    PAGE_DEFINITIONS = (
        ("dashboard", "Home", "⌂"),
        ("new_analysis", "New Analysis", "+"),
        ("datasets", "Datasets", "▤"),
        ("results", "Results", "▥"),
        ("calibration", "Calibration", "⌁"),
        ("tools", "Tools", "◇"),
        ("reports", "Reports", "▧"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(252)
        self.buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        mark = QLabel("◇")
        mark.setObjectName("BrandMark")
        mark.setFixedWidth(30)
        name = QLabel("SensorQA")
        name.setObjectName("BrandName")
        brand.addWidget(mark)
        brand.addWidget(name)
        brand.addStretch(1)
        layout.addLayout(brand)

        descriptor = QLabel("Sensor data analysis")
        descriptor.setProperty("role", "eyebrow")
        layout.addWidget(descriptor)
        layout.addSpacing(18)

        for page_id, label, glyph in self.PAGE_DEFINITIONS:
            button = self._button(page_id, label, glyph)
            layout.addWidget(button)

        layout.addStretch(1)
        divider = QFrame()
        divider.setProperty("role", "divider")
        layout.addWidget(divider)
        layout.addSpacing(8)
        layout.addWidget(self._button("settings", "Settings", "⚙"))

        version = QLabel("Desktop app")
        version.setProperty("role", "muted")
        version.setStyleSheet("font-size: 10px;")
        layout.addWidget(version)

    def _button(self, page_id: str, label: str, glyph: str) -> QPushButton:
        button = QPushButton(f"{glyph:<3}  {label}")
        button.setProperty("role", "navButton")
        button.setProperty("active", False)
        button.setMinimumHeight(46)
        button.clicked.connect(lambda checked=False, pid=page_id: self.navigate_requested.emit(pid))
        self.buttons[page_id] = button
        return button

    def set_active(self, page_id: str) -> None:
        for pid, button in self.buttons.items():
            button.setProperty("active", pid == page_id)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()
