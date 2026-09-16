from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from sensorqa.ui import humanize_identifier, humanize_unit
from sensorqa.ui.components import EmptyState, MetricCard, PageHeader
from sensorqa.ui.state import DesktopSession
from sensorqa.ui.widgets import DataFrameTable



def _clear_layout(layout) -> None:
    """Remove every widget, nested layout, and spacer from a Qt layout.

    Args:
        layout: Qt layout to empty. Nested layouts are cleared recursively.

    Returns:
        None.
    """

    while layout.count():
        item = layout.takeAt(0)

        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
            continue

        child_layout = item.layout()
        if child_layout is not None:
            _clear_layout(child_layout)
            child_layout.deleteLater()


_SENSOR_NAMES = {
    "generic": "Generic sensor",
    "imu": "IMU",
    "accelerometer": "Accelerometer",
    "gyroscope": "Gyroscope",
}


class DatasetsPage(QWidget):
    def __init__(self, session: DesktopSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(self.scroll)
        self.content = QWidget()
        self.scroll.setWidget(self.content)
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(36, 30, 36, 36)
        self.layout.setSpacing(18)
        self.refresh()

    def _clear(self) -> None:
        """Clear the page before rebuilding its dynamic content."""

        _clear_layout(self.layout)

    def refresh(self) -> None:
        self._clear()
        self.layout.addWidget(
            PageHeader(
                "Dataset",
                "Dataset Overview",
                "Review the prepared data, assigned units, and a sample of the rows used for analysis.",
            )
        )
        dataset = self.session.dataset
        if dataset is None:
            self.layout.addWidget(
                EmptyState(
                    "No dataset loaded",
                    "Load a sensor file and complete setup to see the prepared dataset here.",
                )
            )
            self.layout.addStretch(1)
            return

        grid = QGridLayout()
        grid.setSpacing(12)
        raw_sensor_type = getattr(dataset.sensor_type, "value", dataset.sensor_type) or "unknown"
        sensor_type = _SENSOR_NAMES.get(str(raw_sensor_type).lower(), humanize_identifier(raw_sensor_type))
        valid = dataset.is_valid
        valid_text = "Ready" if valid is True else "Needs a look" if valid is False else "Not checked"
        grid.addWidget(MetricCard("Rows", f"{dataset.row_count:,}"), 0, 0)
        grid.addWidget(MetricCard("Columns", str(dataset.column_count)), 0, 1)
        grid.addWidget(MetricCard("Sensor", sensor_type), 0, 2)
        grid.addWidget(MetricCard("Data status", valid_text), 0, 3)
        self.layout.addLayout(grid)

        units = QFrame()
        units.setProperty("role", "card")
        u = QVBoxLayout(units)
        u.setContentsMargins(20, 18, 20, 18)
        title = QLabel("Units")
        title.setProperty("role", "sectionTitle")
        u.addWidget(title)
        if dataset.units:
            text = "   ".join(
                f"{humanize_identifier(key)}: {humanize_unit(value)}"
                for key, value in dataset.units.items()
            )
            unit_text = QLabel(text)
            unit_text.setWordWrap(True)
            u.addWidget(unit_text)
        else:
            empty = QLabel("No units were saved with this dataset.")
            empty.setProperty("role", "muted")
            u.addWidget(empty)
        self.layout.addWidget(units)

        table_card = QFrame()
        table_card.setProperty("role", "card")
        t = QVBoxLayout(table_card)
        t.setContentsMargins(18, 18, 18, 18)
        title = QLabel("Preview")
        title.setProperty("role", "sectionTitle")
        note = QLabel("First 30 rows of the prepared dataset.")
        note.setProperty("role", "muted")
        table = DataFrameTable()
        table.setMinimumHeight(340)
        table.set_dataframe(dataset.data, max_rows=30, humanize_headers=True)
        t.addWidget(title)
        t.addWidget(note)
        t.addWidget(table)
        self.layout.addWidget(table_card)
        self.layout.addStretch(1)
