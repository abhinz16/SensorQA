from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sensorqa.reporting import ReportMetadata
from sensorqa.ui.components import EmptyState, PageHeader
from sensorqa.ui.state import DesktopSession



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


class ReportsPage(QWidget):
    def __init__(self, services, session: DesktopSession, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.session = session
        self.last_export: str | None = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(36, 30, 36, 36)
        self.layout.setSpacing(18)
        self.refresh()

    def _clear(self) -> None:
        """Clear the page before rebuilding its dynamic content."""

        _clear_layout(self.layout)

    def refresh(self) -> None:
        self._clear()
        self.layout.addWidget(PageHeader(
            "Reports",
            "Export Results",
            "Save an HTML report for review or export the analysis results as JSON.",
        ))
        if self.session.analysis is None:
            self.layout.addWidget(EmptyState(
                "No analysis results",
                "Run an analysis before exporting a report or results file.",
            ))
            self.layout.addStretch(1)
            return

        card = QFrame()
        card.setProperty("role", "card")
        grid = QGridLayout(card)
        grid.setContentsMargins(20, 18, 20, 20)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(9)

        title = QLabel("Report Information")
        title.setProperty("role", "sectionTitle")
        grid.addWidget(title, 0, 0, 1, 2)

        grid.addWidget(QLabel("Title"), 1, 0)
        self.title = QLineEdit("SensorQA Report")
        grid.addWidget(self.title, 2, 0, 1, 2)

        grid.addWidget(QLabel("Prepared for"), 3, 0)
        grid.addWidget(QLabel("Prepared by"), 3, 1)
        self.prepared_for = QLineEdit()
        self.prepared_by = QLineEdit()
        grid.addWidget(self.prepared_for, 4, 0)
        grid.addWidget(self.prepared_by, 4, 1)

        grid.addWidget(QLabel("Notes"), 5, 0, 1, 2)
        self.notes = QTextEdit()
        self.notes.setMaximumHeight(90)
        grid.addWidget(self.notes, 6, 0, 1, 2)

        export = QPushButton("Export HTML report")
        export.setProperty("role", "primaryButton")
        export.clicked.connect(self._export_html)
        json_button = QPushButton("Export JSON")
        json_button.setProperty("role", "secondaryButton")
        json_button.clicked.connect(self._export_json)
        grid.addWidget(export, 7, 0)
        grid.addWidget(json_button, 7, 1)
        self.layout.addWidget(card)

        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        self.layout.addWidget(self.status)
        self.layout.addStretch(1)

    def _export_html(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save report", "SensorQA_Report.html", "HTML report (*.html)")
        if not path:
            return
        metadata = ReportMetadata(
            title=self.title.text().strip() or "SensorQA Report",
            prepared_for=self.prepared_for.text().strip() or None,
            prepared_by=self.prepared_by.text().strip() or None,
            notes=self.notes.toPlainText().strip() or None,
        )
        result = self.services.reports.export_html(
            self.session.analysis,
            path,
            calibration=self.session.calibration,
            metadata=metadata,
        )
        if result.success:
            self.last_export = result.file_path
            self.status.setText(f"Report saved: {result.file_path}")
            if self.session.preferences.open_report_after_export:
                QDesktopServices.openUrl(QUrl.fromLocalFile(result.file_path))
        else:
            self.status.setText("Could not save the report: " + "  ".join(result.errors))

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save results", "SensorQA_Results.json", "JSON (*.json)")
        if not path:
            return
        result = self.services.export.export_analysis_json(self.session.analysis, path)
        self.status.setText(
            f"Results saved: {result.file_path}" if result.success else "Could not save the results: " + "  ".join(result.errors)
        )
