from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sensorqa.ui import humanize_unit
from sensorqa.ui.components import EmptyState, MetricCard, PageHeader, StatusBadge
from sensorqa.ui.state import DesktopSession
from sensorqa.ui.widgets import AnalysisResultPlot


def _status_text(value: str) -> tuple[str, str]:
    """Return a concise UI label and visual status key.

    Args:
        value: Qualification status value from the presentation model.

    Returns:
        A tuple containing the label shown to the user and the status key used
        by ``StatusBadge``.
    """

    key = value.lower()
    if key == "pass":
        return "Pass", "pass"
    if key == "fail":
        return "Fail", "fail"
    return "Not evaluated", "not_evaluated"


def _qualification_summary_text(qualification) -> tuple[str, str]:
    """Summarize qualification coverage without changing backend status.

    Args:
        qualification: Presentation-layer qualification summary.

    Returns:
        A tuple containing a user-facing summary and a status badge key.
    """

    if qualification is None:
        return "Not available", "not_evaluated"
    if qualification.failed > 0:
        return "Fail", "fail"
    evaluated = qualification.passed + qualification.failed
    if evaluated == 0:
        return "Not evaluated", "not_evaluated"
    if qualification.not_evaluated > 0:
        return "Partially evaluated", "not_evaluated"
    return "Pass", "pass"


def _analysis_status(analysis) -> tuple[str, str]:
    """Choose the clearest status label for an analysis card.

    Args:
        analysis: Analysis summary view for one tool.

    Returns:
        A tuple containing the display label and visual status key.
    """

    execution = str(analysis.execution_status).lower()
    if execution == "error":
        return "Error", "fail"
    if execution == "skipped":
        return "Skipped", "not_evaluated"
    return _status_text(analysis.qualification_status)


def _format_value(value) -> str:
    """Format a metric value for compact human-readable display.

    Args:
        value: Numeric metric value or ``None``.

    Returns:
        Formatted metric value.
    """

    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        magnitude = abs(value)
        if value == 0:
            return "0"
        if magnitude >= 10000 or magnitude < 0.0001:
            return f"{value:.4g}"
        if magnitude >= 100:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        if magnitude >= 1:
            return f"{value:.5g}"
        return f"{value:.6g}"
    return str(value)


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


class ResultsPage(QWidget):
    """Display analysis, qualification, plots, and detailed metrics.

    Args:
        session: Shared desktop session containing the latest analysis.
        parent: Optional Qt parent widget.
    """

    def __init__(self, session: DesktopSession, parent=None) -> None:
        """Initialize the results page.

        Args:
            session: Shared desktop session containing analysis state.
            parent: Optional Qt parent widget.

        Returns:
            None.
        """

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
        self.layout.setSpacing(14)
        self.refresh()

    def _clear(self) -> None:
        """Clear dynamic content before rebuilding the page.

        Returns:
            None.
        """

        _clear_layout(self.layout)

    def refresh(self) -> None:
        """Rebuild the page from the current desktop session.

        Returns:
            None.
        """

        self._clear()
        self.layout.addWidget(
            PageHeader(
                "Results",
                "Analysis Results",
                "Review the summary first, then open the details for any check you want to inspect.",
            )
        )

        result = self.session.analysis
        if result is None:
            self.layout.addWidget(
                EmptyState(
                    "No results yet",
                    "Run an analysis and the results will appear here.",
                )
            )
            self.layout.addStretch(1)
            return

        summary = result.summary
        qualification = summary.qualification

        overview = QGridLayout()
        overview.setSpacing(10)
        overview.addWidget(
            MetricCard("Analyses", str(summary.analysis_count), "Checks returned by the analysis pipeline."),
            0,
            0,
        )
        overview.addWidget(
            MetricCard("Completed", str(summary.successful_analysis_count), "Checks that finished successfully."),
            0,
            1,
        )
        overview.addWidget(
            MetricCard(
                "Warnings",
                str(len(summary.workflow_warnings) + len(summary.pipeline_warnings)),
                "Warnings reported during analysis.",
            ),
            0,
            2,
        )
        qualification_text, _ = _qualification_summary_text(qualification)
        overview.addWidget(
            MetricCard("Qualification", qualification_text, "Overall status of the active requirements."),
            0,
            3,
        )
        self.layout.addLayout(overview)

        if qualification is not None:
            self.layout.addWidget(self._qualification_card(qualification))

        for analysis in summary.analyses:
            self.layout.addWidget(self._analysis_card(analysis))

        self.layout.addStretch(1)

    def _qualification_card(self, qualification) -> QFrame:
        """Build the qualification summary card.

        Args:
            qualification: Qualification summary view.

        Returns:
            Card containing qualification coverage and status.
        """

        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        row = QHBoxLayout()
        title = QLabel("Qualification")
        title.setProperty("role", "sectionTitle")
        badge_text, status_key = _qualification_summary_text(qualification)
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(StatusBadge(badge_text, status_key))
        layout.addLayout(row)

        evaluated = qualification.passed + qualification.failed
        coverage = QLabel(
            f"{qualification.passed} passed   {qualification.failed} failed   "
            f"{qualification.not_evaluated} not evaluated   "
            f"({evaluated}/{qualification.total_requirements} evaluated)"
        )
        coverage.setProperty("role", "muted")
        layout.addWidget(coverage)
        return card

    def _analysis_card(self, analysis) -> QFrame:
        """Build one analysis card with a plot and compact metric summary.

        Args:
            analysis: Presentation-layer analysis result.

        Returns:
            Qt frame containing the analysis result.
        """

        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(12)

        top = QHBoxLayout()
        title = QLabel(analysis.tool_name)
        title.setProperty("role", "sectionTitle")
        badge_text, status_key = _analysis_status(analysis)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(StatusBadge(badge_text, status_key))
        layout.addLayout(top)

        messages = [*analysis.messages, *analysis.warnings]
        if messages:
            banner = QFrame()
            banner.setProperty("role", "messageBanner")
            banner.setProperty("state", "warning" if analysis.warnings else "info")
            banner_layout = QVBoxLayout(banner)
            banner_layout.setContentsMargins(14, 10, 14, 10)
            banner_layout.setSpacing(3)
            for message in messages[:3]:
                label = QLabel(str(message))
                label.setWordWrap(True)
                label.setProperty("role", "messageBody")
                banner_layout.addWidget(label)
            layout.addWidget(banner)

        if AnalysisResultPlot.supports(analysis) and str(analysis.execution_status).lower() == "success":
            plot = AnalysisResultPlot()
            plot.setMinimumHeight(300)
            if plot.set_analysis(analysis, self.session.dataset):
                layout.addWidget(plot)
            else:
                plot.deleteLater()

        metrics = list(analysis.metrics)
        if metrics:
            key_title = QLabel("Key metrics")
            key_title.setProperty("role", "eyebrow")
            layout.addWidget(key_title)
            layout.addLayout(self._metric_grid(metrics[:6]))

        if len(metrics) > 6:
            details_button = QPushButton(f"Show all metrics ({len(metrics)})")
            details_button.setProperty("role", "ghostButton")
            details_button.setCheckable(True)
            details_button.setMaximumWidth(180)
            table = self._metric_table(metrics)
            table.setVisible(False)

            def toggle_details(checked: bool) -> None:
                table.setVisible(checked)
                details_button.setText(
                    "Hide metric details" if checked else f"Show all metrics ({len(metrics)})"
                )

            details_button.toggled.connect(toggle_details)
            layout.addWidget(details_button)
            layout.addWidget(table)
        elif metrics:
            # The key metric grid already contains every metric.
            pass

        if analysis.evidence:
            evidence_button = QPushButton(f"Evidence ({len(analysis.evidence)})")
            evidence_button.setProperty("role", "ghostButton")
            evidence_button.setCheckable(True)
            evidence_button.setMaximumWidth(150)
            evidence_box = QFrame()
            evidence_box.setProperty("role", "mutedCard")
            evidence_box.setVisible(False)
            evidence_layout = QVBoxLayout(evidence_box)
            evidence_layout.setContentsMargins(14, 10, 14, 10)
            for item in analysis.evidence:
                text = QLabel(f"{item.strength.title()}: {item.statement}")
                text.setWordWrap(True)
                evidence_layout.addWidget(text)
            evidence_button.toggled.connect(evidence_box.setVisible)
            layout.addWidget(evidence_button)
            layout.addWidget(evidence_box)

        return card

    def _metric_grid(self, metrics) -> QGridLayout:
        """Build a compact two-column grid of key metrics.

        Args:
            metrics: Metrics to display.

        Returns:
            Populated Qt grid layout.
        """

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        for index, metric in enumerate(metrics):
            tile = QFrame()
            tile.setProperty("role", "mutedCard")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(14, 10, 14, 10)
            tile_layout.setSpacing(3)

            label = QLabel(metric.name)
            label.setProperty("role", "metricLabel")
            value = _format_value(metric.value)
            unit = humanize_unit(metric.unit)
            display = f"{value} {unit}".strip()
            value_label = QLabel(display)
            value_label.setStyleSheet("font-size: 16px; font-weight: 650;")
            value_label.setTextInteractionFlags(value_label.textInteractionFlags())

            tile_layout.addWidget(label)
            tile_layout.addWidget(value_label)
            grid.addWidget(tile, index // 3, index % 3)

        return grid

    def _metric_table(self, metrics) -> QTableWidget:
        """Build the full metric table used by the expandable details view.

        Args:
            metrics: All metrics produced by an analysis.

        Returns:
            Read-only metric table.
        """

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Metric", "Result", "Unit"])
        table.setRowCount(len(metrics))
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setMinimumHeight(min(360, max(140, 42 + 29 * len(metrics))))

        for row, metric in enumerate(metrics):
            table.setItem(row, 0, QTableWidgetItem(metric.name))
            table.setItem(row, 1, QTableWidgetItem(_format_value(metric.value)))
            table.setItem(row, 2, QTableWidgetItem(humanize_unit(metric.unit)))

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        return table
