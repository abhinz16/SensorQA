from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sensorqa.ui.components import MetricCard, PageHeader, StatusBadge
from sensorqa.ui.state import DesktopSession


class DashboardPage(QWidget):
    """Desktop home page showing the current SensorQA status."""
    navigate_requested = Signal(str)

    def __init__(self, startup, services, session: DesktopSession, parent=None) -> None:
        """Initialize the dashboard page.
        
        Args:
            startup: Startup used by this function.
            services: Services used by this function.
            session: Session used by this function.
            parent: Optional parent widget.
        
        Returns:
            None.
        """
        super().__init__(parent)
        self.startup = startup
        self.services = services
        self.session = session

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        self.layout = QVBoxLayout(content)
        self.layout.setContentsMargins(36, 30, 36, 36)
        self.layout.setSpacing(22)

        self.layout.addWidget(
            PageHeader(
                "Home",
                "Start with your sensor data",
                "Load a test file, tell SensorQA what the columns mean, and review the checks that matter for that sensor.",
            )
        )

        hero = QFrame()
        hero.setProperty("role", "accentCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 20, 24, 20)
        copy = QVBoxLayout()
        eyebrow = QLabel("New analysis")
        eyebrow.setProperty("role", "eyebrow")
        title = QLabel("Open a sensor test")
        title.setProperty("role", "sectionTitle")
        text = QLabel(
            "Choose a CSV, check the column mapping and units, add the test details, then run the checks that apply."
        )
        text.setProperty("role", "muted")
        text.setWordWrap(True)
        copy.addWidget(eyebrow)
        copy.addWidget(title)
        copy.addWidget(text)
        hero_layout.addLayout(copy, 1)
        start = QPushButton("Start analysis")
        start.setProperty("role", "primaryButton")
        start.setMinimumWidth(220)
        start.clicked.connect(lambda: self.navigate_requested.emit("new_analysis"))
        hero_layout.addWidget(start)
        self.layout.addWidget(hero)

        grid = QGridLayout()
        grid.setSpacing(14)
        self.status_card = MetricCard("App", "Ready", "SensorQA started normally.", accent=True)
        self.tools_card = MetricCard("Tools", "0", "Checks available in this installation.")
        self.active_card = MetricCard("Enabled", "0", "Checks currently turned on.")
        self.issue_card = MetricCard("Tool issues", "0", "Problems found while loading tools.")
        grid.addWidget(self.status_card, 0, 0)
        grid.addWidget(self.tools_card, 0, 1)
        grid.addWidget(self.active_card, 0, 2)
        grid.addWidget(self.issue_card, 0, 3)
        self.layout.addLayout(grid)

        lower = QGridLayout()
        lower.setSpacing(14)
        lower.addWidget(self._workflow_card(), 0, 0)
        self.health_card = self._health_card()
        lower.addWidget(self.health_card, 0, 1)
        lower.setColumnStretch(0, 3)
        lower.setColumnStretch(1, 2)
        self.layout.addLayout(lower)
        self.layout.addStretch(1)
        self.refresh()

    def _workflow_card(self) -> QFrame:
        """Build the workflow summary card on the home page.
        
        Returns:
            QFrame returned by this function.
        """
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 22)
        layout.setSpacing(10)
        label = QLabel("Workflow")
        label.setProperty("role", "eyebrow")
        title = QLabel("From file to results")
        title.setProperty("role", "sectionTitle")
        layout.addWidget(label)
        layout.addWidget(title)
        steps = [
            ("01", "Choose a file", "Load the sensor data and check the preview."),
            ("02", "Match the columns", "Tell SensorQA what each column represents."),
            ("03", "Add units and test details", "Give the data the context it needs."),
            ("04", "Run the checks", "SensorQA runs the enabled tools that fit the dataset."),
            ("05", "Review the results", "Look at the metrics, warnings, and requirements."),
            ("06", "Save a report", "Export the results when you are ready to share them."),
        ]
        for number, name, text in steps:
            row = QHBoxLayout()
            n = QLabel(number)
            n.setProperty("role", "accent")
            n.setFixedWidth(34)
            block = QVBoxLayout()
            block.setSpacing(1)
            a = QLabel(name)
            a.setStyleSheet("font-weight: 600;")
            b = QLabel(text)
            b.setProperty("role", "muted")
            block.addWidget(a)
            block.addWidget(b)
            row.addWidget(n)
            row.addLayout(block, 1)
            layout.addLayout(row)
        layout.addStretch(1)
        return card

    def _health_card(self) -> QFrame:
        """Build the tool-health card on the home page.
        
        Returns:
            QFrame returned by this function.
        """
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 22)
        layout.setSpacing(10)
        label = QLabel("Tools")
        label.setProperty("role", "eyebrow")
        title = QLabel("Tool status")
        title.setProperty("role", "sectionTitle")
        self.health_badge = StatusBadge("Ready", "pass")
        self.health_text = QLabel("")
        self.health_text.setProperty("role", "muted")
        self.health_text.setWordWrap(True)
        button = QPushButton("Open tools")
        button.setProperty("role", "secondaryButton")
        button.clicked.connect(lambda: self.navigate_requested.emit("tools"))
        layout.addWidget(label)
        layout.addWidget(title)
        layout.addWidget(self.health_badge)
        layout.addWidget(self.health_text)
        layout.addStretch(1)
        layout.addWidget(button)
        return card

    def refresh(self) -> None:
        """Refresh the page from the current session state.
        
        Returns:
            None.
        """
        if self.startup.ready and self.services is not None:
            health = self.services.tools.health()
            tools = self.services.tools.list_tools()
            enabled = sum(1 for tool in tools if tool.enabled)
            issues = health.rejected_count + health.load_error_count
            self.status_card.set_value("Ready", "SensorQA started normally.")
            self.tools_card.set_value(str(health.registered_count))
            self.active_card.set_value(str(enabled))
            self.issue_card.set_value(
                str(issues),
                "Everything loaded normally." if issues == 0 else "Open Tools to see what needs attention.",
            )
            self.health_badge.set_status(
                "pass" if issues == 0 else "warning",
                "Ready" if issues == 0 else "Check tools",
            )
            self.health_text.setText(
                f"{health.registered_count} tools are available. "
                + ("Everything loaded normally." if issues == 0 else f"{issues} tool issue(s) need attention.")
            )
        else:
            self.status_card.set_value("Needs attention", "SensorQA did not finish starting up.")
            self.health_badge.set_status("fail", "Needs attention")
            self.health_text.setText(getattr(self.startup, "error", None) or "SensorQA could not finish starting up.")
