#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SensorQA desktop application.

Run the GUI with:

    python -m sensorqa.ui.app

Run only the backend startup check with:

    python -m sensorqa.ui.app --check
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.bootstrap import SensorQABootstrapResult, bootstrap_application
from sensorqa.services import SensorQAServices, create_services
from sensorqa.ui.theme import apply_theme

try:
    from PySide6.QtCore import QEasingCurve, QPropertyAnimation
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGraphicsOpacityEffect,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
    QT_AVAILABLE = True
    QT_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = exc


@dataclass(frozen=True)
class DesktopStartupState:
    bootstrap: SensorQABootstrapResult | None
    services: SensorQAServices | None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return bool(
            self.bootstrap is not None
            and self.bootstrap.initialized
            and self.services is not None
            and self.error is None
        )


def start_backend(project_root: str | Path | None = None) -> DesktopStartupState:
    try:
        bootstrap = bootstrap_application(project_root=project_root, raise_on_error=False)
        if not bootstrap.initialized:
            messages: list[str] = []
            initialization = bootstrap.initialization
            if initialization is not None:
                config_report = getattr(initialization, "config_report", None)
                if config_report is not None:
                    for error in getattr(config_report, "errors", []):
                        location = getattr(error, "location", "configuration")
                        message = getattr(error, "message", str(error))
                        messages.append(f"{location}: {message}")
            if not messages:
                messages.append("SensorQA backend initialization did not complete successfully.")
            return DesktopStartupState(bootstrap=bootstrap, services=None, error="\n".join(messages))

        return DesktopStartupState(
            bootstrap=bootstrap,
            services=create_services(bootstrap),
            error=None,
        )
    except Exception as exc:
        return DesktopStartupState(
            bootstrap=None,
            services=None,
            error=f"{type(exc).__name__}: {exc}",
        )


if QT_AVAILABLE:
    from sensorqa.ui.components import Sidebar, StatusBadge
    from sensorqa.ui.pages import (
        CalibrationPage,
        DashboardPage,
        DatasetsPage,
        NewAnalysisPage,
        ReportsPage,
        ResultsPage,
        SettingsPage,
        ToolManagerPage,
    )
    from sensorqa.ui.state import DesktopSession


    class SensorQAMainWindow(QMainWindow):
        PAGE_LABELS = {
            "dashboard": "Home",
            "new_analysis": "New Analysis",
            "datasets": "Datasets",
            "results": "Results",
            "calibration": "Calibration",
            "tools": "Tools",
            "reports": "Reports",
            "settings": "Settings",
        }

        def __init__(self, startup: DesktopStartupState, parent=None) -> None:
            super().__init__(parent)
            self.startup = startup
            self.services = startup.services
            self.session = DesktopSession()
            self.pages: dict[str, QWidget] = {}
            self.page_indices: dict[str, int] = {}
            self._animation = None

            self.setWindowTitle("SensorQA")
            self.resize(1480, 900)
            self.setMinimumSize(1120, 720)

            central = QWidget()
            self.setCentralWidget(central)
            shell = QHBoxLayout(central)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.setSpacing(0)

            self.sidebar = Sidebar()
            self.sidebar.navigate_requested.connect(self.navigate)
            shell.addWidget(self.sidebar)

            main = QWidget()
            main_layout = QVBoxLayout(main)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            main_layout.addWidget(self._build_topbar())

            self.stack = QStackedWidget()
            main_layout.addWidget(self.stack, 1)
            shell.addWidget(main, 1)

            self._build_pages()
            self.navigate("dashboard", animate=False)

        def _build_topbar(self) -> QFrame:
            bar = QFrame()
            bar.setObjectName("TopBar")
            bar.setFixedHeight(68)
            layout = QHBoxLayout(bar)
            layout.setContentsMargins(28, 0, 28, 0)
            self.context_label = QLabel("Home")
            self.context_label.setProperty("role", "topbarContext")
            layout.addWidget(self.context_label)
            layout.addStretch(1)
            self.health_badge = StatusBadge(
                "Ready" if self.startup.ready else "Needs attention",
                "pass" if self.startup.ready else "fail",
            )
            layout.addWidget(self.health_badge)
            return bar

        def _build_pages(self) -> None:
            if self.services is None:
                dashboard = DashboardPage(self.startup, None, self.session)
                self._add_page("dashboard", dashboard)
                for page_id in self.PAGE_LABELS:
                    if page_id != "dashboard":
                        placeholder = SettingsPage(self.startup, self.session)
                        self._add_page(page_id, placeholder)
                dashboard.navigate_requested.connect(self.navigate)
                return

            dashboard = DashboardPage(self.startup, self.services, self.session)
            new_analysis = NewAnalysisPage(self.services, self.session)
            datasets = DatasetsPage(self.session)
            results = ResultsPage(self.session)
            calibration = CalibrationPage(self.services, self.session)
            tools = ToolManagerPage(self.services)
            reports = ReportsPage(self.services, self.session)
            settings = SettingsPage(self.startup, self.session)

            dashboard.navigate_requested.connect(self.navigate)
            new_analysis.dataset_built.connect(self._dataset_built)
            new_analysis.analysis_completed.connect(self._analysis_completed)
            calibration.calibration_completed.connect(self._calibration_completed)

            page_map = {
                "dashboard": dashboard,
                "new_analysis": new_analysis,
                "datasets": datasets,
                "results": results,
                "calibration": calibration,
                "tools": tools,
                "reports": reports,
                "settings": settings,
            }
            for page_id, page in page_map.items():
                self._add_page(page_id, page)

        def _add_page(self, page_id: str, page: QWidget) -> None:
            self.pages[page_id] = page
            self.page_indices[page_id] = self.stack.addWidget(page)

        def navigate(self, page_id: str, animate: bool = True) -> None:
            if page_id not in self.page_indices:
                return
            page = self.pages[page_id]
            refresh = getattr(page, "refresh", None)
            if callable(refresh):
                refresh()
            self.stack.setCurrentIndex(self.page_indices[page_id])
            self.context_label.setText(self.PAGE_LABELS.get(page_id, page_id))
            self.sidebar.set_active(page_id)
            if animate:
                self._fade(page)

        def _fade(self, page: QWidget) -> None:
            effect = QGraphicsOpacityEffect(page)
            page.setGraphicsEffect(effect)
            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setDuration(150)
            animation.setStartValue(0.25)
            animation.setEndValue(1.0)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.finished.connect(lambda: page.setGraphicsEffect(None))
            self._animation = animation
            animation.start()

        def _dataset_built(self, dataset) -> None:
            self.session.dataset = dataset
            page = self.pages.get("datasets")
            if page is not None and hasattr(page, "refresh"):
                page.refresh()

        def _analysis_completed(self, result) -> None:
            self.session.analysis = result
            for page_id in ("dashboard", "datasets", "results", "calibration", "reports"):
                page = self.pages.get(page_id)
                if page is not None and hasattr(page, "refresh"):
                    page.refresh()
            self.navigate("results")

        def _calibration_completed(self, result) -> None:
            self.session.calibration = result
            page = self.pages.get("reports")
            if page is not None and hasattr(page, "refresh"):
                page.refresh()


def _backend_check(startup: DesktopStartupState) -> int:
    if not startup.ready or startup.services is None:
        print("SensorQA desktop backend check: FAILED")
        print(startup.error or "Unknown startup problem")
        return 1

    health = startup.services.tools.health()
    tools = startup.services.tools.list_tools()
    enabled = sum(1 for tool in tools if tool.enabled)
    print("SensorQA desktop backend check: OK")
    print(f"Registered tools : {health.registered_count}")
    print(f"Enabled tools    : {enabled}")
    print(f"Rejected tools   : {health.rejected_count}")
    print(f"Load errors      : {health.load_error_count}")
    if startup.bootstrap is not None:
        print(f"Project root     : {startup.bootstrap.paths.project_root}")
        print(f"Startup warnings : {len(startup.bootstrap.warnings)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SensorQA desktop application")
    parser.add_argument("--check", action="store_true", help="check backend startup without opening Qt")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)

    startup = start_backend(args.project_root)
    if args.check:
        return _backend_check(startup)

    if not QT_AVAILABLE:
        print("PySide6 is required for the SensorQA desktop application.")
        if QT_IMPORT_ERROR is not None:
            print(f"Import error: {QT_IMPORT_ERROR}")
        print("Install it with: python -m pip install PySide6")
        return 2

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("SensorQA")
    app.setOrganizationName("SensorQA")
    apply_theme(app)
    window = SensorQAMainWindow(startup)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
