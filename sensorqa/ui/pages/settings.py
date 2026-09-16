from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sensorqa.ui.components import PageHeader
from sensorqa.ui.state import DesktopSession


class SettingsPage(QWidget):
    def __init__(self, startup, session: DesktopSession, parent=None) -> None:
        super().__init__(parent)
        self.startup = startup
        self.session = session

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 30, 36, 36)
        layout.setSpacing(18)

        layout.addWidget(
            PageHeader(
                "Settings",
                "Preferences",
                "Configure analysis, calibration, and report behavior for this session.",
            )
        )

        layout.addWidget(self._analysis_card())
        layout.addWidget(self._calibration_card())
        layout.addWidget(self._reports_card())
        layout.addWidget(self._session_card())
        layout.addWidget(self._about_card())
        layout.addStretch(1)

    def _analysis_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(8)

        title = QLabel("Requirement Checks")
        title.setProperty("role", "sectionTitle")
        layout.addWidget(title)

        self.qualification_check = QCheckBox("Run requirement checks after analysis")
        self.qualification_check.setChecked(self.session.preferences.run_qualification)
        self.qualification_check.toggled.connect(self._set_qualification)
        layout.addWidget(self.qualification_check)

        note = QLabel(
            "When enabled, SensorQA compares available metrics with the active requirements. Requirements that cannot be evaluated remain marked as not evaluated."
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    def _calibration_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(10)

        title = QLabel("Calibration")
        title.setProperty("role", "sectionTitle")
        layout.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(14)

        split_block = QVBoxLayout()
        split_label = QLabel("Default validation split")
        split_label.setProperty("role", "muted")
        self.split_combo = QComboBox()
        self.split_combo.addItem("Keep data in order", "sequential")
        self.split_combo.addItem("Random split", "random")
        index = self.split_combo.findData(self.session.preferences.calibration_split_strategy)
        self.split_combo.setCurrentIndex(max(index, 0))
        self.split_combo.currentIndexChanged.connect(self._set_split_strategy)
        split_block.addWidget(split_label)
        split_block.addWidget(self.split_combo)

        validation_block = QVBoxLayout()
        validation_label = QLabel("Validation data")
        validation_label.setProperty("role", "muted")
        self.validation_percent = QDoubleSpinBox()
        self.validation_percent.setRange(5.0, 50.0)
        self.validation_percent.setDecimals(0)
        self.validation_percent.setSingleStep(5.0)
        self.validation_percent.setSuffix(" %")
        self.validation_percent.setValue(self.session.preferences.calibration_validation_percent)
        self.validation_percent.valueChanged.connect(self._set_validation_percent)
        validation_block.addWidget(validation_label)
        validation_block.addWidget(self.validation_percent)

        row.addLayout(split_block, 1)
        row.addLayout(validation_block, 1)
        row.addStretch(1)
        layout.addLayout(row)

        note = QLabel(
            "Validation rows are held out from model fitting and used only to evaluate calibration performance."
        )
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    def _reports_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(8)

        title = QLabel("Reports")
        title.setProperty("role", "sectionTitle")
        layout.addWidget(title)

        self.open_report_check = QCheckBox("Open HTML reports after saving")
        self.open_report_check.setChecked(self.session.preferences.open_report_after_export)
        self.open_report_check.toggled.connect(self._set_open_report)
        layout.addWidget(self.open_report_check)

        note = QLabel("Disable this option to save reports without opening them automatically.")
        note.setProperty("role", "muted")
        layout.addWidget(note)
        return card

    def _session_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(16)

        copy = QVBoxLayout()
        title = QLabel("Current work")
        title.setProperty("role", "sectionTitle")
        text = QLabel("Remove the loaded dataset, analysis results, and calibration while keeping your preferences.")
        text.setProperty("role", "muted")
        text.setWordWrap(True)
        copy.addWidget(title)
        copy.addWidget(text)

        button = QPushButton("Clear current work")
        button.setProperty("role", "secondaryButton")
        button.clicked.connect(self._clear_session)

        layout.addLayout(copy, 1)
        layout.addWidget(button)
        return card

    def _about_card(self) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(6)

        title = QLabel("About SensorQA")
        title.setProperty("role", "sectionTitle")
        text = QLabel(
            "SensorQA is a desktop application for sensor characterization, calibration, diagnostics, qualification, and reporting."
        )
        text.setWordWrap(True)
        text.setProperty("role", "muted")
        layout.addWidget(title)
        layout.addWidget(text)
        return card

    def _set_qualification(self, checked: bool) -> None:
        self.session.preferences.run_qualification = bool(checked)

    def _set_open_report(self, checked: bool) -> None:
        self.session.preferences.open_report_after_export = bool(checked)

    def _set_split_strategy(self) -> None:
        self.session.preferences.calibration_split_strategy = str(self.split_combo.currentData())

    def _set_validation_percent(self, value: float) -> None:
        self.session.preferences.calibration_validation_percent = float(value)

    def _clear_session(self) -> None:
        answer = QMessageBox.question(
            self,
            "Clear current work?",
            "This will remove the loaded dataset, analysis results, and calibration from this session.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.session.clear_all()
