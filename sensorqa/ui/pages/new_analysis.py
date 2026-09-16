from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sensorqa.core.tool_contract import SensorType
from sensorqa.ingestion.column_mapper import ColumnMapping, StandardField
from sensorqa.ingestion.metadata import (
    DatasetSourceInformation,
    ReferenceType,
    SensorInformation,
    SensorQAMetadata,
    TestInformation,
    TestMode,
)
from sensorqa.services import CSVPreviewRequest, DatasetBuildRequest
from sensorqa.ui import humanize_identifier, humanize_unit, machine_unit
from sensorqa.ui.components import FileDropZone, PageHeader, WorkflowStepper
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


_GENERIC_FIELDS = [
    StandardField.TIMESTAMP,
    StandardField.MEASUREMENT,
    StandardField.REFERENCE,
    StandardField.TEMPERATURE,
    StandardField.HUMIDITY,
    StandardField.SUPPLY_VOLTAGE,
    StandardField.REVISION,
    StandardField.TEST_CYCLE,
    StandardField.SWEEP_DIRECTION,
    StandardField.SENSOR_ID,
]

_IMU_FIELDS = [
    StandardField.TIMESTAMP,
    StandardField.AX,
    StandardField.AY,
    StandardField.AZ,
    StandardField.GX,
    StandardField.GY,
    StandardField.GZ,
    StandardField.TEMPERATURE,
    StandardField.AX_REFERENCE,
    StandardField.AY_REFERENCE,
    StandardField.AZ_REFERENCE,
    StandardField.GX_REFERENCE,
    StandardField.GY_REFERENCE,
    StandardField.GZ_REFERENCE,
    StandardField.ORIENTATION_LABEL,
    StandardField.MOTION_STATE,
    StandardField.EXPERIMENT_LABEL,
    StandardField.SENSOR_ID,
]

_UNIT_DEFAULTS = {
    "timestamp": "s",
    "ax": "m/s^2",
    "ay": "m/s^2",
    "az": "m/s^2",
    "ax_reference": "m/s^2",
    "ay_reference": "m/s^2",
    "az_reference": "m/s^2",
    "gx": "rad/s",
    "gy": "rad/s",
    "gz": "rad/s",
    "gx_reference": "rad/s",
    "gy_reference": "rad/s",
    "gz_reference": "rad/s",
    "temperature": "degC",
    "supply_voltage": "V",
}

_UNIT_PLACEHOLDERS = {
    "measurement": "for example: V, psi, ppm, N",
    "reference": "same physical unit as the reference source",
}


class NewAnalysisPage(QWidget):
    analysis_completed = Signal(object)
    dataset_built = Signal(object)

    STEPS = ["File", "Columns", "Units", "Test", "Review"]

    def __init__(self, services, session: DesktopSession, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.session = session
        self.current_step = 0
        self.source_path: str | None = None
        self.preview_result = None
        self.mapping_combos: dict[StandardField, QComboBox] = {}
        self.unit_edits: dict[str, QLineEdit] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(36, 30, 36, 30)
        outer.setSpacing(16)

        outer.addWidget(PageHeader(
            "New analysis",
            "Choose a sensor file",
            "Start with a CSV. You can check the columns, units, and test details before SensorQA runs anything.",
        ))

        self.stepper = WorkflowStepper(self.STEPS)
        outer.addWidget(self.stepper)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_import_step())
        self.stack.addWidget(self._build_mapping_step())
        self.stack.addWidget(self._build_units_step())
        self.stack.addWidget(self._build_setup_step())
        self.stack.addWidget(self._build_review_step())
        outer.addWidget(self.stack, 1)

        self.message_frame = QFrame()
        self.message_frame.setProperty("role", "messageBanner")
        self.message_frame.setProperty("state", "info")
        message_layout = QVBoxLayout(self.message_frame)
        message_layout.setContentsMargins(16, 12, 16, 12)
        message_layout.setSpacing(4)

        self.message_title = QLabel("")
        self.message_title.setProperty("role", "messageTitle")
        self.message = QLabel("")
        self.message.setProperty("role", "messageBody")
        self.message.setWordWrap(True)
        message_layout.addWidget(self.message_title)
        message_layout.addWidget(self.message)
        self.message_frame.hide()
        outer.addWidget(self.message_frame)

        footer = QHBoxLayout()
        footer.addStretch(1)

        self.back_button = QPushButton("Back")
        self.back_button.setProperty("role", "secondaryButton")
        self.back_button.clicked.connect(self._back)
        self.next_button = QPushButton("Continue")
        self.next_button.setProperty("role", "primaryButton")
        self.next_button.clicked.connect(self._next)
        footer.addWidget(self.back_button)
        footer.addWidget(self.next_button)
        outer.addLayout(footer)
        self._update_navigation()

    # ------------------------------------------------------------------
    # Step builders
    # ------------------------------------------------------------------

    def _build_import_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(14)

        self.drop_zone = FileDropZone()
        self.drop_zone.browse_requested.connect(self._browse)
        self.drop_zone.file_selected.connect(self._load_preview)
        layout.addWidget(self.drop_zone)

        self.file_card = QFrame()
        self.file_card.setProperty("role", "card")
        file_layout = QVBoxLayout(self.file_card)
        file_layout.setContentsMargins(18, 16, 18, 18)
        self.file_title = QLabel("No file selected")
        self.file_title.setProperty("role", "sectionTitle")
        self.file_meta = QLabel("Choose a CSV and we will show you the first few rows.")
        self.file_meta.setProperty("role", "muted")
        self.file_meta.setWordWrap(True)
        self.preview_table = DataFrameTable()
        self.preview_table.setMinimumHeight(230)
        file_layout.addWidget(self.file_title)
        file_layout.addWidget(self.file_meta)
        file_layout.addWidget(self.preview_table)
        layout.addWidget(self.file_card)
        return page

    def _build_mapping_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)

        intro = QFrame()
        intro.setProperty("role", "card")
        top = QGridLayout(intro)
        top.setContentsMargins(18, 16, 18, 16)
        top.addWidget(QLabel("Sensor type"), 0, 0)
        self.sensor_type_combo = QComboBox()
        self.sensor_type_combo.addItem("Generic sensor", SensorType.GENERIC.value)
        self.sensor_type_combo.addItem("IMU", SensorType.IMU.value)
        self.sensor_type_combo.currentIndexChanged.connect(self._rebuild_mapping_rows)
        top.addWidget(self.sensor_type_combo, 1, 0)
        note = QLabel("We made a best guess from the column names. Check each one and change anything that looks wrong.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        top.addWidget(note, 0, 1, 2, 1)
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.mapping_container = QWidget()
        self.mapping_layout = QGridLayout(self.mapping_container)
        self.mapping_layout.setContentsMargins(0, 0, 0, 0)
        self.mapping_layout.setHorizontalSpacing(14)
        self.mapping_layout.setVerticalSpacing(8)
        scroll.setWidget(self.mapping_container)
        layout.addWidget(scroll, 1)
        return page

    def _build_units_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)

        card = QFrame()
        card.setProperty("role", "card")
        self.units_form = QFormLayout(card)
        self.units_form.setContentsMargins(20, 18, 20, 20)
        self.units_form.setSpacing(10)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_setup_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)

        card = QFrame()
        card.setProperty("role", "card")
        form = QFormLayout(card)
        form.setContentsMargins(20, 18, 20, 20)
        form.setSpacing(10)

        self.test_name = QLineEdit()
        self.test_name.setPlaceholderText("for example: Bench test 01")
        self.test_mode = QComboBox()
        self.manufacturer = QLineEdit()
        self.model = QLineEdit()
        self.sample_rate = QLineEdit()
        self.sample_rate.setPlaceholderText("optional, in Hz")
        self.reference_type = QComboBox()
        self.reference_type.addItem("No external reference", ReferenceType.NONE.value)
        self.reference_type.addItem("User provided", ReferenceType.USER_PROVIDED.value)
        self.reference_type.addItem("Calibrated instrument", ReferenceType.CALIBRATED_INSTRUMENT.value)
        self.reference_type.addItem("Motion capture", ReferenceType.MOTION_CAPTURE.value)
        self.reference_type.addItem("GNSS", ReferenceType.GNSS.value)
        self.reference_type.addItem("Encoder", ReferenceType.ENCODER.value)
        self.reference_type.addItem("Simulation", ReferenceType.SIMULATION.value)
        self.reference_type.addItem("Other", ReferenceType.OTHER.value)

        form.addRow("Test name", self.test_name)
        form.addRow("Test mode", self.test_mode)
        form.addRow("Manufacturer", self.manufacturer)
        form.addRow("Sensor model", self.model)
        form.addRow("Nominal sample rate", self.sample_rate)
        form.addRow("Reference source", self.reference_type)
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _build_review_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(12)

        self.review_card = QFrame()
        self.review_card.setProperty("role", "card")
        self.review_layout = QVBoxLayout(self.review_card)
        self.review_layout.setContentsMargins(20, 18, 20, 20)
        self.review_layout.setSpacing(8)
        layout.addWidget(self.review_card)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # File preview
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose sensor data", "", "CSV files (*.csv);;All files (*)")
        if path:
            self._load_preview(path)

    def _load_preview(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.services.ingestion.preview_csv(CSVPreviewRequest(source=path, preview_rows=20))
        finally:
            QApplication.restoreOverrideCursor()

        self.preview_result = result
        self.session.preview = result
        if not result.success:
            self.source_path = None
            self.file_title.setText("Could not open this file")
            self.file_meta.setText("  ".join(result.errors))
            self.preview_table.set_dataframe(None)
            self._show_message("Could not open this file", "Choose another CSV or check the file format.", "error")
            return

        self.source_path = result.source_file
        meta = result.metadata
        self.file_title.setText(Path(result.source_file).name)
        if meta is not None:
            self.file_meta.setText(f"{meta.row_count:,} rows   {meta.column_count} columns   delimiter: {repr(meta.delimiter)}")
        self.preview_table.set_dataframe(result.preview)
        self._show_message("File loaded", "The preview is ready. Continue when the columns look right.", "success")
        self._detect_sensor_type()

    def _detect_sensor_type(self) -> None:
        if self.preview_result is None or self.preview_result.suggested_mapping is None:
            return
        fields = {field.value for field in self.preview_result.suggested_mapping.mapped_fields}
        imu_hits = len(fields & {"ax", "ay", "az", "gx", "gy", "gz"})
        if imu_hits >= 3:
            self.sensor_type_combo.setCurrentIndex(self.sensor_type_combo.findData(SensorType.IMU.value))
        else:
            self.sensor_type_combo.setCurrentIndex(self.sensor_type_combo.findData(SensorType.GENERIC.value))

    # ------------------------------------------------------------------
    # Mapping / units / setup
    # ------------------------------------------------------------------

    def _rebuild_mapping_rows(self) -> None:
        _clear_layout(self.mapping_layout)
        self.mapping_combos = {}

        if self.preview_result is None:
            return

        fields = _IMU_FIELDS if self.sensor_type_combo.currentData() == SensorType.IMU.value else _GENERIC_FIELDS
        columns = list(self.preview_result.columns)
        suggested = self.preview_result.suggested_mapping

        header1 = QLabel("Use this as")
        header1.setProperty("role", "eyebrow")
        header2 = QLabel("Column in your file")
        header2.setProperty("role", "eyebrow")
        self.mapping_layout.addWidget(header1, 0, 0)
        self.mapping_layout.addWidget(header2, 0, 1)

        for row, field in enumerate(fields, start=1):
            label = QLabel(humanize_identifier(field.value))
            combo = QComboBox()
            combo.addItem("Skip this field", None)
            for column in columns:
                combo.addItem(column, column)
            if suggested is not None:
                source = suggested.get(field)
                if source in columns:
                    combo.setCurrentIndex(combo.findData(source))
            self.mapping_layout.addWidget(label, row, 0)
            self.mapping_layout.addWidget(combo, row, 1)
            self.mapping_combos[field] = combo
        self.mapping_layout.setColumnStretch(1, 1)

    def _current_mapping(self) -> ColumnMapping:
        mapping = {}
        for field, combo in self.mapping_combos.items():
            source = combo.currentData()
            if source:
                mapping[field] = source
        return ColumnMapping(mapping=mapping)

    def _rebuild_units(self) -> None:
        while self.units_form.rowCount():
            self.units_form.removeRow(0)
        self.unit_edits = {}
        mapping = self._current_mapping()
        required, optional = self.services.ingestion.unit_fields_for_mapping(mapping)
        fields = list(required) + [name for name in optional if name not in required]

        if not fields:
            note = QLabel("No units are needed for the fields you mapped.")
            note.setProperty("role", "muted")
            self.units_form.addRow(note)
            return

        intro = QLabel("Enter the units exactly as they appear in your test. SensorQA will convert supported quantities when it prepares the dataset.")
        intro.setProperty("role", "muted")
        intro.setWordWrap(True)
        self.units_form.addRow(intro)

        for field_name in fields:
            edit = QLineEdit()
            if field_name in _UNIT_DEFAULTS:
                edit.setText(humanize_unit(_UNIT_DEFAULTS[field_name]))
            edit.setPlaceholderText(_UNIT_PLACEHOLDERS.get(field_name, "engineering unit used in the CSV"))
            label = humanize_identifier(field_name)
            if field_name in required:
                label += " *"
            self.units_form.addRow(label, edit)
            self.unit_edits[field_name] = edit

    def _unit_assignments(self) -> dict[str, str]:
        return {
            name: machine_unit(edit.text().strip())
            for name, edit in self.unit_edits.items()
            if edit.text().strip()
        }

    def _update_test_modes(self) -> None:
        current = self.sensor_type_combo.currentData()
        self.test_mode.clear()
        if current == SensorType.IMU.value:
            modes = [
                ("Stationary", TestMode.IMU_STATIONARY.value),
                ("Controlled orientation", TestMode.IMU_CONTROLLED_ORIENTATION.value),
                ("Dynamic", TestMode.IMU_DYNAMIC.value),
            ]
        else:
            modes = [
                ("Static", TestMode.GENERIC_STATIC.value),
                ("Dynamic", TestMode.GENERIC_DYNAMIC.value),
            ]
        for label, value in modes:
            self.test_mode.addItem(label, value)

    def _metadata(self) -> SensorQAMetadata:
        sample_rate = None
        text = self.sample_rate.text().strip()
        if text:
            sample_rate = float(text)
        sensor_type = SensorType(self.sensor_type_combo.currentData())
        mode = TestMode(self.test_mode.currentData())
        return SensorQAMetadata(
            sensor=SensorInformation(
                sensor_type=sensor_type,
                manufacturer=self.manufacturer.text().strip() or None,
                model=self.model.text().strip() or None,
                nominal_sampling_rate_hz=sample_rate,
            ),
            test=TestInformation(
                test_name=self.test_name.text().strip() or None,
                test_mode=mode,
                reference_type=ReferenceType(self.reference_type.currentData()),
                expected_stationary=(mode == TestMode.IMU_STATIONARY),
            ),
            dataset_source=DatasetSourceInformation(
                file_name=Path(self.source_path).name if self.source_path else None,
            ),
        )

    def _render_review(self) -> None:
        _clear_layout(self.review_layout)

        title = QLabel("Ready to run")
        title.setProperty("role", "sectionTitle")
        self.review_layout.addWidget(title)

        mapping = self._current_mapping()
        sensor_type = SensorType(self.sensor_type_combo.currentData())
        tools = [tool for tool in self.services.tools.list_tools(sensor_type) if tool.enabled]
        details = [
            ("File", Path(self.source_path).name if self.source_path else "None"),
            ("Sensor type", sensor_type.value.upper()),
            ("Columns mapped", str(len(mapping))),
            ("Units entered", str(len(self._unit_assignments()))),
            ("Checks that will run", str(len(tools))),
        ]
        for name, value in details:
            row = QHBoxLayout()
            a = QLabel(name)
            a.setProperty("role", "muted")
            b = QLabel(value)
            b.setStyleSheet("font-weight: 600;")
            row.addWidget(a)
            row.addStretch(1)
            row.addWidget(b)
            self.review_layout.addLayout(row)

        note = QLabel("SensorQA will prepare the data, run the enabled checks that fit this sensor type, and then open the results.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        self.review_layout.addWidget(note)

    def _show_message(self, title: str, body: str, state: str = "info") -> None:
        """Show a prominent inline message on the current workflow step."""

        self.message_title.setText(title)
        self.message.setText(body)
        self.message_frame.setProperty("state", state)
        self.message_frame.style().unpolish(self.message_frame)
        self.message_frame.style().polish(self.message_frame)
        self.message_frame.show()

    def _clear_message(self) -> None:
        """Hide the workflow message banner and clear its text."""

        self.message_title.clear()
        self.message.clear()
        self.message_frame.hide()

    @staticmethod
    def _analysis_error_details(analysis) -> str:
        """Build readable error details from the service and tool results."""

        details: list[str] = []

        workflow = getattr(analysis, "workflow", None)
        pipeline = getattr(workflow, "pipeline_result", None)
        error_results = getattr(pipeline, "error_results", []) if pipeline is not None else []

        for result in error_results:
            tool_name = getattr(result, "tool_name", None) or getattr(result, "tool_id", "Analysis check")
            messages = list(getattr(result, "messages", []) or [])
            if messages:
                details.append(f"• {tool_name}: {messages[0]}")
            else:
                details.append(f"• {tool_name}: This check returned an error.")

        for error in getattr(analysis, "errors", ()):
            text = str(error).strip()
            if text and text != "SensorQA analysis did not complete successfully.":
                bullet = f"• {text}"
                if bullet not in details:
                    details.append(bullet)

        if not details:
            details.append("• SensorQA could not complete all of the selected checks.")

        return "\n".join(details)

    # ------------------------------------------------------------------
    # Navigation and execution
    # ------------------------------------------------------------------

    def _next(self) -> None:
        if self.current_step == 0:
            if self.preview_result is None or not self.preview_result.success:
                self._show_message("Choose a CSV first", "Select a CSV that SensorQA can open before continuing.", "warning")
                return
            self._rebuild_mapping_rows()
        elif self.current_step == 1:
            mapping = self._current_mapping()
            if len(mapping) == 0:
                self._show_message("Check the column mapping", "Map at least one column before continuing.", "warning")
                return
            self._rebuild_units()
        elif self.current_step == 2:
            required, _ = self.services.ingestion.unit_fields_for_mapping(self._current_mapping())
            missing = [name for name in required if not self.unit_edits.get(name) or not self.unit_edits[name].text().strip()]
            if missing:
                self._show_message("Units are missing", "Add units for: " + ", ".join(missing), "warning")
                return
            self._update_test_modes()
        elif self.current_step == 3:
            try:
                self._metadata()
            except Exception as exc:
                self._show_message("Check the test details", str(exc), "warning")
                return
            self._render_review()
        elif self.current_step == 4:
            self._run_analysis()
            return

        self.current_step = min(self.current_step + 1, len(self.STEPS) - 1)
        self.stack.setCurrentIndex(self.current_step)
        self.stepper.set_current(self.current_step)
        self._clear_message()
        self._update_navigation()

    def _back(self) -> None:
        self.current_step = max(0, self.current_step - 1)
        self.stack.setCurrentIndex(self.current_step)
        self.stepper.set_current(self.current_step)
        self._clear_message()
        self._update_navigation()

    def _update_navigation(self) -> None:
        self.back_button.setEnabled(self.current_step > 0)
        self.next_button.setText("Run analysis" if self.current_step == len(self.STEPS) - 1 else "Continue")

    def _run_analysis(self) -> None:
        if not self.source_path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            build = self.services.ingestion.build_dataset(
                DatasetBuildRequest(
                    source=self.source_path,
                    column_mapping=self._current_mapping(),
                    unit_assignments=self._unit_assignments(),
                    metadata=self._metadata(),
                )
            )
            if not build.success or build.dataset is None:
                self._show_message("SensorQA could not prepare the dataset", "\n".join(build.errors), "error")
                return
            self.session.dataset = build.dataset
            self.session.calibration = None
            self.dataset_built.emit(build.dataset)

            analysis = self.services.analysis.analyze_dataset(
                build.dataset,
                qualify=self.session.preferences.run_qualification,
            )
            self.session.analysis = analysis
            if not analysis.success:
                details = self._analysis_error_details(analysis)
                body = (
                    "One or more checks failed while SensorQA was running the analysis."
                    "\n\n"
                    + details
                    + "\n\nCheck the timestamp unit, sensor type, and test mode, then try again."
                )
                self._show_message(
                    "Analysis could not finish",
                    body,
                    "error",
                )
                return
            self._show_message("Analysis finished", f"{analysis.analysis_count} check(s) returned results.", "success")
            self.analysis_completed.emit(analysis)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "SensorQA",
                f"The analysis could not finish.\n\n{type(exc).__name__}: {exc}",
            )
        finally:
            QApplication.restoreOverrideCursor()
