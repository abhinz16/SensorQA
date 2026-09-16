from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sensorqa.core.tool_contract import SensorType
from sensorqa.ui import humanize_identifier, humanize_unit
from sensorqa.ui.components import MetricCard, PageHeader, StatusBadge



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


TOOL_COPY: dict[str, str] = {
    "generic_accuracy": "How far are the readings from the reference? Reports bias, MAE, RMSE, and worst-case error.",
    "generic_linearity": "Fits a straight line across the tested range and shows where the response bends away from it.",
    "generic_repeatability": "At the same reference point, do repeated readings land in the same place? This check measures that spread.",
    "generic_precision": "Shows the random spread in the measurement error, separate from a steady offset.",
    "generic_sensitivity": "Shows how much the sensor output changes for a change in the reference value.",
    "generic_drift": "Looks for a steady change in measurement error as the test runs.",
    "generic_snr": "Compares the useful reference-driven signal with the leftover measurement noise.",
    "generic_psd": "Breaks measurement error down by frequency so periodic noise is easier to spot.",
    "generic_environmental_stability": "Looks for changes in error as temperature, humidity, or supply voltage changes.",
    "generic_outlier_detection": "Flags unusually large errors without removing or changing any rows.",
    "generic_hysteresis": "Compares upward and downward sweeps at the same reference values.",
    "generic_revision_comparison": "Compares the main error metrics across hardware or firmware revisions.",
    "imu_sampling_integrity": "Checks sample timing, measured rate, gaps, and irregular timestamps before timing-sensitive IMU checks run.",
    "imu_gyro_bias": "Measures the stationary offset on each gyroscope axis.",
    "imu_accelerometer_bias": "Measures stationary accelerometer offset and checks the gravity vector.",
    "imu_stationary_noise": "Measures short-term accelerometer and gyroscope noise while the IMU is still.",
    "imu_allan_deviation": "Shows how IMU noise changes with averaging time and reports Allan regions when the recording is long enough.",
    "imu_axis_correlation": "Looks for axes that move together statistically. It does not treat correlation as proof of misalignment.",
    "imu_psd": "Shows dominant frequencies and broadband noise in accelerometer and gyroscope signals.",
    "imu_saturation": "Finds readings that reach or stay close to the configured sensor limits.",
    "imu_gravity_error": "Checks the measured acceleration vector against gravity during static or labeled-orientation tests.",
    "imu_six_position_calibration": "Uses six static orientations to estimate accelerometer gain and offset corrections.",
    "imu_temperature_stability": "Looks for accelerometer or gyroscope output that changes with temperature during the test.",
}

CATEGORY_LABELS = {
    "accuracy": "Accuracy and response",
    "precision": "Precision",
    "stability": "Stability",
    "noise": "Noise",
    "frequency": "Frequency",
    "environmental": "Environment",
    "data_quality": "Data quality",
    "diagnostic": "Comparison",
    "imu": "IMU checks",
    "calibration": "Calibration",
}


FIELD_LABELS = {
    "measurement": "Measurement",
    "reference": "Reference",
    "timestamp": "Timestamp",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "supply_voltage": "Supply voltage",
    "revision": "Revision",
    "sweep_direction": "Sweep direction",
    "orientation_label": "Orientation label",
    "ax": "Accel X",
    "ay": "Accel Y",
    "az": "Accel Z",
    "gx": "Gyro X",
    "gy": "Gyro Y",
    "gz": "Gyro Z",
}


PARAMETER_LABELS = {
    "minimum_samples": "Minimum samples",
    "minimum_samples_per_orientation": "Samples per orientation",
    "minimum_samples_per_revision": "Samples per revision",
    "minimum_repeats_per_level": "Repeats per level",
    "minimum_repeat_levels": "Repeated levels",
    "minimum_unique_reference_values": "Reference levels",
    "minimum_reference_levels": "Reference levels",
    "minimum_overlap_levels": "Matched levels",
    "minimum_duration_s": "Minimum duration",
    "minimum_duration_seconds": "Minimum duration",
    "minimum_temperature_span_degc": "Temperature span",
    "minimum_samples_per_bin": "Samples per temperature bin",
    "temperature_bins": "Temperature bins",
    "reference_group_tolerance": "Reference grouping tolerance",
    "reference_match_tolerance": "Reference matching tolerance",
    "modified_z_threshold": "Outlier threshold",
    "segment_length": "Segment length",
    "overlap_fraction": "Segment overlap",
    "minimum_frequency_hz": "Minimum frequency",
    "max_sampling_cv": "Sampling variation limit",
    "large_gap_multiplier": "Large-gap threshold",
    "irregular_sampling_cv_threshold": "Irregular sampling limit",
    "nominal_rate_warning_percent": "Sample-rate warning",
    "gravity_magnitude": "Gravity reference",
    "warn_gravity_error_percent": "Gravity warning",
    "warn_relative_axis_noise_spread": "Axis-noise warning",
    "warn_relative_axis_spread": "Axis-noise warning",
    "warn_std_to_robust_ratio": "Outlier-tail warning",
    "magnitude_warning_percent": "Gravity-magnitude warning",
    "orientation_angle_warning_deg": "Orientation-angle warning",
    "correlation_warning_threshold": "Correlation warning",
    "difference_correlation_warning_threshold": "Fast-change correlation warning",
    "allow_irregular_sampling": "Allow irregular sampling",
    "tau_points": "Averaging-time points",
    "minimum_allan_pairs": "Minimum Allan pairs",
    "slope_tolerance": "Allan slope tolerance",
    "bias_instability_min_duration_seconds": "Bias-instability duration",
    "window": "Window",
    "peak_prominence_ratio": "Peak prominence",
    "max_reported_peaks": "Reported peaks",
    "near_limit_margin_fraction": "Near-limit margin",
    "exact_limit_tolerance_fraction": "Limit tolerance",
    "minimum_clipping_run_samples": "Clipping run length",
    "minimum_asymmetry_samples": "Asymmetry samples",
    "asymmetry_fraction_threshold": "Asymmetry threshold",
    "minimum_primary_axis_span_fraction": "Minimum axis span",
    "minimum_span": "Minimum environmental span",
    "association_r2_threshold": "Association R squared warning",
}


SENSOR_LABELS = {
    "generic": "Generic sensor",
    "imu": "IMU",
    "accelerometer": "Accelerometer",
    "gyroscope": "Gyroscope",
    "any": "Any sensor",
}


class ToolManagerPage(QWidget):
    def __init__(self, services, parent=None) -> None:
        super().__init__(parent)
        self.services = services

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

        self.layout.addWidget(
            PageHeader(
                "Tools",
                "See what SensorQA can check",
                "Search by name or sensor type. Open a tool to see what data it needs and the settings it uses.",
            )
        )

        grid = QGridLayout()
        grid.setSpacing(12)
        self.registered = MetricCard("Available", "0", "Tools ready to use.")
        self.rejected = MetricCard("Need attention", "0", "Tools that were not accepted.")
        self.errors = MetricCard("Could not load", "0", "Tool files SensorQA could not open.")
        grid.addWidget(self.registered, 0, 0)
        grid.addWidget(self.rejected, 0, 1)
        grid.addWidget(self.errors, 0, 2)
        self.layout.addLayout(grid)

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a tool")
        self.search.textChanged.connect(self.refresh)

        self.sensor_filter = QComboBox()
        self.sensor_filter.addItem("All sensors", None)
        self.sensor_filter.addItem("Generic sensors", SensorType.GENERIC)
        self.sensor_filter.addItem("IMUs", SensorType.IMU)
        self.sensor_filter.currentIndexChanged.connect(self.refresh)

        reload_button = QPushButton("Reload tools")
        reload_button.setProperty("role", "secondaryButton")
        reload_button.clicked.connect(self._reload)

        filters.addWidget(self.search, 1)
        filters.addWidget(self.sensor_filter)
        filters.addWidget(reload_button)
        self.layout.addLayout(filters)

        self.tool_container = QWidget()
        self.tool_layout = QVBoxLayout(self.tool_container)
        self.tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_layout.setSpacing(12)
        self.layout.addWidget(self.tool_container)
        self.layout.addStretch(1)

        self.refresh()

    def _clear_tools(self) -> None:
        """Clear the dynamic tool list before rebuilding it."""

        _clear_layout(self.tool_layout)

    def refresh(self) -> None:
        health = self.services.tools.health()
        self.registered.set_value(str(health.registered_count))
        self.rejected.set_value(str(health.rejected_count))
        self.errors.set_value(str(health.load_error_count))

        self._clear_tools()

        sensor_type = self.sensor_filter.currentData() if hasattr(self, "sensor_filter") else None
        query = self.search.text().strip().lower() if hasattr(self, "search") else ""

        tools = self.services.tools.list_tools(sensor_type)
        visible = []

        for tool in tools:
            summary = self._summary_for(tool)
            haystack = " ".join(
                [
                    tool.name,
                    tool.tool_id,
                    summary,
                    tool.category,
                    " ".join(tool.required_columns),
                ]
            ).lower()

            if query and query not in haystack:
                continue

            visible.append(tool)

        if not visible:
            empty = QLabel("No tools match that search.")
            empty.setProperty("role", "muted")
            self.tool_layout.addWidget(empty)
            self.tool_layout.addStretch(1)
            return

        groups: list[tuple[str, list]] = []

        generic_tools = [
            tool for tool in visible
            if "generic" in tool.compatible_sensor_types
        ]
        imu_tools = [
            tool for tool in visible
            if "generic" not in tool.compatible_sensor_types
        ]

        if generic_tools:
            groups.append(("Generic sensors", generic_tools))
        if imu_tools:
            groups.append(("IMUs", imu_tools))

        for heading, group_tools in groups:
            section = QLabel(heading)
            section.setProperty("role", "sectionTitle")
            self.tool_layout.addWidget(section)

            for tool in group_tools:
                self.tool_layout.addWidget(self._tool_card(tool))

        self.tool_layout.addStretch(1)

    def _tool_card(self, tool) -> QFrame:
        card = QFrame()
        card.setProperty("role", "card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(9)

        top = QHBoxLayout()

        title = QLabel(tool.name)
        title.setProperty("role", "sectionTitle")

        badge = StatusBadge(
            "On" if tool.enabled else "Off",
            "pass" if tool.enabled else "not_evaluated",
        )

        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(badge)
        layout.addLayout(top)

        summary = QLabel(self._summary_for(tool))
        summary.setWordWrap(True)
        layout.addWidget(summary)

        context = QHBoxLayout()
        context.setSpacing(18)

        category = QLabel(CATEGORY_LABELS.get(tool.category, humanize_identifier(tool.category)))
        category.setProperty("role", "muted")

        sensors = ", ".join(
            SENSOR_LABELS.get(value, humanize_identifier(value))
            for value in tool.compatible_sensor_types
        )
        sensor_label = QLabel(sensors)
        sensor_label.setProperty("role", "muted")

        context.addWidget(category)
        context.addWidget(sensor_label)
        context.addStretch(1)
        layout.addLayout(context)

        if tool.required_columns:
            fields = ", ".join(self._field_label(name) for name in tool.required_columns)
            needed = QLabel(f"Required data: {fields}")
            needed.setProperty("role", "muted")
            needed.setWordWrap(True)
            layout.addWidget(needed)

        if tool.optional_columns:
            fields = ", ".join(self._field_label(name) for name in tool.optional_columns)
            optional = QLabel(f"Optional data: {fields}")
            optional.setProperty("role", "muted")
            optional.setWordWrap(True)
            layout.addWidget(optional)

        detail_button = QPushButton(
            "Show settings" if tool.parameters else "Show details"
        )
        detail_button.setProperty("role", "ghostButton")
        detail_button.setCheckable(True)
        detail_button.setMaximumWidth(150)
        layout.addWidget(detail_button)

        details = QFrame()
        details.setProperty("role", "mutedCard")
        details.setVisible(False)

        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(16, 14, 16, 14)
        detail_layout.setSpacing(8)

        if tool.parameters:
            settings_title = QLabel("Current settings")
            settings_title.setStyleSheet("font-weight: 600;")
            detail_layout.addWidget(settings_title)

            for parameter in tool.parameters:
                row = QHBoxLayout()
                name = QLabel(self._parameter_label(parameter.name))
                value = QLabel(self._format_parameter_value(parameter))
                name.setProperty("role", "muted")
                value.setStyleSheet("font-weight: 600;")
                row.addWidget(name)
                row.addStretch(1)
                row.addWidget(value)
                detail_layout.addLayout(row)

        technical = QLabel(f"Version {tool.version}")
        technical.setProperty("role", "muted")
        technical.setWordWrap(True)
        detail_layout.addWidget(technical)

        if tool.warnings:
            warning = QLabel("  ".join(tool.warnings))
            warning.setWordWrap(True)
            warning.setProperty("role", "muted")
            detail_layout.addWidget(warning)

        layout.addWidget(details)

        def toggle_details(checked: bool) -> None:
            details.setVisible(checked)
            if tool.parameters:
                detail_button.setText("Hide settings" if checked else "Show settings")
            else:
                detail_button.setText("Hide details" if checked else "Show details")

        detail_button.toggled.connect(toggle_details)

        return card

    @staticmethod
    def _summary_for(tool) -> str:
        return TOOL_COPY.get(
            tool.tool_id,
            tool.description.strip(),
        )

    @staticmethod
    def _field_label(name: str) -> str:
        return FIELD_LABELS.get(
            name,
            humanize_identifier(name),
        )

    @staticmethod
    def _parameter_label(name: str) -> str:
        return PARAMETER_LABELS.get(
            name,
            humanize_identifier(name),
        )

    @staticmethod
    def _format_parameter_value(parameter) -> str:
        value = parameter.current_value

        if isinstance(value, bool):
            text = "Yes" if value else "No"
        elif isinstance(value, float):
            if value == 0:
                text = "0"
            elif abs(value) >= 1000 or abs(value) < 0.001:
                text = f"{value:.3g}"
            else:
                text = f"{value:g}"
        else:
            text = str(value)

        if parameter.unit:
            text = f"{text} {humanize_unit(parameter.unit)}"

        return text

    def _reload(self) -> None:
        try:
            self.services.tools.reload_tools()
        finally:
            self.refresh()
