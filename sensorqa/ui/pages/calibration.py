from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sensorqa.calibration import (
    CalibrationEvaluationConfig,
    CalibrationFitConfig,
    CalibrationMethod,
    CalibrationRunConfig,
    DataSplitConfig,
    SplitStrategy,
)
from sensorqa.services import CalibrationRequest
from sensorqa.ui.components import EmptyState, MetricCard, PageHeader
from sensorqa.ui.state import DesktopSession
from sensorqa.ui.widgets import CalibrationComparisonPlot



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


class CalibrationPage(QWidget):
    calibration_completed = Signal(object)

    def __init__(self, services, session: DesktopSession, parent=None) -> None:
        super().__init__(parent)
        self.services = services
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
        self.layout.addWidget(PageHeader(
            "Calibration",
            "Calibration Validation",
            "Fit a correction using training data and evaluate it on held-out validation data.",
        ))

        dataset = self.session.dataset
        if dataset is None:
            self.layout.addWidget(EmptyState(
                "No dataset loaded",
                "Run an analysis first, then return here to fit and evaluate a calibration.",
            ))
            self.layout.addStretch(1)
            return

        if "measurement" not in dataset.data.columns or "reference" not in dataset.data.columns:
            self.layout.addWidget(EmptyState(
                "Calibration needs a measurement and a reference",
                "Map both columns in New Analysis before running calibration.",
            ))
            self.layout.addStretch(1)
            return

        controls = QFrame()
        controls.setProperty("role", "card")
        c = QGridLayout(controls)
        c.setContentsMargins(20, 18, 20, 20)
        c.setHorizontalSpacing(16)
        c.setVerticalSpacing(10)

        title = QLabel("Calibration Settings")
        title.setProperty("role", "sectionTitle")
        c.addWidget(title, 0, 0, 1, 4)

        c.addWidget(QLabel("Calibration model"), 1, 0)
        self.method = QComboBox()
        self.method.addItem("Gain and offset", CalibrationMethod.AFFINE.value)
        self.method.addItem("Offset only", CalibrationMethod.OFFSET.value)
        c.addWidget(self.method, 2, 0)

        c.addWidget(QLabel("Validation split"), 1, 1)
        self.split = QComboBox()
        self.split.addItem("Keep data in order", SplitStrategy.SEQUENTIAL.value)
        self.split.addItem("Random split", SplitStrategy.RANDOM.value)
        preferred_split = self.session.preferences.calibration_split_strategy
        preferred_index = self.split.findData(preferred_split)
        if preferred_index >= 0:
            self.split.setCurrentIndex(preferred_index)
        c.addWidget(self.split, 2, 1)

        c.addWidget(QLabel("Validation data"), 1, 2)
        self.validation_fraction = QDoubleSpinBox()
        self.validation_fraction.setRange(5.0, 50.0)
        self.validation_fraction.setValue(self.session.preferences.calibration_validation_percent)
        self.validation_fraction.setSuffix(" %")
        c.addWidget(self.validation_fraction, 2, 2)

        run = QPushButton("Run calibration")
        run.setProperty("role", "primaryButton")
        run.clicked.connect(self._run)
        c.addWidget(run, 2, 3)

        note = QLabel("Validation rows are not used to fit the calibration model.")
        note.setProperty("role", "muted")
        c.addWidget(note, 3, 0, 1, 4)
        self.layout.addWidget(controls)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setProperty("role", "muted")
        self.layout.addWidget(self.message)

        self._render_result()
        self.layout.addStretch(1)

    def _run(self) -> None:
        method = CalibrationMethod(self.method.currentData())
        strategy = SplitStrategy(self.split.currentData())
        config = CalibrationRunConfig(
            split=DataSplitConfig(
                validation_fraction=self.validation_fraction.value() / 100.0,
                strategy=strategy,
            ),
            fit=CalibrationFitConfig(method=method),
            evaluation=CalibrationEvaluationConfig(),
        )

        result = self.services.calibration.run(
            CalibrationRequest(
                data=self.session.dataset,
                measurement_column="measurement",
                reference_column="reference",
                config=config,
            )
        )
        self.session.calibration = result
        if result.success:
            self.message.setText("Calibration completed. The metrics below are calculated on held-out validation data.")
            self.calibration_completed.emit(result)
        else:
            self.message.setText("Calibration could not complete: " + "  ".join(result.errors))
        self.refresh()

    def _render_result(self) -> None:
        result = self.session.calibration
        if result is None or not result.success or result.run_result is None:
            return

        run = result.run_result
        before = run.evaluation.before
        after = run.evaluation.after
        improvement = run.evaluation.improvement

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(MetricCard("Validation rows", str(run.split.validation_row_count)), 0, 0)
        grid.addWidget(MetricCard("RMSE before", f"{before.rmse:.5g}"), 0, 1)
        grid.addWidget(MetricCard("RMSE after", f"{after.rmse:.5g}", "Measured on validation rows.", accent=True), 0, 2)
        change = improvement.rmse_reduction_percent
        change_text = "N/A" if change is None else f"{change:.1f}%"
        grid.addWidget(MetricCard("RMSE improvement", change_text), 0, 3)
        self.layout.addLayout(grid)

        model = QFrame()
        model.setProperty("role", "card")
        m = QVBoxLayout(model)
        m.setContentsMargins(20, 18, 20, 18)
        title = QLabel("Calibration Model")
        title.setProperty("role", "sectionTitle")
        m.addWidget(title)
        note = QLabel("Model fitted using training data only.")
        note.setProperty("role", "muted")
        m.addWidget(note)
        equation = QLabel(f"corrected = {run.model.gain:.6g} × measurement + {run.model.offset:.6g}")
        equation.setProperty("role", "mono")
        m.addWidget(equation)
        self.layout.addWidget(model)

        plot_card = QFrame()
        plot_card.setProperty("role", "card")
        p = QVBoxLayout(plot_card)
        p.setContentsMargins(18, 18, 18, 18)
        title = QLabel("Validation Error Comparison")
        title.setProperty("role", "sectionTitle")
        p.addWidget(title)
        plot = CalibrationComparisonPlot()
        plot.setMinimumHeight(320)
        plot.set_data(
            run.evaluated_validation_data,
            unit=self.session.dataset.units.get("measurement") if self.session.dataset is not None else None,
        )
        p.addWidget(plot)
        self.layout.addWidget(plot_card)
