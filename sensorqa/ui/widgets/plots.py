from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from sensorqa.ui import humanize_identifier, humanize_unit
from sensorqa.ui.theme import apply_matplotlib_theme

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


def _downsample(x: np.ndarray, y: np.ndarray, maximum: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a long curve for responsive desktop plotting.

    Args:
        x: X-axis values.
        y: Y-axis values.
        maximum: Maximum number of displayed samples.

    Returns:
        Downsampled ``x`` and ``y`` arrays.
    """

    if len(x) <= maximum:
        return x, y
    step = max(1, int(np.ceil(len(x) / maximum)))
    return x[::step], y[::step]


class CalibrationComparisonPlot(QWidget):
    """Plot calibration error before and after correction.

    Args:
        parent: Optional Qt parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the calibration comparison plot.

        Args:
            parent: Optional Qt parent widget.

        Returns:
            None.
        """

        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if MATPLOTLIB_AVAILABLE:
            apply_matplotlib_theme()
            self.figure = Figure(figsize=(6, 3.5), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas)
            self.message = None
        else:
            self.figure = None
            self.canvas = None
            self.message = QLabel(
                "Plot preview is unavailable because matplotlib Qt support is not installed."
            )
            self.message.setProperty("role", "muted")
            self.message.setWordWrap(True)
            layout.addWidget(self.message)

    def set_data(self, evaluated_data, unit: str | None = None) -> None:
        """Update the plot with held-out calibration errors.

        Args:
            evaluated_data: DataFrame containing ``raw_error`` and
                ``corrected_error`` columns.
            unit: Optional engineering unit for the error axis.

        Returns:
            None.
        """

        if self.figure is None or self.canvas is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if evaluated_data is None or evaluated_data.empty:
            ax.text(
                0.5,
                0.5,
                "No calibration data yet",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            x = np.arange(len(evaluated_data))
            before = np.asarray(evaluated_data["raw_error"], dtype=float)
            after = np.asarray(evaluated_data["corrected_error"], dtype=float)
            x_before, before = _downsample(x, before)
            x_after, after = _downsample(x, after)
            ax.plot(x_before, before, label="Before", linewidth=1.2)
            ax.plot(x_after, after, label="After", linewidth=1.2)
            ax.axhline(0.0, linewidth=1)
            ax.set_title("Validation error")
            ax.set_xlabel("Validation sample")
            readable_unit = humanize_unit(unit)
            ax.set_ylabel(f"Error ({readable_unit})" if readable_unit else "Error")
            ax.legend(frameon=False)
        self.canvas.draw_idle()


class AnalysisResultPlot(QWidget):
    """Render a useful plot for supported SensorQA analysis results.

    The widget only visualizes data already produced by the analysis tools or
    present in the current dataset. It does not recalculate qualification or
    alter the analysis result.

    Args:
        parent: Optional Qt parent widget.
    """

    SUPPORTED_TOOL_IDS = {
        "imu_sampling_integrity",
        "imu_psd",
        "imu_axis_correlation",
        "imu_allan_deviation",
        "generic_psd",
        "generic_sensitivity",
        "generic_repeatability",
        "generic_hysteresis",
        "generic_revision_comparison",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the analysis plot widget.

        Args:
            parent: Optional Qt parent widget.

        Returns:
            None.
        """

        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if MATPLOTLIB_AVAILABLE:
            apply_matplotlib_theme()
            self.figure = Figure(figsize=(7, 3.2), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas)
            self.message = None
        else:
            self.figure = None
            self.canvas = None
            self.message = QLabel("Plots are unavailable because matplotlib Qt support is not installed.")
            self.message.setProperty("role", "muted")
            self.message.setWordWrap(True)
            layout.addWidget(self.message)

    @classmethod
    def supports(cls, analysis) -> bool:
        """Return whether the result has a plot implemented by this widget.

        Args:
            analysis: Presentation-layer analysis summary.

        Returns:
            ``True`` when the result can be visualized.
        """

        return getattr(analysis, "tool_id", "") in cls.SUPPORTED_TOOL_IDS

    def set_analysis(self, analysis, dataset=None) -> bool:
        """Render the most useful plot for an analysis result.

        Args:
            analysis: Presentation-layer analysis summary.
            dataset: Optional current ``SensorQADataset`` used only for plots
                that require the original normalized samples.

        Returns:
            ``True`` if a plot was rendered, otherwise ``False``.
        """

        if self.figure is None or self.canvas is None:
            return False

        self.figure.clear()
        tool_id = getattr(analysis, "tool_id", "")
        metadata = getattr(analysis, "metadata", {}) or {}

        rendered = False
        if tool_id == "imu_sampling_integrity":
            rendered = self._sampling_integrity(dataset)
        elif tool_id == "imu_psd":
            rendered = self._imu_psd(metadata)
        elif tool_id == "imu_axis_correlation":
            rendered = self._axis_correlation(metadata)
        elif tool_id == "imu_allan_deviation":
            rendered = self._allan(metadata)
        elif tool_id == "generic_psd":
            rendered = self._generic_psd(metadata, analysis)
        elif tool_id == "generic_sensitivity":
            rendered = self._sensitivity(metadata, analysis)
        elif tool_id == "generic_repeatability":
            rendered = self._repeatability(metadata, analysis)
        elif tool_id == "generic_hysteresis":
            rendered = self._hysteresis(metadata, analysis)
        elif tool_id == "generic_revision_comparison":
            rendered = self._revision_comparison(metadata, analysis)

        if rendered:
            self.canvas.draw_idle()
        return rendered

    def _sampling_integrity(self, dataset) -> bool:
        if dataset is None or "timestamp" not in dataset.data.columns:
            return False
        timestamp = pd.to_numeric(dataset.data["timestamp"], errors="coerce").to_numpy(dtype=float)
        timestamp = timestamp[np.isfinite(timestamp)]
        if len(timestamp) < 3:
            return False
        intervals = np.diff(timestamp)
        mask = np.isfinite(intervals) & (intervals > 0)
        intervals = intervals[mask]
        if len(intervals) < 2:
            return False
        x = np.arange(1, len(intervals) + 1)
        x, intervals = _downsample(x, intervals)
        milliseconds = intervals * 1000.0
        ax = self.figure.add_subplot(111)
        ax.plot(x, milliseconds, linewidth=1.0)
        ax.axhline(float(np.median(milliseconds)), linewidth=1.0, linestyle="--", label="Median")
        ax.set_title("Sample timing")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Sample interval (ms)")
        ax.legend(frameon=False)
        return True

    def _imu_psd(self, metadata: dict[str, Any]) -> bool:
        spectra = metadata.get("spectra")
        if not isinstance(spectra, dict) or not spectra:
            return False
        ax = self.figure.add_subplot(111)
        plotted = 0
        unit = ""
        for axis_name, spectrum in spectra.items():
            if not isinstance(spectrum, dict):
                continue
            frequency = np.asarray(spectrum.get("frequency_hz", []), dtype=float)
            psd = np.asarray(spectrum.get("psd", []), dtype=float)
            mask = np.isfinite(frequency) & np.isfinite(psd) & (frequency > 0) & (psd > 0)
            if np.count_nonzero(mask) < 2:
                continue
            frequency, psd = _downsample(frequency[mask], psd[mask])
            ax.semilogy(frequency, psd, linewidth=1.0, label=humanize_identifier(axis_name))
            unit = unit or humanize_unit(spectrum.get("psd_unit"))
            plotted += 1
        if not plotted:
            return False
        ax.set_title("Power spectral density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"PSD ({unit})" if unit else "PSD")
        ax.legend(frameon=False, ncol=2)
        return True

    def _generic_psd(self, metadata: dict[str, Any], analysis) -> bool:
        frequency = np.asarray(metadata.get("frequencies_hz", []), dtype=float)
        psd = np.asarray(metadata.get("psd", []), dtype=float)
        mask = np.isfinite(frequency) & np.isfinite(psd) & (frequency > 0) & (psd > 0)
        if np.count_nonzero(mask) < 2:
            return False
        frequency, psd = _downsample(frequency[mask], psd[mask])
        ax = self.figure.add_subplot(111)
        ax.semilogy(frequency, psd, linewidth=1.1)
        unit = ""
        for metric in getattr(analysis, "metrics", ()):
            if "PSD" in metric.name and metric.unit:
                unit = humanize_unit(metric.unit)
                break
        ax.set_title("Error power spectral density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"PSD ({unit})" if unit else "PSD")
        return True

    def _axis_correlation(self, metadata: dict[str, Any]) -> bool:
        results = metadata.get("correlation_results")
        if not isinstance(results, dict) or not results:
            return False
        groups = [(name, value) for name, value in results.items() if isinstance(value, dict)]
        if not groups:
            return False
        groups = groups[:2]
        image = None
        for index, (name, group) in enumerate(groups, start=1):
            matrix = np.asarray(group.get("correlation_matrix", []), dtype=float)
            axes = list(group.get("axes", []))
            if matrix.shape != (3, 3):
                continue
            ax = self.figure.add_subplot(1, len(groups), index)
            image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            labels = [humanize_identifier(value) for value in axes]
            ax.set_xticks(range(3), labels=labels)
            ax.set_yticks(range(3), labels=labels)
            ax.set_title(humanize_identifier(name))
            for row in range(3):
                for col in range(3):
                    ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
        if image is None:
            return False
        self.figure.colorbar(image, ax=self.figure.axes, fraction=0.025, pad=0.04, label="Correlation")
        return True

    def _allan(self, metadata: dict[str, Any]) -> bool:
        curves = metadata.get("allan_curves")
        if not isinstance(curves, dict) or not curves:
            return False
        ax = self.figure.add_subplot(111)
        plotted = 0
        for axis_name, curve in curves.items():
            if not isinstance(curve, dict):
                continue
            tau = np.asarray(curve.get("tau_seconds", []), dtype=float)
            adev = np.asarray(curve.get("allan_deviation", []), dtype=float)
            mask = np.isfinite(tau) & np.isfinite(adev) & (tau > 0) & (adev > 0)
            if np.count_nonzero(mask) < 2:
                continue
            ax.loglog(tau[mask], adev[mask], linewidth=1.1, label=humanize_identifier(axis_name))
            plotted += 1
        if not plotted:
            return False
        ax.set_title("Allan deviation")
        ax.set_xlabel("Averaging time (s)")
        ax.set_ylabel("Allan deviation")
        ax.legend(frameon=False, ncol=2)
        return True

    def _sensitivity(self, metadata: dict[str, Any], analysis) -> bool:
        reference = np.asarray(metadata.get("reference_levels", []), dtype=float)
        measurement = np.asarray(metadata.get("mean_measurements_by_level", []), dtype=float)
        mask = np.isfinite(reference) & np.isfinite(measurement)
        if np.count_nonzero(mask) < 2:
            return False
        ax = self.figure.add_subplot(111)
        ax.plot(reference[mask], measurement[mask], marker="o", linewidth=1.1)
        ax.set_title("Response by reference level")
        ax.set_xlabel("Reference")
        ax.set_ylabel("Mean measurement")
        return True

    def _repeatability(self, metadata: dict[str, Any], analysis) -> bool:
        groups = metadata.get("repeatability_groups")
        if not isinstance(groups, list) or not groups:
            return False
        reference = np.asarray([item.get("reference_level", np.nan) for item in groups], dtype=float)
        spread = np.asarray([item.get("measurement_std", np.nan) for item in groups], dtype=float)
        mask = np.isfinite(reference) & np.isfinite(spread)
        if np.count_nonzero(mask) < 1:
            return False
        ax = self.figure.add_subplot(111)
        ax.bar(reference[mask].astype(str), spread[mask])
        ax.set_title("Repeatability by reference level")
        ax.set_xlabel("Reference level")
        unit = humanize_unit(metadata.get("measurement_unit"))
        ax.set_ylabel(f"Standard deviation ({unit})" if unit else "Standard deviation")
        return True

    def _hysteresis(self, metadata: dict[str, Any], analysis) -> bool:
        levels = metadata.get("matched_levels")
        if not isinstance(levels, list) or not levels:
            return False
        reference = np.asarray([item.get("reference", np.nan) for item in levels], dtype=float)
        first = np.asarray([item.get("first_mean", np.nan) for item in levels], dtype=float)
        second = np.asarray([item.get("second_mean", np.nan) for item in levels], dtype=float)
        mask = np.isfinite(reference) & np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(mask) < 2:
            return False
        labels = metadata.get("sweep_labels", ["Sweep 1", "Sweep 2"])
        ax = self.figure.add_subplot(111)
        ax.plot(reference[mask], first[mask], marker="o", label=str(labels[0]))
        ax.plot(reference[mask], second[mask], marker="o", label=str(labels[1]))
        ax.set_title("Hysteresis comparison")
        ax.set_xlabel("Reference")
        ax.set_ylabel("Mean measurement")
        ax.legend(frameon=False)
        return True

    def _revision_comparison(self, metadata: dict[str, Any], analysis) -> bool:
        summaries = metadata.get("revision_summaries")
        if not isinstance(summaries, dict) or len(summaries) < 2:
            return False
        names = list(summaries)
        rmse = [float(summaries[name].get("rmse", np.nan)) for name in names]
        if not any(np.isfinite(rmse)):
            return False
        ax = self.figure.add_subplot(111)
        ax.bar(names, rmse)
        ax.set_title("RMSE by revision")
        ax.set_xlabel("Revision")
        unit = ""
        for metric in getattr(analysis, "metrics", ()):
            if metric.name.endswith("RMSE") and metric.unit:
                unit = humanize_unit(metric.unit)
                break
        ax.set_ylabel(f"RMSE ({unit})" if unit else "RMSE")
        return True
