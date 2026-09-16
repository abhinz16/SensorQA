#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import BaseAnalysisTool, ParameterType, SensorType, ToolCategory, ToolContext, ToolMetadata, ToolParameter
from sensorqa.tools.generic._utils import finite_many, numeric_column, sampling_information



def _trapezoidal_integral(
    values: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """Integrate values using the NumPy API available at runtime.

    Args:
        values: Samples to integrate.
        coordinates: Coordinate values associated with ``values``.

    Returns:
        The trapezoidal numerical integral as a float.
    """

    trapezoid = getattr(np, "trapezoid", None)

    if callable(trapezoid):
        return float(trapezoid(values, coordinates))

    return float(np.trapz(values, coordinates))


class GenericPSDTool(BaseAnalysisTool):
    """Welch PSD of reference-relative error for generic time-series sensors."""

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            tool_id="generic_psd", name="Error Power Spectral Density", version="1.0.0",
            description=(
                "Computes Welch power spectral density of measurement-reference error for uniformly sampled generic sensor data."
            ),
            category=ToolCategory.FREQUENCY,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["timestamp", "measurement", "reference"], optional_columns=[],
            parameters=[
                ToolParameter(name="segment_length", parameter_type=ParameterType.INTEGER, default=1024,
                              description="Samples per Welch segment.", minimum=32, maximum=1_048_576),
                ToolParameter(name="overlap_fraction", parameter_type=ParameterType.FLOAT, default=0.5,
                              description="Fractional overlap between segments.", minimum=0.0, maximum=0.95),
                ToolParameter(name="minimum_frequency_hz", parameter_type=ParameterType.FLOAT, default=0.0,
                              description="Minimum frequency considered for dominant-peak reporting.", unit="Hz",
                              minimum=0.0, maximum=1_000_000.0),
                ToolParameter(name="max_sampling_cv", parameter_type=ParameterType.FLOAT, default=0.02,
                              description="Maximum interval CV for standard Welch assumptions.", minimum=0.0, maximum=1.0),
            ], dependencies=[], author="SensorQA",
        )

    @staticmethod
    def _welch(signal: np.ndarray, fs: float, segment_length: int, overlap_fraction: float):
        nperseg = min(segment_length, len(signal))
        if nperseg < 8:
            raise ValueError("At least 8 samples are required for PSD estimation.")
        noverlap = int(round(nperseg * overlap_fraction))
        noverlap = min(noverlap, nperseg - 1)
        step = nperseg - noverlap
        starts = list(range(0, len(signal) - nperseg + 1, step))
        if not starts:
            starts = [0]
        window = np.hanning(nperseg)
        window_power = float(np.sum(window ** 2))
        accum = None
        for start in starts:
            segment = signal[start:start + nperseg]
            segment = segment - np.mean(segment)
            spectrum = np.fft.rfft(segment * window)
            psd = (np.abs(spectrum) ** 2) / (fs * window_power)
            if nperseg % 2 == 0:
                if len(psd) > 2:
                    psd[1:-1] *= 2.0
            elif len(psd) > 1:
                psd[1:] *= 2.0
            accum = psd if accum is None else accum + psd
        averaged = accum / len(starts)
        frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        return frequencies, averaged, len(starts), nperseg

    def validate(self, data: pd.DataFrame, context: ToolContext):
        validation = super().validate(data, context)
        if not validation.valid:
            return validation
        cols = [context.column_mapping.get(name) for name in ("timestamp", "measurement", "reference")]
        if any(column is None for column in cols):
            validation.errors.append("Timestamp, measurement, and reference mappings are required.")
            validation.valid = False
            return validation
        arrays, _ = finite_many(*(numeric_column(data, column) for column in cols))
        timestamps = arrays[0]
        info = sampling_information(timestamps)
        if info is None:
            validation.errors.append("A valid positive sampling interval could not be determined.")
        else:
            max_cv = float(context.parameters.get("max_sampling_cv", 0.02))
            if info["coefficient_of_variation"] > max_cv:
                validation.errors.append(
                    "Sampling intervals are too irregular for uniformly sampled Welch PSD analysis."
                )
        if len(timestamps) < 32:
            validation.errors.append("At least 32 finite samples are required for PSD analysis.")
        if validation.errors:
            validation.valid = False
        return validation

    def run(self, data: pd.DataFrame, context: ToolContext) -> AnalysisResult:
        cols = [context.column_mapping[name] for name in ("timestamp", "measurement", "reference")]
        arrays, mask = finite_many(*(numeric_column(data, column) for column in cols))
        timestamp, measurement, reference = arrays
        order = np.argsort(timestamp, kind="stable")
        timestamp, measurement, reference = timestamp[order], measurement[order], reference[order]
        info = sampling_information(timestamp)
        if info is None:
            raise ValueError("Sampling information is unavailable.")
        error = measurement - reference
        frequencies, psd, segment_count, nperseg = self._welch(
            error, info["sampling_frequency_hz"],
            int(context.parameters.get("segment_length", 1024)),
            float(context.parameters.get("overlap_fraction", 0.5)),
        )
        min_freq = float(context.parameters.get("minimum_frequency_hz", 0.0))
        eligible = (frequencies >= min_freq) & (frequencies > 0.0)
        dominant_frequency = None
        dominant_psd = None
        if np.any(eligible):
            eligible_idx = np.flatnonzero(eligible)
            idx = int(eligible_idx[np.argmax(psd[eligible])])
            dominant_frequency = float(frequencies[idx])
            dominant_psd = float(psd[idx])
        integrated_rms = (
            float(np.sqrt(_trapezoidal_integral(psd, frequencies)))
            if len(frequencies) > 1
            else 0.0
        )
        median_psd = float(np.median(psd[frequencies > 0.0])) if np.any(frequencies > 0.0) else float(psd[0])
        resolution = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else None
        unit = context.units.get("measurement")
        psd_unit = f"{unit}^2/Hz" if unit else None

        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        result.add_metric("PSD Sampling Frequency", info["sampling_frequency_hz"], "Hz")
        result.add_metric("PSD Frequency Resolution", resolution, "Hz" if resolution is not None else None)
        result.add_metric("PSD Segment Count", segment_count)
        result.add_metric("PSD Segment Length", nperseg)
        result.add_metric("Dominant Error Frequency", dominant_frequency, "Hz" if dominant_frequency is not None else None)
        result.add_metric("Dominant Error PSD", dominant_psd, psd_unit)
        result.add_metric("Median Error PSD", median_psd, psd_unit)
        result.add_metric("PSD Integrated Error RMS", integrated_rms, unit)
        result.metadata.update({
            "signal_definition": "measurement - reference",
            "method": "Welch PSD with Hann window",
            "frequencies_hz": [float(v) for v in frequencies],
            "psd": [float(v) for v in psd],
            "sampling_interval_cv": info["coefficient_of_variation"],
            "excluded_rows": len(data) - len(error),
        })
        if len(data) - len(error) > 0:
            result.add_warning(f"{len(data) - len(error)} row(s) were excluded because required values were non-finite.")
        return result
