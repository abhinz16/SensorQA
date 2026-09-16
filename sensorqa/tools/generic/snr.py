#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import BaseAnalysisTool, ParameterType, SensorType, ToolCategory, ToolContext, ToolMetadata, ToolParameter
from sensorqa.tools.generic._utils import add_exclusion_warning, finite_pair, numeric_column


class SNRTool(BaseAnalysisTool):
    """Estimate reference-based signal-to-noise ratio using residual error as noise."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_snr", name="Signal-to-Noise Ratio", version="1.0.0",
            description=(
                "Estimates SNR by treating the reference variation as signal and measurement-reference residual as noise."
            ),
            category=ToolCategory.NOISE,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference"], optional_columns=[],
            parameters=[ToolParameter(
                name="minimum_samples", parameter_type=ParameterType.INTEGER, default=10,
                description="Minimum finite measurement/reference pairs.", minimum=3, maximum=100_000_000,
            )], dependencies=[], author="SensorQA",
        )

    def validate(self, data: pd.DataFrame, context: ToolContext):
        """Validate validate.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Validation result.
        """
        validation = super().validate(data, context)
        if not validation.valid:
            return validation
        m = context.column_mapping.get("measurement")
        r = context.column_mapping.get("reference")
        if m is None or r is None:
            validation.errors.append("Measurement and reference mappings are required.")
            validation.valid = False
            return validation
        measurement, reference, _ = finite_pair(numeric_column(data, m), numeric_column(data, r))
        minimum = int(context.parameters.get("minimum_samples", 10))
        if len(measurement) < minimum:
            validation.errors.append(f"SNR analysis requires at least {minimum} finite pairs.")
        if len(reference) and np.isclose(float(np.var(reference)), 0.0, atol=1e-15, rtol=0.0):
            validation.errors.append("Reference signal has effectively zero variance; SNR is undefined.")
        if validation.errors:
            validation.valid = False
        return validation

    def run(self, data: pd.DataFrame, context: ToolContext) -> AnalysisResult:
        """Run run.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing the calculated metrics and messages.
        """
        m = context.column_mapping["measurement"]
        r = context.column_mapping["reference"]
        measurement, reference, mask = finite_pair(numeric_column(data, m), numeric_column(data, r))
        noise = measurement - reference
        signal_centered = reference - np.mean(reference)
        signal_power = float(np.mean(signal_centered ** 2))
        noise_power = float(np.mean((noise - np.mean(noise)) ** 2))
        signal_rms = float(np.sqrt(signal_power))
        noise_rms = float(np.sqrt(noise_power))
        snr_db = None if np.isclose(noise_power, 0.0, atol=1e-30, rtol=0.0) else float(10.0 * np.log10(signal_power / noise_power))
        unit = context.units.get("measurement")

        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        result.add_metric("Signal RMS", signal_rms, context.units.get("reference"))
        result.add_metric("Noise RMS", noise_rms, unit,
                          "RMS of demeaned measurement-reference residual.")
        result.add_metric("Signal Power", signal_power)
        result.add_metric("Noise Power", noise_power)
        result.add_metric("SNR", snr_db, "dB" if snr_db is not None else None,
                          "10 log10(reference variance / residual-noise variance).")
        result.add_metric("SNR Sample Count", len(noise))
        if snr_db is None:
            result.add_warning("Residual noise variance is effectively zero, so finite SNR in dB is undefined.")
        add_exclusion_warning(result, len(data) - len(noise), len(data))
        result.metadata.update({
            "signal_definition": "demeaned reference",
            "noise_definition": "demeaned (measurement - reference)",
            "snr_definition": "10*log10(signal_power/noise_power)",
        })
        return result
