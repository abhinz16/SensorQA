#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import (
    BaseAnalysisTool, ParameterType, SensorType, ToolCategory,
    ToolContext, ToolMetadata, ToolParameter,
)
from sensorqa.tools.generic._utils import add_exclusion_warning, finite_many, linear_fit, numeric_column, unit_ratio


class DriftTool(BaseAnalysisTool):
    """Characterize change in reference-relative error with elapsed time."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_drift", name="Drift", version="1.0.0",
            description=(
                "Fits reference-relative sensor error versus elapsed time to quantify observed temporal drift."
            ),
            category=ToolCategory.STABILITY,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["timestamp", "measurement", "reference"],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples", parameter_type=ParameterType.INTEGER, default=10,
                    description="Minimum finite timestamp/measurement/reference rows.", minimum=3,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_duration_s", parameter_type=ParameterType.FLOAT, default=1.0,
                    description="Minimum elapsed test duration after unit normalization.", unit="s",
                    minimum=0.0, maximum=1.0e15,
                ),
            ], dependencies=[], author="SensorQA",
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
        cols = [context.column_mapping.get(name) for name in ("timestamp", "measurement", "reference")]
        if any(column is None for column in cols):
            validation.errors.append("Timestamp, measurement, and reference mappings are required.")
            validation.valid = False
            return validation
        arrays, _ = finite_many(*(numeric_column(data, column) for column in cols))
        timestamp = arrays[0]
        minimum = int(context.parameters.get("minimum_samples", 10))
        duration_min = float(context.parameters.get("minimum_duration_s", 1.0))
        if len(timestamp) < minimum:
            validation.errors.append(f"Drift analysis requires at least {minimum} finite rows.")
        if len(timestamp) >= 2:
            duration = float(np.max(timestamp) - np.min(timestamp))
            if duration < duration_min:
                validation.errors.append(
                    f"Drift analysis requires at least {duration_min:g} s duration; observed {duration:g} s."
                )
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
        cols = [context.column_mapping[name] for name in ("timestamp", "measurement", "reference")]
        arrays, mask = finite_many(*(numeric_column(data, column) for column in cols))
        timestamp, measurement, reference = arrays
        order = np.argsort(timestamp, kind="stable")
        timestamp, measurement, reference = timestamp[order], measurement[order], reference[order]
        elapsed = timestamp - timestamp[0]
        error = measurement - reference
        slope, intercept, r2 = linear_fit(elapsed, error)
        duration = float(elapsed[-1] - elapsed[0])
        predicted_total = float(slope * duration)
        observed_total = float(error[-1] - error[0])
        unit = context.units.get("measurement")
        drift_unit = unit_ratio(unit, context.units.get("timestamp", "s"))

        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        result.add_metric("Drift Rate", slope, drift_unit,
                          "Slope of measurement-reference error versus elapsed time.")
        result.add_metric("Predicted Drift Over Test", predicted_total, unit,
                          "Linear-model change in error across the test duration.")
        result.add_metric("Observed Endpoint Error Change", observed_total, unit,
                          "Last finite error minus first finite error after timestamp ordering.")
        result.add_metric("Drift Fit R Squared", r2,
                          description="R² of the linear error-versus-time relationship.")
        result.add_metric("Drift Fit Intercept", intercept, unit)
        result.add_metric("Drift Test Duration", duration, context.units.get("timestamp", "s"))
        result.add_metric("Drift Sample Count", len(error))
        add_exclusion_warning(result, len(data) - len(error), len(data))
        result.add_warning(
            "A time trend is an observed statistical relationship and does not by itself identify the physical cause of drift."
        )
        result.metadata.update({
            "error_definition": "measurement - reference",
            "fit_definition": "error = drift_rate * elapsed_time + intercept",
            "timestamp_column": cols[0],
            "measurement_column": cols[1],
            "reference_column": cols[2],
        })
        return result
