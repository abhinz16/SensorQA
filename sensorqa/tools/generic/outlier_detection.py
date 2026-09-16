#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import BaseAnalysisTool, ParameterType, SensorType, ToolCategory, ToolContext, ToolMetadata, ToolParameter
from sensorqa.tools.generic._utils import add_exclusion_warning, finite_pair, numeric_column


class OutlierDetectionTool(BaseAnalysisTool):
    """Flag unusually large reference-relative errors using a robust MAD score."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_outlier_detection", name="Outlier Detection", version="1.0.0",
            description=(
                "Flags unusually large measurement-reference errors using median absolute deviation without deleting or correcting rows."
            ),
            category=ToolCategory.DATA_QUALITY,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference"], optional_columns=[],
            parameters=[
                ToolParameter(name="minimum_samples", parameter_type=ParameterType.INTEGER, default=10,
                              description="Minimum finite measurement/reference pairs.", minimum=3, maximum=100_000_000),
                ToolParameter(name="modified_z_threshold", parameter_type=ParameterType.FLOAT, default=3.5,
                              description="Absolute modified-Z threshold for outlier flagging.", minimum=0.1, maximum=1000.0),
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
        m, r = context.column_mapping.get("measurement"), context.column_mapping.get("reference")
        if m is None or r is None:
            validation.errors.append("Measurement and reference mappings are required.")
            validation.valid = False
            return validation
        measurement, reference, _ = finite_pair(numeric_column(data, m), numeric_column(data, r))
        minimum = int(context.parameters.get("minimum_samples", 10))
        if len(measurement) < minimum:
            validation.errors.append(f"Outlier analysis requires at least {minimum} finite pairs.")
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
        m, r = context.column_mapping["measurement"], context.column_mapping["reference"]
        measurement_all = numeric_column(data, m)
        reference_all = numeric_column(data, r)
        measurement, reference, mask = finite_pair(measurement_all, reference_all)
        error = measurement - reference
        median_error = float(np.median(error))
        mad = float(np.median(np.abs(error - median_error)))
        threshold = float(context.parameters.get("modified_z_threshold", 3.5))
        if np.isclose(mad, 0.0, atol=1e-15, rtol=0.0):
            scores = np.zeros_like(error)
            nonmedian = ~np.isclose(error, median_error, atol=1e-15, rtol=0.0)
            scores[nonmedian] = np.inf
        else:
            scores = 0.6744897501960817 * (error - median_error) / mad
        outlier_mask = np.abs(scores) > threshold
        outlier_count = int(np.sum(outlier_mask))
        outlier_percent = float(100.0 * outlier_count / len(error))
        finite_positions = np.flatnonzero(mask)
        outlier_positions = finite_positions[outlier_mask]
        max_score = float(np.max(np.abs(scores))) if len(scores) else 0.0
        unit = context.units.get("measurement")

        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        result.add_metric("Outlier Count", outlier_count)
        result.add_metric("Outlier Fraction", outlier_percent, "%")
        result.add_metric("Median Error", median_error, unit)
        result.add_metric("Error MAD", mad, unit)
        result.add_metric("Maximum Absolute Modified Z Score", max_score)
        result.add_metric("Outlier Analysis Sample Count", len(error))
        add_exclusion_warning(result, len(data) - len(error), len(data))
        if outlier_count > 0:
            result.add_warning(
                f"{outlier_count} row(s) exceeded the robust outlier threshold. Flagging is diagnostic only; rows were not removed."
            )
        result.metadata.update({
            "error_definition": "measurement - reference",
            "method": "modified Z score based on error median absolute deviation",
            "modified_z_threshold": threshold,
            "outlier_row_positions": [int(v) for v in outlier_positions],
            "outlier_scores": [float(v) for v in scores[outlier_mask]],
        })
        return result
