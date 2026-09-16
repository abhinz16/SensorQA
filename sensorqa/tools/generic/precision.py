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
from sensorqa.tools.generic._utils import add_exclusion_warning, finite_pair, numeric_column


class PrecisionTool(BaseAnalysisTool):
    """Characterize scatter of reference-relative error independent of mean bias."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_precision",
            name="Precision",
            version="1.0.0",
            description=(
                "Characterizes random scatter of measurement error after removing "
                "the mean error, using standard deviation and robust dispersion metrics."
            ),
            category=ToolCategory.PRECISION,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference"],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples", parameter_type=ParameterType.INTEGER,
                    default=5, description="Minimum finite measurement/reference pairs.",
                    minimum=2, maximum=100_000_000,
                )
            ],
            dependencies=[],
            author="SensorQA",
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
        measurement = numeric_column(data, m)
        reference = numeric_column(data, r)
        _, _, mask = finite_pair(measurement, reference)
        minimum = int(context.parameters.get("minimum_samples", 5))
        if int(mask.sum()) < minimum:
            validation.errors.append(
                f"Precision analysis requires at least {minimum} finite pairs; "
                f"only {int(mask.sum())} are available."
            )
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
        error = measurement - reference
        centered = error - np.mean(error)
        std = float(np.std(error, ddof=1))
        robust_sigma = float(1.4826 * np.median(np.abs(error - np.median(error))))
        q75, q25 = np.percentile(error, [75.0, 25.0])
        iqr = float(q75 - q25)
        p95_abs_centered = float(np.percentile(np.abs(centered), 95.0))
        measurement_unit = context.units.get("measurement")

        result = AnalysisResult(
            tool_id=self.metadata.tool_id, tool_name=self.metadata.name,
            tool_version=self.metadata.version, status=ExecutionStatus.SUCCESS,
        )
        result.add_metric("Precision Error Standard Deviation", std, measurement_unit,
                          "Sample standard deviation of measurement-reference error.")
        result.add_metric("Robust Precision Sigma", robust_sigma, measurement_unit,
                          "MAD-based robust dispersion estimate, 1.4826 × median absolute deviation.")
        result.add_metric("Precision Error IQR", iqr, measurement_unit,
                          "Interquartile range of measurement-reference error.")
        result.add_metric("95th Percentile Absolute Demeaned Error", p95_abs_centered, measurement_unit,
                          "95th percentile of absolute error after removing mean bias.")
        result.add_metric("Precision Sample Count", len(error), description="Finite paired samples used.")
        add_exclusion_warning(result, len(data) - len(error), len(data))
        result.metadata.update({
            "error_definition": "measurement - reference",
            "precision_definition": "dispersion of reference-relative error independent of mean bias",
            "measurement_column": m,
            "reference_column": r,
            "valid_rows": len(error),
            "excluded_rows": len(data) - len(error),
        })
        return result
