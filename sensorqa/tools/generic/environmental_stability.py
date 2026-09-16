#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import BaseAnalysisTool, ParameterType, SensorType, ToolCategory, ToolContext, ToolMetadata, ToolParameter
from sensorqa.tools.generic._utils import linear_fit, numeric_column, unit_ratio


class EnvironmentalStabilityTool(BaseAnalysisTool):
    """Characterize association of reference-relative error with environmental variables."""

    ENVIRONMENT_FIELDS = ("temperature", "humidity", "supply_voltage")

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_environmental_stability", name="Environmental Stability", version="1.0.0",
            description=(
                "Quantifies linear associations between measurement-reference error and available temperature, humidity, or supply-voltage data."
            ),
            category=ToolCategory.ENVIRONMENTAL,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference"], optional_columns=[],
            parameters=[
                ToolParameter(name="minimum_samples", parameter_type=ParameterType.INTEGER, default=10,
                              description="Minimum finite rows per environmental variable.", minimum=3, maximum=100_000_000),
                ToolParameter(name="minimum_span", parameter_type=ParameterType.FLOAT, default=0.0,
                              description="Minimum numeric span required for an environmental variable.", minimum=0.0, maximum=1.0e100),
                ToolParameter(name="association_r2_threshold", parameter_type=ParameterType.FLOAT, default=0.5,
                              description="R² threshold used only to flag a notable statistical association.", minimum=0.0, maximum=1.0),
            ], dependencies=[], author="SensorQA",
        )

    def _available_fields(self, data: pd.DataFrame, context: ToolContext):
        """Return available fields.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Calculated value.
        """
        available = []
        for field in self.ENVIRONMENT_FIELDS:
            column = context.column_mapping.get(field)
            if column is not None and column in data.columns:
                available.append((field, column))
        return available

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
        if not self._available_fields(data, context):
            validation.errors.append(
                "Environmental stability requires at least one mapped field: temperature, humidity, or supply_voltage."
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
        measurement = numeric_column(data, context.column_mapping["measurement"])
        reference = numeric_column(data, context.column_mapping["reference"])
        error = measurement - reference
        minimum_samples = int(context.parameters.get("minimum_samples", 10))
        minimum_span = float(context.parameters.get("minimum_span", 0.0))
        r2_threshold = float(context.parameters.get("association_r2_threshold", 0.5))
        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        analyses = {}
        successful = 0
        for field, column in self._available_fields(data, context):
            environment = numeric_column(data, column)
            mask = np.isfinite(environment) & np.isfinite(error)
            x = environment[mask]
            y = error[mask]
            if len(x) < minimum_samples:
                result.add_warning(f"{field}: only {len(x)} finite rows; at least {minimum_samples} are required.")
                continue
            span = float(np.max(x) - np.min(x))
            if span <= minimum_span or np.isclose(span, 0.0, atol=1e-15, rtol=0.0):
                result.add_warning(f"{field}: environmental span ({span:g}) is insufficient for association analysis.")
                continue
            slope, intercept, r2 = linear_fit(x, y)
            predicted_change = float(slope * span)
            prefix = field.replace("_", " ").title()
            slope_unit = unit_ratio(context.units.get("measurement"), context.units.get(field))
            result.add_metric(f"{prefix} Error Coefficient", slope, slope_unit,
                              f"Slope of measurement-reference error versus {field}.")
            result.add_metric(f"{prefix} Association R Squared", r2)
            result.add_metric(f"{prefix} Span", span, context.units.get(field))
            result.add_metric(f"Predicted Error Change Across {prefix} Span", predicted_change, context.units.get("measurement"))
            analyses[field] = {
                "column": column, "sample_count": len(x), "span": span,
                "slope": slope, "intercept": intercept, "r2": r2,
                "predicted_error_change_across_span": predicted_change,
            }
            successful += 1
            if r2 is not None and r2 >= r2_threshold:
                result.add_warning(
                    f"{prefix} shows a notable statistical association with sensor error (R²={r2:.3f}). "
                    "This does not by itself establish causality."
                )
        if successful == 0:
            result.status = ExecutionStatus.SKIPPED
            result.add_message("No environmental variable had sufficient finite samples and span for analysis.")
        result.metadata.update({
            "error_definition": "measurement - reference",
            "relationship_model": "error = coefficient * environmental_variable + intercept",
            "analyses": analyses,
            "association_r2_threshold": r2_threshold,
        })
        return result
