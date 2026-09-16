#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import BaseAnalysisTool, ParameterType, SensorType, ToolCategory, ToolContext, ToolMetadata, ToolParameter
from sensorqa.tools.generic._utils import error_metrics, numeric_column


class RevisionComparisonTool(BaseAnalysisTool):
    """Compare reference-relative performance across sensor or firmware revisions."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_revision_comparison", name="Revision Comparison", version="1.0.0",
            description=(
                "Compares bias, MAE, RMSE, and maximum absolute error across labeled revisions without declaring a preferred revision."
            ),
            category=ToolCategory.DIAGNOSTIC,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference", "revision"], optional_columns=[],
            parameters=[
                ToolParameter(name="minimum_samples_per_revision", parameter_type=ParameterType.INTEGER, default=5,
                              description="Minimum finite pairs required for each revision to be reported.", minimum=2,
                              maximum=100_000_000),
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
        revision_col = context.column_mapping.get("revision")
        if revision_col is None:
            validation.errors.append("Revision mapping is unavailable.")
        elif data[revision_col].dropna().nunique() < 2:
            validation.errors.append("Revision comparison requires at least two revision labels.")
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
        mcol = context.column_mapping["measurement"]
        rcol = context.column_mapping["reference"]
        revcol = context.column_mapping["revision"]
        minimum = int(context.parameters.get("minimum_samples_per_revision", 5))
        unit = context.units.get("measurement")
        summaries = {}
        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        for revision in pd.unique(data[revcol].dropna()):
            subset = data[data[revcol] == revision]
            measurement = numeric_column(subset, mcol)
            reference = numeric_column(subset, rcol)
            mask = np.isfinite(measurement) & np.isfinite(reference)
            error = measurement[mask] - reference[mask]
            if len(error) < minimum:
                result.add_warning(
                    f"Revision '{revision}' has only {len(error)} finite pairs and was not compared; {minimum} are required."
                )
                continue
            metrics = error_metrics(error)
            key = str(revision)
            summaries[key] = {"sample_count": len(error), **metrics}
            prefix = f"Revision {key}"
            result.add_metric(f"{prefix} Bias", metrics["bias"], unit)
            result.add_metric(f"{prefix} MAE", metrics["mae"], unit)
            result.add_metric(f"{prefix} RMSE", metrics["rmse"], unit)
            result.add_metric(f"{prefix} Maximum Absolute Error", metrics["max_abs"], unit)
            result.add_metric(f"{prefix} Sample Count", len(error))
        if len(summaries) < 2:
            result.status = ExecutionStatus.SKIPPED
            result.add_message("Fewer than two revisions had enough valid samples for comparison.")
        result.metadata.update({
            "error_definition": "measurement - reference",
            "revision_summaries": summaries,
            "interpretation": (
                "Metrics are reported descriptively by revision. SensorQA does not automatically declare one revision superior."
            ),
        })
        return result
