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
from sensorqa.tools.generic._utils import (
    add_exclusion_warning, finite_pair, group_reference_indices, numeric_column, unit_ratio,
)


class SensitivityTool(BaseAnalysisTool):
    """Estimate output change per unit reference change globally and locally."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_sensitivity",
            name="Sensitivity",
            version="1.0.0",
            description=(
                "Estimates sensor sensitivity as output change per unit reference change, "
                "including global secant sensitivity and variation among adjacent operating levels."
            ),
            category=ToolCategory.ACCURACY,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference"],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples", parameter_type=ParameterType.INTEGER, default=5,
                    description="Minimum finite measurement/reference pairs.", minimum=3,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_reference_levels", parameter_type=ParameterType.INTEGER, default=3,
                    description="Minimum distinct/grouped reference levels.", minimum=2,
                    maximum=1_000_000,
                ),
                ToolParameter(
                    name="reference_group_tolerance", parameter_type=ParameterType.FLOAT, default=0.0,
                    description="Absolute tolerance used to combine nearby reference values into one level.",
                    minimum=0.0, maximum=1.0e100,
                ),
            ],
            dependencies=[], author="SensorQA",
        )

    def _levels(self, measurement: np.ndarray, reference: np.ndarray, tolerance: float):
        """Return levels.
        
        Args:
            measurement: Measurement used by this function.
            reference: Reference used by this function.
            tolerance: Tolerance used by this function.
        
        Returns:
            Calculated value.
        """
        groups = group_reference_indices(reference, tolerance)
        levels = []
        for idx in groups:
            levels.append((float(np.mean(reference[idx])), float(np.mean(measurement[idx])), len(idx)))
        levels.sort(key=lambda item: item[0])
        return levels

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
        measurement, reference, mask = finite_pair(numeric_column(data, m), numeric_column(data, r))
        minimum_samples = int(context.parameters.get("minimum_samples", 5))
        minimum_levels = int(context.parameters.get("minimum_reference_levels", 3))
        tolerance = float(context.parameters.get("reference_group_tolerance", 0.0))
        if len(measurement) < minimum_samples:
            validation.errors.append(f"Sensitivity analysis requires at least {minimum_samples} finite pairs.")
        if tolerance < 0.0 or not np.isfinite(tolerance):
            validation.errors.append("reference_group_tolerance must be finite and nonnegative.")
        elif len(self._levels(measurement, reference, tolerance)) < minimum_levels:
            validation.errors.append(
                f"Sensitivity analysis requires at least {minimum_levels} reference levels."
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
        m = context.column_mapping["measurement"]
        r = context.column_mapping["reference"]
        measurement, reference, mask = finite_pair(numeric_column(data, m), numeric_column(data, r))
        tolerance = float(context.parameters.get("reference_group_tolerance", 0.0))
        levels = self._levels(measurement, reference, tolerance)
        ref_levels = np.asarray([item[0] for item in levels], dtype=float)
        meas_levels = np.asarray([item[1] for item in levels], dtype=float)
        delta_ref = np.diff(ref_levels)
        delta_meas = np.diff(meas_levels)
        nonzero = ~np.isclose(delta_ref, 0.0, rtol=0.0, atol=1e-15)
        local = delta_meas[nonzero] / delta_ref[nonzero]
        if len(local) == 0:
            raise ValueError("No nonzero reference intervals are available for sensitivity analysis.")
        global_sensitivity = float((meas_levels[-1] - meas_levels[0]) / (ref_levels[-1] - ref_levels[0]))
        mean_local = float(np.mean(local))
        std_local = float(np.std(local, ddof=1)) if len(local) > 1 else 0.0
        cv_percent = None if np.isclose(mean_local, 0.0, atol=1e-15, rtol=0.0) else float(100.0 * std_local / abs(mean_local))
        unit = unit_ratio(context.units.get("measurement"), context.units.get("reference"))

        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        result.add_metric("Global Sensitivity", global_sensitivity, unit,
                          "Output change divided by reference change across the tested span.")
        result.add_metric("Mean Local Sensitivity", mean_local, unit,
                          "Mean sensitivity between adjacent reference levels.")
        result.add_metric("Local Sensitivity Standard Deviation", std_local, unit,
                          "Sample standard deviation of adjacent-level sensitivities.")
        result.add_metric("Minimum Local Sensitivity", float(np.min(local)), unit)
        result.add_metric("Maximum Local Sensitivity", float(np.max(local)), unit)
        result.add_metric("Local Sensitivity CV", cv_percent, "%" if cv_percent is not None else None,
                          "Coefficient of variation of local sensitivity magnitude.")
        result.add_metric("Sensitivity Reference Level Count", len(levels))
        result.add_metric("Sensitivity Interval Count", len(local))
        add_exclusion_warning(result, len(data) - len(measurement), len(data))
        if tolerance > 0.0:
            result.add_warning(
                "Nearby reference values were grouped numerically. Verify that grouped rows represent the same physical condition."
            )
        result.metadata.update({
            "sensitivity_definition": "change in measurement divided by change in reference",
            "reference_group_tolerance": tolerance,
            "reference_levels": [float(value) for value in ref_levels],
            "mean_measurements_by_level": [float(value) for value in meas_levels],
            "local_sensitivities": [float(value) for value in local],
        })
        return result
