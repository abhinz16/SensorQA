#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import (
    AnalysisResult,
    ExecutionStatus,
)
from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
    SensorType,
    ToolCategory,
    ToolContext,
    ToolMetadata,
)


class AccuracyTool(BaseAnalysisTool):
    """
    Generic reference-based sensor accuracy analysis.

    Compares a measured sensor value against a known reference
    value and calculates common engineering error metrics.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Describe this tool to SensorQA.
        
        Returns:
            ToolMetadata returned by the function.
        """

        return ToolMetadata(
            tool_id="generic_accuracy",
            name="Accuracy",
            version="1.0.0",
            description=(
                "Compares sensor measurements against reference "
                "values and calculates bias, MAE, RMSE, maximum "
                "absolute error, and percentage-error statistics."
            ),
            category=ToolCategory.ACCURACY,
            compatible_sensor_types=[
                SensorType.GENERIC,
            ],
            required_columns=[
                "measurement",
                "reference",
            ],
            optional_columns=[],
            parameters=[],
            dependencies=[],
            author="SensorQA",
        )

    def validate(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ):
        """Extend the common SensorQA validation with
        accuracy-specific checks.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Result returned by the function.
        """

        validation = super().validate(
            data=data,
            context=context,
        )

        if not validation.valid:
            return validation

        measurement_column = context.column_mapping.get(
            "measurement"
        )

        reference_column = context.column_mapping.get(
            "reference"
        )

        if measurement_column is None:
            validation.errors.append(
                "Measurement column mapping is unavailable."
            )

        if reference_column is None:
            validation.errors.append(
                "Reference column mapping is unavailable."
            )

        if validation.errors:
            validation.valid = False
            return validation

        measurement = pd.to_numeric(
            data[measurement_column],
            errors="coerce",
        )

        reference = pd.to_numeric(
            data[reference_column],
            errors="coerce",
        )

        valid_pair_count = int(
            (
                measurement.notna()
                & reference.notna()
            ).sum()
        )

        if valid_pair_count == 0:

            validation.errors.append(
                "No valid numerical measurement/reference pairs "
                "are available for accuracy analysis."
            )

            validation.valid = False

        elif valid_pair_count < 3:

            validation.warnings.append(
                "Fewer than three valid measurement/reference "
                "pairs are available. Accuracy statistics may "
                "not be representative."
            )

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Calculate accuracy metrics.
        
        Error convention:
        
            error = measurement - reference
        
        Therefore:
        
            positive bias -> sensor reads high
            negative bias -> sensor reads low
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        measurement_column = context.column_mapping[
            "measurement"
        ]

        reference_column = context.column_mapping[
            "reference"
        ]

        measurement = pd.to_numeric(
            data[measurement_column],
            errors="coerce",
        )

        reference = pd.to_numeric(
            data[reference_column],
            errors="coerce",
        )

        # Keep only rows where both quantities are numerical

        valid_mask = (
            measurement.notna()
            & reference.notna()
        )

        measurement_valid = measurement[
            valid_mask
        ].to_numpy(
            dtype=float
        )

        reference_valid = reference[
            valid_mask
        ].to_numpy(
            dtype=float
        )

        valid_count = len(
            measurement_valid
        )

        total_count = len(
            data
        )

        excluded_count = (
            total_count - valid_count
        )

        # Core error calculation

        error = (
            measurement_valid
            - reference_valid
        )

        absolute_error = np.abs(
            error
        )

        squared_error = (
            error ** 2
        )

        # Basic accuracy metrics

        bias = float(
            np.mean(
                error
            )
        )

        mae = float(
            np.mean(
                absolute_error
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    squared_error
                )
            )
        )

        max_absolute_error = float(
            np.max(
                absolute_error
            )
        )

        median_absolute_error = float(
            np.median(
                absolute_error
            )
        )

        # Percentage error
        #
        # Percentage error is undefined when reference = 0.
        # Those rows are excluded only from percentage calculations.
        # They remain part of absolute-error calculations.

        nonzero_reference_mask = (
            reference_valid != 0.0
        )

        percentage_valid_count = int(
            np.sum(
                nonzero_reference_mask
            )
        )

        percentage_excluded_count = (
            valid_count
            - percentage_valid_count
        )

        mean_absolute_percentage_error = None
        mean_percentage_error = None
        max_absolute_percentage_error = None

        if percentage_valid_count > 0:

            percentage_error = (
                error[
                    nonzero_reference_mask
                ]
                / np.abs(
                    reference_valid[
                        nonzero_reference_mask
                    ]
                )
                * 100.0
            )

            absolute_percentage_error = np.abs(
                percentage_error
            )

            mean_percentage_error = float(
                np.mean(
                    percentage_error
                )
            )

            mean_absolute_percentage_error = float(
                np.mean(
                    absolute_percentage_error
                )
            )

            max_absolute_percentage_error = float(
                np.max(
                    absolute_percentage_error
                )
            )

        # Determine engineering unit

        measurement_unit = context.units.get(
            "measurement"
        )

        reference_unit = context.units.get(
            "reference"
        )

        result_unit = (
            measurement_unit
            or reference_unit
        )

        # Build standardized SensorQA result

        result = AnalysisResult(
            tool_id=self.metadata.tool_id,
            tool_name=self.metadata.name,
            tool_version=self.metadata.version,
            status=ExecutionStatus.SUCCESS,
        )

        result.add_metric(
            name="Valid Sample Count",
            value=valid_count,
            unit=None,
            description=(
                "Number of rows containing both a valid "
                "measurement and reference value."
            ),
        )

        result.add_metric(
            name="Bias",
            value=bias,
            unit=result_unit,
            description=(
                "Mean signed error. Positive values indicate "
                "the sensor reads higher than the reference."
            ),
        )

        result.add_metric(
            name="Mean Absolute Error",
            value=mae,
            unit=result_unit,
            description=(
                "Mean magnitude of measurement error."
            ),
        )

        result.add_metric(
            name="Root Mean Squared Error",
            value=rmse,
            unit=result_unit,
            description=(
                "Root mean squared difference between "
                "measurement and reference."
            ),
        )

        result.add_metric(
            name="Median Absolute Error",
            value=median_absolute_error,
            unit=result_unit,
            description=(
                "Median magnitude of measurement error."
            ),
        )

        result.add_metric(
            name="Maximum Absolute Error",
            value=max_absolute_error,
            unit=result_unit,
            description=(
                "Largest observed absolute measurement error."
            ),
        )

        # Add percentage metrics when they are mathematically valid

        if mean_absolute_percentage_error is not None:

            result.add_metric(
                name="Mean Percentage Error",
                value=mean_percentage_error,
                unit="%",
                description=(
                    "Mean signed error relative to the absolute "
                    "reference value."
                ),
            )

            result.add_metric(
                name="Mean Absolute Percentage Error",
                value=mean_absolute_percentage_error,
                unit="%",
                description=(
                    "Mean absolute measurement error as a "
                    "percentage of the reference magnitude."
                ),
            )

            result.add_metric(
                name="Maximum Absolute Percentage Error",
                value=max_absolute_percentage_error,
                unit="%",
                description=(
                    "Largest absolute percentage error."
                ),
            )

        # Warnings

        if excluded_count > 0:

            result.add_warning(
                f"{excluded_count} of {total_count} rows were "
                "excluded because measurement or reference values "
                "were missing or non-numeric."
            )

        if percentage_excluded_count > 0:

            result.add_warning(
                f"{percentage_excluded_count} valid rows had a "
                "reference value of zero and were excluded from "
                "percentage-error calculations."
            )

        if percentage_valid_count == 0:

            result.add_warning(
                "Percentage-error metrics could not be calculated "
                "because all valid reference values were zero."
            )

        # Analysis metadata

        result.metadata.update(
            {
                "error_definition": (
                    "measurement - reference"
                ),
                "total_rows": total_count,
                "valid_rows": valid_count,
                "excluded_rows": excluded_count,
                "percentage_error_valid_rows":
                    percentage_valid_count,
                "percentage_error_excluded_rows":
                    percentage_excluded_count,
                "measurement_column":
                    measurement_column,
                "reference_column":
                    reference_column,
                "measurement_unit":
                    measurement_unit,
                "reference_unit":
                    reference_unit,
            }
        )

        return result
