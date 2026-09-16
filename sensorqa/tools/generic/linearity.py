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
    ParameterType,
    SensorType,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolParameter,
)


class LinearityTool(BaseAnalysisTool):
    """
    Generic reference-based linearity characterization.

    The tool fits a best-fit straight line (ordinary least squares)
    describing measurement as a function of reference:

        measurement = slope * reference + intercept

    Linearity error is then defined as the deviation of each measured
    value from that best-fit line.

    This tool characterizes linearity only. A highly linear sensor may
    still have incorrect sensitivity, offset, or absolute accuracy.
    Those are separate engineering properties and should be interpreted
    separately.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Describe this tool to SensorQA.
        
        Returns:
            ToolMetadata returned by the function.
        """

        return ToolMetadata(
            tool_id="generic_linearity",
            name="Linearity",
            version="1.0.0",
            description=(
                "Characterizes the linearity of a generic sensor against "
                "reference values using a best-fit straight line and "
                "reports slope, intercept, R-squared, residual error, "
                "and maximum deviation from the fitted line."
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
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=5,
                    description=(
                        "Minimum number of finite measurement/reference "
                        "pairs required for linearity analysis."
                    ),
                    minimum=3,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_unique_reference_values",
                    parameter_type=ParameterType.INTEGER,
                    default=3,
                    description=(
                        "Minimum number of distinct finite reference values "
                        "required to establish a meaningful reference span."
                    ),
                    minimum=2,
                    maximum=100_000_000,
                ),
            ],
            dependencies=[],
            author="SensorQA",
        )

    def validate(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ):
        """Validate that the dataset can support linearity analysis.
        
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
        ).to_numpy(
            dtype=float,
            copy=False,
        )

        reference = pd.to_numeric(
            data[reference_column],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            copy=False,
        )

        finite_mask = (
            np.isfinite(measurement)
            & np.isfinite(reference)
        )

        valid_count = int(
            finite_mask.sum()
        )

        minimum_samples = int(
            context.parameters.get(
                "minimum_samples",
                5,
            )
        )

        if valid_count < minimum_samples:
            validation.errors.append(
                "Linearity analysis requires at least "
                f"{minimum_samples} finite measurement/reference pairs; "
                f"only {valid_count} are available."
            )
            validation.valid = False
            return validation

        reference_valid = reference[
            finite_mask
        ]

        unique_reference_count = int(
            np.unique(
                reference_valid
            ).size
        )

        minimum_unique = int(
            context.parameters.get(
                "minimum_unique_reference_values",
                3,
            )
        )

        if unique_reference_count < minimum_unique:
            validation.errors.append(
                "Linearity analysis requires at least "
                f"{minimum_unique} distinct reference values; only "
                f"{unique_reference_count} are available."
            )

        reference_span = float(
            np.max(reference_valid)
            - np.min(reference_valid)
        )

        if not np.isfinite(reference_span) or np.isclose(
            reference_span,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ):
            validation.errors.append(
                "Linearity analysis requires a nonzero reference span."
            )

        measurement_unit = context.units.get(
            "measurement"
        )
        reference_unit = context.units.get(
            "reference"
        )

        if (
            measurement_unit is not None
            and reference_unit is not None
            and measurement_unit != reference_unit
        ):
            validation.warnings.append(
                "Measurement and reference use different units. The fitted "
                "slope will therefore carry measurement/reference units."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform best-fit-straight-line linearity analysis.
        
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
        ).to_numpy(
            dtype=float,
            copy=True,
        )

        reference = pd.to_numeric(
            data[reference_column],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            copy=True,
        )

        finite_mask = (
            np.isfinite(measurement)
            & np.isfinite(reference)
        )

        y = measurement[
            finite_mask
        ]
        x = reference[
            finite_mask
        ]

        total_count = len(data)
        valid_count = len(x)
        excluded_count = (
            total_count - valid_count
        )

        unique_reference_count = int(
            np.unique(x).size
        )

        reference_min = float(
            np.min(x)
        )
        reference_max = float(
            np.max(x)
        )
        reference_span = float(
            reference_max
            - reference_min
        )

        # Best-fit straight line:
        #     measurement = slope * reference + intercept

        design = np.column_stack(
            (
                x,
                np.ones_like(x),
            )
        )

        coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )

        if rank < 2:
            raise ValueError(
                "Unable to determine an independent slope and intercept "
                "for the supplied reference values."
            )

        slope = float(
            coefficients[0]
        )
        intercept = float(
            coefficients[1]
        )

        fitted = (
            slope * x
            + intercept
        )

        residual = (
            y
            - fitted
        )

        absolute_residual = np.abs(
            residual
        )

        residual_rmse = float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        )

        residual_std = float(
            np.std(
                residual,
                ddof=1,
            )
        )

        max_absolute_deviation = float(
            np.max(
                absolute_residual
            )
        )

        median_absolute_deviation = float(
            np.median(
                absolute_residual
            )
        )

        fitted_output_span = float(
            abs(slope)
            * reference_span
        )

        max_deviation_percent_full_scale = None

        if not np.isclose(
            fitted_output_span,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ):
            max_deviation_percent_full_scale = float(
                100.0
                * max_absolute_deviation
                / fitted_output_span
            )

        # Coefficient of determination.
        #
        # R² is undefined when the measured output itself has
        # effectively zero variance.

        centered_measurement = (
            y
            - np.mean(y)
        )

        total_sum_squares = float(
            np.sum(
                centered_measurement ** 2
            )
        )

        residual_sum_squares = float(
            np.sum(
                residual ** 2
            )
        )

        r_squared = None

        if not np.isclose(
            total_sum_squares,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ):
            r_squared = float(
                1.0
                - residual_sum_squares
                / total_sum_squares
            )

        measurement_unit = context.units.get(
            "measurement"
        )
        reference_unit = context.units.get(
            "reference"
        )

        slope_unit = self._slope_unit(
            measurement_unit=measurement_unit,
            reference_unit=reference_unit,
        )

        result = AnalysisResult(
            tool_id=self.metadata.tool_id,
            tool_name=self.metadata.name,
            tool_version=self.metadata.version,
            status=ExecutionStatus.SUCCESS,
        )

        result.add_metric(
            name="Valid Sample Count",
            value=valid_count,
            description=(
                "Number of finite measurement/reference pairs included "
                "in the best-fit linearity calculation."
            ),
        )

        result.add_metric(
            name="Unique Reference Value Count",
            value=unique_reference_count,
            description=(
                "Number of distinct finite reference values represented "
                "in the linearity fit."
            ),
        )

        result.add_metric(
            name="Reference Span",
            value=reference_span,
            unit=reference_unit,
            description=(
                "Maximum reference value minus minimum reference value."
            ),
        )

        result.add_metric(
            name="Linearity Slope",
            value=slope,
            unit=slope_unit,
            description=(
                "Slope of the best-fit relationship measurement = "
                "slope × reference + intercept."
            ),
        )

        result.add_metric(
            name="Linearity Intercept",
            value=intercept,
            unit=measurement_unit,
            description=(
                "Intercept of the best-fit straight line relating "
                "measurement to reference."
            ),
        )

        result.add_metric(
            name="Linearity R Squared",
            value=r_squared,
            description=(
                "Coefficient of determination for the best-fit straight "
                "line. R² is undefined when measured output has zero "
                "variance."
            ),
        )

        result.add_metric(
            name="Linearity Residual RMSE",
            value=residual_rmse,
            unit=measurement_unit,
            description=(
                "Root mean squared deviation of measured values from the "
                "best-fit straight line."
            ),
        )

        result.add_metric(
            name="Linearity Residual Standard Deviation",
            value=residual_std,
            unit=measurement_unit,
            description=(
                "Sample standard deviation of deviations from the best-fit "
                "straight line."
            ),
        )

        result.add_metric(
            name="Median Absolute Linearity Deviation",
            value=median_absolute_deviation,
            unit=measurement_unit,
            description=(
                "Median absolute deviation of measured values from the "
                "best-fit straight line."
            ),
        )

        result.add_metric(
            name="Maximum Absolute Linearity Deviation",
            value=max_absolute_deviation,
            unit=measurement_unit,
            description=(
                "Largest absolute deviation from the best-fit straight "
                "line over the tested reference range."
            ),
        )

        result.add_metric(
            name="Fitted Output Span",
            value=fitted_output_span,
            unit=measurement_unit,
            description=(
                "Magnitude of the fitted output change across the tested "
                "reference span."
            ),
        )

        result.add_metric(
            name="Maximum Linearity Deviation Percent Full Scale",
            value=max_deviation_percent_full_scale,
            unit="%" if max_deviation_percent_full_scale is not None else None,
            description=(
                "Maximum absolute deviation from the best-fit straight "
                "line expressed as a percentage of the fitted output span."
            ),
        )

        if excluded_count > 0:
            result.add_warning(
                f"{excluded_count} of {total_count} rows were excluded "
                "because measurement or reference values were non-numeric "
                "or non-finite."
            )

        if r_squared is None:
            result.add_warning(
                "Linearity R² is undefined because the measured output "
                "has effectively zero variance."
            )

        if max_deviation_percent_full_scale is None:
            result.add_warning(
                "Percent-full-scale linearity deviation is undefined "
                "because the fitted output span is effectively zero. "
                "Linearity alone should not be interpreted as evidence "
                "of adequate sensor sensitivity."
            )

        result.metadata.update(
            {
                "linearity_method": "best_fit_straight_line",
                "fit_equation": (
                    "measurement = slope * reference + intercept"
                ),
                "residual_definition": (
                    "measurement - fitted_measurement"
                ),
                "full_scale_definition": (
                    "absolute fitted output change across tested "
                    "reference span"
                ),
                "total_rows": total_count,
                "valid_rows": valid_count,
                "excluded_rows": excluded_count,
                "unique_reference_values": unique_reference_count,
                "reference_min": reference_min,
                "reference_max": reference_max,
                "measurement_column": measurement_column,
                "reference_column": reference_column,
                "measurement_unit": measurement_unit,
                "reference_unit": reference_unit,
            }
        )

        return result

    @staticmethod
    def _slope_unit(
        *,
        measurement_unit: str | None,
        reference_unit: str | None,
    ) -> str | None:
        """Return a readable unit for the fitted slope when possible.
        
        Args:
            measurement_unit: Value for `measurement_unit`.
            reference_unit: Value for `reference_unit`.
        
        Returns:
            str | None returned by the function.
        """

        if measurement_unit is None or reference_unit is None:
            return None

        if measurement_unit == reference_unit:
            return None

        return (
            f"{measurement_unit}/{reference_unit}"
        )
