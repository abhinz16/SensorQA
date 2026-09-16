#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
    ParameterType,
    SensorType,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolParameter,
)


class _RepeatabilityGroup:
    """Internal summary for one repeated reference condition."""

    __slots__ = (
        "reference_level",
        "reference_min",
        "reference_max",
        "sample_count",
        "measurement_mean",
        "measurement_std",
        "measurement_range",
    )

    def __init__(
        self,
        *,
        reference_level: float,
        reference_min: float,
        reference_max: float,
        sample_count: int,
        measurement_mean: float,
        measurement_std: float,
        measurement_range: float,
    ) -> None:
        """Initialize the repeatability group.
        
        Args:
            reference_level: Reference level used by this function.
            reference_min: Reference min used by this function.
            reference_max: Reference max used by this function.
            sample_count: Sample count used by this function.
            measurement_mean: Measurement mean used by this function.
            measurement_std: Measurement std used by this function.
            measurement_range: Measurement range used by this function.
        
        Returns:
            None.
        """
        self.reference_level = reference_level
        self.reference_min = reference_min
        self.reference_max = reference_max
        self.sample_count = sample_count
        self.measurement_mean = measurement_mean
        self.measurement_std = measurement_std
        self.measurement_range = measurement_range


class RepeatabilityTool(BaseAnalysisTool):
    """
    Quantify within-condition variation at repeated reference levels.

    Repeatability is calculated only from variation among measurements taken
    at nominally identical reference conditions. Variation between different
    operating points is not counted as repeatability error.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_repeatability",
            name="Repeatability",
            version="1.0.0",
            description=(
                "Quantifies within-condition measurement variation across "
                "repeated reference levels using pooled and per-level "
                "repeatability statistics."
            ),
            category=ToolCategory.PRECISION,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference"],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=6,
                    description=(
                        "Minimum number of finite measurement/reference pairs "
                        "required before repeatability analysis."
                    ),
                    minimum=2,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_repeats_per_level",
                    parameter_type=ParameterType.INTEGER,
                    default=3,
                    description=(
                        "Minimum observations required at a reference level "
                        "for that level to contribute to repeatability metrics."
                    ),
                    minimum=2,
                    maximum=1_000_000,
                ),
                ToolParameter(
                    name="minimum_repeat_levels",
                    parameter_type=ParameterType.INTEGER,
                    default=2,
                    description=(
                        "Minimum number of qualifying repeated reference "
                        "levels required for analysis."
                    ),
                    minimum=1,
                    maximum=1_000_000,
                ),
                ToolParameter(
                    name="reference_group_tolerance",
                    parameter_type=ParameterType.FLOAT,
                    default=0.0,
                    description=(
                        "Absolute reference tolerance for grouping nearby rows "
                        "as the same nominal condition. Zero requires exact "
                        "reference equality."
                    ),
                    minimum=0.0,
                    maximum=1.0e100,
                ),
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
        validation = super().validate(data=data, context=context)
        if not validation.valid:
            return validation

        measurement_column = context.column_mapping.get("measurement")
        reference_column = context.column_mapping.get("reference")

        if measurement_column is None:
            validation.errors.append("Measurement column mapping is unavailable.")
        if reference_column is None:
            validation.errors.append("Reference column mapping is unavailable.")
        if validation.errors:
            validation.valid = False
            return validation

        parameters = self._resolve_parameters(context, validation.errors)
        if parameters is None:
            validation.valid = False
            return validation

        (
            minimum_samples,
            minimum_repeats,
            minimum_levels,
            tolerance,
        ) = parameters

        measurement, reference, finite_mask = self._numeric_pairs(
            data=data,
            measurement_column=measurement_column,
            reference_column=reference_column,
        )

        valid_count = int(finite_mask.sum())
        if valid_count < minimum_samples:
            validation.errors.append(
                f"Repeatability analysis requires at least {minimum_samples} "
                "finite measurement/reference pairs; "
                f"only {valid_count} are available."
            )
            validation.valid = False
            return validation

        groups = self._summarize_repeat_groups(
            measurement=measurement[finite_mask],
            reference=reference[finite_mask],
            tolerance=tolerance,
            minimum_repeats=minimum_repeats,
        )

        if len(groups) < minimum_levels:
            validation.errors.append(
                f"Repeatability analysis requires at least {minimum_levels} "
                f"reference level(s) with at least {minimum_repeats} repeats "
                f"each; only {len(groups)} qualifying level(s) were found."
            )

        repeated_count = sum(group.sample_count for group in groups)
        if groups and repeated_count < valid_count:
            validation.warnings.append(
                f"{valid_count - repeated_count} finite row(s) belong to "
                "reference levels that do not meet the minimum repeat count "
                "and will not contribute to pooled repeatability metrics."
            )

        if tolerance > 0.0:
            validation.warnings.append(
                "A nonzero reference grouping tolerance is active. Numerical "
                "grouping does not by itself prove that the physical test "
                "condition was identical."
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
        measurement_column = context.column_mapping["measurement"]
        reference_column = context.column_mapping["reference"]

        parameters = self._resolve_parameters(context, None)
        if parameters is None:
            raise ValueError("Invalid repeatability parameters.")

        (
            minimum_samples,
            minimum_repeats,
            minimum_levels,
            tolerance,
        ) = parameters

        measurement, reference, finite_mask = self._numeric_pairs(
            data=data,
            measurement_column=measurement_column,
            reference_column=reference_column,
        )

        measurement_valid = measurement[finite_mask]
        reference_valid = reference[finite_mask]
        total_count = len(data)
        valid_count = len(measurement_valid)
        excluded_nonfinite = total_count - valid_count

        groups = self._summarize_repeat_groups(
            measurement=measurement_valid,
            reference=reference_valid,
            tolerance=tolerance,
            minimum_repeats=minimum_repeats,
        )

        if valid_count < minimum_samples:
            raise ValueError("Insufficient finite data for repeatability analysis.")
        if len(groups) < minimum_levels:
            raise ValueError("Insufficient repeated reference levels for analysis.")

        repeated_count = sum(group.sample_count for group in groups)
        excluded_unrepeated = valid_count - repeated_count
        dof = sum(group.sample_count - 1 for group in groups)
        if dof <= 0:
            raise ValueError("Repeatability variance has no within-group degrees of freedom.")

        pooled_variance = sum(
            (group.sample_count - 1) * group.measurement_std**2
            for group in groups
        ) / dof

        pooled_std = float(np.sqrt(pooled_variance))
        level_stds = np.asarray([group.measurement_std for group in groups], dtype=float)
        level_ranges = np.asarray([group.measurement_range for group in groups], dtype=float)

        measurement_unit = context.units.get("measurement")
        reference_unit = context.units.get("reference")

        result = AnalysisResult(
            tool_id=self.metadata.tool_id,
            tool_name=self.metadata.name,
            tool_version=self.metadata.version,
            status=ExecutionStatus.SUCCESS,
        )

        result.add_metric(
            "Repeatability Level Count",
            len(groups),
            description=(
                "Number of reference levels containing enough repeated "
                "measurements to contribute to repeatability analysis."
            ),
        )
        result.add_metric(
            "Repeatability Sample Count",
            repeated_count,
            description="Number of observations used in repeatability metrics.",
        )
        result.add_metric(
            "Average Repeats Per Level",
            float(repeated_count / len(groups)),
            description="Average observations per qualifying reference level.",
        )
        result.add_metric(
            "Pooled Repeatability Standard Deviation",
            pooled_std,
            unit=measurement_unit,
            description=(
                "Pooled sample standard deviation of measurement variation "
                "within repeated reference levels."
            ),
        )
        result.add_metric(
            "Mean Within-Level Standard Deviation",
            float(np.mean(level_stds)),
            unit=measurement_unit,
            description="Mean sample standard deviation across repeated levels.",
        )
        result.add_metric(
            "Maximum Within-Level Standard Deviation",
            float(np.max(level_stds)),
            unit=measurement_unit,
            description="Largest sample standard deviation at any repeated level.",
        )
        result.add_metric(
            "Mean Within-Level Range",
            float(np.mean(level_ranges)),
            unit=measurement_unit,
            description="Mean peak-to-peak range across repeated levels.",
        )
        result.add_metric(
            "Maximum Within-Level Range",
            float(np.max(level_ranges)),
            unit=measurement_unit,
            description="Largest peak-to-peak range at any repeated level.",
        )

        if excluded_nonfinite > 0:
            result.add_warning(
                f"{excluded_nonfinite} of {total_count} rows were excluded "
                "because measurement or reference values were non-numeric "
                "or non-finite."
            )
        if excluded_unrepeated > 0:
            result.add_warning(
                f"{excluded_unrepeated} finite row(s) belonged to levels with "
                f"fewer than {minimum_repeats} repeats and were excluded from "
                "pooled repeatability metrics."
            )
        if tolerance > 0.0:
            result.add_warning(
                "Reference levels were formed using a nonzero numerical "
                "tolerance. Verify that grouped rows represent the same "
                "physical test condition."
            )

        result.metadata.update(
            {
                "repeatability_definition": "within-reference-level measurement variation",
                "pooled_variance_definition": "sum((n_i - 1) * s_i^2) / sum(n_i - 1)",
                "standard_deviation_definition": "sample standard deviation with ddof=1",
                "total_rows": total_count,
                "valid_rows": valid_count,
                "excluded_nonfinite_rows": excluded_nonfinite,
                "repeatability_rows": repeated_count,
                "excluded_insufficient_repeat_rows": excluded_unrepeated,
                "minimum_samples": minimum_samples,
                "minimum_repeats_per_level": minimum_repeats,
                "minimum_repeat_levels": minimum_levels,
                "reference_group_tolerance": tolerance,
                "measurement_column": measurement_column,
                "reference_column": reference_column,
                "measurement_unit": measurement_unit,
                "reference_unit": reference_unit,
                "repeatability_groups": [
                    {
                        "reference_level": group.reference_level,
                        "reference_min": group.reference_min,
                        "reference_max": group.reference_max,
                        "sample_count": group.sample_count,
                        "measurement_mean": group.measurement_mean,
                        "measurement_std": group.measurement_std,
                        "measurement_range": group.measurement_range,
                    }
                    for group in groups
                ],
            }
        )
        return result

    @staticmethod
    def _numeric_pairs(
        *,
        data: pd.DataFrame,
        measurement_column: str,
        reference_column: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate numeric pairs.
        
        Args:
            data: Input data to process.
            measurement_column: Column containing sensor measurements.
            reference_column: Column containing reference values.
        
        Returns:
            Tuple containing the calculated values.
        """
        measurement = pd.to_numeric(
            data[measurement_column], errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        reference = pd.to_numeric(
            data[reference_column], errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        finite_mask = np.isfinite(measurement) & np.isfinite(reference)
        return measurement, reference, finite_mask

    @staticmethod
    def _resolve_parameters(
        context: ToolContext,
        errors: list[str] | None,
    ) -> tuple[int, int, int, float] | None:
        """Resolve parameters.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
            errors: Errors used by this function.
        
        Returns:
            Tuple containing the calculated values.
        """
        try:
            minimum_samples = int(context.parameters.get("minimum_samples", 6))
            minimum_repeats = int(
                context.parameters.get("minimum_repeats_per_level", 3)
            )
            minimum_levels = int(
                context.parameters.get("minimum_repeat_levels", 2)
            )
            tolerance = float(
                context.parameters.get("reference_group_tolerance", 0.0)
            )
        except (TypeError, ValueError):
            if errors is not None:
                errors.append("Repeatability parameters must be numerical.")
            return None

        problems: list[str] = []
        if minimum_samples < 2:
            problems.append("minimum_samples must be at least 2.")
        if minimum_repeats < 2:
            problems.append("minimum_repeats_per_level must be at least 2.")
        if minimum_levels < 1:
            problems.append("minimum_repeat_levels must be at least 1.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            problems.append(
                "reference_group_tolerance must be a finite nonnegative number."
            )

        if problems:
            if errors is not None:
                errors.extend(problems)
            return None
        return minimum_samples, minimum_repeats, minimum_levels, tolerance

    @classmethod
    def _summarize_repeat_groups(
        cls,
        *,
        measurement: np.ndarray,
        reference: np.ndarray,
        tolerance: float,
        minimum_repeats: int,
    ) -> list[_RepeatabilityGroup]:
        """Return summarize repeat groups.
        
        Args:
            measurement: Measurement used by this function.
            reference: Reference used by this function.
            tolerance: Tolerance used by this function.
            minimum_repeats: Minimum repeats accepted.
        
        Returns:
            List of result values.
        """
        summaries: list[_RepeatabilityGroup] = []

        for indices in cls._group_reference_indices(reference, tolerance):
            if len(indices) < minimum_repeats:
                continue

            group_reference = reference[indices]
            group_measurement = measurement[indices]

            summaries.append(
                _RepeatabilityGroup(
                    reference_level=float(np.mean(group_reference)),
                    reference_min=float(np.min(group_reference)),
                    reference_max=float(np.max(group_reference)),
                    sample_count=len(indices),
                    measurement_mean=float(np.mean(group_measurement)),
                    measurement_std=float(np.std(group_measurement, ddof=1)),
                    measurement_range=float(
                        np.max(group_measurement) - np.min(group_measurement)
                    ),
                )
            )

        return summaries

    @staticmethod
    def _group_reference_indices(
        reference: np.ndarray,
        tolerance: float,
    ) -> list[np.ndarray]:
        """Group row positions by repeated reference condition.
        
        For positive tolerance, sorted values join the current group while
        they remain within tolerance of that group's running mean.
        
        Args:
            reference: Value for `reference`.
            tolerance: Value for `tolerance`.
        
        Returns:
            List of result values.
        """
        if reference.size == 0:
            return []

        if tolerance == 0.0:
            return [
                np.flatnonzero(reference == value)
                for value in np.unique(reference)
            ]

        sorted_positions = np.argsort(reference, kind="stable")
        groups: list[list[int]] = []
        current: list[int] = []

        for raw_position in sorted_positions:
            position = int(raw_position)
            if not current:
                current = [position]
                continue

            center = float(np.mean(reference[np.asarray(current, dtype=int)]))
            if abs(float(reference[position]) - center) <= tolerance:
                current.append(position)
            else:
                groups.append(current)
                current = [position]

        if current:
            groups.append(current)

        return [np.asarray(group, dtype=int) for group in groups]
