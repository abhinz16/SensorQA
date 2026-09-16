#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import (
    AnalysisResult,
    EvidenceStrength,
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


class AxisCorrelationTool(BaseAnalysisTool):
    """
    Characterizes statistical relationships between IMU axes.

    For complete accelerometer and/or gyroscope XYZ groups, the tool
    calculates:

        - covariance matrix
        - Pearson correlation matrix
        - strongest absolute off-diagonal correlation
        - first-difference correlation matrix
        - strongest first-difference correlation
        - dominant covariance eigenvalue fraction

    Important:
        Cross-axis correlation does NOT by itself establish:

            - mechanical axis misalignment
            - cross-axis sensitivity
            - sensor nonorthogonality
            - electrical crosstalk

        Correlation may also result from:

            - genuine multi-axis motion
            - vibration
            - gravity projection
            - common thermal drift
            - common-mode disturbances
            - filtering or preprocessing
    """

    AXIS_PAIRS = (
        (0, 1),
        (0, 2),
        (1, 2),
    )

    @property
    def metadata(self) -> ToolMetadata:

        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="imu_axis_correlation",
            name="IMU Axis Correlation",
            version="1.0.0",
            description=(
                "Computes accelerometer and gyroscope covariance "
                "and cross-axis correlation matrices, including "
                "first-difference correlation for short-timescale "
                "comparison."
            ),
            category=ToolCategory.IMU,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.ACCELEROMETER,
                SensorType.GYROSCOPE,
            ],
            required_columns=[],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=500,
                    description=(
                        "Minimum number of simultaneous valid XYZ "
                        "samples required for each sensor group."
                    ),
                    minimum=10,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="correlation_warning_threshold",
                    parameter_type=ParameterType.FLOAT,
                    default=0.70,
                    description=(
                        "Absolute Pearson correlation above this "
                        "value is reported as substantial statistical "
                        "cross-axis correlation."
                    ),
                    minimum=0.0,
                    maximum=1.0,
                ),
                ToolParameter(
                    name="difference_correlation_warning_threshold",
                    parameter_type=ParameterType.FLOAT,
                    default=0.70,
                    description=(
                        "Absolute first-difference correlation above "
                        "this value is reported as substantial "
                        "short-timescale cross-axis correlation."
                    ),
                    minimum=0.0,
                    maximum=1.0,
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
        """Check whether complete XYZ groups and sufficient simultaneous
        valid measurements are available.
        
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

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        if not available_groups:

            validation.errors.append(
                "No complete accelerometer or gyroscope XYZ group "
                "is available for axis-correlation analysis."
            )

            validation.valid = False
            return validation

        minimum_samples = int(
            context.parameters.get(
                "minimum_samples",
                500,
            )
        )

        usable_group_count = 0

        for group_name, axes in available_groups.items():

            matrix = self._valid_xyz_matrix(
                data=data,
                context=context,
                axes=axes,
            )

            valid_count = len(
                matrix
            )

            if valid_count < minimum_samples:

                validation.warnings.append(
                    f"{group_name.capitalize()} has only "
                    f"{valid_count} simultaneous valid XYZ samples; "
                    f"{minimum_samples} are required."
                )

                continue

            # Correlation is undefined for a constant axis.
            standard_deviations = np.std(
                matrix,
                axis=0,
                ddof=1,
            )

            constant_axes = [
                axes[index]
                for index, value
                in enumerate(
                    standard_deviations
                )
                if value == 0.0
            ]

            if constant_axes:

                validation.warnings.append(
                    f"{group_name.capitalize()} contains constant "
                    "axis/axes that cannot support correlation "
                    "analysis: "
                    + ", ".join(
                        axis.upper()
                        for axis in constant_axes
                    )
                )

                continue

            usable_group_count += 1

        if usable_group_count == 0:

            validation.errors.append(
                "No sensor group contains enough varying XYZ data "
                "for axis-correlation analysis."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform cross-axis covariance and correlation analysis.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        result = AnalysisResult(
            tool_id=self.metadata.tool_id,
            tool_name=self.metadata.name,
            tool_version=self.metadata.version,
            status=ExecutionStatus.SUCCESS,
        )

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        minimum_samples = int(
            context.parameters[
                "minimum_samples"
            ]
        )

        correlation_threshold = float(
            context.parameters[
                "correlation_warning_threshold"
            ]
        )

        difference_threshold = float(
            context.parameters[
                "difference_correlation_warning_threshold"
            ]
        )

        processed_group_count = 0

        correlation_results = {}

        # Process accelerometer and/or gyroscope groups

        for group_name, axes in available_groups.items():

            matrix = self._valid_xyz_matrix(
                data=data,
                context=context,
                axes=axes,
            )

            sample_count = len(
                matrix
            )

            if sample_count < minimum_samples:

                result.add_warning(
                    f"{group_name.capitalize()} was skipped because "
                    f"only {sample_count} simultaneous valid XYZ "
                    "samples are available."
                )

                continue

            standard_deviations = np.std(
                matrix,
                axis=0,
                ddof=1,
            )

            if np.any(
                standard_deviations == 0.0
            ):

                result.add_warning(
                    f"{group_name.capitalize()} was skipped because "
                    "one or more axes are constant and therefore "
                    "cannot support a Pearson correlation matrix."
                )

                continue

            group_display_name = (
                "Accelerometer"
                if group_name == "accelerometer"
                else "Gyroscope"
            )

            # Remove means before covariance calculation.
            # np.corrcoef would remove them implicitly, but keeping
            # the centered matrix explicit makes the calculation
            # and later PCA-style covariance interpretation clear.

            centered = (
                matrix
                - np.mean(
                    matrix,
                    axis=0,
                )
            )

            covariance_matrix = np.cov(
                centered,
                rowvar=False,
                ddof=1,
            )

            correlation_matrix = np.corrcoef(
                centered,
                rowvar=False,
            )

            # First differences
            #
            # This provides a complementary short-timescale view.
            # A strong raw correlation that disappears after
            # differencing may be associated more strongly with a
            # common slow trend than with sample-to-sample variation.

            differences = np.diff(
                matrix,
                axis=0,
            )

            difference_correlation_matrix = None

            if len(
                differences
            ) >= 2:

                difference_std = np.std(
                    differences,
                    axis=0,
                    ddof=1,
                )

                if np.all(
                    difference_std > 0.0
                ):

                    difference_correlation_matrix = (
                        np.corrcoef(
                            differences,
                            rowvar=False,
                        )
                    )

            # Strongest raw correlation

            (
                strongest_pair,
                strongest_correlation,
            ) = self._strongest_off_diagonal(
                matrix=correlation_matrix,
                axes=axes,
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Maximum Absolute "
                    "Axis Correlation"
                ),
                value=abs(
                    strongest_correlation
                ),
                unit=None,
                description=(
                    "Largest absolute Pearson correlation "
                    "between any two different sensor axes."
                ),
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Strongest "
                    "Axis Correlation"
                ),
                value=strongest_correlation,
                unit=None,
                description=(
                    f"Signed Pearson correlation for "
                    f"{strongest_pair[0].upper()} versus "
                    f"{strongest_pair[1].upper()}."
                ),
            )

            # Pairwise covariance and correlation

            for i, j in self.AXIS_PAIRS:

                axis_a = axes[
                    i
                ].upper()

                axis_b = axes[
                    j
                ].upper()

                result.add_metric(
                    name=(
                        f"{axis_a}-{axis_b} Covariance"
                    ),
                    value=float(
                        covariance_matrix[
                            i,
                            j,
                        ]
                    ),
                    unit=self._covariance_unit(
                        context.units.get(
                            axes[
                                i
                            ]
                        )
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_a}-{axis_b} Correlation"
                    ),
                    value=float(
                        correlation_matrix[
                            i,
                            j,
                        ]
                    ),
                    unit=None,
                )

            # First-difference correlation

            strongest_difference_pair = None
            strongest_difference_correlation = None

            if (
                difference_correlation_matrix
                is not None
            ):

                (
                    strongest_difference_pair,
                    strongest_difference_correlation,
                ) = self._strongest_off_diagonal(
                    matrix=difference_correlation_matrix,
                    axes=axes,
                )

                result.add_metric(
                    name=(
                        f"{group_display_name} Maximum Absolute "
                        "First-Difference Axis Correlation"
                    ),
                    value=abs(
                        strongest_difference_correlation
                    ),
                    unit=None,
                    description=(
                        "Largest absolute correlation between "
                        "successive changes in different axes."
                    ),
                )

                for i, j in self.AXIS_PAIRS:

                    axis_a = axes[
                        i
                    ].upper()

                    axis_b = axes[
                        j
                    ].upper()

                    result.add_metric(
                        name=(
                            f"{axis_a}-{axis_b} "
                            "First-Difference Correlation"
                        ),
                        value=float(
                            difference_correlation_matrix[
                                i,
                                j,
                            ]
                        ),
                        unit=None,
                    )

            # Covariance eigenvalue structure
            #
            # This measures how much of the covariance is concentrated
            # along one statistical direction. It does not identify
            # the physical origin of that direction.

            eigenvalues = np.linalg.eigvalsh(
                covariance_matrix
            )

            eigenvalues = np.maximum(
                eigenvalues,
                0.0,
            )

            total_variance = float(
                np.sum(
                    eigenvalues
                )
            )

            if total_variance > 0.0:

                dominant_variance_fraction = float(
                    np.max(
                        eigenvalues
                    )
                    / total_variance
                )

                result.add_metric(
                    name=(
                        f"{group_display_name} Dominant "
                        "Covariance Variance Fraction"
                    ),
                    value=(
                        dominant_variance_fraction
                        * 100.0
                    ),
                    unit="%",
                    description=(
                        "Fraction of total three-axis covariance "
                        "represented by the largest covariance "
                        "eigenvalue."
                    ),
                )

            else:

                dominant_variance_fraction = None

            # Evidence and warnings

            if (
                abs(
                    strongest_correlation
                )
                >= correlation_threshold
            ):

                result.add_warning(
                    f"{group_display_name} shows substantial "
                    "cross-axis correlation. Strongest pair = "
                    f"{strongest_pair[0].upper()}-"
                    f"{strongest_pair[1].upper()}, "
                    f"r = {strongest_correlation:.3f}."
                )

                result.add_evidence(
                    statement=(
                        f"{group_display_name} contains substantial "
                        "statistical covariance between at least "
                        "two measurement axes."
                    ),
                    strength=EvidenceStrength.MODERATE,
                    supporting_metrics=[
                        (
                            f"{group_display_name} Maximum "
                            "Absolute Axis Correlation"
                        ),
                        (
                            f"{group_display_name} Strongest "
                            "Axis Correlation"
                        ),
                    ],
                )

            if (
                strongest_difference_correlation
                is not None
                and abs(
                    strongest_difference_correlation
                )
                >= difference_threshold
            ):

                result.add_warning(
                    f"{group_display_name} also shows substantial "
                    "correlation in sample-to-sample changes. "
                    f"Strongest first-difference pair = "
                    f"{strongest_difference_pair[0].upper()}-"
                    f"{strongest_difference_pair[1].upper()}, "
                    f"r = "
                    f"{strongest_difference_correlation:.3f}."
                )

                result.add_evidence(
                    statement=(
                        f"{group_display_name} axes contain "
                        "correlated short-timescale changes in "
                        "addition to any slower shared behavior."
                    ),
                    strength=EvidenceStrength.MODERATE,
                    supporting_metrics=[
                        (
                            f"{group_display_name} Maximum "
                            "Absolute First-Difference "
                            "Axis Correlation"
                        ),
                    ],
                )

            # Raw high correlation but weak difference correlation

            if (
                abs(
                    strongest_correlation
                )
                >= correlation_threshold
                and strongest_difference_correlation
                is not None
                and abs(
                    strongest_difference_correlation
                )
                < difference_threshold
            ):

                result.add_evidence(
                    statement=(
                        f"{group_display_name} shows substantial "
                        "raw cross-axis correlation, while "
                        "first-difference correlation is weaker. "
                        "This is consistent with shared slower "
                        "variation contributing to the raw "
                        "correlation."
                    ),
                    strength=EvidenceStrength.MODERATE,
                    supporting_metrics=[
                        (
                            f"{group_display_name} Maximum "
                            "Absolute Axis Correlation"
                        ),
                        (
                            f"{group_display_name} Maximum "
                            "Absolute First-Difference "
                            "Axis Correlation"
                        ),
                    ],
                )

            # Save complete matrices for plots/reports

            group_result = {
                "axes":
                    list(
                        axes
                    ),

                "sample_count":
                    sample_count,

                "covariance_matrix":
                    covariance_matrix.tolist(),

                "correlation_matrix":
                    correlation_matrix.tolist(),

                "strongest_pair":
                    list(
                        strongest_pair
                    ),

                "strongest_correlation":
                    float(
                        strongest_correlation
                    ),

                "dominant_variance_fraction":
                    dominant_variance_fraction,
            }

            if (
                difference_correlation_matrix
                is not None
            ):

                group_result[
                    "first_difference_correlation_matrix"
                ] = (
                    difference_correlation_matrix.tolist()
                )

                group_result[
                    "strongest_first_difference_pair"
                ] = list(
                    strongest_difference_pair
                )

                group_result[
                    "strongest_first_difference_correlation"
                ] = float(
                    strongest_difference_correlation
                )

            correlation_results[
                group_name
            ] = group_result

            processed_group_count += 1

        # Final state

        if processed_group_count == 0:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "No accelerometer or gyroscope group produced "
                "a valid axis-correlation result."
            )

            return result

        result.add_warning(
            "Cross-axis correlation is a statistical observation. "
            "It must not be interpreted by itself as proof of "
            "sensor-axis misalignment, nonorthogonality, "
            "cross-axis sensitivity, or electrical crosstalk."
        )

        result.metadata.update(
            {
                "minimum_samples":
                    minimum_samples,

                "correlation_warning_threshold":
                    correlation_threshold,

                "difference_correlation_warning_threshold":
                    difference_threshold,

                "interpretation":
                    "statistical_association_only",

                "physical_misalignment_inferred":
                    False,

                "correlation_results":
                    correlation_results,
            }
        )

        return result

    # Simultaneous valid XYZ samples

    @staticmethod
    def _valid_xyz_matrix(
        data: pd.DataFrame,
        context: ToolContext,
        axes: tuple[
            str,
            str,
            str,
        ],
    ) -> np.ndarray:
        """Return N x 3 matrix containing only rows where all three
        axes are finite.
        
        Listwise validity is used so every covariance/correlation
        matrix is based on the same observations.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
            axes: Value for `axes`.
        
        Returns:
            np.ndarray returned by the function.
        """

        arrays = []

        for axis in axes:

            column = context.column_mapping[
                axis
            ]

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            arrays.append(
                values
            )

        matrix = np.column_stack(
            arrays
        )

        valid_mask = np.all(
            np.isfinite(
                matrix
            ),
            axis=1,
        )

        return matrix[
            valid_mask
        ]

    # Strongest pair

    @classmethod
    def _strongest_off_diagonal(
        cls,
        matrix: np.ndarray,
        axes: tuple[
            str,
            str,
            str,
        ],
    ) -> tuple[
        tuple[str, str],
        float,
    ]:
        """Return the off-diagonal axis pair having the largest
        absolute matrix value.
        
        Args:
            matrix: Value for `matrix`.
            axes: Value for `axes`.
        
        Returns:
            Tuple containing the result values.
        """

        best_pair = None
        best_value = None

        for i, j in cls.AXIS_PAIRS:

            value = float(
                matrix[
                    i,
                    j,
                ]
            )

            if (
                best_value is None
                or abs(
                    value
                )
                > abs(
                    best_value
                )
            ):

                best_value = value

                best_pair = (
                    axes[
                        i
                    ],
                    axes[
                        j
                    ],
                )

        if (
            best_pair is None
            or best_value is None
        ):

            raise ValueError(
                "Could not identify an off-diagonal axis pair."
            )

        return (
            best_pair,
            best_value,
        )

    # Sensor-group discovery

    @staticmethod
    def _available_sensor_groups(
        data: pd.DataFrame,
        context: ToolContext,
    ) -> dict[
        str,
        tuple[
            str,
            str,
            str,
        ],
    ]:

        """Return sensor groups that contain all required axis columns.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Dictionary containing the result values.
        """
        groups = {
            "accelerometer": (
                "ax",
                "ay",
                "az",
            ),
            "gyroscope": (
                "gx",
                "gy",
                "gz",
            ),
        }

        available = {}

        for group_name, axes in groups.items():

            if all(
                (
                    context.column_mapping.get(
                        axis
                    )
                    in data.columns
                )
                for axis
                in axes
            ):

                available[
                    group_name
                ] = axes

        return available

    # Units

    @staticmethod
    def _covariance_unit(
        input_unit: str | None,
    ) -> str | None:

        """Calculate covariance unit.
        
        Args:
            input_unit: Input unit used by this function.
        
        Returns:
            str | None returned by this function.
        """
        if input_unit is None:
            return None

        return (
            f"({input_unit})^2"
        )
