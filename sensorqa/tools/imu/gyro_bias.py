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


class GyroBiasTool(BaseAnalysisTool):
    """
    Characterizes zero-rate gyroscope output for stationary data.

    For each available gyroscope axis, the tool calculates:

        - mean zero-rate output
        - standard deviation
        - RMS output
        - peak-to-peak output

    When the experiment is confirmed to be stationary, the mean
    zero-rate output is interpreted as an estimate of gyroscope bias.

    SensorQA canonical gyroscope units are rad/s.
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Describe the analysis tool to SensorQA.
        
        Returns:
            ToolMetadata returned by the function.
        """

        return ToolMetadata(
            tool_id="imu_gyro_bias",
            name="Gyroscope Bias",
            version="1.0.0",
            description=(
                "Characterizes stationary gyroscope zero-rate output "
                "and estimates per-axis bias, stationary noise, RMS "
                "output, peak-to-peak variation, and three-axis bias "
                "magnitude."
            ),
            category=ToolCategory.IMU,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.GYROSCOPE,
            ],
            required_columns=[
                "gx",
                "gy",
                "gz",
            ],
            optional_columns=[
                "temperature",
            ],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=100,
                    description=(
                        "Minimum number of valid samples required "
                        "per gyroscope axis."
                    ),
                    minimum=2,
                    maximum=10_000_000,
                ),
                ToolParameter(
                    name="warn_relative_axis_spread",
                    parameter_type=ParameterType.FLOAT,
                    default=5.0,
                    description=(
                        "Generate a warning when one axis has a "
                        "stationary-noise standard deviation more than "
                        "this multiple of the quietest axis."
                    ),
                    minimum=1.0,
                    maximum=1000.0,
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
        """Extend common SensorQA validation with gyroscope-specific
        scientific checks.
        
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

        minimum_samples = int(
            context.parameters.get(
                "minimum_samples",
                100,
            )
        )

        # Confirm enough numerical data exist for every axis

        for field_name in ("gx", "gy", "gz"):

            column = context.column_mapping.get(
                field_name
            )

            if column is None:
                continue

            numeric = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            valid_count = int(
                numeric.notna().sum()
            )

            if valid_count < minimum_samples:

                validation.errors.append(
                    f"Gyroscope field '{field_name}' contains only "
                    f"{valid_count} valid samples. At least "
                    f"{minimum_samples} are required."
                )

        # Check test-condition metadata

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is False:

            validation.errors.append(
                "Gyroscope bias analysis requires stationary "
                "zero-rate data, but the dataset metadata indicates "
                "that this experiment is not stationary."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm from metadata that the "
                "IMU was stationary. Mean angular-rate output will "
                "be calculated, but interpretation as true "
                "zero-rate bias should be treated cautiously."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform stationary gyroscope characterization.
        
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

        gyro_unit = context.units.get(
            "gx",
            "rad/s",
        )

        stationary_state = self._stationary_state(
            context
        )

        axis_results: dict[
            str,
            dict[str, float],
        ] = {}

        # Analyze each gyroscope axis

        for axis in ("gx", "gy", "gz"):

            column = context.column_mapping[
                axis
            ]

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            ).dropna().to_numpy(
                dtype=float
            )

            mean_output = float(
                np.mean(
                    values
                )
            )

            standard_deviation = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

            rms_output = float(
                np.sqrt(
                    np.mean(
                        values ** 2
                    )
                )
            )

            minimum = float(
                np.min(
                    values
                )
            )

            maximum = float(
                np.max(
                    values
                )
            )

            peak_to_peak = (
                maximum
                - minimum
            )

            median_output = float(
                np.median(
                    values
                )
            )

            axis_results[
                axis
            ] = {
                "mean": mean_output,
                "standard_deviation":
                    standard_deviation,
                "rms": rms_output,
                "minimum": minimum,
                "maximum": maximum,
                "peak_to_peak": peak_to_peak,
                "median": median_output,
                "sample_count": float(
                    len(values)
                ),
            }

            axis_label = axis.upper()

            # Mean / bias

            if stationary_state is True:

                metric_name = (
                    f"{axis_label} Bias"
                )

                description = (
                    "Mean zero-rate gyroscope output while the "
                    "sensor is confirmed stationary."
                )

            else:

                metric_name = (
                    f"{axis_label} Mean Zero-Rate Output"
                )

                description = (
                    "Mean measured angular-rate output. Stationarity "
                    "was not confirmed, so this value should not be "
                    "interpreted as definitive gyroscope bias."
                )

            result.add_metric(
                name=metric_name,
                value=mean_output,
                unit=gyro_unit,
                description=description,
            )

            # Stationary noise

            result.add_metric(
                name=(
                    f"{axis_label} Stationary Noise "
                    "Standard Deviation"
                ),
                value=standard_deviation,
                unit=gyro_unit,
                description=(
                    "Standard deviation of gyroscope output "
                    "around its mean."
                ),
            )

            result.add_metric(
                name=f"{axis_label} RMS Output",
                value=rms_output,
                unit=gyro_unit,
                description=(
                    "Root mean square angular-rate output."
                ),
            )

            result.add_metric(
                name=f"{axis_label} Peak-to-Peak Output",
                value=peak_to_peak,
                unit=gyro_unit,
                description=(
                    "Difference between maximum and minimum "
                    "observed angular-rate output."
                ),
            )

            result.add_metric(
                name=f"{axis_label} Median Output",
                value=median_output,
                unit=gyro_unit,
            )

        # Three-axis bias vector

        mean_vector = np.array(
            [
                axis_results["gx"]["mean"],
                axis_results["gy"]["mean"],
                axis_results["gz"]["mean"],
            ],
            dtype=float,
        )

        mean_vector_magnitude = float(
            np.linalg.norm(
                mean_vector
            )
        )

        if stationary_state is True:

            result.add_metric(
                name="Gyroscope Bias Vector Magnitude",
                value=mean_vector_magnitude,
                unit=gyro_unit,
                description=(
                    "Euclidean magnitude of the three-axis "
                    "stationary bias vector."
                ),
            )

        else:

            result.add_metric(
                name="Mean Angular-Rate Vector Magnitude",
                value=mean_vector_magnitude,
                unit=gyro_unit,
                description=(
                    "Magnitude of the three-axis mean angular-rate "
                    "vector. Stationarity was not confirmed."
                ),
            )

        # Combined stationary-noise statistics

        axis_noise_std = np.array(
            [
                axis_results["gx"][
                    "standard_deviation"
                ],
                axis_results["gy"][
                    "standard_deviation"
                ],
                axis_results["gz"][
                    "standard_deviation"
                ],
            ],
            dtype=float,
        )

        mean_axis_noise_std = float(
            np.mean(
                axis_noise_std
            )
        )

        max_axis_noise_std = float(
            np.max(
                axis_noise_std
            )
        )

        min_axis_noise_std = float(
            np.min(
                axis_noise_std
            )
        )

        result.add_metric(
            name="Mean Axis Noise Standard Deviation",
            value=mean_axis_noise_std,
            unit=gyro_unit,
            description=(
                "Mean of the stationary-noise standard deviations "
                "across the three gyroscope axes."
            ),
        )

        result.add_metric(
            name="Maximum Axis Noise Standard Deviation",
            value=max_axis_noise_std,
            unit=gyro_unit,
        )

        # Axis imbalance warning

        spread_threshold = float(
            context.parameters[
                "warn_relative_axis_spread"
            ]
        )

        if min_axis_noise_std > 0:

            noise_spread_ratio = (
                max_axis_noise_std
                / min_axis_noise_std
            )

        else:

            noise_spread_ratio = (
                np.inf
                if max_axis_noise_std > 0
                else 1.0
            )

        if np.isfinite(
            noise_spread_ratio
        ):

            result.add_metric(
                name="Gyroscope Axis Noise Spread Ratio",
                value=float(
                    noise_spread_ratio
                ),
                unit=None,
                description=(
                    "Ratio between the noisiest and quietest "
                    "gyroscope-axis standard deviations."
                ),
            )

        if (
            noise_spread_ratio
            > spread_threshold
        ):

            result.add_warning(
                "Gyroscope stationary-noise levels differ "
                "substantially between axes. "
                f"Noisiest/quietest standard-deviation ratio = "
                f"{noise_spread_ratio:.3f}."
            )

            result.add_evidence(
                statement=(
                    "One or more gyroscope axes exhibit substantially "
                    "higher stationary variation than the quietest axis."
                ),
                strength=EvidenceStrength.MODERATE,
                supporting_metrics=[
                    "GX Stationary Noise Standard Deviation",
                    "GY Stationary Noise Standard Deviation",
                    "GZ Stationary Noise Standard Deviation",
                    "Gyroscope Axis Noise Spread Ratio",
                ],
            )

        # Stationary-test interpretation

        if stationary_state is True:

            result.add_evidence(
                statement=(
                    "The experiment metadata identifies this as a "
                    "stationary test, allowing mean angular-rate "
                    "output to be interpreted as zero-rate bias."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[
                    "GX Bias",
                    "GY Bias",
                    "GZ Bias",
                    "Gyroscope Bias Vector Magnitude",
                ],
            )

        else:

            result.add_warning(
                "Stationarity was not confirmed. Reported mean "
                "angular-rate values should not be treated as "
                "definitive zero-rate bias estimates."
            )

        # Missing-data warnings

        for axis in ("gx", "gy", "gz"):

            column = context.column_mapping[
                axis
            ]

            total_count = len(
                data[column]
            )

            valid_count = int(
                axis_results[
                    axis
                ][
                    "sample_count"
                ]
            )

            missing_count = (
                total_count
                - valid_count
            )

            if missing_count > 0:

                result.add_warning(
                    f"{axis.upper()} contains {missing_count} "
                    "missing or non-numeric values that were "
                    "excluded from gyroscope statistics."
                )

        # Reproducibility metadata

        sampling_metadata = context.metadata.get(
            "sampling",
            {},
        )

        result.metadata.update(
            {
                "stationary_confirmed":
                    stationary_state,

                "gyro_unit":
                    gyro_unit,

                "minimum_samples":
                    context.parameters[
                        "minimum_samples"
                    ],

                "warn_relative_axis_spread":
                    spread_threshold,

                "estimated_sampling_rate_hz":
                    sampling_metadata.get(
                        "estimated_sampling_rate_hz"
                    ),

                "duration_seconds":
                    sampling_metadata.get(
                        "duration_seconds"
                    ),

                "gx_sample_count":
                    int(
                        axis_results["gx"][
                            "sample_count"
                        ]
                    ),

                "gy_sample_count":
                    int(
                        axis_results["gy"][
                            "sample_count"
                        ]
                    ),

                "gz_sample_count":
                    int(
                        axis_results["gz"][
                            "sample_count"
                        ]
                    ),
            }
        )

        return result

    # Stationarity interpretation

    @staticmethod
    def _stationary_state(
        context: ToolContext,
    ) -> bool | None:
        """Determine whether experiment metadata confirms stationarity.
        
        Returns:
        
            True
                Experiment is explicitly stationary.
        
            False
                Experiment is explicitly non-stationary.
        
            None
                Metadata is insufficient to determine the state.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
        """

        metadata = context.metadata.get(
            "sensorqa_metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        test_metadata = metadata.get(
            "test"
        )

        if not isinstance(
            test_metadata,
            dict,
        ):
            return None

        expected_stationary = (
            test_metadata.get(
                "expected_stationary"
            )
        )

        if expected_stationary is True:
            return True

        if expected_stationary is False:
            return False

        test_mode = test_metadata.get(
            "test_mode"
        )

        # asdict() preserves Enum objects in nested structures,
        # so handle both Enum-style and string representations.

        if hasattr(
            test_mode,
            "value",
        ):
            test_mode = (
                test_mode.value
            )

        if test_mode == "imu_stationary":
            return True

        if test_mode == "imu_dynamic":
            return False

        return None
