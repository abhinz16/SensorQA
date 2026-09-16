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


class AccelerometerBiasTool(BaseAnalysisTool):
    """
    Characterizes stationary accelerometer behavior.

    Always available for stationary data:
        - mean output by axis
        - standard deviation by axis
        - RMS output by axis
        - peak-to-peak output by axis
        - mean acceleration-vector magnitude
        - gravity-magnitude error

    Per-axis bias is only calculated when SensorQA can determine
    the expected gravity vector from a recognized orientation label.

    SensorQA canonical acceleration units are m/s^2.
    """

    STANDARD_GRAVITY = 9.80665

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata exposed to SensorQA.
        
        Returns:
            ToolMetadata returned by the function.
        """

        return ToolMetadata(
            tool_id="imu_accelerometer_bias",
            name="Accelerometer Bias",
            version="1.0.0",
            description=(
                "Characterizes stationary accelerometer output, "
                "noise, gravity-vector magnitude, gravity-magnitude "
                "error, and per-axis bias when the sensor orientation "
                "is explicitly known."
            ),
            category=ToolCategory.IMU,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.ACCELEROMETER,
            ],
            required_columns=[
                "ax",
                "ay",
                "az",
            ],
            optional_columns=[
                "orientation_label",
                "temperature",
            ],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=100,
                    description=(
                        "Minimum number of valid samples required "
                        "per accelerometer axis."
                    ),
                    minimum=2,
                    maximum=10_000_000,
                ),
                ToolParameter(
                    name="gravity_magnitude",
                    parameter_type=ParameterType.FLOAT,
                    default=self.STANDARD_GRAVITY,
                    description=(
                        "Expected gravity magnitude used for static "
                        "gravity-error calculations."
                    ),
                    unit="m/s^2",
                    minimum=1.0,
                    maximum=20.0,
                ),
                ToolParameter(
                    name="warn_gravity_error_percent",
                    parameter_type=ParameterType.FLOAT,
                    default=2.0,
                    description=(
                        "Generate a warning when the mean measured "
                        "gravity-vector magnitude differs from the "
                        "configured gravity magnitude by more than "
                        "this percentage."
                    ),
                    unit="%",
                    minimum=0.0,
                    maximum=100.0,
                ),
                ToolParameter(
                    name="warn_relative_axis_noise_spread",
                    parameter_type=ParameterType.FLOAT,
                    default=5.0,
                    description=(
                        "Generate a warning when the noisiest axis "
                        "standard deviation exceeds the quietest axis "
                        "by more than this factor."
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
        """Extend SensorQA's common validation with stationary
        accelerometer requirements.
        
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

        # Check numerical data availability

        for field_name in ("ax", "ay", "az"):

            column = context.column_mapping.get(
                field_name
            )

            if column is None:
                continue

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            valid_count = int(
                values.notna().sum()
            )

            if valid_count < minimum_samples:

                validation.errors.append(
                    f"Accelerometer field '{field_name}' contains "
                    f"only {valid_count} valid samples. At least "
                    f"{minimum_samples} are required."
                )

        # Confirm experiment is compatible with static analysis

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is False:

            validation.errors.append(
                "Accelerometer bias analysis requires stationary "
                "or static-orientation data, but the experiment "
                "metadata indicates a dynamic test."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm that this dataset was "
                "collected while stationary. Static accelerometer "
                "interpretations should therefore be treated "
                "cautiously."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform stationary accelerometer characterization.
        
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

        acceleration_unit = context.units.get(
            "ax",
            "m/s^2",
        )

        gravity = float(
            context.parameters[
                "gravity_magnitude"
            ]
        )

        stationary_state = self._stationary_state(
            context
        )

        axis_results: dict[
            str,
            dict[str, float],
        ] = {}

        # Analyze each accelerometer axis

        for axis in ("ax", "ay", "az"):

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

            peak_to_peak = float(
                maximum - minimum
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

            result.add_metric(
                name=f"{axis_label} Mean Output",
                value=mean_output,
                unit=acceleration_unit,
                description=(
                    "Mean measured acceleration along this axis."
                ),
            )

            result.add_metric(
                name=(
                    f"{axis_label} Stationary Noise "
                    "Standard Deviation"
                ),
                value=standard_deviation,
                unit=acceleration_unit,
                description=(
                    "Standard deviation of accelerometer output "
                    "around its mean."
                ),
            )

            result.add_metric(
                name=f"{axis_label} RMS Output",
                value=rms_output,
                unit=acceleration_unit,
            )

            result.add_metric(
                name=f"{axis_label} Peak-to-Peak Output",
                value=peak_to_peak,
                unit=acceleration_unit,
            )

            result.add_metric(
                name=f"{axis_label} Median Output",
                value=median_output,
                unit=acceleration_unit,
            )

        # Mean acceleration vector

        mean_vector = np.array(
            [
                axis_results["ax"]["mean"],
                axis_results["ay"]["mean"],
                axis_results["az"]["mean"],
            ],
            dtype=float,
        )

        mean_vector_magnitude = float(
            np.linalg.norm(
                mean_vector
            )
        )

        gravity_magnitude_error = (
            mean_vector_magnitude
            - gravity
        )

        gravity_magnitude_error_percent = (
            gravity_magnitude_error
            / gravity
            * 100.0
        )

        result.add_metric(
            name="Mean Acceleration Vector Magnitude",
            value=mean_vector_magnitude,
            unit=acceleration_unit,
            description=(
                "Magnitude of the vector formed from the three "
                "mean accelerometer outputs."
            ),
        )

        result.add_metric(
            name="Gravity Magnitude Error",
            value=gravity_magnitude_error,
            unit=acceleration_unit,
            description=(
                "Difference between the measured mean acceleration "
                "magnitude and configured gravity magnitude."
            ),
        )

        result.add_metric(
            name="Gravity Magnitude Error Percent",
            value=gravity_magnitude_error_percent,
            unit="%",
        )

        # Determine whether orientation is explicitly known

        orientation_label = (
            self._get_orientation_label(
                data=data,
                context=context,
            )
        )

        expected_vector = None

        if orientation_label is not None:

            expected_vector = (
                self._expected_gravity_vector(
                    orientation_label=orientation_label,
                    gravity=gravity,
                )
            )

        # Per-axis bias, only when expected orientation is known

        if expected_vector is not None:

            bias_vector = (
                mean_vector
                - expected_vector
            )

            bias_magnitude = float(
                np.linalg.norm(
                    bias_vector
                )
            )

            axis_names = (
                "AX",
                "AY",
                "AZ",
            )

            for index, axis_name in enumerate(
                axis_names
            ):

                result.add_metric(
                    name=f"{axis_name} Bias",
                    value=float(
                        bias_vector[
                            index
                        ]
                    ),
                    unit=acceleration_unit,
                    description=(
                        "Mean measured acceleration minus the "
                        "expected static acceleration component "
                        "for the declared orientation."
                    ),
                )

            result.add_metric(
                name="Accelerometer Bias Vector Magnitude",
                value=bias_magnitude,
                unit=acceleration_unit,
                description=(
                    "Magnitude of the three-axis bias vector "
                    "relative to the expected gravity vector."
                ),
            )

            result.add_evidence(
                statement=(
                    "A recognized static orientation was available, "
                    "so SensorQA could compare the measured mean "
                    "acceleration vector with an explicit expected "
                    "gravity vector."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[
                    "AX Bias",
                    "AY Bias",
                    "AZ Bias",
                    "Accelerometer Bias Vector Magnitude",
                ],
            )

        else:

            result.add_warning(
                "The static sensor orientation was not explicitly "
                "known or was not recognized. SensorQA therefore "
                "did not report per-axis accelerometer bias. "
                "Gravity-magnitude error is still reported."
            )

            result.add_evidence(
                statement=(
                    "The dataset supports gravity-magnitude "
                    "characterization, but not a unique per-axis "
                    "bias estimate because the expected gravity "
                    "vector is unknown."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[
                    "Mean Acceleration Vector Magnitude",
                    "Gravity Magnitude Error",
                ],
            )

        # Axis stationary-noise comparison

        axis_noise_std = np.array(
            [
                axis_results["ax"][
                    "standard_deviation"
                ],
                axis_results["ay"][
                    "standard_deviation"
                ],
                axis_results["az"][
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

        maximum_axis_noise_std = float(
            np.max(
                axis_noise_std
            )
        )

        minimum_axis_noise_std = float(
            np.min(
                axis_noise_std
            )
        )

        result.add_metric(
            name="Mean Axis Noise Standard Deviation",
            value=mean_axis_noise_std,
            unit=acceleration_unit,
        )

        result.add_metric(
            name="Maximum Axis Noise Standard Deviation",
            value=maximum_axis_noise_std,
            unit=acceleration_unit,
        )

        # Axis noise spread

        if minimum_axis_noise_std > 0.0:

            axis_noise_spread_ratio = (
                maximum_axis_noise_std
                / minimum_axis_noise_std
            )

        elif maximum_axis_noise_std > 0.0:

            axis_noise_spread_ratio = np.inf

        else:

            axis_noise_spread_ratio = 1.0

        if np.isfinite(
            axis_noise_spread_ratio
        ):

            result.add_metric(
                name="Accelerometer Axis Noise Spread Ratio",
                value=float(
                    axis_noise_spread_ratio
                ),
                unit=None,
                description=(
                    "Ratio between the noisiest and quietest "
                    "accelerometer-axis standard deviations."
                ),
            )

        noise_spread_threshold = float(
            context.parameters[
                "warn_relative_axis_noise_spread"
            ]
        )

        if (
            axis_noise_spread_ratio
            > noise_spread_threshold
        ):

            result.add_warning(
                "Accelerometer stationary-noise levels differ "
                "substantially between axes. "
                f"Noisiest/quietest ratio = "
                f"{axis_noise_spread_ratio:.3f}."
            )

            result.add_evidence(
                statement=(
                    "One or more accelerometer axes exhibit "
                    "substantially higher stationary variation "
                    "than the quietest axis."
                ),
                strength=EvidenceStrength.MODERATE,
                supporting_metrics=[
                    "AX Stationary Noise Standard Deviation",
                    "AY Stationary Noise Standard Deviation",
                    "AZ Stationary Noise Standard Deviation",
                    "Accelerometer Axis Noise Spread Ratio",
                ],
            )

        # Gravity error warning

        gravity_warning_threshold = float(
            context.parameters[
                "warn_gravity_error_percent"
            ]
        )

        if (
            abs(
                gravity_magnitude_error_percent
            )
            > gravity_warning_threshold
        ):

            result.add_warning(
                "Measured mean acceleration-vector magnitude differs "
                "from the configured gravity magnitude by "
                f"{gravity_magnitude_error_percent:.3f}%."
            )

            result.add_evidence(
                statement=(
                    "The measured stationary acceleration magnitude "
                    "differs materially from the configured gravity "
                    "magnitude."
                ),
                strength=EvidenceStrength.MODERATE,
                supporting_metrics=[
                    "Mean Acceleration Vector Magnitude",
                    "Gravity Magnitude Error Percent",
                ],
            )

        # Stationarity interpretation

        if stationary_state is True:

            result.add_evidence(
                statement=(
                    "Experiment metadata identifies the dataset as "
                    "stationary or static-orientation data, supporting "
                    "static accelerometer characterization."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[
                    "Mean Acceleration Vector Magnitude",
                    "Gravity Magnitude Error",
                ],
            )

        else:

            result.add_warning(
                "Stationarity was not confirmed from metadata. "
                "Static gravity-based results should therefore "
                "be interpreted cautiously."
            )

        # Missing-data warnings

        for axis in ("ax", "ay", "az"):

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
                    "excluded from accelerometer statistics."
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

                "orientation_label":
                    orientation_label,

                "orientation_recognized":
                    expected_vector is not None,

                "gravity_magnitude":
                    gravity,

                "acceleration_unit":
                    acceleration_unit,

                "minimum_samples":
                    context.parameters[
                        "minimum_samples"
                    ],

                "warn_gravity_error_percent":
                    gravity_warning_threshold,

                "warn_relative_axis_noise_spread":
                    noise_spread_threshold,

                "estimated_sampling_rate_hz":
                    sampling_metadata.get(
                        "estimated_sampling_rate_hz"
                    ),

                "duration_seconds":
                    sampling_metadata.get(
                        "duration_seconds"
                    ),

                "ax_sample_count":
                    int(
                        axis_results["ax"][
                            "sample_count"
                        ]
                    ),

                "ay_sample_count":
                    int(
                        axis_results["ay"][
                            "sample_count"
                        ]
                    ),

                "az_sample_count":
                    int(
                        axis_results["az"][
                            "sample_count"
                        ]
                    ),
            }
        )

        return result

    # Stationarity

    @staticmethod
    def _stationary_state(
        context: ToolContext,
    ) -> bool | None:
        """Determine whether metadata supports a static experiment.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            bool | None returned by the function.
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

        if hasattr(
            test_mode,
            "value",
        ):
            test_mode = test_mode.value

        if test_mode in {
            "imu_stationary",
            "imu_controlled_orientation",
        }:
            return True

        if test_mode == "imu_dynamic":
            return False

        return None

    # Orientation handling

    @staticmethod
    def _get_orientation_label(
        data: pd.DataFrame,
        context: ToolContext,
    ) -> str | None:
        """Retrieve a single static orientation label from the dataset.
        
        The current V1 implementation only accepts a dataset that
        contains one unique non-null orientation.
        
        Multi-position calibration will be handled by a separate
        tool later.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            str | None returned by the function.
        """

        orientation_column = (
            context.column_mapping.get(
                "orientation_label"
            )
        )

        if orientation_column is None:
            return None

        if orientation_column not in data.columns:
            return None

        labels = (
            data[orientation_column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        labels = labels[
            labels != ""
        ]

        unique_labels = list(
            labels.unique()
        )

        if len(
            unique_labels
        ) != 1:
            return None

        return unique_labels[0]

    @staticmethod
    def _expected_gravity_vector(
        orientation_label: str,
        gravity: float,
    ) -> np.ndarray | None:
        """Convert a recognized orientation label into the expected
        static acceleration vector.
        
        IMPORTANT:
        Labels describe the expected accelerometer output directly.
        
        Examples:
        
            "+X" means expected measurement:
                [ +g, 0, 0 ]
        
            "-Z" means expected measurement:
                [ 0, 0, -g ]
        
        This avoids guessing the user's mechanical definition of
        "sensor facing up" or "sensor facing down".
        
        Args:
            orientation_label: Value for `orientation_label`.
            gravity: Value for `gravity`.
        
        Returns:
            np.ndarray | None returned by the function.
        """

        normalized = (
            orientation_label
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        aliases = {
            "+x": np.array(
                [gravity, 0.0, 0.0]
            ),
            "x+": np.array(
                [gravity, 0.0, 0.0]
            ),
            "plusx": np.array(
                [gravity, 0.0, 0.0]
            ),
            "positiveX".lower(): np.array(
                [gravity, 0.0, 0.0]
            ),

            "-x": np.array(
                [-gravity, 0.0, 0.0]
            ),
            "x-": np.array(
                [-gravity, 0.0, 0.0]
            ),
            "minusx": np.array(
                [-gravity, 0.0, 0.0]
            ),
            "negativeX".lower(): np.array(
                [-gravity, 0.0, 0.0]
            ),

            "+y": np.array(
                [0.0, gravity, 0.0]
            ),
            "y+": np.array(
                [0.0, gravity, 0.0]
            ),
            "plusy": np.array(
                [0.0, gravity, 0.0]
            ),
            "positiveY".lower(): np.array(
                [0.0, gravity, 0.0]
            ),

            "-y": np.array(
                [0.0, -gravity, 0.0]
            ),
            "y-": np.array(
                [0.0, -gravity, 0.0]
            ),
            "minusy": np.array(
                [0.0, -gravity, 0.0]
            ),
            "negativeY".lower(): np.array(
                [0.0, -gravity, 0.0]
            ),

            "+z": np.array(
                [0.0, 0.0, gravity]
            ),
            "z+": np.array(
                [0.0, 0.0, gravity]
            ),
            "plusz": np.array(
                [0.0, 0.0, gravity]
            ),
            "positiveZ".lower(): np.array(
                [0.0, 0.0, gravity]
            ),

            "-z": np.array(
                [0.0, 0.0, -gravity]
            ),
            "z-": np.array(
                [0.0, 0.0, -gravity]
            ),
            "minusz": np.array(
                [0.0, 0.0, -gravity]
            ),
            "negativeZ".lower(): np.array(
                [0.0, 0.0, -gravity]
            ),
        }

        return aliases.get(
            normalized
        )
