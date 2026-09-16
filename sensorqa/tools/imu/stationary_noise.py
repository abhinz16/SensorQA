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


class StationaryNoiseTool(BaseAnalysisTool):
    """
    Characterizes short-term stationary noise in accelerometer
    and/or gyroscope channels.

    For every available axis, SensorQA calculates:

        - mean-removed standard deviation
        - variance
        - RMS of the mean-removed residual
        - median absolute deviation (MAD)
        - robust MAD-based sigma estimate
        - 5th to 95th percentile span
        - successive-difference noise estimate

    This tool characterizes observed stationary variation.

    It does NOT attempt to separate individual stochastic IMU
    processes such as:

        - angle random walk
        - velocity random walk
        - bias instability
        - rate random walk

    Those require Allan-deviation analysis.
    """

    MAD_TO_SIGMA = 1.4826

    @property
    def metadata(self) -> ToolMetadata:
        """Describe this tool to the SensorQA plug-in system.
        
        Returns:
            ToolMetadata returned by the function.
        """

        return ToolMetadata(
            tool_id="imu_stationary_noise",
            name="Stationary Noise",
            version="1.0.0",
            description=(
                "Characterizes short-term stationary accelerometer "
                "and gyroscope variation using conventional and "
                "robust statistics."
            ),
            category=ToolCategory.NOISE,
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
                        "Minimum valid samples required for each "
                        "axis included in stationary-noise analysis."
                    ),
                    minimum=10,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="warn_relative_axis_noise_spread",
                    parameter_type=ParameterType.FLOAT,
                    default=5.0,
                    description=(
                        "Warn when the noisiest axis has a standard "
                        "deviation more than this multiple of the "
                        "quietest axis in the same sensor group."
                    ),
                    minimum=1.0,
                    maximum=1000.0,
                ),
                ToolParameter(
                    name="warn_std_to_robust_ratio",
                    parameter_type=ParameterType.FLOAT,
                    default=2.0,
                    description=(
                        "Warn when conventional standard deviation "
                        "is more than this multiple of the robust "
                        "MAD-derived sigma estimate. A large ratio "
                        "can indicate outliers or non-Gaussian tails."
                    ),
                    minimum=1.0,
                    maximum=100.0,
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
        """Verify that the experiment and available channels support
        stationary-noise characterization.
        
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

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is False:

            validation.errors.append(
                "Stationary-noise analysis requires stationary "
                "data, but experiment metadata indicates a "
                "dynamic test."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm from metadata that "
                "the experiment was stationary. Reported variation "
                "may therefore contain real motion as well as "
                "sensor noise."
            )

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        if not available_groups:

            validation.errors.append(
                "No complete accelerometer or gyroscope axis group "
                "is available. Stationary-noise analysis requires "
                "AX/AY/AZ and/or GX/GY/GZ."
            )

        minimum_samples = int(
            context.parameters.get(
                "minimum_samples",
                500,
            )
        )

        for group_name, axes in available_groups.items():

            for axis in axes:

                column = context.column_mapping.get(
                    axis
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
                        f"Field '{axis}' contains only "
                        f"{valid_count} valid samples. At least "
                        f"{minimum_samples} are required for "
                        "stationary-noise analysis."
                    )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform stationary-noise characterization.
        
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

        stationary_state = self._stationary_state(
            context
        )

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        std_to_robust_threshold = float(
            context.parameters[
                "warn_std_to_robust_ratio"
            ]
        )

        axis_spread_threshold = float(
            context.parameters[
                "warn_relative_axis_noise_spread"
            ]
        )

        all_axis_statistics: dict[
            str,
            dict[str, float],
        ] = {}

        # Analyze accelerometer and/or gyroscope groups

        for group_name, axes in available_groups.items():

            group_standard_deviations: list[
                float
            ] = []

            group_axis_names: list[str] = []

            for axis in axes:

                column = context.column_mapping[
                    axis
                ]

                values_series = pd.to_numeric(
                    data[column],
                    errors="coerce",
                )

                valid_values = (
                    values_series
                    .dropna()
                    .to_numpy(
                        dtype=float
                    )
                )

                sample_count = len(
                    valid_values
                )

                mean_value = float(
                    np.mean(
                        valid_values
                    )
                )

                # Remove only the channel mean.
                #
                # We intentionally do not remove a fitted trend
                # because drift is real behavior that should not
                # silently disappear from the analysis.

                residual = (
                    valid_values
                    - mean_value
                )

                standard_deviation = float(
                    np.std(
                        residual,
                        ddof=1,
                    )
                )

                variance = float(
                    np.var(
                        residual,
                        ddof=1,
                    )
                )

                residual_rms = float(
                    np.sqrt(
                        np.mean(
                            residual ** 2
                        )
                    )
                )

                residual_median = float(
                    np.median(
                        residual
                    )
                )

                mad = float(
                    np.median(
                        np.abs(
                            residual
                            - residual_median
                        )
                    )
                )

                robust_sigma = float(
                    self.MAD_TO_SIGMA
                    * mad
                )

                percentile_05 = float(
                    np.percentile(
                        residual,
                        5.0,
                    )
                )

                percentile_95 = float(
                    np.percentile(
                        residual,
                        95.0,
                    )
                )

                percentile_span = (
                    percentile_95
                    - percentile_05
                )

                # Successive-difference noise estimate
                #
                # For uncorrelated white noise x[k] with variance
                # sigma^2:
                #
                # Var(x[k+1] - x[k]) = 2 sigma^2
                #
                # so std(diff) / sqrt(2) provides a useful
                # short-timescale estimate.
                #
                # It is NOT a substitute for Allan deviation.

                differences = np.diff(
                    valid_values
                )

                if len(differences) >= 2:

                    difference_noise_estimate = float(
                        np.std(
                            differences,
                            ddof=1,
                        )
                        / np.sqrt(
                            2.0
                        )
                    )

                else:

                    difference_noise_estimate = (
                        None
                    )

                unit = context.units.get(
                    axis
                )

                axis_label = axis.upper()

                all_axis_statistics[
                    axis
                ] = {
                    "mean": mean_value,
                    "standard_deviation":
                        standard_deviation,
                    "variance":
                        variance,
                    "residual_rms":
                        residual_rms,
                    "mad":
                        mad,
                    "robust_sigma":
                        robust_sigma,
                    "percentile_05":
                        percentile_05,
                    "percentile_95":
                        percentile_95,
                    "percentile_span":
                        percentile_span,
                    "sample_count":
                        float(sample_count),
                }

                if (
                    difference_noise_estimate
                    is not None
                ):
                    all_axis_statistics[
                        axis
                    ][
                        "difference_noise_estimate"
                    ] = (
                        difference_noise_estimate
                    )

                group_standard_deviations.append(
                    standard_deviation
                )

                group_axis_names.append(
                    axis
                )

                # Add metrics

                result.add_metric(
                    name=(
                        f"{axis_label} Noise "
                        "Standard Deviation"
                    ),
                    value=standard_deviation,
                    unit=unit,
                    description=(
                        "Standard deviation after subtracting "
                        "the channel mean."
                    ),
                )

                result.add_metric(
                    name=f"{axis_label} Noise Variance",
                    value=variance,
                    unit=(
                        f"({unit})^2"
                        if unit
                        else None
                    ),
                    description=(
                        "Variance of the mean-removed "
                        "stationary signal."
                    ),
                )

                result.add_metric(
                    name=f"{axis_label} Noise RMS",
                    value=residual_rms,
                    unit=unit,
                    description=(
                        "RMS magnitude of the mean-removed "
                        "stationary signal."
                    ),
                )

                result.add_metric(
                    name=f"{axis_label} Noise MAD",
                    value=mad,
                    unit=unit,
                    description=(
                        "Median absolute deviation of the "
                        "mean-removed signal."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Robust Noise "
                        "Sigma Estimate"
                    ),
                    value=robust_sigma,
                    unit=unit,
                    description=(
                        "Robust noise-scale estimate equal to "
                        "1.4826 times the median absolute "
                        "deviation."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} 5-95 Percentile "
                        "Noise Span"
                    ),
                    value=percentile_span,
                    unit=unit,
                    description=(
                        "Span between the 5th and 95th "
                        "percentiles of the mean-removed signal."
                    ),
                )

                if (
                    difference_noise_estimate
                    is not None
                ):

                    result.add_metric(
                        name=(
                            f"{axis_label} Successive-Difference "
                            "Noise Estimate"
                        ),
                        value=difference_noise_estimate,
                        unit=unit,
                        description=(
                            "Short-timescale noise estimate "
                            "calculated as std(diff(signal)) / "
                            "sqrt(2). This approximation is most "
                            "meaningful for uncorrelated noise."
                        ),
                    )

                # Conventional vs robust noise comparison

                if robust_sigma > 0.0:

                    std_to_robust_ratio = (
                        standard_deviation
                        / robust_sigma
                    )

                elif standard_deviation > 0.0:

                    std_to_robust_ratio = np.inf

                else:

                    std_to_robust_ratio = 1.0

                if np.isfinite(
                    std_to_robust_ratio
                ):

                    result.add_metric(
                        name=(
                            f"{axis_label} Standard-to-Robust "
                            "Noise Ratio"
                        ),
                        value=float(
                            std_to_robust_ratio
                        ),
                        unit=None,
                        description=(
                            "Ratio of conventional standard "
                            "deviation to the MAD-derived robust "
                            "sigma estimate."
                        ),
                    )

                if (
                    std_to_robust_ratio
                    > std_to_robust_threshold
                ):

                    result.add_warning(
                        f"{axis_label} conventional noise standard "
                        "deviation is substantially larger than "
                        "its robust MAD-based estimate "
                        f"(ratio = {std_to_robust_ratio:.3f})."
                    )

                    result.add_evidence(
                        statement=(
                            f"{axis_label} stationary variation has "
                            "a substantially larger conventional "
                            "standard deviation than robust "
                            "MAD-based noise estimate, which is "
                            "consistent with outliers or "
                            "non-Gaussian tails."
                        ),
                        strength=EvidenceStrength.MODERATE,
                        supporting_metrics=[
                            (
                                f"{axis_label} Noise "
                                "Standard Deviation"
                            ),
                            (
                                f"{axis_label} Robust Noise "
                                "Sigma Estimate"
                            ),
                            (
                                f"{axis_label} Standard-to-Robust "
                                "Noise Ratio"
                            ),
                        ],
                    )

                # Missing values

                missing_count = (
                    len(values_series)
                    - sample_count
                )

                if missing_count > 0:

                    result.add_warning(
                        f"{axis_label} contains {missing_count} "
                        "missing or non-numeric samples that were "
                        "excluded from stationary-noise statistics."
                    )

            # Group-level axis comparison

            group_std = np.array(
                group_standard_deviations,
                dtype=float,
            )

            mean_group_std = float(
                np.mean(
                    group_std
                )
            )

            maximum_group_std = float(
                np.max(
                    group_std
                )
            )

            minimum_group_std = float(
                np.min(
                    group_std
                )
            )

            group_display_name = (
                "Accelerometer"
                if group_name == "accelerometer"
                else "Gyroscope"
            )

            group_unit = context.units.get(
                group_axis_names[0]
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Mean Axis "
                    "Noise Standard Deviation"
                ),
                value=mean_group_std,
                unit=group_unit,
                description=(
                    "Mean of the three per-axis stationary "
                    "noise standard deviations."
                ),
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Maximum Axis "
                    "Noise Standard Deviation"
                ),
                value=maximum_group_std,
                unit=group_unit,
            )

            if minimum_group_std > 0.0:

                axis_spread_ratio = (
                    maximum_group_std
                    / minimum_group_std
                )

            elif maximum_group_std > 0.0:

                axis_spread_ratio = np.inf

            else:

                axis_spread_ratio = 1.0

            if np.isfinite(
                axis_spread_ratio
            ):

                result.add_metric(
                    name=(
                        f"{group_display_name} Axis "
                        "Noise Spread Ratio"
                    ),
                    value=float(
                        axis_spread_ratio
                    ),
                    unit=None,
                    description=(
                        "Ratio of the noisiest-axis standard "
                        "deviation to the quietest-axis "
                        "standard deviation."
                    ),
                )

            if (
                axis_spread_ratio
                > axis_spread_threshold
            ):

                result.add_warning(
                    f"{group_display_name} stationary noise "
                    "differs substantially between axes "
                    f"(spread ratio = "
                    f"{axis_spread_ratio:.3f})."
                )

                result.add_evidence(
                    statement=(
                        f"{group_display_name} stationary "
                        "variation is strongly axis-dependent."
                    ),
                    strength=EvidenceStrength.MODERATE,
                    supporting_metrics=[
                        (
                            f"{group_display_name} Axis "
                            "Noise Spread Ratio"
                        ),
                        *[
                            (
                                f"{axis.upper()} Noise "
                                "Standard Deviation"
                            )
                            for axis
                            in group_axis_names
                        ],
                    ],
                )

        # Stationarity evidence

        if stationary_state is True:

            result.add_evidence(
                statement=(
                    "Experiment metadata identifies the dataset "
                    "as stationary or static, supporting "
                    "interpretation of measured short-term "
                    "variation as stationary sensor behavior."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[],
            )

        else:

            result.add_warning(
                "Stationarity was not confirmed. Reported variation "
                "may include true sensor motion and should not be "
                "interpreted as pure sensor noise without additional "
                "evidence."
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

                "minimum_samples":
                    context.parameters[
                        "minimum_samples"
                    ],

                "warn_relative_axis_noise_spread":
                    axis_spread_threshold,

                "warn_std_to_robust_ratio":
                    std_to_robust_threshold,

                "mad_to_sigma_factor":
                    self.MAD_TO_SIGMA,

                "available_sensor_groups":
                    list(
                        available_groups.keys()
                    ),

                "estimated_sampling_rate_hz":
                    sampling_metadata.get(
                        "estimated_sampling_rate_hz"
                    ),

                "duration_seconds":
                    sampling_metadata.get(
                        "duration_seconds"
                    ),

                "mean_removed_only":
                    True,

                "linear_detrending_applied":
                    False,
            }
        )

        return result

    # Channel discovery

    @staticmethod
    def _available_sensor_groups(
        data: pd.DataFrame,
        context: ToolContext,
    ) -> dict[str, tuple[str, str, str]]:
        """Determine whether complete accelerometer and/or
        gyroscope groups are available.
        
        A partial XYZ group is not analyzed as a complete
        stationary sensor group.
        
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

        available: dict[
            str,
            tuple[str, str, str],
        ] = {}

        for group_name, axes in groups.items():

            all_available = True

            for axis in axes:

                mapped_column = (
                    context.column_mapping.get(
                        axis
                    )
                )

                if (
                    mapped_column is None
                    or mapped_column
                    not in data.columns
                ):

                    all_available = False
                    break

            if all_available:

                available[
                    group_name
                ] = axes

        return available

    # Stationarity handling

    @staticmethod
    def _stationary_state(
        context: ToolContext,
    ) -> bool | None:
        """Determine whether experiment metadata confirms that
        the recording is stationary/static.
        
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
            test_mode = (
                test_mode.value
            )

        if test_mode in {
            "imu_stationary",
            "imu_controlled_orientation",
        }:
            return True

        if test_mode == "imu_dynamic":
            return False

        return None
