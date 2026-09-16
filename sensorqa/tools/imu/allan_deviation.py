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


class AllanDeviationTool(BaseAnalysisTool):
    """
    Computes overlapping Allan deviation for stationary
    accelerometer and/or gyroscope data.

    SensorQA uses Allan deviation to characterize stochastic
    inertial-sensor behavior across different averaging times.

    The tool can identify evidence consistent with:

        Gyroscope:
            - angle random walk / white rate noise
            - bias-instability region

        Accelerometer:
            - velocity random walk / white acceleration noise
            - bias-instability region

    Important:
        SensorQA only reports a coefficient when the observed
        log-log Allan slope is sufficiently close to the
        theoretical slope expected for that process.

    The tool does not resample the input data.
    """

    BIAS_INSTABILITY_FACTOR = 0.664

    @property
    def metadata(self) -> ToolMetadata:

        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="imu_allan_deviation",
            name="Allan Deviation",
            version="1.0.0",
            description=(
                "Computes overlapping Allan deviation for stationary "
                "accelerometer and gyroscope channels and cautiously "
                "extracts supported stochastic-noise coefficients."
            ),
            category=ToolCategory.NOISE,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.ACCELEROMETER,
                SensorType.GYROSCOPE,
            ],
            required_columns=[
                "timestamp",
            ],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=2000,
                    description=(
                        "Minimum number of continuous valid samples "
                        "required for Allan-deviation analysis."
                    ),
                    minimum=100,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_duration_seconds",
                    parameter_type=ParameterType.FLOAT,
                    default=30.0,
                    description=(
                        "Minimum continuous recording duration "
                        "required to calculate an Allan curve."
                    ),
                    unit="s",
                    minimum=1.0,
                    maximum=1_000_000.0,
                ),
                ToolParameter(
                    name="tau_points",
                    parameter_type=ParameterType.INTEGER,
                    default=40,
                    description=(
                        "Approximate number of logarithmically spaced "
                        "averaging times used for the Allan curve."
                    ),
                    minimum=10,
                    maximum=200,
                ),
                ToolParameter(
                    name="minimum_allan_pairs",
                    parameter_type=ParameterType.INTEGER,
                    default=20,
                    description=(
                        "Minimum number of overlapping Allan "
                        "difference pairs required at each "
                        "averaging time."
                    ),
                    minimum=5,
                    maximum=10_000,
                ),
                ToolParameter(
                    name="max_sampling_cv",
                    parameter_type=ParameterType.FLOAT,
                    default=0.02,
                    description=(
                        "Maximum allowed coefficient of variation "
                        "of positive timestamp intervals."
                    ),
                    minimum=0.0,
                    maximum=1.0,
                ),
                ToolParameter(
                    name="allow_irregular_sampling",
                    parameter_type=ParameterType.BOOLEAN,
                    default=False,
                    description=(
                        "Allow Allan analysis despite sampling "
                        "irregularity. SensorQA does not resample "
                        "the signal when this option is enabled."
                    ),
                ),
                ToolParameter(
                    name="slope_tolerance",
                    parameter_type=ParameterType.FLOAT,
                    default=0.15,
                    description=(
                        "Maximum log-log slope deviation from a "
                        "theoretical Allan-region slope before "
                        "SensorQA declines to identify that process."
                    ),
                    minimum=0.02,
                    maximum=0.5,
                ),
                ToolParameter(
                    name="bias_instability_min_duration_seconds",
                    parameter_type=ParameterType.FLOAT,
                    default=300.0,
                    description=(
                        "Minimum continuous recording duration "
                        "required before SensorQA attempts to report "
                        "a bias-instability estimate."
                    ),
                    unit="s",
                    minimum=10.0,
                    maximum=10_000_000.0,
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
        """Verify that the dataset supports meaningful Allan analysis.
        
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
                "Allan-deviation characterization requires "
                "stationary data, but metadata identifies this "
                "experiment as dynamic."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm that the recording "
                "was stationary. Allan results may therefore "
                "contain true motion in addition to stochastic "
                "sensor behavior."
            )

        available_groups = (
            self._available_sensor_groups(
                data=data,
                context=context,
            )
        )

        if not available_groups:

            validation.errors.append(
                "No complete accelerometer or gyroscope XYZ group "
                "is available for Allan-deviation analysis."
            )

        timestamp_column = context.column_mapping.get(
            "timestamp"
        )

        if timestamp_column is None:

            validation.errors.append(
                "Timestamp mapping is unavailable."
            )

            validation.valid = False
            return validation

        timestamps = pd.to_numeric(
            data[timestamp_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_timestamps = timestamps[
            np.isfinite(
                timestamps
            )
        ]

        if len(finite_timestamps) < 3:

            validation.errors.append(
                "At least three valid timestamps are required."
            )

            validation.valid = False
            return validation

        intervals = np.diff(
            finite_timestamps
        )

        positive_intervals = intervals[
            intervals > 0
        ]

        if len(positive_intervals) < 2:

            validation.errors.append(
                "Insufficient positive timestamp intervals "
                "for Allan-deviation analysis."
            )

            validation.valid = False
            return validation

        mean_dt = float(
            np.mean(
                positive_intervals
            )
        )

        std_dt = float(
            np.std(
                positive_intervals
            )
        )

        sampling_cv = (
            std_dt / mean_dt
            if mean_dt > 0
            else np.inf
        )

        max_sampling_cv = float(
            context.parameters.get(
                "max_sampling_cv",
                0.02,
            )
        )

        allow_irregular = bool(
            context.parameters.get(
                "allow_irregular_sampling",
                False,
            )
        )

        if sampling_cv > max_sampling_cv:

            if allow_irregular:

                validation.warnings.append(
                    "Sampling irregularity exceeds the configured "
                    "Allan-deviation threshold, but execution was "
                    "explicitly allowed. SensorQA will not resample "
                    "the input data."
                )

            else:

                validation.errors.append(
                    "Sampling intervals are too irregular for "
                    "standard Allan-deviation analysis. "
                    f"Interval CV = {sampling_cv:.6f}, "
                    f"allowed maximum = {max_sampling_cv:.6f}."
                )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Calculate overlapping Allan deviation.
        
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

        timestamp_column = context.column_mapping[
            "timestamp"
        ]

        timestamps = pd.to_numeric(
            data[timestamp_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        sampling_info = self._sampling_information(
            timestamps
        )

        if sampling_info is None:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "Sampling period could not be determined."
            )

            return result

        sample_interval = sampling_info[
            "median_interval"
        ]

        sampling_rate = (
            1.0 / sample_interval
        )

        sampling_cv = sampling_info[
            "coefficient_of_variation"
        ]

        available_groups = (
            self._available_sensor_groups(
                data=data,
                context=context,
            )
        )

        minimum_samples = int(
            context.parameters[
                "minimum_samples"
            ]
        )

        minimum_duration = float(
            context.parameters[
                "minimum_duration_seconds"
            ]
        )

        tau_points = int(
            context.parameters[
                "tau_points"
            ]
        )

        minimum_allan_pairs = int(
            context.parameters[
                "minimum_allan_pairs"
            ]
        )

        slope_tolerance = float(
            context.parameters[
                "slope_tolerance"
            ]
        )

        bias_minimum_duration = float(
            context.parameters[
                "bias_instability_min_duration_seconds"
            ]
        )

        curves: dict[
            str,
            dict[str, list[float]],
        ] = {}

        processed_axis_count = 0

        # Analyze each complete sensor group

        for group_name, axes in available_groups.items():

            group_values = {
                axis: pd.to_numeric(
                    data[
                        context.column_mapping[
                            axis
                        ]
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
                for axis in axes
            }

            # Find a continuous region in which:
            #
            # - timestamp is valid
            # - all three sensor axes are valid
            # - timestamps move forward
            # - no large acquisition gap is present
            #
            # Allan analysis is performed on the longest such
            # region rather than silently deleting isolated rows.

            segment = self._longest_continuous_group_segment(
                timestamps=timestamps,
                group_values=group_values,
                expected_dt=sample_interval,
            )

            if segment is None:

                result.add_warning(
                    f"No usable continuous {group_name} segment "
                    "was found for Allan-deviation analysis."
                )

                continue

            start_index, end_index = segment

            segment_length = (
                end_index - start_index
            )

            segment_duration = (
                (
                    segment_length - 1
                )
                * sample_interval
            )

            if segment_length < minimum_samples:

                result.add_warning(
                    f"The longest continuous {group_name} segment "
                    f"contains only {segment_length} samples. "
                    f"At least {minimum_samples} are required."
                )

                continue

            if segment_duration < minimum_duration:

                result.add_warning(
                    f"The longest continuous {group_name} segment "
                    f"is only {segment_duration:.2f} s long. "
                    f"At least {minimum_duration:.2f} s are "
                    "required."
                )

                continue

            # Build valid averaging factors once for the group

            averaging_factors = self._build_averaging_factors(
                sample_count=segment_length,
                requested_points=tau_points,
                minimum_pairs=minimum_allan_pairs,
            )

            if len(averaging_factors) < 3:

                result.add_warning(
                    f"Insufficient valid averaging times for "
                    f"{group_name} Allan analysis."
                )

                continue

            tau = (
                averaging_factors
                * sample_interval
            )

            group_display_name = (
                "Accelerometer"
                if group_name == "accelerometer"
                else "Gyroscope"
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Allan Segment "
                    "Duration"
                ),
                value=segment_duration,
                unit="s",
                description=(
                    "Duration of the longest continuous valid "
                    "XYZ segment used for Allan analysis."
                ),
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Allan Segment "
                    "Sample Count"
                ),
                value=segment_length,
                unit=None,
            )

            # Per-axis Allan calculation

            for axis in axes:

                values = group_values[
                    axis
                ][
                    start_index:end_index
                ]

                allan_deviation = (
                    self._overlapping_allan_deviation(
                        values=values,
                        averaging_factors=averaging_factors,
                    )
                )

                valid_curve_mask = (
                    np.isfinite(
                        allan_deviation
                    )
                    & (
                        allan_deviation > 0
                    )
                )

                axis_tau = tau[
                    valid_curve_mask
                ]

                axis_adev = allan_deviation[
                    valid_curve_mask
                ]

                if len(axis_tau) < 3:

                    result.add_warning(
                        f"{axis.upper()} did not produce enough "
                        "valid Allan-deviation points."
                    )

                    continue

                slopes = self._local_log_slopes(
                    tau=axis_tau,
                    deviation=axis_adev,
                )

                curves[
                    axis
                ] = {
                    "tau_seconds":
                        axis_tau.tolist(),

                    "allan_deviation":
                        axis_adev.tolist(),

                    "local_log_slopes":
                        [
                            (
                                float(value)
                                if np.isfinite(
                                    value
                                )
                                else None
                            )
                            for value
                            in slopes
                        ],
                }

                processed_axis_count += 1

                axis_label = axis.upper()

                axis_unit = context.units.get(
                    axis
                )

                minimum_index = int(
                    np.argmin(
                        axis_adev
                    )
                )

                minimum_adev = float(
                    axis_adev[
                        minimum_index
                    ]
                )

                minimum_tau = float(
                    axis_tau[
                        minimum_index
                    ]
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Minimum Allan Deviation"
                    ),
                    value=minimum_adev,
                    unit=axis_unit,
                    description=(
                        "Minimum observed Allan deviation within "
                        "the supported averaging-time range."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Tau at Minimum "
                        "Allan Deviation"
                    ),
                    value=minimum_tau,
                    unit="s",
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Maximum Allan "
                        "Averaging Time"
                    ),
                    value=float(
                        axis_tau[-1]
                    ),
                    unit="s",
                )

                # White-noise / random-walk region
                #
                # Allan slope approximately -1/2.

                white_region = (
                    self._extract_slope_region(
                        tau=axis_tau,
                        deviation=axis_adev,
                        slopes=slopes,
                        target_slope=-0.5,
                        tolerance=slope_tolerance,
                    )
                )

                if white_region is not None:

                    white_coefficient = float(
                        np.median(
                            white_region[
                                "deviation"
                            ]
                            * np.sqrt(
                                white_region[
                                    "tau"
                                ]
                            )
                        )
                    )

                    representative_slope = float(
                        np.median(
                            white_region[
                                "slope"
                            ]
                        )
                    )

                    if group_name == "gyroscope":

                        coefficient_name = (
                            f"{axis_label} Angle Random Walk "
                            "Coefficient"
                        )

                        coefficient_unit = (
                            self._gyro_white_noise_unit(
                                axis_unit
                            )
                        )

                        evidence_statement = (
                            f"{axis_label} Allan behavior contains "
                            "a region consistent with white angular-"
                            "rate noise / angle random walk."
                        )

                    else:

                        coefficient_name = (
                            f"{axis_label} Velocity Random Walk "
                            "Coefficient"
                        )

                        coefficient_unit = (
                            self._accelerometer_white_noise_unit(
                                axis_unit
                            )
                        )

                        evidence_statement = (
                            f"{axis_label} Allan behavior contains "
                            "a region consistent with white "
                            "acceleration noise / velocity random "
                            "walk."
                        )

                    result.add_metric(
                        name=coefficient_name,
                        value=white_coefficient,
                        unit=coefficient_unit,
                        description=(
                            "Noise coefficient estimated only from "
                            "Allan points whose local log-log slope "
                            "is consistent with -1/2."
                        ),
                    )

                    result.add_metric(
                        name=(
                            f"{axis_label} White-Noise "
                            "Region Median Slope"
                        ),
                        value=representative_slope,
                        unit=None,
                    )

                    result.add_evidence(
                        statement=evidence_statement,
                        strength=EvidenceStrength.MODERATE,
                        supporting_metrics=[
                            coefficient_name,
                            (
                                f"{axis_label} White-Noise "
                                "Region Median Slope"
                            ),
                        ],
                    )

                else:

                    result.add_warning(
                        f"{axis_label} did not contain a sufficiently "
                        "clear -1/2 Allan-slope region. SensorQA "
                        "therefore did not report a random-walk "
                        "coefficient for this axis."
                    )

                # Bias-instability region
                #
                # Allan slope approximately zero.
                #
                # The commonly used rate-sensor conversion:
                #
                #     B ≈ sigma_A / 0.664
                #
                # is only used when:
                #
                # 1. recording duration is sufficiently long, and
                # 2. the Allan curve actually exhibits a near-zero
                #    slope region.

                if (
                    segment_duration
                    >= bias_minimum_duration
                ):

                    bias_region = (
                        self._extract_slope_region(
                            tau=axis_tau,
                            deviation=axis_adev,
                            slopes=slopes,
                            target_slope=0.0,
                            tolerance=slope_tolerance,
                        )
                    )

                    if bias_region is not None:

                        bias_instability = float(
                            np.median(
                                bias_region[
                                    "deviation"
                                ]
                            )
                            / self.BIAS_INSTABILITY_FACTOR
                        )

                        representative_slope = float(
                            np.median(
                                bias_region[
                                    "slope"
                                ]
                            )
                        )

                        result.add_metric(
                            name=(
                                f"{axis_label} Bias Instability "
                                "Estimate"
                            ),
                            value=bias_instability,
                            unit=axis_unit,
                            description=(
                                "Approximate bias-instability "
                                "coefficient derived from a "
                                "near-zero-slope Allan region "
                                "using sigma_A / 0.664."
                            ),
                        )

                        result.add_metric(
                            name=(
                                f"{axis_label} Bias-Instability "
                                "Region Median Slope"
                            ),
                            value=representative_slope,
                            unit=None,
                        )

                        result.add_evidence(
                            statement=(
                                f"{axis_label} Allan behavior "
                                "contains a near-zero-slope region "
                                "consistent with bias-instability "
                                "behavior."
                            ),
                            strength=EvidenceStrength.MODERATE,
                            supporting_metrics=[
                                (
                                    f"{axis_label} Bias "
                                    "Instability Estimate"
                                ),
                                (
                                    f"{axis_label} "
                                    "Bias-Instability Region "
                                    "Median Slope"
                                ),
                            ],
                        )

                    else:

                        result.add_warning(
                            f"{axis_label} did not contain a clear "
                            "near-zero Allan-slope region. SensorQA "
                            "did not report a bias-instability "
                            "coefficient for this axis."
                        )

                else:

                    result.add_warning(
                        f"{axis_label} recording duration "
                        f"({segment_duration:.1f} s) is shorter "
                        "than the configured "
                        f"{bias_minimum_duration:.1f} s requirement "
                        "for bias-instability estimation."
                    )

        # Overall tool status

        if processed_axis_count == 0:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "No accelerometer or gyroscope axis produced a "
                "valid Allan-deviation curve."
            )

            return result

        # Sampling warnings

        max_sampling_cv = float(
            context.parameters[
                "max_sampling_cv"
            ]
        )

        if sampling_cv > max_sampling_cv:

            result.add_warning(
                "Allan deviation was calculated despite sampling "
                "irregularity because the advanced override was "
                "enabled. No resampling was performed."
            )

            result.add_evidence(
                statement=(
                    "Sampling irregularity exceeds the normal "
                    "Allan-analysis acceptance threshold, reducing "
                    "confidence in the derived Allan coefficients."
                ),
                strength=EvidenceStrength.WEAK,
                supporting_metrics=[],
            )

        if self._stationary_state(
            context
        ) is True:

            result.add_evidence(
                statement=(
                    "Experiment metadata identifies this recording "
                    "as stationary, supporting stochastic inertial-"
                    "sensor characterization."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[],
            )

        # Store complete curves for plotting/reporting

        result.metadata.update(
            {
                "method":
                    "overlapping_allan_deviation",

                "input_resampled":
                    False,

                "sample_interval_seconds":
                    sample_interval,

                "estimated_sampling_rate_hz":
                    sampling_rate,

                "sampling_interval_cv":
                    sampling_cv,

                "minimum_samples":
                    minimum_samples,

                "minimum_duration_seconds":
                    minimum_duration,

                "minimum_allan_pairs":
                    minimum_allan_pairs,

                "requested_tau_points":
                    tau_points,

                "slope_tolerance":
                    slope_tolerance,

                "bias_instability_factor":
                    self.BIAS_INSTABILITY_FACTOR,

                "bias_instability_min_duration_seconds":
                    bias_minimum_duration,

                "allan_curves":
                    curves,
            }
        )

        return result

    # Overlapping Allan calculation

    @staticmethod
    def _overlapping_allan_deviation(
        values: np.ndarray,
        averaging_factors: np.ndarray,
    ) -> np.ndarray:
        """Compute overlapping Allan deviation for a sequence of
        uniformly sampled rate/acceleration measurements.
        
        For averaging factor m:
        
            y_bar[i] =
                mean(y[i : i + m])
        
        Allan variance:
        
            sigma_A^2 =
                1/2 * mean(
                    (y_bar[i + m] - y_bar[i])^2
                )
        
        This formulation operates directly on rate-like sensor
        measurements such as angular rate and acceleration.
        
        Args:
            values: Values to process.
            averaging_factors: Value for `averaging_factors`.
        
        Returns:
            np.ndarray returned by the function.
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        cumulative = np.concatenate(
            (
                np.array(
                    [0.0]
                ),
                np.cumsum(
                    values,
                    dtype=float,
                ),
            )
        )

        deviations: list[float] = []

        sample_count = len(
            values
        )

        for m in averaging_factors:

            m = int(
                m
            )

            if (
                m < 1
                or 2 * m
                > sample_count
            ):

                deviations.append(
                    np.nan
                )

                continue

            moving_average = (
                cumulative[
                    m:
                ]
                - cumulative[
                    :-m
                ]
            ) / m

            allan_difference = (
                moving_average[
                    m:
                ]
                - moving_average[
                    :-m
                ]
            )

            if len(
                allan_difference
            ) == 0:

                deviations.append(
                    np.nan
                )

                continue

            allan_variance = (
                0.5
                * np.mean(
                    allan_difference ** 2
                )
            )

            deviations.append(
                float(
                    np.sqrt(
                        allan_variance
                    )
                )
            )

        return np.asarray(
            deviations,
            dtype=float,
        )

    # Averaging-time construction

    @staticmethod
    def _build_averaging_factors(
        sample_count: int,
        requested_points: int,
        minimum_pairs: int,
    ) -> np.ndarray:
        """Generate logarithmically spaced integer cluster sizes.
        
        For overlapping Allan deviation:
        
            number of available differences
            = N - 2m + 1
        
        so m is limited to preserve minimum_pairs.
        
        Args:
            sample_count: Value for `sample_count`.
            requested_points: Value for `requested_points`.
            minimum_pairs: Minimum accepted pairs.
        
        Returns:
            np.ndarray returned by the function.
        """

        maximum_m_from_pairs = int(
            (
                sample_count
                - minimum_pairs
                + 1
            )
            // 2
        )

        maximum_m = max(
            1,
            maximum_m_from_pairs,
        )

        if maximum_m < 2:

            return np.array(
                [],
                dtype=int,
            )

        raw = np.logspace(
            start=0.0,
            stop=np.log10(
                maximum_m
            ),
            num=requested_points,
        )

        averaging_factors = np.unique(
            np.clip(
                np.rint(
                    raw
                ).astype(
                    int
                ),
                1,
                maximum_m,
            )
        )

        return averaging_factors

    # Local slope calculations

    @staticmethod
    def _local_log_slopes(
        tau: np.ndarray,
        deviation: np.ndarray,
        window: int = 5,
    ) -> np.ndarray:
        """Estimate local slope on the log10(Allan deviation) versus
        log10(tau) curve using small local linear regressions.
        
        Args:
            tau: Value for `tau`.
            deviation: Value for `deviation`.
            window: Value for `window`.
        
        Returns:
            np.ndarray returned by the function.
        """

        tau = np.asarray(
            tau,
            dtype=float,
        )

        deviation = np.asarray(
            deviation,
            dtype=float,
        )

        slopes = np.full(
            len(tau),
            np.nan,
            dtype=float,
        )

        half_window = (
            window // 2
        )

        for index in range(
            len(tau)
        ):

            start = max(
                0,
                index - half_window,
            )

            stop = min(
                len(tau),
                index + half_window + 1,
            )

            if (
                stop - start
                < 3
            ):
                continue

            x = np.log10(
                tau[
                    start:stop
                ]
            )

            y = np.log10(
                deviation[
                    start:stop
                ]
            )

            if not (
                np.all(
                    np.isfinite(
                        x
                    )
                )
                and np.all(
                    np.isfinite(
                        y
                    )
                )
            ):
                continue

            slope, _ = np.polyfit(
                x,
                y,
                1,
            )

            slopes[
                index
            ] = float(
                slope
            )

        return slopes

    @staticmethod
    def _extract_slope_region(
        tau: np.ndarray,
        deviation: np.ndarray,
        slopes: np.ndarray,
        target_slope: float,
        tolerance: float,
    ) -> dict[
        str,
        np.ndarray,
    ] | None:
        """Return Allan points whose local log-log slope agrees with
        a theoretical process slope within the configured tolerance.
        
        Args:
            tau: Value for `tau`.
            deviation: Value for `deviation`.
            slopes: Value for `slopes`.
            target_slope: Value for `target_slope`.
            tolerance: Value for `tolerance`.
        
        Returns:
            Dictionary containing the result values.
        """

        mask = (
            np.isfinite(
                slopes
            )
            & (
                np.abs(
                    slopes
                    - target_slope
                )
                <= tolerance
            )
        )

        if np.sum(
            mask
        ) < 2:

            return None

        return {
            "tau":
                tau[
                    mask
                ],

            "deviation":
                deviation[
                    mask
                ],

            "slope":
                slopes[
                    mask
                ],
        }

    # Continuous-segment handling

    @staticmethod
    def _longest_continuous_group_segment(
        timestamps: np.ndarray,
        group_values: dict[
            str,
            np.ndarray,
        ],
        expected_dt: float,
    ) -> tuple[int, int] | None:
        """Find the longest continuous block where:
        
            - timestamp is finite
            - every group axis is finite
            - timestamp advances
            - sample gap is no larger than 1.5 times expected dt
        
        The returned end index is exclusive.
        
        This avoids silently deleting isolated invalid samples and
        concatenating physically separated portions of a recording.
        
        Args:
            timestamps: Value for `timestamps`.
            group_values: Value for `group_values`.
            expected_dt: Value for `expected_dt`.
        
        Returns:
            Tuple containing the result values.
        """

        valid = np.isfinite(
            timestamps
        )

        for values in group_values.values():

            valid &= np.isfinite(
                values
            )

        sample_count = len(
            timestamps
        )

        best_start = None
        best_end = None
        best_length = 0

        current_start = None

        for index in range(
            sample_count
        ):

            if not valid[
                index
            ]:

                if current_start is not None:

                    length = (
                        index
                        - current_start
                    )

                    if length > best_length:

                        best_start = (
                            current_start
                        )

                        best_end = index

                        best_length = length

                current_start = None
                continue

            if current_start is None:

                current_start = index
                continue

            dt = (
                timestamps[
                    index
                ]
                - timestamps[
                    index - 1
                ]
            )

            continuous = (
                np.isfinite(
                    dt
                )
                and dt > 0.0
                and dt
                <= (
                    1.5
                    * expected_dt
                )
            )

            if not continuous:

                length = (
                    index
                    - current_start
                )

                if length > best_length:

                    best_start = (
                        current_start
                    )

                    best_end = index

                    best_length = length

                current_start = index

        if current_start is not None:

            length = (
                sample_count
                - current_start
            )

            if length > best_length:

                best_start = (
                    current_start
                )

                best_end = sample_count

                best_length = length

        if (
            best_start is None
            or best_end is None
            or best_length < 2
        ):

            return None

        return (
            best_start,
            best_end,
        )

    # Sampling information

    @staticmethod
    def _sampling_information(
        timestamps: np.ndarray,
    ) -> dict[
        str,
        float,
    ] | None:
        """Estimate sampling characteristics without resampling.
        
        Args:
            timestamps: Value for `timestamps`.
        
        Returns:
            Dictionary containing the result values.
        """

        timestamps = timestamps[
            np.isfinite(
                timestamps
            )
        ]

        if len(
            timestamps
        ) < 3:

            return None

        intervals = np.diff(
            timestamps
        )

        positive_intervals = intervals[
            intervals > 0
        ]

        if len(
            positive_intervals
        ) < 2:

            return None

        median_interval = float(
            np.median(
                positive_intervals
            )
        )

        mean_interval = float(
            np.mean(
                positive_intervals
            )
        )

        std_interval = float(
            np.std(
                positive_intervals
            )
        )

        coefficient_of_variation = (
            std_interval
            / mean_interval
            if mean_interval > 0.0
            else np.inf
        )

        return {
            "median_interval":
                median_interval,

            "mean_interval":
                mean_interval,

            "std_interval":
                std_interval,

            "coefficient_of_variation":
                coefficient_of_variation,
        }

    # Sensor-group discovery

    @staticmethod
    def _available_sensor_groups(
        data: pd.DataFrame,
        context: ToolContext,
    ) -> dict[
        str,
        tuple[str, str, str],
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
                for axis in axes
            ):

                available[
                    group_name
                ] = axes

        return available

    # Stationarity metadata

    @staticmethod
    def _stationary_state(
        context: ToolContext,
    ) -> bool | None:

        """Read the stationary-state metadata used by this analysis.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            bool | None returned by this function.
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

    # Output-unit helpers

    @staticmethod
    def _gyro_white_noise_unit(
        input_unit: str | None,
    ) -> str | None:

        """Calculate gyro white noise unit.
        
        Args:
            input_unit: Input unit used by this function.
        
        Returns:
            str | None returned by this function.
        """
        if input_unit == "rad/s":
            return "rad/sqrt(s)"

        if input_unit is None:
            return None

        return (
            f"{input_unit}*sqrt(s)"
        )

    @staticmethod
    def _accelerometer_white_noise_unit(
        input_unit: str | None,
    ) -> str | None:

        """Calculate accelerometer white noise unit.
        
        Args:
            input_unit: Input unit used by this function.
        
        Returns:
            str | None returned by this function.
        """
        if input_unit == "m/s^2":
            return "m/s/sqrt(s)"

        if input_unit is None:
            return None

        return (
            f"{input_unit}*sqrt(s)"
        )
