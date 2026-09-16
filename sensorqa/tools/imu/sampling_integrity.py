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


class SamplingIntegrityTool(BaseAnalysisTool):
    """
    Evaluates timing and sampling quality for IMU datasets.

    The tool examines:

        - sample count
        - recording duration
        - median sample interval
        - mean sample interval
        - sampling-rate estimate
        - interval jitter
        - duplicate timestamps
        - backward timestamps
        - large gaps
        - estimated missing samples
        - deviation from nominal sampling rate, when available
    """

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata exposed to the SensorQA registry and UI.
        
        Returns:
            ToolMetadata returned by the function.
        """

        return ToolMetadata(
            tool_id="imu_sampling_integrity",
            name="Sampling Integrity",
            version="1.0.0",
            description=(
                "Evaluates IMU timestamp quality, effective sampling "
                "frequency, interval jitter, timestamp duplication, "
                "time-order errors, large gaps, and estimated missing "
                "samples."
            ),
            category=ToolCategory.DATA_QUALITY,
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
                    name="large_gap_multiplier",
                    parameter_type=ParameterType.FLOAT,
                    default=5.0,
                    description=(
                        "An interval larger than this multiple of the "
                        "median sample interval is classified as a "
                        "large gap."
                    ),
                    minimum=1.5,
                    maximum=100.0,
                ),
                ToolParameter(
                    name="irregular_sampling_cv_threshold",
                    parameter_type=ParameterType.FLOAT,
                    default=0.05,
                    description=(
                        "Coefficient-of-variation threshold used to "
                        "flag irregular sampling."
                    ),
                    minimum=0.0,
                    maximum=1.0,
                ),
                ToolParameter(
                    name="nominal_rate_warning_percent",
                    parameter_type=ParameterType.FLOAT,
                    default=2.0,
                    description=(
                        "Warn when the measured sampling rate differs "
                        "from the nominal rate by more than this "
                        "percentage."
                    ),
                    unit="%",
                    minimum=0.0,
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
        """Extend SensorQA's standard tool validation with
        timestamp-specific checks.
        
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
        )

        valid_timestamp_count = int(
            timestamps.notna().sum()
        )

        if valid_timestamp_count < 2:

            validation.errors.append(
                "At least two valid timestamps are required "
                "for sampling-integrity analysis."
            )

            validation.valid = False

            return validation

        if valid_timestamp_count < 10:

            validation.warnings.append(
                "Fewer than 10 valid timestamps are available. "
                "Sampling statistics may be poorly representative."
            )

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform sampling-integrity analysis.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        timestamp_column = context.column_mapping[
            "timestamp"
        ]

        timestamps = pd.to_numeric(
            data[timestamp_column],
            errors="coerce",
        )

        valid_timestamp_mask = timestamps.notna()

        valid_timestamps = timestamps[
            valid_timestamp_mask
        ].to_numpy(
            dtype=float
        )

        total_rows = len(
            data
        )

        valid_timestamp_count = len(
            valid_timestamps
        )

        invalid_timestamp_count = (
            total_rows
            - valid_timestamp_count
        )

        # Compute intervals

        intervals = np.diff(
            valid_timestamps
        )

        duplicate_mask = (
            intervals == 0.0
        )

        backward_mask = (
            intervals < 0.0
        )

        positive_intervals = intervals[
            intervals > 0.0
        ]

        duplicate_count = int(
            np.sum(
                duplicate_mask
            )
        )

        backward_count = int(
            np.sum(
                backward_mask
            )
        )

        result = AnalysisResult(
            tool_id=self.metadata.tool_id,
            tool_name=self.metadata.name,
            tool_version=self.metadata.version,
            status=ExecutionStatus.SUCCESS,
        )

        # Handle pathological case

        if len(positive_intervals) == 0:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "No positive timestamp intervals were found. "
                "Sampling statistics cannot be calculated."
            )

            return result

        # Main timing statistics

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

        interval_std = float(
            np.std(
                positive_intervals,
                ddof=0,
            )
        )

        min_interval = float(
            np.min(
                positive_intervals
            )
        )

        max_interval = float(
            np.max(
                positive_intervals
            )
        )

        sampling_rate_hz = (
            1.0 / median_interval
        )

        duration_seconds = float(
            valid_timestamps[-1]
            - valid_timestamps[0]
        )

        coefficient_of_variation = (
            interval_std / mean_interval
            if mean_interval > 0.0
            else None
        )

        # Timing jitter
        #
        # Jitter is measured relative to the median interval,
        # which is more robust to occasional large gaps.

        interval_deviation = (
            positive_intervals
            - median_interval
        )

        rms_jitter_seconds = float(
            np.sqrt(
                np.mean(
                    interval_deviation ** 2
                )
            )
        )

        mean_absolute_jitter_seconds = float(
            np.mean(
                np.abs(
                    interval_deviation
                )
            )
        )

        max_absolute_jitter_seconds = float(
            np.max(
                np.abs(
                    interval_deviation
                )
            )
        )

        # Large-gap analysis

        large_gap_multiplier = float(
            context.parameters[
                "large_gap_multiplier"
            ]
        )

        large_gap_threshold = (
            median_interval
            * large_gap_multiplier
        )

        large_gap_mask = (
            positive_intervals
            > large_gap_threshold
        )

        large_gaps = positive_intervals[
            large_gap_mask
        ]

        large_gap_count = int(
            len(
                large_gaps
            )
        )

        largest_gap_seconds = (
            float(
                np.max(
                    large_gaps
                )
            )
            if large_gap_count > 0
            else None
        )

        # Estimated missing samples
        #
        # For each positive interval, estimate the number of
        # expected nominal intervals that fit inside it.
        #
        # Example:
        #
        # nominal dt = 0.005 s
        # observed dt = 0.020 s
        #
        # 0.020 / 0.005 = 4 intervals
        #
        # Therefore approximately 3 intermediate samples
        # may be missing.

        interval_ratios = (
            positive_intervals
            / median_interval
        )

        estimated_interval_counts = np.rint(
            interval_ratios
        ).astype(
            int
        )

        estimated_interval_counts = np.maximum(
            estimated_interval_counts,
            1,
        )

        estimated_missing_per_gap = np.maximum(
            estimated_interval_counts - 1,
            0,
        )

        estimated_missing_samples = int(
            np.sum(
                estimated_missing_per_gap
            )
        )

        # Effective completeness estimate

        expected_sample_count = (
            valid_timestamp_count
            + estimated_missing_samples
        )

        estimated_capture_fraction = (
            valid_timestamp_count
            / expected_sample_count
            if expected_sample_count > 0
            else None
        )

        # Nominal sampling-rate comparison

        nominal_sampling_rate_hz = (
            self._get_nominal_sampling_rate(
                context
            )
        )

        sampling_rate_error_percent = None

        if (
            nominal_sampling_rate_hz is not None
            and nominal_sampling_rate_hz > 0.0
        ):

            sampling_rate_error_percent = (
                (
                    sampling_rate_hz
                    - nominal_sampling_rate_hz
                )
                / nominal_sampling_rate_hz
                * 100.0
            )

        # Add metrics

        result.add_metric(
            name="Valid Timestamp Count",
            value=valid_timestamp_count,
            unit=None,
            description=(
                "Number of valid timestamps used for timing analysis."
            ),
        )

        result.add_metric(
            name="Recording Duration",
            value=duration_seconds,
            unit="s",
            description=(
                "Elapsed time between the first and last valid timestamp."
            ),
        )

        result.add_metric(
            name="Estimated Sampling Rate",
            value=sampling_rate_hz,
            unit="Hz",
            description=(
                "Sampling frequency estimated from the median "
                "positive timestamp interval."
            ),
        )

        result.add_metric(
            name="Median Sample Interval",
            value=median_interval,
            unit="s",
            description=(
                "Median positive interval between consecutive timestamps."
            ),
        )

        result.add_metric(
            name="Mean Sample Interval",
            value=mean_interval,
            unit="s",
            description=(
                "Mean positive interval between consecutive timestamps."
            ),
        )

        result.add_metric(
            name="Sample Interval Standard Deviation",
            value=interval_std,
            unit="s",
            description=(
                "Standard deviation of positive sample intervals."
            ),
        )

        result.add_metric(
            name="Minimum Sample Interval",
            value=min_interval,
            unit="s",
        )

        result.add_metric(
            name="Maximum Sample Interval",
            value=max_interval,
            unit="s",
        )

        if coefficient_of_variation is not None:

            result.add_metric(
                name="Sample Interval Coefficient of Variation",
                value=coefficient_of_variation,
                unit=None,
                description=(
                    "Standard deviation of sample interval divided "
                    "by mean sample interval."
                ),
            )

        result.add_metric(
            name="RMS Timing Jitter",
            value=rms_jitter_seconds,
            unit="s",
            description=(
                "RMS deviation of positive sample intervals from "
                "the median sample interval."
            ),
        )

        result.add_metric(
            name="Mean Absolute Timing Jitter",
            value=mean_absolute_jitter_seconds,
            unit="s",
        )

        result.add_metric(
            name="Maximum Absolute Timing Jitter",
            value=max_absolute_jitter_seconds,
            unit="s",
        )

        result.add_metric(
            name="Duplicate Timestamp Count",
            value=duplicate_count,
            unit=None,
        )

        result.add_metric(
            name="Backward Timestamp Count",
            value=backward_count,
            unit=None,
        )

        result.add_metric(
            name="Large Gap Count",
            value=large_gap_count,
            unit=None,
        )

        if largest_gap_seconds is not None:

            result.add_metric(
                name="Largest Time Gap",
                value=largest_gap_seconds,
                unit="s",
            )

        result.add_metric(
            name="Estimated Missing Samples",
            value=estimated_missing_samples,
            unit=None,
            description=(
                "Approximate number of missing samples inferred "
                "from timestamp gaps relative to the median sample interval."
            ),
        )

        if estimated_capture_fraction is not None:

            result.add_metric(
                name="Estimated Capture Fraction",
                value=(
                    estimated_capture_fraction
                    * 100.0
                ),
                unit="%",
                description=(
                    "Approximate fraction of expected samples "
                    "present in the recording."
                ),
            )

        if nominal_sampling_rate_hz is not None:

            result.add_metric(
                name="Nominal Sampling Rate",
                value=nominal_sampling_rate_hz,
                unit="Hz",
            )

        if sampling_rate_error_percent is not None:

            result.add_metric(
                name="Sampling Rate Error",
                value=sampling_rate_error_percent,
                unit="%",
                description=(
                    "Relative difference between measured and "
                    "nominal sampling frequency."
                ),
            )

        # Warnings and engineering observations

        if invalid_timestamp_count > 0:

            result.add_warning(
                f"{invalid_timestamp_count} rows contain invalid "
                "or missing timestamps and were excluded from "
                "timing calculations."
            )

        if duplicate_count > 0:

            result.add_warning(
                f"{duplicate_count} zero-length timestamp intervals "
                "were detected."
            )

        if backward_count > 0:

            result.add_warning(
                f"{backward_count} timestamp intervals move "
                "backward in time."
            )

        if large_gap_count > 0:

            result.add_warning(
                f"{large_gap_count} intervals exceed "
                f"{large_gap_multiplier:.2f} times the median "
                "sample interval."
            )

        irregular_threshold = float(
            context.parameters[
                "irregular_sampling_cv_threshold"
            ]
        )

        if (
            coefficient_of_variation is not None
            and coefficient_of_variation
            > irregular_threshold
        ):

            result.add_warning(
                "Sampling intervals are more irregular than the "
                "configured threshold. "
                f"CV = {coefficient_of_variation:.5f}, "
                f"threshold = {irregular_threshold:.5f}."
            )

        nominal_rate_warning_percent = float(
            context.parameters[
                "nominal_rate_warning_percent"
            ]
        )

        if (
            sampling_rate_error_percent is not None
            and abs(
                sampling_rate_error_percent
            )
            > nominal_rate_warning_percent
        ):

            result.add_warning(
                "Measured sampling frequency differs from the "
                "nominal sampling frequency by "
                f"{sampling_rate_error_percent:.3f}%."
            )

        # Evidence for later diagnostic reasoning

        if backward_count > 0:

            result.add_evidence(
                statement=(
                    "Timestamp ordering contains backward time steps, "
                    "which may indicate logger resets, corrupted timing, "
                    "or merged acquisition segments."
                ),
                strength=self._moderate_evidence(),
                supporting_metrics=[
                    "Backward Timestamp Count",
                ],
            )

        if large_gap_count > 0:

            result.add_evidence(
                statement=(
                    "The timestamp sequence contains unusually large "
                    "sampling gaps relative to the typical sample period."
                ),
                strength=self._moderate_evidence(),
                supporting_metrics=[
                    "Large Gap Count",
                    "Largest Time Gap",
                    "Estimated Missing Samples",
                ],
            )

        if (
            estimated_missing_samples == 0
            and duplicate_count == 0
            and backward_count == 0
            and (
                coefficient_of_variation is None
                or coefficient_of_variation
                <= irregular_threshold
            )
        ):

            result.add_evidence(
                statement=(
                    "Timestamp behavior is consistent with a "
                    "continuous and regular acquisition."
                ),
                strength=self._strong_evidence(),
                supporting_metrics=[
                    "Estimated Sampling Rate",
                    "Sample Interval Coefficient of Variation",
                ],
            )

        # Metadata for reproducibility

        result.metadata.update(
            {
                "timestamp_column":
                    timestamp_column,

                "timestamp_unit":
                    context.units.get(
                        "timestamp"
                    ),

                "large_gap_multiplier":
                    large_gap_multiplier,

                "large_gap_threshold_seconds":
                    large_gap_threshold,

                "irregular_sampling_cv_threshold":
                    irregular_threshold,

                "nominal_rate_warning_percent":
                    nominal_rate_warning_percent,

                "total_rows":
                    total_rows,

                "valid_timestamp_count":
                    valid_timestamp_count,

                "invalid_timestamp_count":
                    invalid_timestamp_count,
            }
        )

        return result

    # Metadata helper

    @staticmethod
    def _get_nominal_sampling_rate(
        context: ToolContext,
    ) -> float | None:
        """Retrieve the nominal sampling frequency from SensorQA
        metadata when available.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Numeric result.
        """

        metadata = context.metadata.get(
            "sensorqa_metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        sensor_metadata = metadata.get(
            "sensor"
        )

        if not isinstance(
            sensor_metadata,
            dict,
        ):
            return None

        nominal_rate = sensor_metadata.get(
            "nominal_sampling_rate_hz"
        )

        if nominal_rate is None:
            return None

        try:

            nominal_rate = float(
                nominal_rate
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

        if nominal_rate <= 0.0:
            return None

        return nominal_rate

    # Evidence helpers

    @staticmethod
    def _moderate_evidence():
        """Import locally to keep the main import section focused.
        
        Returns:
            Result returned by the function.
        """

        from sensorqa.core.result_schema import (
            EvidenceStrength,
        )

        return EvidenceStrength.MODERATE

    @staticmethod
    def _strong_evidence():

        """Return strong evidence.
        
        Returns:
            Calculated value.
        """
        from sensorqa.core.result_schema import (
            EvidenceStrength,
        )

        return EvidenceStrength.STRONG
