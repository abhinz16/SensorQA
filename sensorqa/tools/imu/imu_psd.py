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



def _trapezoidal_integral(
    values: np.ndarray,
    coordinates: np.ndarray,
) -> float:
    """Integrate values using the NumPy API available at runtime.

    Args:
        values: Samples to integrate.
        coordinates: Coordinate values associated with ``values``.

    Returns:
        The trapezoidal numerical integral as a float.
    """

    trapezoid = getattr(np, "trapezoid", None)

    if callable(trapezoid):
        return float(trapezoid(values, coordinates))

    return float(np.trapz(values, coordinates))


class IMUPSDTool(BaseAnalysisTool):
    """
    Frequency-domain characterization of accelerometer and
    gyroscope signals using Welch power spectral density.

    The tool reports:

        - sampling frequency
        - frequency resolution
        - Welch segment count
        - PSD and amplitude spectral density curves
        - dominant non-DC spectral peak
        - strongest narrowband peaks
        - integrated RMS from the PSD
        - median spectral density

    Important:
        A spectral peak is an observation.

        SensorQA does not automatically label a peak as:
            - mechanical vibration
            - electrical interference
            - resonance
            - motor harmonics
            - aliasing

        Those interpretations require additional evidence.
    """

    @property
    def metadata(self) -> ToolMetadata:

        return ToolMetadata(
            tool_id="imu_psd",
            name="IMU Power Spectral Density",
            version="1.0.0",
            description=(
                "Computes Welch power spectral density for "
                "accelerometer and gyroscope channels and identifies "
                "prominent narrowband spectral features."
            ),
            category=ToolCategory.FREQUENCY,
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
                    name="segment_length",
                    parameter_type=ParameterType.INTEGER,
                    default=2048,
                    description=(
                        "Number of samples in each Welch segment."
                    ),
                    minimum=32,
                    maximum=1_048_576,
                ),
                ToolParameter(
                    name="overlap_fraction",
                    parameter_type=ParameterType.FLOAT,
                    default=0.50,
                    description=(
                        "Fractional overlap between adjacent "
                        "Welch segments."
                    ),
                    minimum=0.0,
                    maximum=0.95,
                ),
                ToolParameter(
                    name="window",
                    parameter_type=ParameterType.CHOICE,
                    default="hann",
                    description=(
                        "Window function applied to each "
                        "Welch segment."
                    ),
                    choices=[
                        "hann",
                        "hamming",
                        "rectangular",
                    ],
                ),
                ToolParameter(
                    name="minimum_frequency_hz",
                    parameter_type=ParameterType.FLOAT,
                    default=0.1,
                    description=(
                        "Frequencies below this value are excluded "
                        "when searching for dominant spectral peaks."
                    ),
                    unit="Hz",
                    minimum=0.0,
                    maximum=1_000_000.0,
                ),
                ToolParameter(
                    name="peak_prominence_ratio",
                    parameter_type=ParameterType.FLOAT,
                    default=10.0,
                    description=(
                        "A local PSD maximum must exceed the median "
                        "PSD level by at least this ratio to be "
                        "reported as a prominent spectral peak."
                    ),
                    minimum=1.0,
                    maximum=1_000_000.0,
                ),
                ToolParameter(
                    name="max_reported_peaks",
                    parameter_type=ParameterType.INTEGER,
                    default=5,
                    description=(
                        "Maximum number of prominent peaks reported "
                        "for each axis."
                    ),
                    minimum=1,
                    maximum=50,
                ),
                ToolParameter(
                    name="max_sampling_cv",
                    parameter_type=ParameterType.FLOAT,
                    default=0.02,
                    description=(
                        "Maximum allowed coefficient of variation "
                        "of positive sample intervals before "
                        "uniform-sampling PSD assumptions are "
                        "considered violated."
                    ),
                    minimum=0.0,
                    maximum=1.0,
                ),
                ToolParameter(
                    name="allow_irregular_sampling",
                    parameter_type=ParameterType.BOOLEAN,
                    default=False,
                    description=(
                        "Allow PSD calculation despite irregular "
                        "sampling. SensorQA does not resample the "
                        "data when this override is enabled."
                    ),
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
        """
        Check channel availability and sampling suitability.
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
                "is available for PSD analysis."
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

        sampling = self._sampling_information(
            timestamps
        )

        if sampling is None:

            validation.errors.append(
                "SensorQA could not determine a valid "
                "sampling interval."
            )

            validation.valid = False
            return validation

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

        if (
            sampling["coefficient_of_variation"]
            > max_sampling_cv
        ):

            if allow_irregular:

                validation.warnings.append(
                    "Sampling irregularity exceeds the configured "
                    "PSD threshold, but execution was explicitly "
                    "allowed. SensorQA will not resample the data."
                )

            else:

                validation.errors.append(
                    "Sampling intervals are too irregular for "
                    "standard uniformly sampled Welch PSD analysis. "
                    f"Interval CV = "
                    f"{sampling['coefficient_of_variation']:.6f}, "
                    f"allowed maximum = "
                    f"{max_sampling_cv:.6f}."
                )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """
        Perform Welch PSD analysis.
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

        sampling = self._sampling_information(
            timestamps
        )

        if sampling is None:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "Sampling characteristics could not be determined."
            )

            return result

        sample_interval = sampling[
            "median_interval"
        ]

        sampling_rate = (
            1.0 / sample_interval
        )

        nyquist_frequency = (
            sampling_rate / 2.0
        )

        segment_length_requested = int(
            context.parameters[
                "segment_length"
            ]
        )

        overlap_fraction = float(
            context.parameters[
                "overlap_fraction"
            ]
        )

        window_name = str(
            context.parameters[
                "window"
            ]
        )

        minimum_frequency = float(
            context.parameters[
                "minimum_frequency_hz"
            ]
        )

        peak_prominence_ratio = float(
            context.parameters[
                "peak_prominence_ratio"
            ]
        )

        max_reported_peaks = int(
            context.parameters[
                "max_reported_peaks"
            ]
        )

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        spectra: dict[
            str,
            dict[str, object],
        ] = {}

        processed_axis_count = 0

        # =========================================================
        # Process each sensor group
        # =========================================================

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

            # -----------------------------------------------------
            # Use the longest continuous valid XYZ segment.
            #
            # We do not drop isolated rows and concatenate the
            # remaining pieces because doing so changes timing.
            # -----------------------------------------------------

            segment = self._longest_continuous_group_segment(
                timestamps=timestamps,
                group_values=group_values,
                expected_dt=sample_interval,
            )

            if segment is None:

                result.add_warning(
                    f"No continuous {group_name} segment is "
                    "available for PSD analysis."
                )

                continue

            start_index, end_index = segment

            segment_sample_count = (
                end_index - start_index
            )

            if segment_sample_count < 32:

                result.add_warning(
                    f"The longest continuous {group_name} segment "
                    f"contains only {segment_sample_count} samples, "
                    "which is insufficient for meaningful PSD "
                    "analysis."
                )

                continue

            segment_length = min(
                segment_length_requested,
                segment_sample_count,
            )

            # Use an even segment length where possible.
            if (
                segment_length > 32
                and segment_length % 2 != 0
            ):
                segment_length -= 1

            overlap_samples = int(
                round(
                    segment_length
                    * overlap_fraction
                )
            )

            overlap_samples = min(
                overlap_samples,
                segment_length - 1,
            )

            step_size = (
                segment_length
                - overlap_samples
            )

            segment_count = (
                1
                + (
                    segment_sample_count
                    - segment_length
                )
                // step_size
            )

            if segment_count < 1:

                result.add_warning(
                    f"No valid Welch segments could be constructed "
                    f"for {group_name}."
                )

                continue

            frequency_resolution = (
                sampling_rate
                / segment_length
            )

            group_display_name = (
                "Accelerometer"
                if group_name == "accelerometer"
                else "Gyroscope"
            )

            result.add_metric(
                name=(
                    f"{group_display_name} PSD Segment Sample Count"
                ),
                value=segment_sample_count,
                unit=None,
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Welch Segment Length"
                ),
                value=segment_length,
                unit="samples",
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Welch Segment Count"
                ),
                value=segment_count,
                unit=None,
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Frequency Resolution"
                ),
                value=frequency_resolution,
                unit="Hz",
            )

            # =====================================================
            # Analyze individual axes
            # =====================================================

            for axis in axes:

                values = group_values[
                    axis
                ][
                    start_index:end_index
                ]

                frequencies, psd = self._welch_psd(
                    values=values,
                    sampling_rate=sampling_rate,
                    segment_length=segment_length,
                    overlap_samples=overlap_samples,
                    window_name=window_name,
                )

                if (
                    len(frequencies) < 2
                    or len(psd) < 2
                ):

                    result.add_warning(
                        f"{axis.upper()} PSD could not be "
                        "calculated."
                    )

                    continue

                amplitude_spectral_density = np.sqrt(
                    np.maximum(
                        psd,
                        0.0,
                    )
                )

                axis_unit = context.units.get(
                    axis
                )

                psd_unit = self._psd_unit(
                    axis_unit
                )

                asd_unit = self._asd_unit(
                    axis_unit
                )

                axis_label = axis.upper()

                # -------------------------------------------------
                # Integrated spectral RMS
                # -------------------------------------------------

                spectral_variance = float(
                    _trapezoidal_integral(
                        psd,
                        frequencies,
                    )
                )

                spectral_variance = max(
                    spectral_variance,
                    0.0,
                )

                spectral_rms = float(
                    np.sqrt(
                        spectral_variance
                    )
                )

                result.add_metric(
                    name=(
                        f"{axis_label} PSD Integrated RMS"
                    ),
                    value=spectral_rms,
                    unit=axis_unit,
                    description=(
                        "RMS signal variation estimated by "
                        "integrating the one-sided PSD."
                    ),
                )

                # -------------------------------------------------
                # Median broadband spectral density
                # -------------------------------------------------

                non_dc_mask = (
                    frequencies
                    >= minimum_frequency
                )

                non_dc_psd = psd[
                    non_dc_mask
                ]

                non_dc_frequencies = frequencies[
                    non_dc_mask
                ]

                if len(
                    non_dc_psd
                ) == 0:

                    result.add_warning(
                        f"{axis_label} contains no PSD bins at or "
                        "above the configured minimum frequency."
                    )

                    continue

                positive_psd = non_dc_psd[
                    non_dc_psd > 0
                ]

                median_psd = (
                    float(
                        np.median(
                            positive_psd
                        )
                    )
                    if len(
                        positive_psd
                    ) > 0
                    else 0.0
                )

                median_asd = float(
                    np.sqrt(
                        max(
                            median_psd,
                            0.0,
                        )
                    )
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Median Spectral Density"
                    ),
                    value=median_psd,
                    unit=psd_unit,
                    description=(
                        "Median non-DC Welch PSD level above the "
                        "configured minimum analysis frequency."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Median Amplitude "
                        "Spectral Density"
                    ),
                    value=median_asd,
                    unit=asd_unit,
                )

                # -------------------------------------------------
                # Dominant non-DC frequency
                # -------------------------------------------------

                dominant_index = int(
                    np.argmax(
                        non_dc_psd
                    )
                )

                dominant_frequency = float(
                    non_dc_frequencies[
                        dominant_index
                    ]
                )

                dominant_psd = float(
                    non_dc_psd[
                        dominant_index
                    ]
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Dominant Spectral Frequency"
                    ),
                    value=dominant_frequency,
                    unit="Hz",
                    description=(
                        "Frequency of the largest PSD value above "
                        "the configured minimum frequency."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Dominant Spectral PSD"
                    ),
                    value=dominant_psd,
                    unit=psd_unit,
                )

                if median_psd > 0:

                    dominant_ratio = (
                        dominant_psd
                        / median_psd
                    )

                    result.add_metric(
                        name=(
                            f"{axis_label} Dominant Peak-to-Median "
                            "PSD Ratio"
                        ),
                        value=float(
                            dominant_ratio
                        ),
                        unit=None,
                    )

                else:

                    dominant_ratio = np.inf

                # -------------------------------------------------
                # Narrowband peak detection
                # -------------------------------------------------

                peaks = self._find_prominent_peaks(
                    frequencies=non_dc_frequencies,
                    psd=non_dc_psd,
                    median_psd=median_psd,
                    minimum_ratio=peak_prominence_ratio,
                    max_peaks=max_reported_peaks,
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Prominent Spectral Peak Count"
                    ),
                    value=len(
                        peaks
                    ),
                    unit=None,
                )

                for peak_number, peak in enumerate(
                    peaks,
                    start=1,
                ):

                    result.add_metric(
                        name=(
                            f"{axis_label} Spectral Peak "
                            f"{peak_number} Frequency"
                        ),
                        value=peak[
                            "frequency_hz"
                        ],
                        unit="Hz",
                    )

                    result.add_metric(
                        name=(
                            f"{axis_label} Spectral Peak "
                            f"{peak_number} PSD"
                        ),
                        value=peak[
                            "psd"
                        ],
                        unit=psd_unit,
                    )

                    result.add_metric(
                        name=(
                            f"{axis_label} Spectral Peak "
                            f"{peak_number} Ratio"
                        ),
                        value=peak[
                            "median_ratio"
                        ],
                        unit=None,
                        description=(
                            "Peak PSD divided by the median "
                            "non-DC PSD level."
                        ),
                    )

                if peaks:

                    result.add_evidence(
                        statement=(
                            f"{axis_label} contains one or more "
                            "prominent narrowband spectral features "
                            "above the surrounding broadband PSD "
                            "level."
                        ),
                        strength=EvidenceStrength.MODERATE,
                        supporting_metrics=[
                            (
                                f"{axis_label} Prominent "
                                "Spectral Peak Count"
                            ),
                            (
                                f"{axis_label} Dominant "
                                "Spectral Frequency"
                            ),
                        ],
                    )

                # -------------------------------------------------
                # Save complete spectral curves
                # -------------------------------------------------

                spectra[
                    axis
                ] = {
                    "frequency_hz":
                        frequencies.tolist(),

                    "psd":
                        psd.tolist(),

                    "amplitude_spectral_density":
                        amplitude_spectral_density.tolist(),

                    "psd_unit":
                        psd_unit,

                    "asd_unit":
                        asd_unit,

                    "prominent_peaks":
                        peaks,
                }

                processed_axis_count += 1

        # =========================================================
        # Check whether anything succeeded
        # =========================================================

        if processed_axis_count == 0:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "No IMU axis produced a valid PSD result."
            )

            return result

        # =========================================================
        # Sampling-quality interpretation
        # =========================================================

        max_sampling_cv = float(
            context.parameters[
                "max_sampling_cv"
            ]
        )

        if (
            sampling[
                "coefficient_of_variation"
            ]
            > max_sampling_cv
        ):

            result.add_warning(
                "PSD was calculated despite sampling irregularity "
                "because the advanced override was enabled. "
                "SensorQA did not resample the signal."
            )

            result.add_evidence(
                statement=(
                    "Sampling irregularity exceeds the normal "
                    "uniform-sampling PSD threshold, reducing "
                    "confidence in exact spectral-frequency "
                    "interpretation."
                ),
                strength=EvidenceStrength.WEAK,
                supporting_metrics=[],
            )

        # =========================================================
        # Global metrics
        # =========================================================

        result.add_metric(
            name="PSD Estimated Sampling Rate",
            value=sampling_rate,
            unit="Hz",
        )

        result.add_metric(
            name="PSD Nyquist Frequency",
            value=nyquist_frequency,
            unit="Hz",
            description=(
                "Highest frequency representable under the "
                "uniform-sampling assumption."
            ),
        )

        # =========================================================
        # Reproducibility metadata
        # =========================================================

        result.metadata.update(
            {
                "method":
                    "welch",

                "input_resampled":
                    False,

                "sampling_rate_hz":
                    sampling_rate,

                "sample_interval_seconds":
                    sample_interval,

                "sampling_interval_cv":
                    sampling[
                        "coefficient_of_variation"
                    ],

                "requested_segment_length":
                    segment_length_requested,

                "overlap_fraction":
                    overlap_fraction,

                "window":
                    window_name,

                "minimum_frequency_hz":
                    minimum_frequency,

                "peak_prominence_ratio":
                    peak_prominence_ratio,

                "max_reported_peaks":
                    max_reported_peaks,

                "spectra":
                    spectra,
            }
        )

        return result

    # =============================================================
    # Welch PSD
    # =============================================================

    @staticmethod
    def _welch_psd(
        values: np.ndarray,
        sampling_rate: float,
        segment_length: int,
        overlap_samples: int,
        window_name: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Compute a one-sided Welch PSD using NumPy only.

        Each segment has its mean removed before windowing.
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        step_size = (
            segment_length
            - overlap_samples
        )

        if step_size <= 0:

            raise ValueError(
                "Welch step size must be positive."
            )

        window = IMUPSDTool._window(
            window_name,
            segment_length,
        )

        window_power = float(
            np.sum(
                window ** 2
            )
        )

        if window_power <= 0:

            raise ValueError(
                "Window power must be positive."
            )

        accumulated_psd = None
        segment_count = 0

        start = 0

        while (
            start + segment_length
            <= len(values)
        ):

            segment = values[
                start:
                start + segment_length
            ].copy()

            # Remove DC independently from every Welch segment.
            segment -= np.mean(
                segment
            )

            segment *= window

            spectrum = np.fft.rfft(
                segment
            )

            segment_psd = (
                np.abs(
                    spectrum
                ) ** 2
                / (
                    sampling_rate
                    * window_power
                )
            )

            # Convert two-sided energy into a one-sided PSD.
            if segment_length % 2 == 0:

                if len(
                    segment_psd
                ) > 2:

                    segment_psd[
                        1:-1
                    ] *= 2.0

            else:

                if len(
                    segment_psd
                ) > 1:

                    segment_psd[
                        1:
                    ] *= 2.0

            if accumulated_psd is None:

                accumulated_psd = (
                    segment_psd
                )

            else:

                accumulated_psd += (
                    segment_psd
                )

            segment_count += 1

            start += (
                step_size
            )

        if (
            accumulated_psd is None
            or segment_count == 0
        ):

            return (
                np.array(
                    [],
                    dtype=float,
                ),
                np.array(
                    [],
                    dtype=float,
                ),
            )

        psd = (
            accumulated_psd
            / segment_count
        )

        frequencies = np.fft.rfftfreq(
            segment_length,
            d=(
                1.0
                / sampling_rate
            ),
        )

        return (
            frequencies,
            psd,
        )

    # =============================================================
    # Windows
    # =============================================================

    @staticmethod
    def _window(
        name: str,
        length: int,
    ) -> np.ndarray:

        if name == "hann":

            return np.hanning(
                length
            )

        if name == "hamming":

            return np.hamming(
                length
            )

        if name == "rectangular":

            return np.ones(
                length,
                dtype=float,
            )

        raise ValueError(
            f"Unsupported PSD window '{name}'."
        )

    # =============================================================
    # Peak detection
    # =============================================================

    @staticmethod
    def _find_prominent_peaks(
        frequencies: np.ndarray,
        psd: np.ndarray,
        median_psd: float,
        minimum_ratio: float,
        max_peaks: int,
    ) -> list[
        dict[str, float]
    ]:
        """
        Identify simple local PSD maxima substantially above
        the median spectral level.

        This intentionally avoids assigning physical causes.
        """

        if len(
            psd
        ) < 3:

            return []

        if median_psd <= 0:

            return []

        candidates: list[
            dict[str, float]
        ] = []

        for index in range(
            1,
            len(psd) - 1,
        ):

            current = psd[
                index
            ]

            is_local_maximum = (
                current
                > psd[
                    index - 1
                ]
                and current
                >= psd[
                    index + 1
                ]
            )

            if not is_local_maximum:
                continue

            ratio = (
                current
                / median_psd
            )

            if ratio < minimum_ratio:
                continue

            candidates.append(
                {
                    "frequency_hz":
                        float(
                            frequencies[
                                index
                            ]
                        ),

                    "psd":
                        float(
                            current
                        ),

                    "median_ratio":
                        float(
                            ratio
                        ),
                }
            )

        candidates.sort(
            key=lambda item:
                item[
                    "psd"
                ],
            reverse=True,
        )

        return candidates[
            :max_peaks
        ]

    # =============================================================
    # Sampling information
    # =============================================================

    @staticmethod
    def _sampling_information(
        timestamps: np.ndarray,
    ) -> dict[
        str,
        float,
    ] | None:

        valid_timestamps = timestamps[
            np.isfinite(
                timestamps
            )
        ]

        if len(
            valid_timestamps
        ) < 3:

            return None

        intervals = np.diff(
            valid_timestamps
        )

        positive = intervals[
            intervals > 0
        ]

        if len(
            positive
        ) < 2:

            return None

        median_interval = float(
            np.median(
                positive
            )
        )

        mean_interval = float(
            np.mean(
                positive
            )
        )

        std_interval = float(
            np.std(
                positive
            )
        )

        cv = (
            std_interval
            / mean_interval
            if mean_interval > 0
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
                cv,
        }

    # =============================================================
    # Continuous-segment handling
    # =============================================================

    @staticmethod
    def _longest_continuous_group_segment(
        timestamps: np.ndarray,
        group_values: dict[
            str,
            np.ndarray,
        ],
        expected_dt: float,
    ) -> tuple[
        int,
        int,
    ] | None:
        """
        Return longest contiguous valid region.

        Continuity requires:

            valid timestamp
            valid XYZ samples
            positive timestamp progression
            dt <= 1.5 * expected_dt

        End index is exclusive.
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
                and dt > 0
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

    # =============================================================
    # Sensor-group discovery
    # =============================================================

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

    # =============================================================
    # Unit helpers
    # =============================================================

    @staticmethod
    def _psd_unit(
        input_unit: str | None,
    ) -> str | None:

        if input_unit is None:
            return None

        return (
            f"({input_unit})^2/Hz"
        )

    @staticmethod
    def _asd_unit(
        input_unit: str | None,
    ) -> str | None:

        if input_unit is None:
            return None

        return (
            f"{input_unit}/sqrt(Hz)"
        )