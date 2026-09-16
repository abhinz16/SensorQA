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


class SaturationTool(BaseAnalysisTool):
    """
    Evaluates accelerometer and gyroscope measurements relative
    to configured sensor full-scale ranges.

    The tool distinguishes between:

        exact-limit observations
        near-limit observations
        values outside the configured range
        contiguous limit-contact runs
        upper/lower limit asymmetry
        overall configured-range utilization

    Important:
        Operating near a sensor limit is not automatically the same
        as clipping.

        Repeated values at a configured boundary provide stronger
        evidence of clipping/saturation than values merely close
        to that boundary.
    """

    STANDARD_GRAVITY = 9.80665

    @property
    def metadata(self) -> ToolMetadata:

        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="imu_saturation",
            name="IMU Saturation",
            version="1.0.0",
            description=(
                "Evaluates accelerometer and gyroscope data against "
                "configured full-scale ranges and identifies exact "
                "boundary hits, near-limit operation, contiguous "
                "clipping runs, and out-of-range observations."
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
                    name="near_limit_margin_fraction",
                    parameter_type=ParameterType.FLOAT,
                    default=0.005,
                    description=(
                        "Fraction of the complete configured sensor "
                        "span treated as the near-limit region at "
                        "each boundary."
                    ),
                    minimum=0.0,
                    maximum=0.25,
                ),
                ToolParameter(
                    name="exact_limit_tolerance_fraction",
                    parameter_type=ParameterType.FLOAT,
                    default=1e-6,
                    description=(
                        "Numerical tolerance, expressed as a fraction "
                        "of full configured span, used when deciding "
                        "whether a measurement coincides with a "
                        "configured full-scale boundary."
                    ),
                    minimum=0.0,
                    maximum=0.01,
                ),
                ToolParameter(
                    name="minimum_clipping_run_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=3,
                    description=(
                        "Minimum number of consecutive exact-limit "
                        "samples used as stronger evidence of clipping."
                    ),
                    minimum=1,
                    maximum=100_000,
                ),
                ToolParameter(
                    name="minimum_asymmetry_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=5,
                    description=(
                        "Minimum number of near-limit samples required "
                        "before upper-versus-lower limit asymmetry is "
                        "evaluated."
                    ),
                    minimum=1,
                    maximum=1_000_000,
                ),
                ToolParameter(
                    name="asymmetry_fraction_threshold",
                    parameter_type=ParameterType.FLOAT,
                    default=0.90,
                    description=(
                        "Fraction of near-limit samples that must occur "
                        "on one boundary before limit interaction is "
                        "reported as strongly asymmetric."
                    ),
                    minimum=0.50,
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
        """Verify that usable channels and corresponding full-scale
        metadata are available.
        
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
                "is available for saturation analysis."
            )

            validation.valid = False
            return validation

        usable_group_count = 0

        for group_name, axes in available_groups.items():

            range_metadata = self._get_range_metadata(
                context=context,
                group_name=group_name,
            )

            if range_metadata is None:

                validation.warnings.append(
                    f"No configured full-scale range is available "
                    f"for the {group_name}. That sensor group will "
                    "be skipped."
                )

                continue

            target_unit = context.units.get(
                axes[0]
            )

            try:

                minimum, maximum = self._convert_range(
                    group_name=group_name,
                    range_metadata=range_metadata,
                    target_unit=target_unit,
                )

            except ValueError as exc:

                validation.warnings.append(
                    f"{group_name.capitalize()} range could not be "
                    f"interpreted: {exc}"
                )

                continue

            if minimum >= maximum:

                validation.warnings.append(
                    f"{group_name.capitalize()} full-scale minimum "
                    "must be less than its maximum."
                )

                continue

            usable_group_count += 1

        if usable_group_count == 0:

            validation.errors.append(
                "No available sensor group has usable full-scale "
                "metadata. Saturation analysis cannot proceed."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform full-scale and clipping analysis.
        
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

        near_fraction = float(
            context.parameters[
                "near_limit_margin_fraction"
            ]
        )

        exact_fraction = float(
            context.parameters[
                "exact_limit_tolerance_fraction"
            ]
        )

        minimum_clipping_run = int(
            context.parameters[
                "minimum_clipping_run_samples"
            ]
        )

        minimum_asymmetry_samples = int(
            context.parameters[
                "minimum_asymmetry_samples"
            ]
        )

        asymmetry_threshold = float(
            context.parameters[
                "asymmetry_fraction_threshold"
            ]
        )

        processed_axes = 0

        saturation_summary = {}

        # Process accelerometer / gyroscope groups

        for group_name, axes in available_groups.items():

            range_metadata = self._get_range_metadata(
                context=context,
                group_name=group_name,
            )

            if range_metadata is None:

                result.add_warning(
                    f"{group_name.capitalize()} saturation analysis "
                    "was skipped because no full-scale range was "
                    "provided."
                )

                continue

            target_unit = context.units.get(
                axes[0]
            )

            try:

                configured_minimum, configured_maximum = (
                    self._convert_range(
                        group_name=group_name,
                        range_metadata=range_metadata,
                        target_unit=target_unit,
                    )
                )

            except ValueError as exc:

                result.add_warning(
                    f"{group_name.capitalize()} saturation analysis "
                    f"was skipped: {exc}"
                )

                continue

            configured_span = (
                configured_maximum
                - configured_minimum
            )

            near_margin = (
                near_fraction
                * configured_span
            )

            exact_tolerance = (
                exact_fraction
                * configured_span
            )

            lower_near_threshold = (
                configured_minimum
                + near_margin
            )

            upper_near_threshold = (
                configured_maximum
                - near_margin
            )

            group_display_name = (
                "Accelerometer"
                if group_name == "accelerometer"
                else "Gyroscope"
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Configured Minimum"
                ),
                value=configured_minimum,
                unit=target_unit,
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Configured Maximum"
                ),
                value=configured_maximum,
                unit=target_unit,
            )

            result.add_metric(
                name=(
                    f"{group_display_name} Configured Span"
                ),
                value=configured_span,
                unit=target_unit,
            )

            group_summary = {}

            # Per-axis analysis

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

                finite_mask = np.isfinite(
                    values
                )

                finite_values = values[
                    finite_mask
                ]

                if len(finite_values) == 0:

                    result.add_warning(
                        f"{axis.upper()} contains no valid numerical "
                        "samples for saturation analysis."
                    )

                    continue

                axis_label = axis.upper()

                observed_minimum = float(
                    np.min(
                        finite_values
                    )
                )

                observed_maximum = float(
                    np.max(
                        finite_values
                    )
                )

                observed_span = (
                    observed_maximum
                    - observed_minimum
                )

                range_utilization_percent = (
                    observed_span
                    / configured_span
                    * 100.0
                )

                # Exact boundary masks

                exact_lower = (
                    finite_mask
                    & (
                        np.abs(
                            values
                            - configured_minimum
                        )
                        <= exact_tolerance
                    )
                )

                exact_upper = (
                    finite_mask
                    & (
                        np.abs(
                            values
                            - configured_maximum
                        )
                        <= exact_tolerance
                    )
                )

                exact_any = (
                    exact_lower
                    | exact_upper
                )

                # Near-limit masks

                near_lower = (
                    finite_mask
                    & (
                        values
                        <= lower_near_threshold
                    )
                )

                near_upper = (
                    finite_mask
                    & (
                        values
                        >= upper_near_threshold
                    )
                )

                near_any = (
                    near_lower
                    | near_upper
                )

                near_nonexact = (
                    near_any
                    & ~exact_any
                )

                # Out-of-configured-range samples

                below_range = (
                    finite_mask
                    & (
                        values
                        < (
                            configured_minimum
                            - exact_tolerance
                        )
                    )
                )

                above_range = (
                    finite_mask
                    & (
                        values
                        > (
                            configured_maximum
                            + exact_tolerance
                        )
                    )
                )

                outside_range = (
                    below_range
                    | above_range
                )

                exact_lower_count = int(
                    np.sum(
                        exact_lower
                    )
                )

                exact_upper_count = int(
                    np.sum(
                        exact_upper
                    )
                )

                exact_total_count = (
                    exact_lower_count
                    + exact_upper_count
                )

                near_lower_count = int(
                    np.sum(
                        near_lower
                    )
                )

                near_upper_count = int(
                    np.sum(
                        near_upper
                    )
                )

                near_total_count = int(
                    np.sum(
                        near_any
                    )
                )

                near_nonexact_count = int(
                    np.sum(
                        near_nonexact
                    )
                )

                outside_range_count = int(
                    np.sum(
                        outside_range
                    )
                )

                valid_count = len(
                    finite_values
                )

                exact_fraction_percent = (
                    exact_total_count
                    / valid_count
                    * 100.0
                )

                near_fraction_percent = (
                    near_total_count
                    / valid_count
                    * 100.0
                )

                outside_fraction_percent = (
                    outside_range_count
                    / valid_count
                    * 100.0
                )

                # Consecutive clipping runs

                longest_exact_run = self._longest_true_run(
                    exact_any
                )

                longest_near_run = self._longest_true_run(
                    near_any
                )

                # Add metrics

                result.add_metric(
                    name=f"{axis_label} Observed Minimum",
                    value=observed_minimum,
                    unit=target_unit,
                )

                result.add_metric(
                    name=f"{axis_label} Observed Maximum",
                    value=observed_maximum,
                    unit=target_unit,
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Configured Range Utilization"
                    ),
                    value=range_utilization_percent,
                    unit="%",
                    description=(
                        "Observed signal span divided by configured "
                        "full-scale span. Low utilization is reported "
                        "descriptively and is not automatically "
                        "classified as a fault."
                    ),
                )

                result.add_metric(
                    name=f"{axis_label} Exact Lower Limit Count",
                    value=exact_lower_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Exact Upper Limit Count",
                    value=exact_upper_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Exact Limit Count",
                    value=exact_total_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Exact Limit Fraction",
                    value=exact_fraction_percent,
                    unit="%",
                )

                result.add_metric(
                    name=f"{axis_label} Near Lower Limit Count",
                    value=near_lower_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Near Upper Limit Count",
                    value=near_upper_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Near Limit Count",
                    value=near_total_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Near Limit Fraction",
                    value=near_fraction_percent,
                    unit="%",
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Near-Limit Non-Exact Count"
                    ),
                    value=near_nonexact_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Out-of-Range Count",
                    value=outside_range_count,
                    unit=None,
                )

                result.add_metric(
                    name=f"{axis_label} Out-of-Range Fraction",
                    value=outside_fraction_percent,
                    unit="%",
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Longest Exact-Limit Run"
                    ),
                    value=longest_exact_run,
                    unit="samples",
                    description=(
                        "Longest consecutive sequence of samples "
                        "coinciding with either configured full-scale "
                        "boundary."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Longest Near-Limit Run"
                    ),
                    value=longest_near_run,
                    unit="samples",
                )

                # Exact-limit evidence

                if exact_total_count > 0:

                    if (
                        longest_exact_run
                        >= minimum_clipping_run
                    ):

                        evidence_strength = (
                            EvidenceStrength.STRONG
                        )

                        result.add_warning(
                            f"{axis_label} contains repeated "
                            "measurements at a configured full-scale "
                            f"boundary. Longest run = "
                            f"{longest_exact_run} samples."
                        )

                    else:

                        evidence_strength = (
                            EvidenceStrength.MODERATE
                        )

                        result.add_warning(
                            f"{axis_label} contains "
                            f"{exact_total_count} sample(s) coinciding "
                            "with a configured full-scale boundary."
                        )

                    result.add_evidence(
                        statement=(
                            f"{axis_label} contains measurements "
                            "at the configured full-scale boundary, "
                            "which is consistent with possible "
                            "clipping or saturation."
                        ),
                        strength=evidence_strength,
                        supporting_metrics=[
                            (
                                f"{axis_label} Exact "
                                "Limit Count"
                            ),
                            (
                                f"{axis_label} Longest "
                                "Exact-Limit Run"
                            ),
                        ],
                    )

                # Near-limit operation without exact clipping

                elif near_nonexact_count > 0:

                    result.add_warning(
                        f"{axis_label} operates near a configured "
                        "full-scale boundary but does not contain "
                        "samples that clearly coincide with the "
                        "boundary."
                    )

                    result.add_evidence(
                        statement=(
                            f"{axis_label} operates within the "
                            "configured near-limit margin. This "
                            "indicates limited remaining measurement "
                            "headroom but does not by itself prove "
                            "clipping."
                        ),
                        strength=EvidenceStrength.MODERATE,
                        supporting_metrics=[
                            (
                                f"{axis_label} Near "
                                "Limit Fraction"
                            ),
                            (
                                f"{axis_label} Longest "
                                "Near-Limit Run"
                            ),
                        ],
                    )

                # Values beyond configured limits

                if outside_range_count > 0:

                    result.add_warning(
                        f"{axis_label} contains "
                        f"{outside_range_count} value(s) outside the "
                        "configured full-scale range."
                    )

                    result.add_evidence(
                        statement=(
                            f"{axis_label} includes values beyond "
                            "the configured sensor range. This may "
                            "indicate mismatched range metadata, "
                            "unit/configuration inconsistency, or "
                            "non-nominal reported values and should "
                            "be investigated."
                        ),
                        strength=EvidenceStrength.MODERATE,
                        supporting_metrics=[
                            (
                                f"{axis_label} "
                                "Out-of-Range Count"
                            ),
                            (
                                f"{axis_label} "
                                "Out-of-Range Fraction"
                            ),
                        ],
                    )

                # Positive / negative boundary asymmetry

                if (
                    near_total_count
                    >= minimum_asymmetry_samples
                ):

                    upper_fraction = (
                        near_upper_count
                        / near_total_count
                    )

                    lower_fraction = (
                        near_lower_count
                        / near_total_count
                    )

                    dominant_fraction = max(
                        upper_fraction,
                        lower_fraction,
                    )

                    result.add_metric(
                        name=(
                            f"{axis_label} Upper Near-Limit Share"
                        ),
                        value=(
                            upper_fraction
                            * 100.0
                        ),
                        unit="%",
                    )

                    result.add_metric(
                        name=(
                            f"{axis_label} Lower Near-Limit Share"
                        ),
                        value=(
                            lower_fraction
                            * 100.0
                        ),
                        unit="%",
                    )

                    if (
                        dominant_fraction
                        >= asymmetry_threshold
                    ):

                        dominant_boundary = (
                            "upper"
                            if upper_fraction
                            >= lower_fraction
                            else "lower"
                        )

                        result.add_warning(
                            f"{axis_label} near-limit behavior is "
                            f"strongly concentrated at the "
                            f"{dominant_boundary} boundary."
                        )

                        result.add_evidence(
                            statement=(
                                f"{axis_label} interaction with the "
                                "configured measurement limits is "
                                "strongly asymmetric between the "
                                "upper and lower boundaries."
                            ),
                            strength=EvidenceStrength.MODERATE,
                            supporting_metrics=[
                                (
                                    f"{axis_label} Upper "
                                    "Near-Limit Share"
                                ),
                                (
                                    f"{axis_label} Lower "
                                    "Near-Limit Share"
                                ),
                            ],
                        )

                missing_count = (
                    len(values)
                    - valid_count
                )

                if missing_count > 0:

                    result.add_warning(
                        f"{axis_label} contains {missing_count} "
                        "missing or non-numeric sample(s) excluded "
                        "from saturation statistics."
                    )

                group_summary[
                    axis
                ] = {
                    "configured_minimum":
                        configured_minimum,

                    "configured_maximum":
                        configured_maximum,

                    "observed_minimum":
                        observed_minimum,

                    "observed_maximum":
                        observed_maximum,

                    "range_utilization_percent":
                        range_utilization_percent,

                    "exact_limit_count":
                        exact_total_count,

                    "near_limit_count":
                        near_total_count,

                    "outside_range_count":
                        outside_range_count,

                    "longest_exact_limit_run":
                        longest_exact_run,

                    "longest_near_limit_run":
                        longest_near_run,
                }

                processed_axes += 1

            saturation_summary[
                group_name
            ] = group_summary

        # Overall status

        if processed_axes == 0:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "No IMU axis could be evaluated against a valid "
                "configured full-scale range."
            )

            return result

        result.metadata.update(
            {
                "near_limit_margin_fraction":
                    near_fraction,

                "exact_limit_tolerance_fraction":
                    exact_fraction,

                "minimum_clipping_run_samples":
                    minimum_clipping_run,

                "minimum_asymmetry_samples":
                    minimum_asymmetry_samples,

                "asymmetry_fraction_threshold":
                    asymmetry_threshold,

                "saturation_summary":
                    saturation_summary,
            }
        )

        return result

    # Full-scale metadata

    @staticmethod
    def _get_range_metadata(
        context: ToolContext,
        group_name: str,
    ) -> dict | None:
        """Retrieve accelerometer or gyroscope full-scale metadata.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
            group_name: Value for `group_name`.
        
        Returns:
            Dictionary containing the result values.
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

        if group_name == "accelerometer":

            range_data = sensor_metadata.get(
                "accelerometer_range"
            )

        elif group_name == "gyroscope":

            range_data = sensor_metadata.get(
                "gyroscope_range"
            )

        else:

            return None

        if not isinstance(
            range_data,
            dict,
        ):
            return None

        return range_data

    # Range unit conversion

    @classmethod
    def _convert_range(
        cls,
        group_name: str,
        range_metadata: dict,
        target_unit: str | None,
    ) -> tuple[float, float]:
        """Convert configured range limits into the internal units
        used by the normalized SensorQADataset.
        
        Supported V1 conversions:
        
            acceleration:
                g
                m/s^2
                cm/s^2
        
            gyroscope:
                deg/s
                rad/s
        
        Args:
            group_name: Value for `group_name`.
            range_metadata: Value for `range_metadata`.
            target_unit: Value for `target_unit`.
        
        Returns:
            Tuple containing the result values.
        """

        try:

            minimum = float(
                range_metadata[
                    "minimum"
                ]
            )

            maximum = float(
                range_metadata[
                    "maximum"
                ]
            )

            source_unit = str(
                range_metadata[
                    "unit"
                ]
            ).strip()

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Range metadata must provide numerical minimum, "
                "maximum, and unit values."
            ) from exc

        source_unit = cls._normalize_unit(
            source_unit
        )

        target_unit = cls._normalize_unit(
            target_unit
        )

        if group_name == "accelerometer":

            minimum = cls._convert_acceleration(
                minimum,
                source_unit,
                target_unit,
            )

            maximum = cls._convert_acceleration(
                maximum,
                source_unit,
                target_unit,
            )

        elif group_name == "gyroscope":

            minimum = cls._convert_angular_velocity(
                minimum,
                source_unit,
                target_unit,
            )

            maximum = cls._convert_angular_velocity(
                maximum,
                source_unit,
                target_unit,
            )

        else:

            raise ValueError(
                f"Unsupported sensor group '{group_name}'."
            )

        return (
            minimum,
            maximum,
        )

    @staticmethod
    def _normalize_unit(
        unit: str | None,
    ) -> str | None:

        """Normalize unit.
        
        Args:
            unit: Engineering unit.
        
        Returns:
            str | None returned by this function.
        """
        if unit is None:
            return None

        cleaned = str(
            unit
        ).strip()

        aliases = {
            "m/s²": "m/s^2",
            "m/s2": "m/s^2",
            "mps2": "m/s^2",

            "cm/s²": "cm/s^2",
            "cm/s2": "cm/s^2",

            "g0": "g",
            "gravity": "g",

            "degree/s": "deg/s",
            "degrees/s": "deg/s",
            "deg/sec": "deg/s",
            "°/s": "deg/s",

            "rad/sec": "rad/s",
            "radian/s": "rad/s",
            "radians/s": "rad/s",
        }

        lowercase = cleaned.lower()

        if cleaned in aliases:
            return aliases[
                cleaned
            ]

        if lowercase in aliases:
            return aliases[
                lowercase
            ]

        return cleaned

    @classmethod
    def _convert_acceleration(
        cls,
        value: float,
        source_unit: str | None,
        target_unit: str | None,
    ) -> float:

        """Convert acceleration.
        
        Args:
            value: Value to process.
            source_unit: Source unit used by this function.
            target_unit: Target unit used by this function.
        
        Returns:
            Calculated value.
        """
        if target_unit is None:
            raise ValueError(
                "Internal accelerometer unit is unavailable."
            )

        factors_to_si = {
            "m/s^2": 1.0,
            "cm/s^2": 0.01,
            "g": cls.STANDARD_GRAVITY,
        }

        if source_unit not in factors_to_si:

            raise ValueError(
                f"Unsupported accelerometer range unit "
                f"'{source_unit}'."
            )

        if target_unit not in factors_to_si:

            raise ValueError(
                f"Unsupported internal accelerometer unit "
                f"'{target_unit}'."
            )

        value_si = (
            value
            * factors_to_si[
                source_unit
            ]
        )

        return (
            value_si
            / factors_to_si[
                target_unit
            ]
        )

    @staticmethod
    def _convert_angular_velocity(
        value: float,
        source_unit: str | None,
        target_unit: str | None,
    ) -> float:

        """Convert angular velocity.
        
        Args:
            value: Value to process.
            source_unit: Source unit used by this function.
            target_unit: Target unit used by this function.
        
        Returns:
            Calculated value.
        """
        if target_unit is None:
            raise ValueError(
                "Internal gyroscope unit is unavailable."
            )

        factors_to_si = {
            "rad/s": 1.0,
            "deg/s": np.pi / 180.0,
        }

        if source_unit not in factors_to_si:

            raise ValueError(
                f"Unsupported gyroscope range unit "
                f"'{source_unit}'."
            )

        if target_unit not in factors_to_si:

            raise ValueError(
                f"Unsupported internal gyroscope unit "
                f"'{target_unit}'."
            )

        value_si = (
            value
            * factors_to_si[
                source_unit
            ]
        )

        return (
            value_si
            / factors_to_si[
                target_unit
            ]
        )

    # Consecutive-run analysis

    @staticmethod
    def _longest_true_run(
        mask: np.ndarray,
    ) -> int:
        """Return the longest consecutive True sequence in a
        Boolean mask.
        
        Missing samples are represented by False and therefore
        break a clipping run.
        
        Args:
            mask: Value for `mask`.
        
        Returns:
            Integer result.
        """

        longest = 0
        current = 0

        for value in mask:

            if bool(
                value
            ):

                current += 1

                longest = max(
                    longest,
                    current,
                )

            else:

                current = 0

        return longest

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
