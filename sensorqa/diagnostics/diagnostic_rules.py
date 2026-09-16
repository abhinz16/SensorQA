#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from sensorqa.core.result_schema import (
    EvidenceStrength,
)
from sensorqa.diagnostics.evidence_engine import (
    EvidenceBundle,
    MetricObservation,
)
from sensorqa.diagnostics.hypothesis_engine import (
    ContributionRole,
    EvidenceSourceType,
    HypothesisContribution,
    HypothesisEvidenceReference,
    HypothesisProposal,
    HypothesisState,
    MissingEvidence,
)


# Common helpers


ACCELEROMETER_AXES = (
    "AX",
    "AY",
    "AZ",
)

GYROSCOPE_AXES = (
    "GX",
    "GY",
    "GZ",
)


def _numeric(
    value: Any,
) -> float | None:
    """Convert a finite numerical scalar to float.
    
    Boolean values are explicitly rejected.
    
    Args:
        value: Value to process.
    
    Returns:
        Numeric result.
    """

    if isinstance(
        value,
        bool,
    ):
        return None

    if not isinstance(
        value,
        Real,
    ):
        return None

    number = float(
        value
    )

    if not math.isfinite(
        number
    ):
        return None

    return number


def _metric_reference(
    observation: MetricObservation,
) -> HypothesisEvidenceReference:
    """Build a hypothesis reference from an existing metric.
    
    Args:
        observation: Value for `observation`.
    
    Returns:
        HypothesisEvidenceReference returned by the function.
    """

    return HypothesisEvidenceReference(
        source_type=(
            EvidenceSourceType.METRIC
        ),
        tool_id=observation.tool_id,
        metric_name=observation.metric_name,
    )


def _contribution(
    observation: MetricObservation,
    role: ContributionRole,
    rationale: str,
) -> HypothesisContribution:

    """Return contribution.
    
    Args:
        observation: Observation used by this function.
        role: Role used by this function.
        rationale: Rationale used by this function.
    
    Returns:
        HypothesisContribution returned by this function.
    """
    return HypothesisContribution(
        role=role,
        reference=_metric_reference(
            observation
        ),
        rationale=rationale,
    )


def _axis_metric(
    evidence: EvidenceBundle,
    tool_id: str,
    axis: str,
    required_tokens: tuple[str, ...],
    excluded_tokens: tuple[str, ...] = (),
) -> MetricObservation | None:
    """Locate an axis-specific metric without hardcoding the entire
    displayed metric name.
    
    Example:
    
        axis = "GX"
        required_tokens = (
            "temperature",
            "regression",
            "r2",
        )
    
    Only metrics from the requested tool are considered.
    
    Args:
        evidence: Value for `evidence`.
        tool_id: Registered tool identifier.
        axis: Value for `axis`.
        required_tokens: Value for `required_tokens`.
        excluded_tokens: Value for `excluded_tokens`.
    
    Returns:
        MetricObservation | None returned by the function.
    """

    axis_upper = axis.upper()

    matches = []

    for observation in evidence.metrics_for_tool(
        tool_id
    ):

        name = observation.metric_name

        normalized = (
            name
            .strip()
            .upper()
        )

        if not normalized.startswith(
            axis_upper
        ):
            continue

        if not all(
            token.upper()
            in normalized
            for token
            in required_tokens
        ):
            continue

        if any(
            token.upper()
            in normalized
            for token
            in excluded_tokens
        ):
            continue

        matches.append(
            observation
        )

    if not matches:
        return None

    # Prefer the simplest/shortest matching metric name when several
    # related metrics contain the same tokens.
    matches.sort(
        key=lambda item:
            len(
                item.metric_name
            )
    )

    return matches[
        0
    ]


def _axis_metrics(
    evidence: EvidenceBundle,
    tool_id: str,
    axis: str,
    required_tokens: tuple[str, ...],
    excluded_tokens: tuple[str, ...] = (),
) -> list[MetricObservation]:
    """Return every axis-specific metric matching the requested tokens.
    
    Args:
        evidence: Value for `evidence`.
        tool_id: Registered tool identifier.
        axis: Value for `axis`.
        required_tokens: Value for `required_tokens`.
        excluded_tokens: Value for `excluded_tokens`.
    
    Returns:
        List of result values.
    """

    axis_upper = axis.upper()

    output = []

    for observation in evidence.metrics_for_tool(
        tool_id
    ):

        normalized = (
            observation.metric_name
            .strip()
            .upper()
        )

        if not normalized.startswith(
            axis_upper
        ):
            continue

        if not all(
            token.upper()
            in normalized
            for token
            in required_tokens
        ):
            continue

        if any(
            token.upper()
            in normalized
            for token
            in excluded_tokens
        ):
            continue

        output.append(
            observation
        )

    return output


def _pair_metric(
    evidence: EvidenceBundle,
    tool_id: str,
    axis_a: str,
    axis_b: str,
    required_tokens: tuple[str, ...],
    excluded_tokens: tuple[str, ...] = (),
) -> MetricObservation | None:
    """Locate a pair-specific metric such as:
    
        AX-AY Correlation
    
    Either axis ordering is accepted.
    
    Args:
        evidence: Value for `evidence`.
        tool_id: Registered tool identifier.
        axis_a: Value for `axis_a`.
        axis_b: Value for `axis_b`.
        required_tokens: Value for `required_tokens`.
        excluded_tokens: Value for `excluded_tokens`.
    
    Returns:
        MetricObservation | None returned by the function.
    """

    pair_a = (
        f"{axis_a.upper()}-"
        f"{axis_b.upper()}"
    )

    pair_b = (
        f"{axis_b.upper()}-"
        f"{axis_a.upper()}"
    )

    matches = []

    for observation in evidence.metrics_for_tool(
        tool_id
    ):

        normalized = (
            observation.metric_name
            .strip()
            .upper()
        )

        if not (
            normalized.startswith(
                pair_a
            )
            or normalized.startswith(
                pair_b
            )
        ):
            continue

        if not all(
            token.upper()
            in normalized
            for token
            in required_tokens
        ):
            continue

        if any(
            token.upper()
            in normalized
            for token
            in excluded_tokens
        ):
            continue

        matches.append(
            observation
        )

    if not matches:
        return None

    matches.sort(
        key=lambda item:
            len(
                item.metric_name
            )
    )

    return matches[
        0
    ]


def _dominant_frequency(
    evidence: EvidenceBundle,
    axis: str,
) -> MetricObservation | None:
    """Locate an axis dominant-frequency metric from IMU PSD output.
    
    Args:
        evidence: Value for `evidence`.
        axis: Value for `axis`.
    
    Returns:
        MetricObservation | None returned by the function.
    """

    return _axis_metric(
        evidence=evidence,
        tool_id="imu_psd",
        axis=axis,
        required_tokens=(
            "dominant",
            "frequency",
        ),
        excluded_tokens=(
            "ratio",
            "psd",
            "power",
        ),
    )


def _temperature_r2(
    evidence: EvidenceBundle,
    axis: str,
) -> MetricObservation | None:

    """Calculate temperature r2.
    
    Args:
        evidence: Evidence used by this function.
        axis: Axis used by this function.
    
    Returns:
        MetricObservation | None returned by this function.
    """
    return _axis_metric(
        evidence=evidence,
        tool_id="imu_temperature_stability",
        axis=axis,
        required_tokens=(
            "temperature",
            "regression",
            "r2",
        ),
    )


def _temperature_coefficient(
    evidence: EvidenceBundle,
    axis: str,
) -> MetricObservation | None:

    """Calculate temperature coefficient.
    
    Args:
        evidence: Evidence used by this function.
        axis: Axis used by this function.
    
    Returns:
        MetricObservation | None returned by this function.
    """
    return _axis_metric(
        evidence=evidence,
        tool_id="imu_temperature_stability",
        axis=axis,
        required_tokens=(
            "temperature",
            "coefficient",
        ),
    )


def _predicted_temperature_change(
    evidence: EvidenceBundle,
    axis: str,
) -> MetricObservation | None:

    """Calculate predicted temperature change.
    
    Args:
        evidence: Evidence used by this function.
        axis: Axis used by this function.
    
    Returns:
        MetricObservation | None returned by this function.
    """
    return _axis_metric(
        evidence=evidence,
        tool_id="imu_temperature_stability",
        axis=axis,
        required_tokens=(
            "predicted",
            "change",
            "temperature",
        ),
    )


def _frequencies_match(
    frequency_a: float,
    frequency_b: float,
    absolute_tolerance_hz: float,
    relative_tolerance: float,
) -> bool:
    """Determine whether two detected spectral components occupy
    approximately the same frequency.
    
    The comparison uses the larger of:
    
        absolute frequency tolerance
        relative frequency tolerance
    
    Args:
        frequency_a: Value for `frequency_a`.
        frequency_b: Value for `frequency_b`.
        absolute_tolerance_hz: Value for `absolute_tolerance_hz`.
        relative_tolerance: Value for `relative_tolerance`.
    
    Returns:
        Boolean result.
    """

    reference = max(
        abs(
            frequency_a
        ),
        abs(
            frequency_b
        ),
    )

    tolerance = max(
        absolute_tolerance_hz,
        relative_tolerance
        * reference,
    )

    return (
        abs(
            frequency_a
            - frequency_b
        )
        <= tolerance
    )


def _axis_pairs(
    axes: tuple[str, ...],
) -> list[
    tuple[str, str]
]:

    """Calculate axis pairs.
    
    Args:
        axes: Axes used by this function.
    
    Returns:
        List of result values.
    """
    output = []

    for index_a in range(
        len(
            axes
        )
    ):

        for index_b in range(
            index_a + 1,
            len(
                axes
            ),
        ):

            output.append(
                (
                    axes[
                        index_a
                    ],
                    axes[
                        index_b
                    ],
                )
            )

    return output


# Shared periodic behavior


class SharedPeriodicBehaviorRule:
    """
    Looks for similar dominant PSD frequencies on multiple axes.

    Cross-axis correlation is used as complementary evidence when
    available.

    This rule intentionally uses the term:

        shared periodic behavior

    rather than:

        vibration fault

    because the physical source is not established without an
    independent vibration/motion reference.
    """

    def __init__(
        self,
        sensor_group: str,
        *,
        minimum_axis_correlation: float = 0.70,
        absolute_frequency_tolerance_hz: float = 0.50,
        relative_frequency_tolerance: float = 0.05,
    ) -> None:

        """Initialize the shared periodic behavior rule.
        
        Args:
            sensor_group: Sensor group used by this function.
            minimum_axis_correlation: Minimum axis correlation accepted.
            absolute_frequency_tolerance_hz: Absolute frequency tolerance hz used by this function.
            relative_frequency_tolerance: Relative frequency tolerance used by this function.
        
        Returns:
            None.
        """
        normalized_group = (
            sensor_group
            .strip()
            .lower()
        )

        if normalized_group == "accelerometer":

            self.axes = (
                ACCELEROMETER_AXES
            )

        elif normalized_group == "gyroscope":

            self.axes = (
                GYROSCOPE_AXES
            )

        else:

            raise ValueError(
                "sensor_group must be "
                "'accelerometer' or 'gyroscope'."
            )

        self.sensor_group = (
            normalized_group
        )

        self.minimum_axis_correlation = float(
            minimum_axis_correlation
        )

        self.absolute_frequency_tolerance_hz = float(
            absolute_frequency_tolerance_hz
        )

        self.relative_frequency_tolerance = float(
            relative_frequency_tolerance
        )

    @property
    def rule_id(
        self,
    ) -> str:

        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        return (
            f"shared_periodic_behavior_"
            f"{self.sensor_group}"
        )

    def evaluate(
        self,
        evidence: EvidenceBundle,
    ) -> HypothesisProposal | None:

        """Evaluate evaluate.
        
        Args:
            evidence: Evidence used by this function.
        
        Returns:
            HypothesisProposal | None returned by this function.
        """
        if not evidence.tool_succeeded(
            "imu_psd"
        ):

            return None

        candidates = []

        for axis_a, axis_b in _axis_pairs(
            self.axes
        ):

            frequency_a_observation = (
                _dominant_frequency(
                    evidence=evidence,
                    axis=axis_a,
                )
            )

            frequency_b_observation = (
                _dominant_frequency(
                    evidence=evidence,
                    axis=axis_b,
                )
            )

            if (
                frequency_a_observation is None
                or frequency_b_observation is None
            ):

                continue

            frequency_a = _numeric(
                frequency_a_observation.value
            )

            frequency_b = _numeric(
                frequency_b_observation.value
            )

            if (
                frequency_a is None
                or frequency_b is None
                or frequency_a <= 0.0
                or frequency_b <= 0.0
            ):

                continue

            if not _frequencies_match(
                frequency_a=frequency_a,
                frequency_b=frequency_b,
                absolute_tolerance_hz=(
                    self.absolute_frequency_tolerance_hz
                ),
                relative_tolerance=(
                    self.relative_frequency_tolerance
                ),
            ):

                continue

            correlation_observation = (
                _pair_metric(
                    evidence=evidence,
                    tool_id=(
                        "imu_axis_correlation"
                    ),
                    axis_a=axis_a,
                    axis_b=axis_b,
                    required_tokens=(
                        "correlation",
                    ),
                    excluded_tokens=(
                        "first-difference",
                        "maximum",
                    ),
                )
            )

            correlation = None

            if correlation_observation is not None:

                correlation = _numeric(
                    correlation_observation.value
                )

            correlation_supported = (
                correlation is not None
                and abs(
                    correlation
                )
                >= self.minimum_axis_correlation
            )

            frequency_difference = abs(
                frequency_a
                - frequency_b
            )

            candidates.append(
                {
                    "axis_a":
                        axis_a,

                    "axis_b":
                        axis_b,

                    "frequency_a":
                        frequency_a,

                    "frequency_b":
                        frequency_b,

                    "frequency_a_observation":
                        frequency_a_observation,

                    "frequency_b_observation":
                        frequency_b_observation,

                    "correlation_observation":
                        correlation_observation,

                    "correlation":
                        correlation,

                    "correlation_supported":
                        correlation_supported,

                    "frequency_difference":
                        frequency_difference,
                }
            )

        if not candidates:

            return None

        # Prefer a matching-frequency pair that also has substantial
        # cross-axis correlation. After that, prefer the strongest
        # correlation and closest spectral agreement.
        candidates.sort(
            key=lambda item: (
                0
                if item[
                    "correlation_supported"
                ]
                else 1,

                -abs(
                    item[
                        "correlation"
                    ]
                )
                if item[
                    "correlation"
                ]
                is not None
                else 0.0,

                item[
                    "frequency_difference"
                ],
            )
        )

        best = candidates[
            0
        ]

        axis_a = best[
            "axis_a"
        ]

        axis_b = best[
            "axis_b"
        ]

        supporting = [
            _contribution(
                observation=best[
                    "frequency_a_observation"
                ],
                role=(
                    ContributionRole.SUPPORTING
                ),
                rationale=(
                    f"{axis_a} contains a dominant spectral "
                    f"component near "
                    f"{best['frequency_a']:.3f} Hz."
                ),
            ),

            _contribution(
                observation=best[
                    "frequency_b_observation"
                ],
                role=(
                    ContributionRole.SUPPORTING
                ),
                rationale=(
                    f"{axis_b} contains a dominant spectral "
                    f"component near "
                    f"{best['frequency_b']:.3f} Hz."
                ),
            ),
        ]

        if best[
            "correlation_supported"
        ]:

            supporting.append(
                _contribution(
                    observation=best[
                        "correlation_observation"
                    ],
                    role=(
                        ContributionRole.SUPPORTING
                    ),
                    rationale=(
                        f"{axis_a} and {axis_b} also show "
                        "substantial statistical cross-axis "
                        "correlation."
                    ),
                )
            )

            support_level = (
                EvidenceStrength.MODERATE
            )

        else:

            support_level = (
                EvidenceStrength.WEAK
            )

        mean_frequency = (
            best[
                "frequency_a"
            ]
            + best[
                "frequency_b"
            ]
        ) / 2.0

        return HypothesisProposal(
            hypothesis_id=(
                f"shared_periodic_behavior:"
                f"{self.sensor_group}:"
                f"{axis_a.lower()}_"
                f"{axis_b.lower()}"
            ),
            title=(
                f"Shared Periodic Behavior "
                f"({self.sensor_group.title()})"
            ),
            statement=(
                f"{axis_a} and {axis_b} show behavior "
                f"consistent with a shared periodic component "
                f"near {mean_frequency:.3f} Hz."
            ),
            state=HypothesisState.SUPPORTED,
            support_level=support_level,
            supporting=supporting,
            missing_evidence=[
                MissingEvidence(
                    description=(
                        "An independent vibration, motion, motor, "
                        "or structural reference measurement would "
                        "be needed to associate the shared spectral "
                        "component with a specific physical source."
                    ),
                    importance=(
                        EvidenceStrength.STRONG
                    ),
                )
            ],
            limitations=[
                (
                    "Similar spectral components on multiple axes "
                    "do not by themselves establish that the source "
                    "is mechanical vibration."
                ),
                (
                    "Common motion, control activity, structural "
                    "response, electrical interference, or signal "
                    "processing can also create shared periodic "
                    "content."
                ),
            ],
            metadata={
                "sensor_group":
                    self.sensor_group,

                "axis_pair": [
                    axis_a,
                    axis_b,
                ],

                "mean_frequency_hz":
                    mean_frequency,

                "frequency_difference_hz":
                    best[
                        "frequency_difference"
                    ],

                "cross_axis_correlation":
                    best[
                        "correlation"
                    ],
            },
        )


# Temperature-associated output variation


class TemperatureAssociatedOutputVariationRule:
    """
    Identifies axes whose output has a substantial linear
    statistical relationship with measured temperature.

    This is deliberately NOT named "thermal drift detected."
    """

    @property
    def rule_id(
        self,
    ) -> str:

        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        return (
            "temperature_associated_output_variation"
        )

    def __init__(
        self,
        *,
        minimum_r_squared: float = 0.50,
    ) -> None:

        """Initialize the temperature associated output variation rule.
        
        Args:
            minimum_r_squared: Minimum r squared accepted.
        
        Returns:
            None.
        """
        self.minimum_r_squared = float(
            minimum_r_squared
        )

    def evaluate(
        self,
        evidence: EvidenceBundle,
    ) -> HypothesisProposal | None:

        """Evaluate evaluate.
        
        Args:
            evidence: Evidence used by this function.
        
        Returns:
            HypothesisProposal | None returned by this function.
        """
        if not evidence.tool_succeeded(
            "imu_temperature_stability"
        ):

            return None

        affected_axes = []

        supporting = []

        context = []

        for axis in (
            *ACCELEROMETER_AXES,
            *GYROSCOPE_AXES,
        ):

            r2_observation = _temperature_r2(
                evidence=evidence,
                axis=axis,
            )

            if r2_observation is None:
                continue

            r_squared = _numeric(
                r2_observation.value
            )

            if (
                r_squared is None
                or r_squared
                < self.minimum_r_squared
            ):

                continue

            affected_axes.append(
                (
                    axis,
                    r_squared,
                )
            )

            supporting.append(
                _contribution(
                    observation=r2_observation,
                    role=(
                        ContributionRole.SUPPORTING
                    ),
                    rationale=(
                        f"{axis} has a linear output-temperature "
                        f"relationship with R² = "
                        f"{r_squared:.3f} over the observed test "
                        "range."
                    ),
                )
            )

            coefficient = (
                _temperature_coefficient(
                    evidence=evidence,
                    axis=axis,
                )
            )

            if coefficient is not None:

                supporting.append(
                    _contribution(
                        observation=coefficient,
                        role=(
                            ContributionRole.SUPPORTING
                        ),
                        rationale=(
                            f"{axis} has a measurable fitted "
                            "output-temperature slope."
                        ),
                    )
                )

            predicted_change = (
                _predicted_temperature_change(
                    evidence=evidence,
                    axis=axis,
                )
            )

            if predicted_change is not None:

                context.append(
                    _contribution(
                        observation=predicted_change,
                        role=(
                            ContributionRole.CONTEXT
                        ),
                        rationale=(
                            "This metric expresses the fitted "
                            "output change across the actually "
                            "observed temperature range."
                        ),
                    )
                )

        if not affected_axes:

            return None

        axis_text = ", ".join(
            axis
            for axis, _
            in affected_axes
        )

        return HypothesisProposal(
            hypothesis_id=(
                "temperature_associated_output_variation"
            ),
            title=(
                "Temperature-Associated Output Variation"
            ),
            statement=(
                f"Output from {axis_text} shows a substantial "
                "statistical relationship with measured "
                "temperature over the observed experiment."
            ),
            state=HypothesisState.SUPPORTED,
            support_level=(
                EvidenceStrength.MODERATE
            ),
            supporting=supporting,
            context=context,
            missing_evidence=[
                MissingEvidence(
                    description=(
                        "A controlled thermal experiment with "
                        "temperature varied independently of elapsed "
                        "time would help separate temperature effects "
                        "from sensor warm-up and ordinary long-term "
                        "drift."
                    ),
                    importance=(
                        EvidenceStrength.STRONG
                    ),
                    related_tool_id=(
                        "imu_temperature_stability"
                    ),
                ),
                MissingEvidence(
                    description=(
                        "Heating and cooling cycles or repeated "
                        "temperature plateaus would help evaluate "
                        "repeatability and possible thermal "
                        "hysteresis."
                    ),
                    importance=(
                        EvidenceStrength.MODERATE
                    ),
                ),
            ],
            limitations=[
                (
                    "The fitted relationship is statistical and "
                    "does not establish temperature as the causal "
                    "source of the output change."
                ),
                (
                    "Temperature may covary with elapsed time, "
                    "warm-up behavior, supply conditions, or other "
                    "environmental variables."
                ),
            ],
            metadata={
                "minimum_r_squared":
                    self.minimum_r_squared,

                "affected_axes": [
                    {
                        "axis":
                            axis,

                        "r_squared":
                            r_squared,
                    }
                    for axis, r_squared
                    in affected_axes
                ],
            },
        )


# Range-boundary / clipping behavior


class RangeBoundaryClippingRule:
    """
    Identifies repeated observations at configured sensor
    full-scale boundaries.

    Exact boundary behavior can be consistent with clipping or
    saturation, but the rule keeps the configured-range metadata
    caveat explicit.
    """

    @property
    def rule_id(
        self,
    ) -> str:

        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        return (
            "range_boundary_clipping_behavior"
        )

    def __init__(
        self,
        *,
        strong_run_length: int = 3,
    ) -> None:

        """Initialize the range boundary clipping rule.
        
        Args:
            strong_run_length: Strong run length used by this function.
        
        Returns:
            None.
        """
        self.strong_run_length = int(
            strong_run_length
        )

    def evaluate(
        self,
        evidence: EvidenceBundle,
    ) -> HypothesisProposal | None:

        """Evaluate evaluate.
        
        Args:
            evidence: Evidence used by this function.
        
        Returns:
            HypothesisProposal | None returned by this function.
        """
        if not evidence.tool_succeeded(
            "imu_saturation"
        ):

            return None

        candidates = []

        for axis in (
            *ACCELEROMETER_AXES,
            *GYROSCOPE_AXES,
        ):

            exact_count_observations = (
                _axis_metrics(
                    evidence=evidence,
                    tool_id="imu_saturation",
                    axis=axis,
                    required_tokens=(
                        "exact",
                        "limit",
                        "count",
                    ),
                    excluded_tokens=(
                        "fraction",
                    ),
                )
            )

            longest_run_observations = (
                _axis_metrics(
                    evidence=evidence,
                    tool_id="imu_saturation",
                    axis=axis,
                    required_tokens=(
                        "longest",
                        "exact",
                        "run",
                    ),
                )
            )

            out_of_range_observations = (
                _axis_metrics(
                    evidence=evidence,
                    tool_id="imu_saturation",
                    axis=axis,
                    required_tokens=(
                        "out",
                        "range",
                        "count",
                    ),
                    excluded_tokens=(
                        "fraction",
                    ),
                )
            )

            exact_count_observation = (
                self._largest_numeric_observation(
                    exact_count_observations
                )
            )

            longest_run_observation = (
                self._largest_numeric_observation(
                    longest_run_observations
                )
            )

            out_of_range_observation = (
                self._largest_numeric_observation(
                    out_of_range_observations
                )
            )

            exact_count = (
                _numeric(
                    exact_count_observation.value
                )
                if exact_count_observation
                is not None
                else 0.0
            )

            longest_run = (
                _numeric(
                    longest_run_observation.value
                )
                if longest_run_observation
                is not None
                else 0.0
            )

            out_of_range_count = (
                _numeric(
                    out_of_range_observation.value
                )
                if out_of_range_observation
                is not None
                else 0.0
            )

            if (
                exact_count is None
                or exact_count <= 0
            ):

                continue

            candidates.append(
                {
                    "axis":
                        axis,

                    "exact_count":
                        exact_count,

                    "exact_count_observation":
                        exact_count_observation,

                    "longest_run":
                        (
                            longest_run
                            if longest_run
                            is not None
                            else 0.0
                        ),

                    "longest_run_observation":
                        longest_run_observation,

                    "out_of_range_count":
                        (
                            out_of_range_count
                            if out_of_range_count
                            is not None
                            else 0.0
                        ),

                    "out_of_range_observation":
                        out_of_range_observation,
                }
            )

        if not candidates:

            return None

        candidates.sort(
            key=lambda item: (
                -item[
                    "longest_run"
                ],
                -item[
                    "exact_count"
                ],
            )
        )

        best = candidates[
            0
        ]

        axis = best[
            "axis"
        ]

        supporting = []

        if best[
            "exact_count_observation"
        ] is not None:

            supporting.append(
                _contribution(
                    observation=best[
                        "exact_count_observation"
                    ],
                    role=(
                        ContributionRole.SUPPORTING
                    ),
                    rationale=(
                        f"{axis} contains repeated samples at a "
                        "configured full-scale boundary."
                    ),
                )
            )

        if (
            best[
                "longest_run_observation"
            ]
            is not None
        ):

            supporting.append(
                _contribution(
                    observation=best[
                        "longest_run_observation"
                    ],
                    role=(
                        ContributionRole.SUPPORTING
                    ),
                    rationale=(
                        f"{axis} contains a consecutive run of "
                        "samples at the configured boundary."
                    ),
                )
            )

        conflicting = []

        if (
            best[
                "out_of_range_count"
            ]
            > 0
            and best[
                "out_of_range_observation"
            ]
            is not None
        ):

            conflicting.append(
                _contribution(
                    observation=best[
                        "out_of_range_observation"
                    ],
                    role=(
                        ContributionRole.CONFLICTING
                    ),
                    rationale=(
                        f"{axis} also contains measurements beyond "
                        "the configured range. This can indicate "
                        "incorrect range metadata, unit mapping, or "
                        "configuration rather than a true hardware "
                        "clipping boundary."
                    ),
                )
            )

        # Out-of-range data undermine the assumption that the
        # configured limit represents the actual clipping boundary.
        if conflicting:

            state = (
                HypothesisState.CONFLICTING_EVIDENCE
            )

            support_level = None

        elif (
            best[
                "longest_run"
            ]
            >= self.strong_run_length
        ):

            state = (
                HypothesisState.SUPPORTED
            )

            support_level = (
                EvidenceStrength.STRONG
            )

        else:

            state = (
                HypothesisState.SUPPORTED
            )

            support_level = (
                EvidenceStrength.MODERATE
            )

        return HypothesisProposal(
            hypothesis_id=(
                f"range_boundary_behavior:"
                f"{axis.lower()}"
            ),
            title=(
                f"Range-Boundary Behavior ({axis})"
            ),
            statement=(
                f"{axis} contains measurements at a configured "
                "full-scale boundary that are consistent with "
                "range limiting or clipping-like behavior."
            ),
            state=state,
            support_level=support_level,
            supporting=supporting,
            conflicting=conflicting,
            missing_evidence=[
                MissingEvidence(
                    description=(
                        "Confirmation of the sensor's active "
                        "hardware or firmware full-scale setting "
                        "would strengthen interpretation of the "
                        "configured range boundary."
                    ),
                    importance=(
                        EvidenceStrength.STRONG
                    ),
                ),
                MissingEvidence(
                    description=(
                        "Raw ADC counts or device saturation/status "
                        "flags would provide more direct evidence "
                        "of hardware or digital clipping."
                    ),
                    importance=(
                        EvidenceStrength.MODERATE
                    ),
                ),
            ],
            limitations=[
                (
                    "Repeated values at a configured boundary are "
                    "consistent with clipping but do not by "
                    "themselves identify whether limiting occurred "
                    "in the sensing element, analog electronics, "
                    "ADC, firmware, or downstream processing."
                ),
            ],
            metadata={
                "axis":
                    axis,

                "exact_limit_count":
                    best[
                        "exact_count"
                    ],

                "longest_exact_limit_run":
                    best[
                        "longest_run"
                    ],

                "out_of_range_count":
                    best[
                        "out_of_range_count"
                    ],
            },
        )

    @staticmethod
    def _largest_numeric_observation(
        observations: list[
            MetricObservation
        ],
    ) -> MetricObservation | None:

        """Return largest numeric observation.
        
        Args:
            observations: Observations used by this function.
        
        Returns:
            MetricObservation | None returned by this function.
        """
        valid = []

        for observation in observations:

            value = _numeric(
                observation.value
            )

            if value is None:
                continue

            valid.append(
                (
                    value,
                    observation,
                )
            )

        if not valid:

            return None

        valid.sort(
            key=lambda item:
                item[
                    0
                ],
            reverse=True,
        )

        return valid[
            0
        ][
            1
        ]


# Unresolved cross-axis covariance


class UnexplainedCrossAxisCovarianceRule:
    """
    Identifies substantial cross-axis statistical covariance while
    explicitly avoiding the conclusion that it represents physical
    misalignment.

    The rule can characterize whether the relationship also appears
    in first differences, which helps distinguish slower shared
    variation from short-timescale shared changes.
    """

    def __init__(
        self,
        sensor_group: str,
        *,
        minimum_correlation: float = 0.70,
        minimum_first_difference_correlation: float = 0.70,
    ) -> None:

        """Initialize the unexplained cross axis covariance rule.
        
        Args:
            sensor_group: Sensor group used by this function.
            minimum_correlation: Minimum correlation accepted.
            minimum_first_difference_correlation: Minimum first difference correlation accepted.
        
        Returns:
            None.
        """
        normalized_group = (
            sensor_group
            .strip()
            .lower()
        )

        if normalized_group == "accelerometer":

            self.axes = (
                ACCELEROMETER_AXES
            )

        elif normalized_group == "gyroscope":

            self.axes = (
                GYROSCOPE_AXES
            )

        else:

            raise ValueError(
                "sensor_group must be "
                "'accelerometer' or 'gyroscope'."
            )

        self.sensor_group = (
            normalized_group
        )

        self.minimum_correlation = float(
            minimum_correlation
        )

        self.minimum_first_difference_correlation = float(
            minimum_first_difference_correlation
        )

    @property
    def rule_id(
        self,
    ) -> str:

        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        return (
            "unresolved_cross_axis_covariance_"
            f"{self.sensor_group}"
        )

    def evaluate(
        self,
        evidence: EvidenceBundle,
    ) -> HypothesisProposal | None:

        """Evaluate evaluate.
        
        Args:
            evidence: Evidence used by this function.
        
        Returns:
            HypothesisProposal | None returned by this function.
        """
        if not evidence.tool_succeeded(
            "imu_axis_correlation"
        ):

            return None

        candidates = []

        for axis_a, axis_b in _axis_pairs(
            self.axes
        ):

            correlation_observation = (
                _pair_metric(
                    evidence=evidence,
                    tool_id=(
                        "imu_axis_correlation"
                    ),
                    axis_a=axis_a,
                    axis_b=axis_b,
                    required_tokens=(
                        "correlation",
                    ),
                    excluded_tokens=(
                        "first-difference",
                        "maximum",
                    ),
                )
            )

            if correlation_observation is None:
                continue

            correlation = _numeric(
                correlation_observation.value
            )

            if (
                correlation is None
                or abs(
                    correlation
                )
                < self.minimum_correlation
            ):

                continue

            first_difference_observation = (
                _pair_metric(
                    evidence=evidence,
                    tool_id=(
                        "imu_axis_correlation"
                    ),
                    axis_a=axis_a,
                    axis_b=axis_b,
                    required_tokens=(
                        "first-difference",
                        "correlation",
                    ),
                )
            )

            first_difference = None

            if (
                first_difference_observation
                is not None
            ):

                first_difference = _numeric(
                    first_difference_observation.value
                )

            candidates.append(
                {
                    "axis_a":
                        axis_a,

                    "axis_b":
                        axis_b,

                    "correlation":
                        correlation,

                    "correlation_observation":
                        correlation_observation,

                    "first_difference":
                        first_difference,

                    "first_difference_observation":
                        first_difference_observation,
                }
            )

        if not candidates:

            return None

        candidates.sort(
            key=lambda item:
                abs(
                    item[
                        "correlation"
                    ]
                ),
            reverse=True,
        )

        best = candidates[
            0
        ]

        axis_a = best[
            "axis_a"
        ]

        axis_b = best[
            "axis_b"
        ]

        supporting = [
            _contribution(
                observation=best[
                    "correlation_observation"
                ],
                role=(
                    ContributionRole.SUPPORTING
                ),
                rationale=(
                    f"{axis_a} and {axis_b} show substantial "
                    "statistical cross-axis covariance "
                    f"(r = {best['correlation']:.3f})."
                ),
            )
        ]

        first_difference = best[
            "first_difference"
        ]

        behavior_description = (
            "shared variation"
        )

        if (
            first_difference is not None
            and best[
                "first_difference_observation"
            ]
            is not None
        ):

            if (
                abs(
                    first_difference
                )
                >= self.minimum_first_difference_correlation
            ):

                supporting.append(
                    _contribution(
                        observation=best[
                            "first_difference_observation"
                        ],
                        role=(
                            ContributionRole.SUPPORTING
                        ),
                        rationale=(
                            "The relationship is also present in "
                            "sample-to-sample changes, indicating "
                            "shared short-timescale behavior."
                        ),
                    )
                )

                behavior_description = (
                    "shared short-timescale variation"
                )

            else:

                supporting.append(
                    _contribution(
                        observation=best[
                            "first_difference_observation"
                        ],
                        role=(
                            ContributionRole.SUPPORTING
                        ),
                        rationale=(
                            "First-difference correlation is weaker "
                            "than the raw correlation, which is "
                            "consistent with slower shared behavior "
                            "contributing to the relationship."
                        ),
                    )
                )

                behavior_description = (
                    "shared slower variation"
                )

        # Context from other tools

        context = []

        temperature_related_axes = []

        for axis in (
            axis_a,
            axis_b,
        ):

            r2_observation = _temperature_r2(
                evidence=evidence,
                axis=axis,
            )

            if r2_observation is None:
                continue

            r_squared = _numeric(
                r2_observation.value
            )

            if (
                r_squared is None
                or r_squared < 0.50
            ):

                continue

            temperature_related_axes.append(
                axis
            )

            context.append(
                _contribution(
                    observation=r2_observation,
                    role=(
                        ContributionRole.CONTEXT
                    ),
                    rationale=(
                        f"{axis} also has a substantial statistical "
                        "relationship with temperature. This is a "
                        "possible shared covariate, not proof that "
                        "temperature causes the cross-axis "
                        "relationship."
                    ),
                )
            )

        frequency_a_observation = (
            _dominant_frequency(
                evidence=evidence,
                axis=axis_a,
            )
        )

        frequency_b_observation = (
            _dominant_frequency(
                evidence=evidence,
                axis=axis_b,
            )
        )

        matching_periodic_content = False

        if (
            frequency_a_observation is not None
            and frequency_b_observation is not None
        ):

            frequency_a = _numeric(
                frequency_a_observation.value
            )

            frequency_b = _numeric(
                frequency_b_observation.value
            )

            if (
                frequency_a is not None
                and frequency_b is not None
                and frequency_a > 0.0
                and frequency_b > 0.0
            ):

                matching_periodic_content = (
                    _frequencies_match(
                        frequency_a=frequency_a,
                        frequency_b=frequency_b,
                        absolute_tolerance_hz=0.50,
                        relative_tolerance=0.05,
                    )
                )

                if matching_periodic_content:

                    context.extend(
                        [
                            _contribution(
                                observation=(
                                    frequency_a_observation
                                ),
                                role=(
                                    ContributionRole.CONTEXT
                                ),
                                rationale=(
                                    f"{axis_a} contains a dominant "
                                    "spectral component similar in "
                                    "frequency to the component on "
                                    f"{axis_b}."
                                ),
                            ),

                            _contribution(
                                observation=(
                                    frequency_b_observation
                                ),
                                role=(
                                    ContributionRole.CONTEXT
                                ),
                                rationale=(
                                    f"{axis_b} contains a dominant "
                                    "spectral component similar in "
                                    "frequency to the component on "
                                    f"{axis_a}."
                                ),
                            ),
                        ]
                    )

        return HypothesisProposal(
            hypothesis_id=(
                f"cross_axis_covariance:"
                f"{self.sensor_group}:"
                f"{axis_a.lower()}_"
                f"{axis_b.lower()}"
            ),
            title=(
                f"Cross-Axis Covariance "
                f"({self.sensor_group.title()})"
            ),
            statement=(
                f"{axis_a} and {axis_b} show substantial "
                f"{behavior_description}. The available analyses "
                "do not establish a unique physical source for "
                "this relationship."
            ),
            state=HypothesisState.SUPPORTED,
            support_level=(
                EvidenceStrength.MODERATE
            ),
            supporting=supporting,
            context=context,
            missing_evidence=[
                MissingEvidence(
                    description=(
                        "A controlled single-axis excitation test "
                        "would help distinguish genuine cross-axis "
                        "sensor response from common physical motion."
                    ),
                    importance=(
                        EvidenceStrength.STRONG
                    ),
                ),
                MissingEvidence(
                    description=(
                        "Independent fixture or motion-reference "
                        "measurements would help test whether the "
                        "observed covariance originates outside the "
                        "sensor."
                    ),
                    importance=(
                        EvidenceStrength.MODERATE
                    ),
                ),
            ],
            limitations=[
                (
                    "Cross-axis covariance does not establish "
                    "mechanical misalignment, nonorthogonality, "
                    "cross-axis sensitivity, or electrical "
                    "crosstalk."
                ),
                (
                    "Common motion, vibration, thermal behavior, "
                    "processing, or environmental disturbances can "
                    "produce correlated sensor axes."
                ),
            ],
            metadata={
                "sensor_group":
                    self.sensor_group,

                "axis_pair": [
                    axis_a,
                    axis_b,
                ],

                "correlation":
                    best[
                        "correlation"
                    ],

                "first_difference_correlation":
                    first_difference,

                "matching_periodic_content":
                    matching_periodic_content,

                "temperature_related_axes":
                    temperature_related_axes,
            },
        )


# Default SensorQA rule collection


def default_diagnostic_rules(
) -> list:
    """Return SensorQA's built-in conservative diagnostic rules.
    
    Keeping this in one function gives the application a simple
    default while still allowing callers to:
    
        - remove rules
        - replace rules
        - add custom rules
        - run only selected diagnostic rules
    
    Returns:
        List of result values.
    """

    return [
        SharedPeriodicBehaviorRule(
            sensor_group="accelerometer"
        ),

        SharedPeriodicBehaviorRule(
            sensor_group="gyroscope"
        ),

        TemperatureAssociatedOutputVariationRule(),

        RangeBoundaryClippingRule(),

        UnexplainedCrossAxisCovarianceRule(
            sensor_group="accelerometer"
        ),

        UnexplainedCrossAxisCovarianceRule(
            sensor_group="gyroscope"
        ),
    ]
