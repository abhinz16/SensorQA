#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.core.result_schema import (
    EvidenceStrength,
    ExecutionStatus,
)
from sensorqa.diagnostics.diagnostic_rules import (
    RangeBoundaryClippingRule,
    SharedPeriodicBehaviorRule,
    TemperatureAssociatedOutputVariationRule,
    UnexplainedCrossAxisCovarianceRule,
    default_diagnostic_rules,
)
from sensorqa.diagnostics.evidence_engine import (
    EvidenceBundle,
    MetricObservation,
)
from sensorqa.diagnostics.hypothesis_engine import (
    HypothesisEngine,
    HypothesisState,
)


# Helpers


def observation(
    tool_id: str,
    metric_name: str,
    value,
    *,
    unit: str | None = None,
) -> MetricObservation:
    """Construct a normalized metric observation.
    
    Args:
        tool_id: Registered tool identifier.
        metric_name: Value for `metric_name`.
        value: Value to process.
        unit: Engineering unit.
    
    Returns:
        MetricObservation returned by the function.
    """

    return MetricObservation(
        tool_id=tool_id,
        tool_name=tool_id,
        metric_name=metric_name,
        value=value,
        unit=unit,
        description=None,
        tool_version="1.0.0",
        analysis_status=(
            ExecutionStatus.SUCCESS.value
        ),
    )


def bundle(
    *,
    metrics=None,
    successful_tools=None,
) -> EvidenceBundle:
    """Build an EvidenceBundle with explicit successful tool states.
    
    Args:
        metrics: Value for `metrics`.
        successful_tools: Value for `successful_tools`.
    
    Returns:
        EvidenceBundle returned by the function.
    """

    successful_tools = (
        successful_tools
        or []
    )

    return EvidenceBundle(
        metrics=list(
            metrics
            or []
        ),
        analysis_status={
            tool_id:
                ExecutionStatus.SUCCESS.value
            for tool_id
            in successful_tools
        },
    )


# Shared periodic behavior


def test_periodic_rule_requires_successful_psd():
    """Check that periodic rule requires successful psd.
    
    Returns:
        None.
    """
    evidence = bundle(
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
                unit="Hz",
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.1,
                unit="Hz",
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is None


def test_matching_psd_peaks_give_weak_support_without_correlation():
    """Check that matching psd peaks give weak support without correlation.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_psd"
        ],
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
                unit="Hz",
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.2,
                unit="Hz",
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is not None

    assert (
        proposal.state
        == HypothesisState.SUPPORTED
    )

    assert (
        proposal.support_level
        == EvidenceStrength.WEAK
    )


def test_matching_psd_and_high_correlation_give_moderate_support():
    """Check that matching psd and high correlation give moderate support.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_psd",
            "imu_axis_correlation",
        ],
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
                unit="Hz",
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.1,
                unit="Hz",
            ),
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.84,
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is not None

    assert (
        proposal.support_level
        == EvidenceStrength.MODERATE
    )

    assert len(
        proposal.supporting
    ) == 3


def test_low_correlation_does_not_upgrade_periodic_support():
    """Check that low correlation does not upgrade periodic support.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_psd",
            "imu_axis_correlation",
        ],
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.1,
            ),
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.20,
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is not None

    assert (
        proposal.support_level
        == EvidenceStrength.WEAK
    )


def test_high_correlation_alone_does_not_create_periodic_hypothesis():
    """Cross-axis correlation without corresponding spectral evidence
    must not be interpreted as shared periodic behavior.
    
    Returns:
        None.
    """

    evidence = bundle(
        successful_tools=[
            "imu_psd",
            "imu_axis_correlation",
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.95,
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is None


def test_different_psd_frequencies_do_not_create_shared_periodic_hypothesis():
    """Check that different psd frequencies do not create shared periodic hypothesis.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_psd",
            "imu_axis_correlation",
        ],
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                20.0,
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                70.0,
            ),
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is None


def test_shared_periodic_rule_does_not_claim_specific_vibration_source():
    """Check that shared periodic rule does not claim specific vibration source.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_psd"
        ],
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.1,
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "gyroscope"
    ).evaluate(
        evidence
    )

    assert proposal is not None

    statement = proposal.statement.lower()

    assert "shared periodic" in statement

    assert "motor" not in statement
    assert "bearing" not in statement
    assert "vibration detected" not in statement


def test_shared_periodic_rule_requests_independent_reference():
    """Check that shared periodic rule requests independent reference.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_psd"
        ],
        metrics=[
            observation(
                "imu_psd",
                "AX Dominant Frequency",
                30.0,
            ),
            observation(
                "imu_psd",
                "AY Dominant Frequency",
                30.1,
            ),
        ],
    )

    proposal = SharedPeriodicBehaviorRule(
        "accelerometer"
    ).evaluate(
        evidence
    )

    assert proposal is not None

    assert any(
        "independent"
        in missing.description.lower()
        for missing
        in proposal.missing_evidence
    )


# Temperature-associated output variation


def test_temperature_rule_requires_successful_temperature_tool():
    """Check that temperature rule requires successful temperature tool.
    
    Returns:
        None.
    """
    evidence = bundle(
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.90,
            )
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule()
        .evaluate(
            evidence
        )
    )

    assert proposal is None


def test_low_temperature_r2_does_not_create_hypothesis():
    """Check that low temperature r2 does not create hypothesis.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_temperature_stability"
        ],
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.20,
            )
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule(
            minimum_r_squared=0.50
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is None


def test_temperature_r2_at_threshold_is_supported():
    """Check that temperature r2 at threshold is supported.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_temperature_stability"
        ],
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.50,
            ),
            observation(
                "imu_temperature_stability",
                "GX Temperature Coefficient",
                0.001,
            ),
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule(
            minimum_r_squared=0.50
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        proposal.state
        == HypothesisState.SUPPORTED
    )

    assert (
        proposal.support_level
        == EvidenceStrength.MODERATE
    )


def test_temperature_rule_can_identify_multiple_axes():
    """Check that temperature rule can identify multiple axes.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_temperature_stability"
        ],
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.80,
            ),
            observation(
                "imu_temperature_stability",
                "GY Temperature Regression R2",
                0.75,
            ),
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule()
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    affected = {
        item[
            "axis"
        ]
        for item
        in proposal.metadata[
            "affected_axes"
        ]
    }

    assert affected == {
        "GX",
        "GY",
    }


def test_temperature_rule_does_not_claim_temperature_caused_drift():
    """High R² is statistical association, not causal proof.
    
    Returns:
        None.
    """

    evidence = bundle(
        successful_tools=[
            "imu_temperature_stability"
        ],
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.98,
            ),
            observation(
                "imu_temperature_stability",
                "GX Temperature Coefficient",
                0.003,
            ),
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule()
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        "caused"
        not in proposal.statement.lower()
    )

    assert (
        "thermal drift detected"
        not in proposal.statement.lower()
    )

    assert any(
        "does not establish temperature as the causal"
        in limitation.lower()
        for limitation
        in proposal.limitations
    )


def test_temperature_rule_requests_controlled_thermal_experiment():
    """Check that temperature rule requests controlled thermal experiment.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_temperature_stability"
        ],
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.90,
            )
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule()
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    descriptions = [
        missing.description.lower()
        for missing
        in proposal.missing_evidence
    ]

    assert any(
        "controlled thermal experiment"
        in description
        for description
        in descriptions
    )


def test_predicted_temperature_change_is_context_not_support():
    """Check that predicted temperature change is context not support.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_temperature_stability"
        ],
        metrics=[
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.80,
            ),
            observation(
                "imu_temperature_stability",
                "GX Predicted Change Across Temperature Span",
                0.05,
            ),
        ],
    )

    proposal = (
        TemperatureAssociatedOutputVariationRule()
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert len(
        proposal.context
    ) == 1

    assert (
        proposal.context[
            0
        ].reference.metric_name
        == "GX Predicted Change Across Temperature Span"
    )


# Range-boundary behavior


def test_range_rule_requires_successful_saturation_tool():
    """Check that range rule requires successful saturation tool.
    
    Returns:
        None.
    """
    evidence = bundle(
        metrics=[
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                10,
            )
        ],
    )

    proposal = RangeBoundaryClippingRule().evaluate(
        evidence
    )

    assert proposal is None


def test_no_exact_limit_samples_produces_no_range_hypothesis():
    """Check that no exact limit samples produces no range hypothesis.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_saturation"
        ],
        metrics=[
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                0,
            )
        ],
    )

    proposal = RangeBoundaryClippingRule().evaluate(
        evidence
    )

    assert proposal is None


def test_isolated_exact_limit_samples_give_moderate_support():
    """Check that isolated exact limit samples give moderate support.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_saturation"
        ],
        metrics=[
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                2,
            ),
            observation(
                "imu_saturation",
                "GX Longest Exact Limit Run",
                1,
            ),
        ],
    )

    proposal = RangeBoundaryClippingRule().evaluate(
        evidence
    )

    assert proposal is not None

    assert (
        proposal.state
        == HypothesisState.SUPPORTED
    )

    assert (
        proposal.support_level
        == EvidenceStrength.MODERATE
    )


def test_repeated_boundary_run_gives_strong_support():
    """Check that repeated boundary run gives strong support.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_saturation"
        ],
        metrics=[
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                20,
            ),
            observation(
                "imu_saturation",
                "GX Longest Exact Limit Run",
                7,
            ),
        ],
    )

    proposal = RangeBoundaryClippingRule(
        strong_run_length=3
    ).evaluate(
        evidence
    )

    assert proposal is not None

    assert (
        proposal.state
        == HypothesisState.SUPPORTED
    )

    assert (
        proposal.support_level
        == EvidenceStrength.STRONG
    )


def test_out_of_range_data_create_conflicting_range_evidence():
    """Values beyond the supposed range boundary undermine the
    assumption that the configured range represents the actual
    clipping limit.
    
    Returns:
        None.
    """

    evidence = bundle(
        successful_tools=[
            "imu_saturation"
        ],
        metrics=[
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                20,
            ),
            observation(
                "imu_saturation",
                "GX Longest Exact Limit Run",
                8,
            ),
            observation(
                "imu_saturation",
                "GX Out of Range Count",
                3,
            ),
        ],
    )

    proposal = RangeBoundaryClippingRule().evaluate(
        evidence
    )

    assert proposal is not None

    assert (
        proposal.state
        == HypothesisState.CONFLICTING_EVIDENCE
    )

    assert proposal.support_level is None

    assert len(
        proposal.conflicting
    ) == 1


def test_range_rule_does_not_claim_clipping_location():
    """Check that range rule does not claim clipping location.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_saturation"
        ],
        metrics=[
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                20,
            ),
            observation(
                "imu_saturation",
                "GX Longest Exact Limit Run",
                10,
            ),
        ],
    )

    proposal = RangeBoundaryClippingRule().evaluate(
        evidence
    )

    assert proposal is not None

    statement = proposal.statement.lower()

    assert "clipping-like" in statement

    assert "adc failure" not in statement
    assert "hardware clipping detected" not in statement

    assert any(
        "analog electronics"
        in limitation.lower()
        for limitation
        in proposal.limitations
    )


# Cross-axis covariance


def test_cross_axis_rule_requires_successful_correlation_tool():
    """Check that cross axis rule requires successful correlation tool.
    
    Returns:
        None.
    """
    evidence = bundle(
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            )
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is None


def test_cross_axis_correlation_below_threshold_produces_no_hypothesis():
    """Check that cross axis correlation below threshold produces no hypothesis.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.50,
            )
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope",
            minimum_correlation=0.70,
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is None


def test_high_cross_axis_correlation_gives_moderate_support():
    """Check that high cross axis correlation gives moderate support.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            )
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        proposal.state
        == HypothesisState.SUPPORTED
    )

    assert (
        proposal.support_level
        == EvidenceStrength.MODERATE
    )


def test_high_cross_axis_correlation_does_not_claim_misalignment():
    """This is one of SensorQA's central scientific safeguards.
    
    Returns:
        None.
    """

    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GX-GY Correlation",
                0.96,
            )
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        "misalignment detected"
        not in proposal.statement.lower()
    )

    assert (
        "misaligned"
        not in proposal.statement.lower()
    )

    assert any(
        "does not establish mechanical misalignment"
        in limitation.lower()
        for limitation
        in proposal.limitations
    )


def test_high_first_difference_correlation_indicates_short_timescale_behavior():
    """Check that high first difference correlation indicates short timescale behavior.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GX-GY Correlation",
                0.90,
            ),
            observation(
                "imu_axis_correlation",
                "GX-GY First-Difference Correlation",
                0.85,
            ),
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        "short-timescale"
        in proposal.statement.lower()
    )


def test_low_first_difference_correlation_indicates_slower_shared_behavior():
    """Check that low first difference correlation indicates slower shared behavior.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GX-GY Correlation",
                0.90,
            ),
            observation(
                "imu_axis_correlation",
                "GX-GY First-Difference Correlation",
                0.10,
            ),
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        "slower variation"
        in proposal.statement.lower()
    )


def test_negative_high_correlation_is_also_detected():
    """Strong anticorrelation is still substantial statistical
    coupling.
    
    Returns:
        None.
    """

    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "AX-AY Correlation",
                -0.88,
            )
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "accelerometer"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        proposal.metadata[
            "correlation"
        ]
        == pytest.approx(
            -0.88
        )
    )


# Cross-tool context


def test_cross_axis_rule_adds_temperature_as_context_not_cause():
    """Check that cross axis rule adds temperature as context not cause.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation",
            "imu_temperature_stability",
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.88,
            ),
            observation(
                "imu_temperature_stability",
                "GY Temperature Regression R2",
                0.85,
            ),
            observation(
                "imu_temperature_stability",
                "GZ Temperature Regression R2",
                0.82,
            ),
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert len(
        proposal.context
    ) == 2

    rationale_text = " ".join(
        contribution.rationale.lower()
        for contribution
        in proposal.context
    )

    assert (
        "possible shared covariate"
        in rationale_text
    )

    assert (
        "not proof"
        in rationale_text
    )


def test_cross_axis_rule_adds_matching_psd_as_context():
    """Check that cross axis rule adds matching psd as context.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation",
            "imu_psd",
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            ),
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.1,
            ),
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        proposal.metadata[
            "matching_periodic_content"
        ]
        is True
    )

    assert len(
        proposal.context
    ) == 2


def test_cross_axis_rule_does_not_mark_unmatched_psd_as_matching():
    """Check that cross axis rule does not mark unmatched psd as matching.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation",
            "imu_psd",
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            ),
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                20.0,
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                80.0,
            ),
        ],
    )

    proposal = (
        UnexplainedCrossAxisCovarianceRule(
            "gyroscope"
        )
        .evaluate(
            evidence
        )
    )

    assert proposal is not None

    assert (
        proposal.metadata[
            "matching_periodic_content"
        ]
        is False
    )


# Rule construction


def test_invalid_shared_periodic_sensor_group_rejected():
    """Check that invalid shared periodic sensor group rejected.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        SharedPeriodicBehaviorRule(
            "magnetometer"
        )


def test_invalid_cross_axis_sensor_group_rejected():
    """Check that invalid cross axis sensor group rejected.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        UnexplainedCrossAxisCovarianceRule(
            "magnetometer"
        )


# Default diagnostic rule set


def test_default_rules_have_unique_ids():
    """Check that default rules have unique ids.
    
    Returns:
        None.
    """
    rules = default_diagnostic_rules()

    rule_ids = [
        rule.rule_id
        for rule
        in rules
    ]

    assert len(
        rule_ids
    ) == len(
        set(
            rule_ids
        )
    )


def test_default_rule_count():
    """Current V1 built-in diagnostic collection:
        2 periodic
        1 temperature
        1 range
        2 covariance
    
    Returns:
        None.
    """

    assert len(
        default_diagnostic_rules()
    ) == 6


# Integration with HypothesisEngine


def test_builtin_rules_produce_valid_hypotheses_through_engine():
    """A compact integration test confirming that built-in diagnostic
    rules produce references that HypothesisEngine can resolve.
    
    Returns:
        None.
    """

    evidence = bundle(
        successful_tools=[
            "imu_psd",
            "imu_axis_correlation",
            "imu_temperature_stability",
            "imu_saturation",
        ],
        metrics=[
            observation(
                "imu_psd",
                "GY Dominant Frequency",
                42.0,
            ),
            observation(
                "imu_psd",
                "GZ Dominant Frequency",
                42.1,
            ),
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            ),
            observation(
                "imu_axis_correlation",
                "GY-GZ First-Difference Correlation",
                0.80,
            ),
            observation(
                "imu_temperature_stability",
                "GX Temperature Regression R2",
                0.80,
            ),
            observation(
                "imu_temperature_stability",
                "GX Temperature Coefficient",
                0.001,
            ),
            observation(
                "imu_saturation",
                "GX Exact Limit Count",
                20,
            ),
            observation(
                "imu_saturation",
                "GX Longest Exact Limit Run",
                5,
            ),
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=evidence,
        rules=default_diagnostic_rules(),
    )

    assert len(
        result.hypotheses
    ) >= 3

    # Built-in rules should not contain broken metric references.
    for hypothesis in result.hypotheses:

        assert not any(
            "could not be resolved"
            in warning.lower()
            for warning
            in hypothesis.validation_warnings
        )


def test_final_builtin_hypotheses_retain_root_cause_limitation():
    """Check that final builtin hypotheses retain root cause limitation.
    
    Returns:
        None.
    """
    evidence = bundle(
        successful_tools=[
            "imu_axis_correlation"
        ],
        metrics=[
            observation(
                "imu_axis_correlation",
                "GY-GZ Correlation",
                0.90,
            )
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=evidence,
        rules=[
            UnexplainedCrossAxisCovarianceRule(
                "gyroscope"
            )
        ],
    )

    assert len(
        result.hypotheses
    ) == 1

    hypothesis = result.hypotheses[
        0
    ]

    assert any(
        (
            "root cause"
            in limitation.lower()
            or "does not establish"
            in limitation.lower()
        )
        for limitation
        in hypothesis.limitations
    )
