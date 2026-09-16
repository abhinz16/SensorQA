#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from types import SimpleNamespace

import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.core.result_schema import (
    AnalysisResult,
    EvidenceStrength,
    ExecutionStatus,
)
from sensorqa.diagnostics.evidence_engine import (
    EvidenceBundle,
    EvidenceEngine,
    MetricReference,
    ObservationType,
)


# Helpers


def make_analysis(
    *,
    tool_id: str = "test_tool",
    tool_name: str = "Test Tool",
    tool_version: str = "1.0.0",
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
) -> AnalysisResult:
    """Construct one real SensorQA AnalysisResult.
    
    Args:
        tool_id: Registered tool identifier.
        tool_name: Value for `tool_name`.
        tool_version: Value for `tool_version`.
        status: Status value.
    
    Returns:
        Analysis result containing metrics and messages.
    """

    return AnalysisResult(
        tool_id=tool_id,
        tool_name=tool_name,
        tool_version=tool_version,
        status=status,
    )


def add_metric(
    analysis: AnalysisResult,
    name: str,
    value,
    *,
    unit: str | None = None,
    description: str | None = None,
) -> None:
    """Add a metric using AnalysisResult's public API.
    
    Args:
        analysis: Value for `analysis`.
        name: Name of the item.
        value: Value to process.
        unit: Engineering unit.
        description: Value for `description`.
    
    Returns:
        None.
    """

    analysis.add_metric(
        name=name,
        value=value,
        unit=unit,
        description=description,
    )


def add_evidence(
    analysis: AnalysisResult,
    statement: str,
    strength: EvidenceStrength,
    *,
    supporting_metrics: list[str] | None = None,
) -> None:
    """Add evidence using AnalysisResult's public API.
    
    Args:
        analysis: Value for `analysis`.
        statement: Value for `statement`.
        strength: Value for `strength`.
        supporting_metrics: Value for `supporting_metrics`.
    
    Returns:
        None.
    """

    analysis.add_evidence(
        statement=statement,
        strength=strength,
        supporting_metrics=(
            supporting_metrics
            or []
        ),
    )


# Basic metric collection


def test_successful_analysis_metrics_are_collected():
    """Check that successful analysis metrics are collected.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_gyro_bias",
        tool_name="Gyroscope Bias",
    )

    add_metric(
        analysis,
        "GX Bias",
        0.001,
        unit="rad/s",
        description="Mean GX output.",
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert len(
        bundle.metrics
    ) == 1

    metric = bundle.metrics[
        0
    ]

    assert metric.tool_id == "imu_gyro_bias"
    assert metric.tool_name == "Gyroscope Bias"
    assert metric.metric_name == "GX Bias"

    assert metric.value == pytest.approx(
        0.001
    )

    assert metric.unit == "rad/s"

    assert (
        metric.description
        == "Mean GX output."
    )


def test_multiple_metrics_from_same_tool_are_preserved():
    """Check that multiple metrics from same tool are preserved.
    
    Returns:
        None.
    """
    analysis = make_analysis()

    add_metric(
        analysis,
        "Metric A",
        1.0,
    )

    add_metric(
        analysis,
        "Metric B",
        2.0,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert [
        metric.metric_name
        for metric
        in bundle.metrics
    ] == [
        "Metric A",
        "Metric B",
    ]


def test_metrics_from_multiple_tools_are_preserved():
    """Check that metrics from multiple tools are preserved.
    
    Returns:
        None.
    """
    first = make_analysis(
        tool_id="tool_a",
    )

    second = make_analysis(
        tool_id="tool_b",
    )

    add_metric(
        first,
        "Metric A",
        1.0,
    )

    add_metric(
        second,
        "Metric B",
        2.0,
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    assert {
        metric.tool_id
        for metric
        in bundle.metrics
    } == {
        "tool_a",
        "tool_b",
    }


# Metric provenance


def test_metric_reference_contains_tool_and_metric():
    """Check that metric reference contains tool and metric.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_axis_correlation",
    )

    add_metric(
        analysis,
        "GY-GZ Correlation",
        0.82,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    reference = (
        bundle.metrics[
            0
        ].reference
    )

    assert isinstance(
        reference,
        MetricReference,
    )

    assert (
        reference.tool_id
        == "imu_axis_correlation"
    )

    assert (
        reference.metric_name
        == "GY-GZ Correlation"
    )


def test_same_metric_name_from_different_tools_remains_distinct():
    """Check that same metric name from different tools remains distinct.
    
    Returns:
        None.
    """
    first = make_analysis(
        tool_id="tool_a",
    )

    second = make_analysis(
        tool_id="tool_b",
    )

    add_metric(
        first,
        "Sample Count",
        100,
    )

    add_metric(
        second,
        "Sample Count",
        200,
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    results = bundle.metrics_named(
        "Sample Count"
    )

    assert len(
        results
    ) == 2

    assert {
        result.tool_id
        for result
        in results
    } == {
        "tool_a",
        "tool_b",
    }


# Metric querying


def test_metric_lookup_uses_tool_and_metric_name():
    """Check that metric lookup uses tool and metric name.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_temperature_stability",
    )

    add_metric(
        analysis,
        "GX Temperature Regression R2",
        0.86,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    metric = bundle.metric(
        tool_id=(
            "imu_temperature_stability"
        ),
        metric_name=(
            "GX Temperature Regression R2"
        ),
    )

    assert metric is not None

    assert metric.value == pytest.approx(
        0.86
    )


def test_missing_metric_lookup_returns_none():
    """Check that missing metric lookup returns none.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle()

    assert (
        bundle.metric(
            "tool_a",
            "does not exist",
        )
        is None
    )


def test_metrics_for_tool_filters_correctly():
    """Check that metrics for tool filters correctly.
    
    Returns:
        None.
    """
    first = make_analysis(
        tool_id="tool_a",
    )

    second = make_analysis(
        tool_id="tool_b",
    )

    add_metric(
        first,
        "A1",
        1,
    )

    add_metric(
        first,
        "A2",
        2,
    )

    add_metric(
        second,
        "B1",
        3,
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    results = bundle.metrics_for_tool(
        "tool_a"
    )

    assert [
        metric.metric_name
        for metric
        in results
    ] == [
        "A1",
        "A2",
    ]


# Execution status safeguards


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.SKIPPED,
        ExecutionStatus.ERROR,
    ],
)
def test_non_successful_analysis_does_not_contribute_metrics(
    status,
):
    """Partial metrics from unsuccessful analyses must not become
    diagnostic evidence.
    
    Args:
        status: Status value.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        status=status
    )

    add_metric(
        analysis,
        "Partial Metric",
        123.0,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert bundle.metrics == []


def test_successful_tool_status_recorded():
    """Check that successful tool status recorded.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_psd"
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert (
        bundle.analysis_status[
            "imu_psd"
        ]
        == ExecutionStatus.SUCCESS.value
    )

    assert bundle.tool_succeeded(
        "imu_psd"
    )


def test_skipped_tool_status_recorded():
    """Check that skipped tool status recorded.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_allan_deviation",
        status=ExecutionStatus.SKIPPED,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert (
        bundle.analysis_status[
            "imu_allan_deviation"
        ]
        == ExecutionStatus.SKIPPED.value
    )

    assert not bundle.tool_succeeded(
        "imu_allan_deviation"
    )


def test_tool_available_in_bundle():
    """Check that tool available in bundle.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="tool_a"
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert bundle.tool_available_in_bundle(
        "tool_a"
    )

    assert not bundle.tool_available_in_bundle(
        "tool_b"
    )


# Evidence collection


def test_direct_evidence_is_collected():
    """Check that direct evidence is collected.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_axis_correlation"
    )

    add_metric(
        analysis,
        "GY-GZ Correlation",
        0.82,
    )

    add_evidence(
        analysis,
        (
            "GY and GZ exhibit substantial "
            "cross-axis covariance."
        ),
        EvidenceStrength.MODERATE,
        supporting_metrics=[
            "GY-GZ Correlation"
        ],
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert len(
        bundle.evidence
    ) == 1

    record = bundle.evidence[
        0
    ]

    assert (
        record.tool_id
        == "imu_axis_correlation"
    )

    assert (
        record.strength
        == EvidenceStrength.MODERATE
    )

    assert (
        "cross-axis covariance"
        in record.statement
    )


def test_evidence_supporting_metric_is_resolved():
    """Check that evidence supporting metric is resolved.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_axis_correlation"
    )

    add_metric(
        analysis,
        "GY-GZ Correlation",
        0.82,
    )

    add_evidence(
        analysis,
        "Substantial covariance is present.",
        EvidenceStrength.MODERATE,
        supporting_metrics=[
            "GY-GZ Correlation"
        ],
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    record = bundle.evidence[
        0
    ]

    assert (
        record.supporting_metrics
        == [
            MetricReference(
                tool_id=(
                    "imu_axis_correlation"
                ),
                metric_name=(
                    "GY-GZ Correlation"
                ),
            )
        ]
    )

    assert (
        record.unresolved_metric_names
        == []
    )


def test_unresolved_evidence_metric_is_reported():
    """Check that unresolved evidence metric is reported.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="custom_tool"
    )

    add_evidence(
        analysis,
        "A custom pattern was detected.",
        EvidenceStrength.WEAK,
        supporting_metrics=[
            "Metric That Was Never Added"
        ],
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert len(
        bundle.evidence
    ) == 1

    record = bundle.evidence[
        0
    ]

    assert (
        record.supporting_metrics
        == []
    )

    assert (
        record.unresolved_metric_names
        == [
            "Metric That Was Never Added"
        ]
    )

    assert any(
        "not found"
        in warning.lower()
        for warning
        in bundle.warnings
    )


# Evidence strength


def test_evidence_strength_is_preserved():
    """Check that evidence strength is preserved.
    
    Returns:
        None.
    """
    analysis = make_analysis()

    add_evidence(
        analysis,
        "Strong observation.",
        EvidenceStrength.STRONG,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert (
        bundle.evidence[
            0
        ].strength
        == EvidenceStrength.STRONG
    )


def test_engine_does_not_upgrade_evidence_strength():
    """Two moderate observations do not automatically become strong.
    
    Returns:
        None.
    """

    first = make_analysis(
        tool_id="tool_a"
    )

    second = make_analysis(
        tool_id="tool_b"
    )

    add_evidence(
        first,
        "Observation A",
        EvidenceStrength.MODERATE,
    )

    add_evidence(
        second,
        "Observation B",
        EvidenceStrength.MODERATE,
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    assert len(
        bundle.evidence
    ) == 2

    assert all(
        record.strength
        == EvidenceStrength.MODERATE
        for record
        in bundle.evidence
    )


def test_evidence_at_least_filters_without_modifying_strength():
    """Check that evidence at least filters without modifying strength.
    
    Returns:
        None.
    """
    analysis = make_analysis()

    add_evidence(
        analysis,
        "Weak",
        EvidenceStrength.WEAK,
    )

    add_evidence(
        analysis,
        "Moderate",
        EvidenceStrength.MODERATE,
    )

    add_evidence(
        analysis,
        "Strong",
        EvidenceStrength.STRONG,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    filtered = bundle.evidence_at_least(
        EvidenceStrength.MODERATE
    )

    assert [
        record.strength
        for record
        in filtered
    ] == [
        EvidenceStrength.MODERATE,
        EvidenceStrength.STRONG,
    ]


# Failed/skipped evidence safeguard


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.SKIPPED,
        ExecutionStatus.ERROR,
    ],
)
def test_non_successful_analysis_does_not_contribute_evidence(
    status,
):
    """Check that non successful analysis does not contribute evidence.
    
    Args:
        status: Status value.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        status=status
    )

    add_evidence(
        analysis,
        "This should not enter diagnostics.",
        EvidenceStrength.STRONG,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert bundle.evidence == []


# Warnings and messages


def test_analysis_warning_is_preserved_as_notice():
    """Check that analysis warning is preserved as notice.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_temperature_stability"
    )

    analysis.add_warning(
        (
            "Temperature and elapsed time may "
            "be confounded."
        )
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert len(
        bundle.notices
    ) == 1

    notice = bundle.notices[
        0
    ]

    assert (
        notice.notice_type
        == ObservationType.WARNING
    )

    assert (
        "confounded"
        in notice.message
    )


def test_analysis_message_is_preserved_as_notice():
    """Check that analysis message is preserved as notice.
    
    Returns:
        None.
    """
    analysis = make_analysis()

    analysis.add_message(
        "Analysis completed normally."
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert len(
        bundle.notices
    ) == 1

    assert (
        bundle.notices[
            0
        ].notice_type
        == ObservationType.MESSAGE
    )


def test_skipped_analysis_notice_is_still_preserved():
    """A skipped tool contributes no metrics/evidence, but its reason
    should remain available to reports and diagnostics.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="imu_allan_deviation",
        status=ExecutionStatus.SKIPPED,
    )

    analysis.add_warning(
        "Recording duration is too short."
    )

    add_metric(
        analysis,
        "Partial Allan Metric",
        1.0,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert bundle.metrics == []

    assert len(
        bundle.notices
    ) == 1

    assert (
        "too short"
        in bundle.notices[
            0
        ].message
    )


def test_warning_is_not_promoted_to_evidence():
    """Context and evidence must remain separate.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    analysis.add_warning(
        "Possible external disturbance."
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert len(
        bundle.notices
    ) == 1

    assert bundle.evidence == []


# Duplicate tool results


def test_duplicate_tool_results_generate_warning():
    """Check that duplicate tool results generate warning.
    
    Returns:
        None.
    """
    first = make_analysis(
        tool_id="imu_psd"
    )

    second = make_analysis(
        tool_id="imu_psd"
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    assert any(
        "multiple analysisresult"
        in warning.lower()
        for warning
        in bundle.warnings
    )


# Evidence IDs


def test_evidence_ids_are_deterministic_within_tool():
    """Check that evidence ids are deterministic within tool.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="imu_psd"
    )

    add_evidence(
        analysis,
        "First finding.",
        EvidenceStrength.WEAK,
    )

    add_evidence(
        analysis,
        "Second finding.",
        EvidenceStrength.MODERATE,
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    assert [
        record.evidence_id
        for record
        in bundle.evidence
    ] == [
        "imu_psd:evidence:001",
        "imu_psd:evidence:002",
    ]


def test_evidence_id_counter_is_per_tool():
    """Check that evidence id counter is per tool.
    
    Returns:
        None.
    """
    first = make_analysis(
        tool_id="tool_a"
    )

    second = make_analysis(
        tool_id="tool_b"
    )

    add_evidence(
        first,
        "A",
        EvidenceStrength.WEAK,
    )

    add_evidence(
        second,
        "B",
        EvidenceStrength.WEAK,
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    assert [
        record.evidence_id
        for record
        in bundle.evidence
    ] == [
        "tool_a:evidence:001",
        "tool_b:evidence:001",
    ]


# Evidence querying


def test_evidence_for_tool_filters_correctly():
    """Check that evidence for tool filters correctly.
    
    Returns:
        None.
    """
    first = make_analysis(
        tool_id="tool_a"
    )

    second = make_analysis(
        tool_id="tool_b"
    )

    add_evidence(
        first,
        "A",
        EvidenceStrength.WEAK,
    )

    add_evidence(
        second,
        "B",
        EvidenceStrength.MODERATE,
    )

    bundle = EvidenceEngine().build(
        [
            first,
            second,
        ]
    )

    records = bundle.evidence_for_tool(
        "tool_b"
    )

    assert len(
        records
    ) == 1

    assert (
        records[
            0
        ].statement
        == "B"
    )


# Robustness against malformed custom result objects


def test_missing_tool_id_uses_unknown_tool():
    """EvidenceEngine should remain robust to a malformed external
    result object rather than crashing report generation.
    
    Returns:
        None.
    """

    fake = SimpleNamespace(
        tool_id="",
        tool_name="Broken Tool",
        tool_version="1.0",
        status=ExecutionStatus.SUCCESS,
        metrics=[],
        evidence=[],
        warnings=[],
        messages=[],
    )

    bundle = EvidenceEngine().build(
        [
            fake
        ]
    )

    assert (
        "unknown_tool"
        in bundle.analysis_status
    )


def test_invalid_metric_without_name_is_ignored():
    """Check that invalid metric without name is ignored.
    
    Returns:
        None.
    """
    fake_metric = SimpleNamespace(
        name="",
        value=10,
        unit=None,
        description=None,
        metadata={},
    )

    fake_analysis = SimpleNamespace(
        tool_id="custom_tool",
        tool_name="Custom",
        tool_version="1.0",
        status=ExecutionStatus.SUCCESS,
        metrics=[
            fake_metric
        ],
        evidence=[],
        warnings=[],
        messages=[],
    )

    bundle = EvidenceEngine().build(
        [
            fake_analysis
        ]
    )

    assert bundle.metrics == []


def test_invalid_evidence_strength_is_excluded():
    """Malformed custom plug-in evidence must not enter the validated
    evidence bundle.
    
    Returns:
        None.
    """

    fake_evidence = SimpleNamespace(
        statement="Potential observation",
        strength="VERY_STRONG",
        supporting_metrics=[],
        metadata={},
    )

    fake_analysis = SimpleNamespace(
        tool_id="custom_tool",
        tool_name="Custom",
        tool_version="1.0",
        status=ExecutionStatus.SUCCESS,
        metrics=[],
        evidence=[
            fake_evidence
        ],
        warnings=[],
        messages=[],
    )

    bundle = EvidenceEngine().build(
        [
            fake_analysis
        ]
    )

    assert bundle.evidence == []

    assert any(
        "unrecognized strength"
        in warning.lower()
        for warning
        in bundle.warnings
    )


def test_evidence_without_statement_is_excluded():
    """Check that evidence without statement is excluded.
    
    Returns:
        None.
    """
    fake_evidence = SimpleNamespace(
        statement="",
        strength=EvidenceStrength.MODERATE,
        supporting_metrics=[],
        metadata={},
    )

    fake_analysis = SimpleNamespace(
        tool_id="custom_tool",
        tool_name="Custom",
        tool_version="1.0",
        status=ExecutionStatus.SUCCESS,
        metrics=[],
        evidence=[
            fake_evidence
        ],
        warnings=[],
        messages=[],
    )

    bundle = EvidenceEngine().build(
        [
            fake_analysis
        ]
    )

    assert bundle.evidence == []

    assert any(
        "without a usable statement"
        in warning.lower()
        for warning
        in bundle.warnings
    )


# Serialization


def test_bundle_to_dict_contains_main_sections():
    """Check that bundle to dict contains main sections.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        "Metric A",
        1.0,
        unit="V",
    )

    add_evidence(
        analysis,
        "Metric A is elevated.",
        EvidenceStrength.WEAK,
        supporting_metrics=[
            "Metric A"
        ],
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    data = bundle.to_dict()

    assert "metrics" in data
    assert "evidence" in data
    assert "notices" in data
    assert "analysis_status" in data
    assert "tool_names" in data
    assert "warnings" in data

    assert (
        data[
            "metrics"
        ][
            0
        ][
            "metric_name"
        ]
        == "Metric A"
    )


def test_serialized_evidence_keeps_supporting_metric_provenance():
    """Check that serialized evidence keeps supporting metric provenance.
    
    Returns:
        None.
    """
    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        "Metric A",
        1.0,
    )

    add_evidence(
        analysis,
        "Observation A",
        EvidenceStrength.MODERATE,
        supporting_metrics=[
            "Metric A"
        ],
    )

    bundle = EvidenceEngine().build(
        [
            analysis
        ]
    )

    data = bundle.to_dict()

    supporting = (
        data[
            "evidence"
        ][
            0
        ][
            "supporting_metrics"
        ]
    )

    assert supporting == [
        {
            "tool_id":
                "tool_a",

            "metric_name":
                "Metric A",
        }
    ]
