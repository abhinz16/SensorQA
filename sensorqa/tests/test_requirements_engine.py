#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.core.requirements_engine import (
    QualificationSummary,
    RequirementDefinition,
    RequirementOperator,
    RequirementSet,
    RequirementsEngine,
)
from sensorqa.core.result_schema import (
    AnalysisResult,
    ExecutionStatus,
    QualificationStatus,
)


# Test helpers


def make_analysis(
    *,
    tool_id: str = "test_tool",
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
) -> AnalysisResult:
    """Construct a minimal AnalysisResult for requirements tests.
    
    Args:
        tool_id: Registered tool identifier.
        status: Status value.
    
    Returns:
        Analysis result containing metrics and messages.
    """

    return AnalysisResult(
        tool_id=tool_id,
        tool_name="Test Tool",
        tool_version="1.0.0",
        status=status,
    )


def add_metric(
    analysis: AnalysisResult,
    *,
    name: str,
    value,
    unit: str | None = None,
) -> None:
    """Add one metric using the public AnalysisResult API.
    
    Args:
        analysis: Value for `analysis`.
        name: Name of the item.
        value: Value to process.
        unit: Engineering unit.
    
    Returns:
        None.
    """

    analysis.add_metric(
        name=name,
        value=value,
        unit=unit,
    )


def make_requirement(
    *,
    requirement_id: str = "req_1",
    tool_id: str = "test_tool",
    metric_name: str = "Test Metric",
    operator: RequirementOperator = (
        RequirementOperator.LESS_THAN_OR_EQUAL
    ),
    limit_value: float | None = 10.0,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    unit: str | None = None,
    enabled: bool = True,
) -> RequirementDefinition:
    """Construct a requirement with concise defaults.
    
    Args:
        requirement_id: Identifier for requirement.
        tool_id: Registered tool identifier.
        metric_name: Value for `metric_name`.
        operator: Value for `operator`.
        limit_value: Value for `limit_value`.
        lower_limit: Value for `lower_limit`.
        upper_limit: Value for `upper_limit`.
        unit: Engineering unit.
        enabled: Value for `enabled`.
    
    Returns:
        RequirementDefinition returned by the function.
    """

    return RequirementDefinition(
        requirement_id=requirement_id,
        tool_id=tool_id,
        metric_name=metric_name,
        operator=operator,
        description=(
            f"Test requirement {requirement_id}"
        ),
        unit=unit,
        limit_value=limit_value,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        enabled=enabled,
    )


# Basic comparison operators


@pytest.mark.parametrize(
    (
        "operator",
        "measured",
        "limit_value",
        "expected_status",
    ),
    [
        (
            RequirementOperator.LESS_THAN,
            4.9,
            5.0,
            QualificationStatus.PASS,
        ),
        (
            RequirementOperator.LESS_THAN,
            5.0,
            5.0,
            QualificationStatus.FAIL,
        ),
        (
            RequirementOperator.LESS_THAN_OR_EQUAL,
            5.0,
            5.0,
            QualificationStatus.PASS,
        ),
        (
            RequirementOperator.GREATER_THAN,
            5.1,
            5.0,
            QualificationStatus.PASS,
        ),
        (
            RequirementOperator.GREATER_THAN,
            5.0,
            5.0,
            QualificationStatus.FAIL,
        ),
        (
            RequirementOperator.GREATER_THAN_OR_EQUAL,
            5.0,
            5.0,
            QualificationStatus.PASS,
        ),
        (
            RequirementOperator.EQUAL,
            5.0,
            5.0,
            QualificationStatus.PASS,
        ),
        (
            RequirementOperator.EQUAL,
            5.0001,
            5.0,
            QualificationStatus.FAIL,
        ),
    ],
)
def test_basic_requirement_operators(
    operator,
    measured,
    limit_value,
    expected_status,
):
    """Verify ordinary numerical comparisons, including exact
    boundary behavior.
    
    Args:
        operator: Value for `operator`.
        measured: Value for `measured`.
        limit_value: Value for `limit_value`.
        expected_status: Value for `expected_status`.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=measured,
    )

    requirement = make_requirement(
        operator=operator,
        limit_value=limit_value,
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    assert len(
        qualified.requirement_checks
    ) == 1

    check = qualified.requirement_checks[
        0
    ]

    assert check.status == expected_status


# Absolute-value requirement


@pytest.mark.parametrize(
    (
        "measured",
        "expected_status",
    ),
    [
        (
            0.004,
            QualificationStatus.PASS,
        ),
        (
            -0.004,
            QualificationStatus.PASS,
        ),
        (
            0.005,
            QualificationStatus.PASS,
        ),
        (
            -0.005,
            QualificationStatus.PASS,
        ),
        (
            0.0051,
            QualificationStatus.FAIL,
        ),
        (
            -0.0051,
            QualificationStatus.FAIL,
        ),
    ],
)
def test_absolute_less_than_or_equal(
    measured,
    expected_status,
):
    """Bias-style requirements must treat positive and negative
    magnitudes symmetrically.
    
    Args:
        measured: Value for `measured`.
        expected_status: Value for `expected_status`.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="imu_gyro_bias"
    )

    add_metric(
        analysis,
        name="GX Bias",
        value=measured,
        unit="rad/s",
    )

    requirement = make_requirement(
        requirement_id="gyro_bias_x",
        tool_id="imu_gyro_bias",
        metric_name="GX Bias",
        operator=(
            RequirementOperator.ABS_LESS_THAN_OR_EQUAL
        ),
        limit_value=0.005,
        unit="rad/s",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    check = qualified.requirement_checks[
        0
    ]

    assert check.status == expected_status


# BETWEEN


@pytest.mark.parametrize(
    (
        "measured",
        "expected_status",
    ),
    [
        (
            -2.0,
            QualificationStatus.PASS,
        ),
        (
            0.0,
            QualificationStatus.PASS,
        ),
        (
            2.0,
            QualificationStatus.PASS,
        ),
        (
            -2.01,
            QualificationStatus.FAIL,
        ),
        (
            2.01,
            QualificationStatus.FAIL,
        ),
    ],
)
def test_between_is_inclusive(
    measured,
    expected_status,
):
    """BETWEEN uses inclusive lower and upper bounds.
    
    Args:
        measured: Value for `measured`.
        expected_status: Value for `expected_status`.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="imu_six_position_calibration"
    )

    add_metric(
        analysis,
        name="AX Sensitivity Error Percent",
        value=measured,
        unit="%",
    )

    requirement = make_requirement(
        requirement_id="ax_sensitivity",
        tool_id="imu_six_position_calibration",
        metric_name="AX Sensitivity Error Percent",
        operator=RequirementOperator.BETWEEN,
        limit_value=None,
        lower_limit=-2.0,
        upper_limit=2.0,
        unit="%",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    assert (
        qualified.requirement_checks[
            0
        ].status
        == expected_status
    )


# Unit handling


def test_matching_units_are_evaluated():
    """Identical canonical units should allow evaluation.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=0.003,
        unit="rad/s",
    )

    requirement = make_requirement(
        operator=(
            RequirementOperator.LESS_THAN_OR_EQUAL
        ),
        limit_value=0.005,
        unit="rad/s",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    check = qualified.requirement_checks[
        0
    ]

    assert (
        check.status
        == QualificationStatus.PASS
    )


def test_unit_text_normalization():
    """Minor text differences currently normalized by the V1 engine
    should not prevent evaluation.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=0.05,
        unit="m/s²",
    )

    requirement = make_requirement(
        limit_value=0.1,
        unit="m/s^2",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    assert (
        qualified.requirement_checks[
            0
        ].status
        == QualificationStatus.PASS
    )


def test_mismatched_units_are_not_evaluated():
    """V1 requirements must not silently compare different physical
    units.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=0.02,
        unit="rad/s",
    )

    requirement = make_requirement(
        limit_value=0.5,
        unit="deg/s",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    check = qualified.requirement_checks[
        0
    ]

    assert (
        check.status
        == QualificationStatus.NOT_EVALUATED
    )

    assert "does not match" in (
        check.message.lower()
    )


def test_required_unit_with_unitless_metric_is_not_evaluated():
    """A metric with no unit cannot satisfy a requirement that
    explicitly expects one.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=2.0,
        unit=None,
    )

    requirement = make_requirement(
        limit_value=3.0,
        unit="m/s^2",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    check = qualified.requirement_checks[
        0
    ]

    assert (
        check.status
        == QualificationStatus.NOT_EVALUATED
    )


# Missing / invalid metrics


def test_missing_metric_is_not_evaluated():
    """Missing metrics must never silently PASS.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    requirement = make_requirement(
        metric_name="Missing Metric",
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    check = qualified.requirement_checks[
        0
    ]

    assert (
        check.status
        == QualificationStatus.NOT_EVALUATED
    )

    assert check.measured_value is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not numeric",
        [1.0, 2.0],
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_numeric_or_non_finite_metric_is_not_evaluated(
    value,
):
    """Engineering numerical comparisons require finite scalar
    numerical measurements.
    
    Args:
        value: Value to process.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=value,
    )

    requirement = make_requirement(
        limit_value=10.0,
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    assert (
        qualified.requirement_checks[
            0
        ].status
        == QualificationStatus.NOT_EVALUATED
    )


# Analysis execution state


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.SKIPPED,
        ExecutionStatus.ERROR,
    ],
)
def test_non_successful_analysis_is_not_evaluated(
    status,
):
    """A requirement cannot PASS or FAIL when its source analysis did
    not complete successfully.
    
    Args:
        status: Status value.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        status=status
    )

    # Even if a partial metric somehow exists, the analysis result
    # itself is not successful and should not qualify.
    add_metric(
        analysis,
        name="Test Metric",
        value=1.0,
    )

    requirement = make_requirement(
        limit_value=5.0,
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement
        ],
    )

    check = qualified.requirement_checks[
        0
    ]

    assert (
        check.status
        == QualificationStatus.NOT_EVALUATED
    )


# Enabled / disabled requirements


def test_disabled_requirement_is_not_applied():
    """Disabled requirements should not appear in qualification
    checks.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    add_metric(
        analysis,
        name="Test Metric",
        value=999.0,
    )

    requirement = make_requirement(
        limit_value=1.0,
        enabled=False,
    )

    requirement_set = RequirementSet(
        name="Test Profile",
        requirements=[
            requirement
        ],
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=requirement_set,
    )

    assert (
        qualified.requirement_checks
        == []
    )


# Tool filtering


def test_requirements_for_other_tools_are_ignored():
    """qualify_analysis should only apply requirements belonging to
    that AnalysisResult's tool_id.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        name="Metric A",
        value=1.0,
    )

    requirement_a = make_requirement(
        requirement_id="req_a",
        tool_id="tool_a",
        metric_name="Metric A",
        limit_value=2.0,
    )

    requirement_b = make_requirement(
        requirement_id="req_b",
        tool_id="tool_b",
        metric_name="Metric B",
        limit_value=2.0,
    )

    engine = RequirementsEngine()

    qualified = engine.qualify_analysis(
        analysis=analysis,
        requirements=[
            requirement_a,
            requirement_b,
        ],
    )

    assert len(
        qualified.requirement_checks
    ) == 1

    assert (
        qualified.requirement_checks[
            0
        ].metric_name
        == "Metric A"
    )


# Overall profile qualification


def test_all_pass_gives_overall_pass():
    """Every enabled requirement evaluated and passed -> PASS.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        name="Metric 1",
        value=1.0,
    )

    add_metric(
        analysis,
        name="Metric 2",
        value=2.0,
    )

    requirement_set = RequirementSet(
        name="Test Profile",
        requirements=[
            make_requirement(
                requirement_id="req_1",
                tool_id="tool_a",
                metric_name="Metric 1",
                limit_value=2.0,
            ),
            make_requirement(
                requirement_id="req_2",
                tool_id="tool_a",
                metric_name="Metric 2",
                limit_value=3.0,
            ),
        ],
    )

    engine = RequirementsEngine()

    summary = engine.qualify_results(
        analyses=[
            analysis
        ],
        requirements=requirement_set,
    )

    assert isinstance(
        summary,
        QualificationSummary,
    )

    assert (
        summary.overall_status
        == QualificationStatus.PASS
    )

    assert summary.total_requirements == 2
    assert summary.passed == 2
    assert summary.failed == 0
    assert summary.not_evaluated == 0


def test_any_failure_gives_overall_fail():
    """A known FAIL takes precedence over PASS results.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        name="Good Metric",
        value=1.0,
    )

    add_metric(
        analysis,
        name="Bad Metric",
        value=10.0,
    )

    requirement_set = RequirementSet(
        name="Test Profile",
        requirements=[
            make_requirement(
                requirement_id="good",
                tool_id="tool_a",
                metric_name="Good Metric",
                limit_value=2.0,
            ),
            make_requirement(
                requirement_id="bad",
                tool_id="tool_a",
                metric_name="Bad Metric",
                limit_value=2.0,
            ),
        ],
    )

    engine = RequirementsEngine()

    summary = engine.qualify_results(
        analyses=[
            analysis
        ],
        requirements=requirement_set,
    )

    assert (
        summary.overall_status
        == QualificationStatus.FAIL
    )

    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.not_evaluated == 0


def test_failure_takes_precedence_over_not_evaluated():
    """If one requirement definitely fails while another cannot be
    evaluated, the overall profile is still known to FAIL.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        name="Failing Metric",
        value=10.0,
    )

    requirement_set = RequirementSet(
        name="Test Profile",
        requirements=[
            make_requirement(
                requirement_id="failure",
                tool_id="tool_a",
                metric_name="Failing Metric",
                limit_value=2.0,
            ),
            make_requirement(
                requirement_id="missing",
                tool_id="tool_a",
                metric_name="Missing Metric",
                limit_value=2.0,
            ),
        ],
    )

    engine = RequirementsEngine()

    summary = engine.qualify_results(
        analyses=[
            analysis
        ],
        requirements=requirement_set,
    )

    assert (
        summary.overall_status
        == QualificationStatus.FAIL
    )

    assert summary.failed == 1
    assert summary.not_evaluated == 1


def test_unevaluated_without_failures_gives_overall_not_evaluated():
    """An incomplete requirement profile must not receive an overall
    PASS.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        name="Available Metric",
        value=1.0,
    )

    requirement_set = RequirementSet(
        name="Test Profile",
        requirements=[
            make_requirement(
                requirement_id="available",
                tool_id="tool_a",
                metric_name="Available Metric",
                limit_value=2.0,
            ),
            make_requirement(
                requirement_id="missing",
                tool_id="tool_a",
                metric_name="Missing Metric",
                limit_value=2.0,
            ),
        ],
    )

    engine = RequirementsEngine()

    summary = engine.qualify_results(
        analyses=[
            analysis
        ],
        requirements=requirement_set,
    )

    assert (
        summary.overall_status
        == QualificationStatus.NOT_EVALUATED
    )

    assert summary.passed == 1
    assert summary.failed == 0
    assert summary.not_evaluated == 1


# Missing tools


def test_requirement_for_missing_tool_is_counted_not_evaluated():
    """Requirements must not disappear simply because the required
    analysis tool was not executed.
    
    Returns:
        None.
    """

    analysis = make_analysis(
        tool_id="tool_a"
    )

    add_metric(
        analysis,
        name="Metric A",
        value=1.0,
    )

    requirement_set = RequirementSet(
        name="Test Profile",
        requirements=[
            make_requirement(
                requirement_id="req_a",
                tool_id="tool_a",
                metric_name="Metric A",
                limit_value=2.0,
            ),
            make_requirement(
                requirement_id="req_missing_tool",
                tool_id="tool_b",
                metric_name="Metric B",
                limit_value=2.0,
            ),
        ],
    )

    engine = RequirementsEngine()

    summary = engine.qualify_results(
        analyses=[
            analysis
        ],
        requirements=requirement_set,
    )

    assert (
        summary.overall_status
        == QualificationStatus.NOT_EVALUATED
    )

    assert summary.total_requirements == 2
    assert summary.passed == 1
    assert summary.failed == 0
    assert summary.not_evaluated == 1

    assert any(
        "tool_b" in message
        for message
        in summary.messages
    )


# Empty requirement profile


def test_empty_requirement_set_is_not_evaluated():
    """Having zero requirements must not be interpreted as PASS.
    
    Returns:
        None.
    """

    analysis = make_analysis()

    requirement_set = RequirementSet(
        name="Empty Profile",
        requirements=[],
    )

    engine = RequirementsEngine()

    summary = engine.qualify_results(
        analyses=[
            analysis
        ],
        requirements=requirement_set,
    )

    assert summary.total_requirements == 0

    assert (
        summary.overall_status
        == QualificationStatus.NOT_EVALUATED
    )


# Requirement-definition validation


def test_between_requires_lower_limit():
    """BETWEEN is malformed without a lower bound.
    
    Returns:
        None.
    """

    with pytest.raises(
        ValueError
    ):

        make_requirement(
            operator=RequirementOperator.BETWEEN,
            limit_value=None,
            lower_limit=None,
            upper_limit=2.0,
        )


def test_between_requires_upper_limit():
    """BETWEEN is malformed without an upper bound.
    
    Returns:
        None.
    """

    with pytest.raises(
        ValueError
    ):

        make_requirement(
            operator=RequirementOperator.BETWEEN,
            limit_value=None,
            lower_limit=-2.0,
            upper_limit=None,
        )


def test_between_rejects_reversed_bounds():
    """lower_limit > upper_limit is invalid.
    
    Returns:
        None.
    """

    with pytest.raises(
        ValueError
    ):

        make_requirement(
            operator=RequirementOperator.BETWEEN,
            limit_value=None,
            lower_limit=5.0,
            upper_limit=2.0,
        )


def test_non_between_operator_requires_limit():
    """Ordinary comparison operators require limit_value.
    
    Returns:
        None.
    """

    with pytest.raises(
        ValueError
    ):

        make_requirement(
            operator=(
                RequirementOperator.LESS_THAN_OR_EQUAL
            ),
            limit_value=None,
        )
