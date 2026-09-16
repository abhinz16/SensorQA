#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any

from sensorqa.core.result_schema import (
    AnalysisResult,
    ExecutionStatus,
    QualificationStatus,
    QualifiedAnalysisResult,
    RequirementCheck,
)


class RequirementOperator(str, Enum):
    """
    Comparison operators supported by SensorQA requirements.
    """

    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    EQUAL = "=="

    ABS_LESS_THAN = "abs<"
    ABS_LESS_THAN_OR_EQUAL = "abs<="

    BETWEEN = "between"


@dataclass
class RequirementDefinition:
    """
    Defines one engineering qualification requirement.

    Example:

        tool_id = "imu_gyro_bias"
        metric_name = "Gyroscope Bias Vector Magnitude"
        operator = RequirementOperator.LESS_THAN_OR_EQUAL
        limit_value = 0.01
        unit = "rad/s"

    For BETWEEN:

        lower_limit = -2.0
        upper_limit = 2.0
    """

    requirement_id: str

    tool_id: str
    metric_name: str

    operator: RequirementOperator

    description: str

    unit: str | None = None

    limit_value: float | None = None

    lower_limit: float | None = None
    upper_limit: float | None = None

    enabled: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.requirement_id,
            str,
        ) or not self.requirement_id.strip():

            raise ValueError(
                "requirement_id must be a non-empty string."
            )

        if not isinstance(
            self.tool_id,
            str,
        ) or not self.tool_id.strip():

            raise ValueError(
                "tool_id must be a non-empty string."
            )

        if not isinstance(
            self.metric_name,
            str,
        ) or not self.metric_name.strip():

            raise ValueError(
                "metric_name must be a non-empty string."
            )

        if not isinstance(
            self.description,
            str,
        ) or not self.description.strip():

            raise ValueError(
                "description must be a non-empty string."
            )

        if not isinstance(
            self.operator,
            RequirementOperator,
        ):

            raise TypeError(
                "operator must be a RequirementOperator."
            )

        self._validate_limits()

    def _validate_limits(self) -> None:
        """Validate the numerical structure of this requirement.
        
        Returns:
            None.
        """

        if self.operator == RequirementOperator.BETWEEN:

            if self.lower_limit is None:
                raise ValueError(
                    "BETWEEN requirements require lower_limit."
                )

            if self.upper_limit is None:
                raise ValueError(
                    "BETWEEN requirements require upper_limit."
                )

            if self.lower_limit > self.upper_limit:
                raise ValueError(
                    "lower_limit cannot exceed upper_limit."
                )

            return

        if self.limit_value is None:

            raise ValueError(
                f"Requirement operator '{self.operator.value}' "
                "requires limit_value."
            )


@dataclass
class RequirementSet:
    """
    Collection of requirements belonging to one qualification
    profile.

    Examples:

        "Prototype IMU Rev A"
        "Production Acceptance"
        "Navigation IMU Requirements"
    """

    name: str

    requirements: list[
        RequirementDefinition
    ] = field(
        default_factory=list
    )

    description: str | None = None

    version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def enabled_requirements(
        self,
    ) -> list[RequirementDefinition]:

        """Return only the enabled requirement definitions.
        
        Returns:
            List of result values.
        """
        return [
            requirement
            for requirement
            in self.requirements
            if requirement.enabled
        ]

    def for_tool(
        self,
        tool_id: str,
    ) -> list[RequirementDefinition]:

        """Return requirements that apply to a tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """
        return [
            requirement
            for requirement
            in self.enabled_requirements()
            if requirement.tool_id == tool_id
        ]


@dataclass
class QualificationSummary:
    """
    Summary across multiple qualified analysis results.
    """

    overall_status: QualificationStatus

    total_requirements: int

    passed: int
    failed: int
    not_evaluated: int

    qualified_results: list[
        QualifiedAnalysisResult
    ] = field(
        default_factory=list
    )

    messages: list[str] = field(
        default_factory=list
    )


class RequirementsEngine:
    """
    Applies engineering requirements to SensorQA analysis results.

    Important separation:

        Analysis tool:
            "GX bias = 0.004 rad/s"

        Requirements engine:
            "Requirement <= 0.005 rad/s"
            therefore PASS

    Analysis tools therefore remain reusable across organizations
    with different acceptance criteria.
    """

    def qualify_analysis(
        self,
        analysis: AnalysisResult,
        requirements: (
            RequirementSet
            | list[RequirementDefinition]
        ),
    ) -> QualifiedAnalysisResult:
        """Apply requirements relevant to one AnalysisResult.
        
        Args:
            analysis: Value for `analysis`.
            requirements: Value for `requirements`.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        if isinstance(
            requirements,
            RequirementSet,
        ):

            definitions = requirements.for_tool(
                analysis.tool_id
            )

        else:

            definitions = [
                requirement
                for requirement
                in requirements
                if (
                    requirement.enabled
                    and requirement.tool_id
                    == analysis.tool_id
                )
            ]

        checks: list[
            RequirementCheck
        ] = []

        # If the underlying analysis did not succeed, none of its
        # requirements can legitimately pass or fail.

        if (
            analysis.status
            != ExecutionStatus.SUCCESS
        ):

            for requirement in definitions:

                checks.append(
                    self._not_evaluated_check(
                        requirement=requirement,
                        measured_value=None,
                        measured_unit=None,
                        message=(
                            "Requirement was not evaluated because "
                            f"analysis tool '{analysis.tool_id}' "
                            f"finished with status "
                            f"'{analysis.status.value}'."
                        ),
                    )
                )

            return QualifiedAnalysisResult(
                analysis=analysis,
                requirement_checks=checks,
            )

        # Build metric lookup

        metrics_by_name = {
            metric.name: metric
            for metric
            in analysis.metrics
        }

        # Evaluate requirements

        for requirement in definitions:

            metric = metrics_by_name.get(
                requirement.metric_name
            )

            if metric is None:

                checks.append(
                    self._not_evaluated_check(
                        requirement=requirement,
                        measured_value=None,
                        measured_unit=None,
                        message=(
                            "Required metric was not produced by "
                            f"tool '{analysis.tool_id}'."
                        ),
                    )
                )

                continue

            measured_value = metric.value
            measured_unit = metric.unit

            # Numerical requirement checks only operate on finite
            # scalar real values.

            if not self._is_numeric_scalar(
                measured_value
            ):

                checks.append(
                    self._not_evaluated_check(
                        requirement=requirement,
                        measured_value=measured_value,
                        measured_unit=measured_unit,
                        message=(
                            "Metric value is not a numerical scalar "
                            "and cannot be compared with this "
                            "engineering requirement."
                        ),
                    )
                )

                continue

            numeric_value = float(
                measured_value
            )

            # Unit compatibility
            #
            # V1 deliberately does not silently convert requirement
            # units here. Requirements should normally be specified
            # using SensorQA's canonical metric units.

            unit_error = self._unit_compatibility_error(
                required_unit=requirement.unit,
                measured_unit=measured_unit,
            )

            if unit_error is not None:

                checks.append(
                    self._not_evaluated_check(
                        requirement=requirement,
                        measured_value=numeric_value,
                        measured_unit=measured_unit,
                        message=unit_error,
                    )
                )

                continue

            # Actual comparison

            passed = self._compare(
                measured_value=numeric_value,
                requirement=requirement,
            )

            status = (
                QualificationStatus.PASS
                if passed
                else QualificationStatus.FAIL
            )

            checks.append(
                RequirementCheck(
                    metric_name=requirement.metric_name,
                    measured_value=numeric_value,
                    unit=measured_unit,
                    requirement_description=(
                        requirement.description
                    ),
                    status=status,
                    limit_value=self._display_limit_value(
                        requirement
                    ),
                    operator=requirement.operator.value,
                    message=self._comparison_message(
                        measured_value=numeric_value,
                        measured_unit=measured_unit,
                        requirement=requirement,
                        passed=passed,
                    ),
                )
            )

        return QualifiedAnalysisResult(
            analysis=analysis,
            requirement_checks=checks,
        )

    def qualify_results(
        self,
        analyses: list[AnalysisResult],
        requirements: RequirementSet,
    ) -> QualificationSummary:
        """Apply an entire requirement profile to multiple
        AnalysisResult objects.
        
        Args:
            analyses: Value for `analyses`.
            requirements: Value for `requirements`.
        
        Returns:
            QualificationSummary returned by the function.
        """

        qualified_results: list[
            QualifiedAnalysisResult
        ] = []

        analysis_by_tool_id = {
            analysis.tool_id: analysis
            for analysis
            in analyses
        }

        # Qualify analyses that exist

        for analysis in analyses:

            qualified_results.append(
                self.qualify_analysis(
                    analysis=analysis,
                    requirements=requirements,
                )
            )

        # Requirements may refer to a tool that never ran.
        #
        # Those requirements should not disappear from the report.
        # They should be NOT_EVALUATED.

        missing_tool_requirements: dict[
            str,
            list[RequirementDefinition],
        ] = {}

        for requirement in (
            requirements.enabled_requirements()
        ):

            if (
                requirement.tool_id
                not in analysis_by_tool_id
            ):

                missing_tool_requirements.setdefault(
                    requirement.tool_id,
                    [],
                ).append(
                    requirement
                )

        messages: list[str] = []

        for tool_id, definitions in (
            missing_tool_requirements.items()
        ):

            messages.append(
                f"{len(definitions)} requirement(s) reference "
                f"tool '{tool_id}', but no analysis result from "
                "that tool is available."
            )

        # Count all evaluated requirement checks

        all_checks = [
            check
            for qualified
            in qualified_results
            for check
            in qualified.requirement_checks
        ]

        # Missing-tool requirements have no corresponding
        # QualifiedAnalysisResult, but still count as not evaluated.

        missing_tool_count = sum(
            len(
                definitions
            )
            for definitions
            in missing_tool_requirements.values()
        )

        passed = sum(
            check.status
            == QualificationStatus.PASS
            for check
            in all_checks
        )

        failed = sum(
            check.status
            == QualificationStatus.FAIL
            for check
            in all_checks
        )

        not_evaluated = (
            sum(
                check.status
                == QualificationStatus.NOT_EVALUATED
                for check
                in all_checks
            )
            + missing_tool_count
        )

        total_requirements = (
            passed
            + failed
            + not_evaluated
        )

        # Overall qualification logic
        #
        # Any failed requirement -> FAIL
        #
        # No failures but at least one unevaluated -> NOT_EVALUATED
        #
        # Every requirement evaluated and passed -> PASS
        #
        # An empty requirement set -> NOT_EVALUATED

        if failed > 0:

            overall_status = (
                QualificationStatus.FAIL
            )

        elif (
            total_requirements == 0
            or not_evaluated > 0
        ):

            overall_status = (
                QualificationStatus.NOT_EVALUATED
            )

        else:

            overall_status = (
                QualificationStatus.PASS
            )

        return QualificationSummary(
            overall_status=overall_status,
            total_requirements=total_requirements,
            passed=passed,
            failed=failed,
            not_evaluated=not_evaluated,
            qualified_results=qualified_results,
            messages=messages,
        )

    # Comparisons

    @staticmethod
    def _compare(
        measured_value: float,
        requirement: RequirementDefinition,
    ) -> bool:
        """Apply one requirement operator.
        
        Args:
            measured_value: Value for `measured_value`.
            requirement: Value for `requirement`.
        
        Returns:
            Boolean result.
        """

        operator = requirement.operator

        if operator == RequirementOperator.LESS_THAN:

            return (
                measured_value
                < float(
                    requirement.limit_value
                )
            )

        if (
            operator
            == RequirementOperator.LESS_THAN_OR_EQUAL
        ):

            return (
                measured_value
                <= float(
                    requirement.limit_value
                )
            )

        if operator == RequirementOperator.GREATER_THAN:

            return (
                measured_value
                > float(
                    requirement.limit_value
                )
            )

        if (
            operator
            == RequirementOperator.GREATER_THAN_OR_EQUAL
        ):

            return (
                measured_value
                >= float(
                    requirement.limit_value
                )
            )

        if operator == RequirementOperator.EQUAL:

            return (
                measured_value
                == float(
                    requirement.limit_value
                )
            )

        if operator == RequirementOperator.ABS_LESS_THAN:

            return (
                abs(
                    measured_value
                )
                < float(
                    requirement.limit_value
                )
            )

        if (
            operator
            == RequirementOperator.ABS_LESS_THAN_OR_EQUAL
        ):

            return (
                abs(
                    measured_value
                )
                <= float(
                    requirement.limit_value
                )
            )

        if operator == RequirementOperator.BETWEEN:

            return (
                float(
                    requirement.lower_limit
                )
                <= measured_value
                <= float(
                    requirement.upper_limit
                )
            )

        raise ValueError(
            f"Unsupported requirement operator "
            f"'{operator.value}'."
        )

    # Requirement messages

    @classmethod
    def _comparison_message(
        cls,
        measured_value: float,
        measured_unit: str | None,
        requirement: RequirementDefinition,
        passed: bool,
    ) -> str:

        """Return comparison message.
        
        Args:
            measured_value: Measured value used by this function.
            measured_unit: Measured unit used by this function.
            requirement: Requirement used by this function.
            passed: Passed used by this function.
        
        Returns:
            Requested text value.
        """
        status_text = (
            "passed"
            if passed
            else "failed"
        )

        measured_text = cls._format_value(
            measured_value,
            measured_unit,
        )

        requirement_text = (
            cls._requirement_expression(
                requirement
            )
        )

        return (
            f"Measured value {measured_text} "
            f"{status_text} requirement "
            f"'{requirement_text}'."
        )

    @staticmethod
    def _requirement_expression(
        requirement: RequirementDefinition,
    ) -> str:

        """Calculate requirement expression.
        
        Args:
            requirement: Requirement used by this function.
        
        Returns:
            Requested text value.
        """
        unit = (
            f" {requirement.unit}"
            if requirement.unit
            else ""
        )

        if (
            requirement.operator
            == RequirementOperator.BETWEEN
        ):

            return (
                f"{requirement.lower_limit}"
                f"{unit} <= value <= "
                f"{requirement.upper_limit}"
                f"{unit}"
            )

        return (
            f"value "
            f"{requirement.operator.value} "
            f"{requirement.limit_value}"
            f"{unit}"
        )

    @staticmethod
    def _format_value(
        value: Any,
        unit: str | None,
    ) -> str:

        """Format value.
        
        Args:
            value: Value to process.
            unit: Engineering unit.
        
        Returns:
            Requested text value.
        """
        if isinstance(
            value,
            Real,
        ) and not isinstance(
            value,
            bool,
        ):

            text = (
                f"{float(value):.6g}"
            )

        else:

            text = str(
                value
            )

        if unit:

            return (
                f"{text} {unit}"
            )

        return text

    # NOT_EVALUATED handling

    @staticmethod
    def _not_evaluated_check(
        requirement: RequirementDefinition,
        measured_value: Any,
        measured_unit: str | None,
        message: str,
    ) -> RequirementCheck:

        """Return not evaluated check.
        
        Args:
            requirement: Requirement used by this function.
            measured_value: Measured value used by this function.
            measured_unit: Measured unit used by this function.
            message: Message text.
        
        Returns:
            RequirementCheck returned by this function.
        """
        return RequirementCheck(
            metric_name=requirement.metric_name,
            measured_value=measured_value,
            unit=measured_unit,
            requirement_description=(
                requirement.description
            ),
            status=QualificationStatus.NOT_EVALUATED,
            limit_value=(
                RequirementsEngine._display_limit_value(
                    requirement
                )
            ),
            operator=requirement.operator.value,
            message=message,
        )

    # Unit handling

    @staticmethod
    def _unit_compatibility_error(
        required_unit: str | None,
        measured_unit: str | None,
    ) -> str | None:
        """V1 uses strict unit matching.
        
        We intentionally avoid silently assuming that, for example:
        
            deg/s == rad/s
        
        Requirements should normally use canonical SensorQA units.
        
        Args:
            required_unit: Value for `required_unit`.
            measured_unit: Value for `measured_unit`.
        
        Returns:
            str | None returned by the function.
        """

        if required_unit is None:

            return None

        if measured_unit is None:

            return (
                f"Requirement expects unit '{required_unit}', "
                "but the analysis metric has no unit."
            )

        if (
            RequirementsEngine._normalize_unit_text(
                required_unit
            )
            != RequirementsEngine._normalize_unit_text(
                measured_unit
            )
        ):

            return (
                f"Requirement unit '{required_unit}' does not "
                f"match measured metric unit '{measured_unit}'. "
                "The requirement was not evaluated."
            )

        return None

    @staticmethod
    def _normalize_unit_text(
        unit: str,
    ) -> str:

        """Normalize unit text.
        
        Args:
            unit: Engineering unit.
        
        Returns:
            Requested text value.
        """
        return (
            unit
            .strip()
            .replace(
                "²",
                "^2",
            )
            .replace(
                " ",
                "",
            )
            .lower()
        )

    # Helpers

    @staticmethod
    def _is_numeric_scalar(
        value: Any,
    ) -> bool:

        """Return whether numeric scalar.
        
        Args:
            value: Value to process.
        
        Returns:
            Boolean result.
        """
        if isinstance(
            value,
            bool,
        ):

            return False

        if not isinstance(
            value,
            Real,
        ):

            return False

        # Handles NaN and infinity without depending on NumPy.
        numeric = float(
            value
        )

        return (
            numeric == numeric
            and numeric
            not in (
                float("inf"),
                float("-inf"),
            )
        )

    @staticmethod
    def _display_limit_value(
        requirement: RequirementDefinition,
    ) -> Any:
        """RequirementCheck currently has a single limit_value field.
        
        For BETWEEN requirements we preserve both bounds as a
        serializable dictionary.
        
        Args:
            requirement: Value for `requirement`.
        
        Returns:
            Any returned by the function.
        """

        if (
            requirement.operator
            == RequirementOperator.BETWEEN
        ):

            return {
                "lower":
                    requirement.lower_limit,

                "upper":
                    requirement.upper_limit,
            }

        return requirement.limit_value
