#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from sensorqa.core.result_schema import (
    AnalysisResult,
    EvidenceStrength,
    ExecutionStatus,
)


# Observation types


class ObservationType(str, Enum):
    """
    Types of information collected from analysis results.
    """

    METRIC = "metric"
    EVIDENCE = "evidence"
    WARNING = "warning"
    MESSAGE = "message"


# Metric references


@dataclass(frozen=True)
class MetricReference:
    """
    Stable reference to one metric produced by one analysis tool.
    """

    tool_id: str
    metric_name: str


@dataclass
class MetricObservation:
    """
    Normalized metric produced by an analysis tool.

    No interpretation is added here. This object preserves the
    engineering measurement and its provenance.
    """

    tool_id: str

    tool_name: str | None

    metric_name: str

    value: Any

    unit: str | None

    description: str | None = None

    tool_version: str | None = None

    analysis_status: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def reference(
        self,
    ) -> MetricReference:

        """Return the evidence reference represented by this object.
        
        Returns:
            MetricReference returned by this function.
        """
        return MetricReference(
            tool_id=self.tool_id,
            metric_name=self.metric_name,
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "tool_id":
                self.tool_id,

            "tool_name":
                self.tool_name,

            "tool_version":
                self.tool_version,

            "metric_name":
                self.metric_name,

            "value":
                self.value,

            "unit":
                self.unit,

            "description":
                self.description,

            "analysis_status":
                self.analysis_status,

            "metadata":
                dict(
                    self.metadata
                ),
        }


# Direct evidence records


@dataclass
class EvidenceRecord:
    """
    Normalized evidence statement originating from an analysis tool.

    Important:
        EvidenceEngine preserves the strength assigned by the
        originating tool. It does not upgrade evidence strength
        simply because several metrics exist.
    """

    evidence_id: str

    tool_id: str

    tool_name: str | None

    statement: str

    strength: EvidenceStrength

    supporting_metrics: list[
        MetricReference
    ] = field(
        default_factory=list
    )

    unresolved_metric_names: list[str] = field(
        default_factory=list
    )

    tool_version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "evidence_id":
                self.evidence_id,

            "tool_id":
                self.tool_id,

            "tool_name":
                self.tool_name,

            "tool_version":
                self.tool_version,

            "statement":
                self.statement,

            "strength":
                self.strength.value,

            "supporting_metrics": [
                {
                    "tool_id":
                        reference.tool_id,

                    "metric_name":
                        reference.metric_name,
                }
                for reference
                in self.supporting_metrics
            ],

            "unresolved_metric_names":
                list(
                    self.unresolved_metric_names
                ),

            "metadata":
                dict(
                    self.metadata
                ),
        }


# Notices


@dataclass
class EvidenceNotice:
    """
    Warning or informational message preserved from an analysis.
    """

    tool_id: str

    notice_type: ObservationType

    message: str

    def to_dict(
        self,
    ) -> dict[str, str]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "tool_id":
                self.tool_id,

            "type":
                self.notice_type.value,

            "message":
                self.message,
        }


# Evidence bundle


@dataclass
class EvidenceBundle:
    """
    Cross-tool evidence representation.

    This becomes the input to later diagnostic and hypothesis
    engines.
    """

    metrics: list[
        MetricObservation
    ] = field(
        default_factory=list
    )

    evidence: list[
        EvidenceRecord
    ] = field(
        default_factory=list
    )

    notices: list[
        EvidenceNotice
    ] = field(
        default_factory=list
    )

    analysis_status: dict[
        str,
        str,
    ] = field(
        default_factory=dict
    )

    tool_names: dict[
        str,
        str,
    ] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    # Metric querying

    def metric(
        self,
        tool_id: str,
        metric_name: str,
    ) -> MetricObservation | None:
        """Return one specific tool/metric observation.
        
        Args:
            tool_id: Registered tool identifier.
            metric_name: Value for `metric_name`.
        
        Returns:
            MetricObservation | None returned by the function.
        """

        for observation in self.metrics:

            if (
                observation.tool_id == tool_id
                and observation.metric_name
                == metric_name
            ):

                return observation

        return None

    def metrics_named(
        self,
        metric_name: str,
    ) -> list[MetricObservation]:
        """Find every metric having a particular name.
        
        Args:
            metric_name: Value for `metric_name`.
        
        Returns:
            List of result values.
        """

        return [
            observation
            for observation
            in self.metrics
            if observation.metric_name
            == metric_name
        ]

    def metrics_for_tool(
        self,
        tool_id: str,
    ) -> list[MetricObservation]:

        """Return metric observations produced by one tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """
        return [
            observation
            for observation
            in self.metrics
            if observation.tool_id
            == tool_id
        ]

    # Evidence querying

    def evidence_for_tool(
        self,
        tool_id: str,
    ) -> list[EvidenceRecord]:

        """Return evidence records produced by one tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            List of result values.
        """
        return [
            record
            for record
            in self.evidence
            if record.tool_id
            == tool_id
        ]

    def evidence_at_least(
        self,
        minimum_strength: EvidenceStrength,
    ) -> list[EvidenceRecord]:
        """Return evidence at or above a specified strength.
        
        This is filtering only. It does not combine or modify
        evidence strength.
        
        Args:
            minimum_strength: Minimum accepted strength.
        
        Returns:
            List of result values.
        """

        ranking = {
            EvidenceStrength.WEAK:
                1,

            EvidenceStrength.MODERATE:
                2,

            EvidenceStrength.STRONG:
                3,
        }

        threshold = ranking[
            minimum_strength
        ]

        return [
            record
            for record
            in self.evidence
            if ranking[
                record.strength
            ] >= threshold
        ]

    # Tool-state querying

    def tool_succeeded(
        self,
        tool_id: str,
    ) -> bool:

        """Return whether a tool completed successfully.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            True when the condition is met; otherwise False.
        """
        return (
            self.analysis_status.get(
                tool_id
            )
            == ExecutionStatus.SUCCESS.value
        )

    def tool_available_in_bundle(
        self,
        tool_id: str,
    ) -> bool:

        """Return whether results from a tool are present in the evidence bundle.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            True when the condition is met; otherwise False.
        """
        return (
            tool_id
            in self.analysis_status
        )

    # Serialization

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "metrics": [
                observation.to_dict()
                for observation
                in self.metrics
            ],

            "evidence": [
                record.to_dict()
                for record
                in self.evidence
            ],

            "notices": [
                notice.to_dict()
                for notice
                in self.notices
            ],

            "analysis_status":
                dict(
                    self.analysis_status
                ),

            "tool_names":
                dict(
                    self.tool_names
                ),

            "warnings":
                list(
                    self.warnings
                ),
        }


# Evidence engine


class EvidenceEngine:
    """
    Converts AnalysisResult objects into a cross-tool evidence
    representation.

    Responsibilities:

        - collect metrics
        - preserve tool provenance
        - preserve evidence strength
        - resolve evidence -> metric references
        - preserve warnings/messages
        - track skipped/error tools
        - detect duplicate tool results
        - provide normalized queryable evidence

    Explicit non-responsibilities:

        - no physical fault diagnosis
        - no causal inference
        - no automatic evidence-strength inflation
        - no PASS/FAIL qualification
        - no probability estimates
    """

    def build(
        self,
        analyses: Iterable[
            AnalysisResult
        ],
    ) -> EvidenceBundle:
        """Build a normalized evidence bundle.
        
        Args:
            analyses: Value for `analyses`.
        
        Returns:
            EvidenceBundle returned by the function.
        """

        bundle = EvidenceBundle()

        analysis_list = list(
            analyses
        )

        seen_tool_ids: set[str] = set()

        # First pass:
        # metrics + tool state
        #
        # Evidence is processed in the second pass so metric
        # references already exist.

        for analysis in analysis_list:

            tool_id = self._tool_id(
                analysis
            )

            if tool_id in seen_tool_ids:

                bundle.warnings.append(
                    "Multiple AnalysisResult objects were supplied "
                    f"for tool '{tool_id}'. Evidence provenance "
                    "may therefore be ambiguous."
                )

            seen_tool_ids.add(
                tool_id
            )

            tool_name = getattr(
                analysis,
                "tool_name",
                None,
            )

            tool_version = getattr(
                analysis,
                "tool_version",
                None,
            )

            status = self._status_text(
                getattr(
                    analysis,
                    "status",
                    None,
                )
            )

            bundle.analysis_status[
                tool_id
            ] = status

            if tool_name:

                bundle.tool_names[
                    tool_id
                ] = str(
                    tool_name
                )

            # Metrics
            #
            # Metrics from non-successful analyses are deliberately
            # not treated as usable cross-tool evidence.

            if (
                status
                == ExecutionStatus.SUCCESS.value
            ):

                for metric in self._collection(
                    analysis,
                    "metrics",
                ):

                    observation = (
                        self._metric_observation(
                            tool_id=tool_id,
                            tool_name=tool_name,
                            tool_version=tool_version,
                            analysis_status=status,
                            metric=metric,
                        )
                    )

                    if observation is None:
                        continue

                    bundle.metrics.append(
                        observation
                    )

            # Preserve warnings/messages regardless of success.
            # They may explain why a tool was skipped or failed.

            self._collect_notices(
                analysis=analysis,
                tool_id=tool_id,
                bundle=bundle,
            )

        # Second pass:
        # direct evidence records

        evidence_counter: dict[
            str,
            int,
        ] = {}

        for analysis in analysis_list:

            tool_id = self._tool_id(
                analysis
            )

            status = self._status_text(
                getattr(
                    analysis,
                    "status",
                    None,
                )
            )

            # Do not promote evidence from a failed/skipped result
            # into cross-tool reasoning.
            if (
                status
                != ExecutionStatus.SUCCESS.value
            ):

                continue

            tool_name = getattr(
                analysis,
                "tool_name",
                None,
            )

            tool_version = getattr(
                analysis,
                "tool_version",
                None,
            )

            direct_evidence = (
                self._direct_evidence_collection(
                    analysis
                )
            )

            for source_evidence in direct_evidence:

                evidence_counter[
                    tool_id
                ] = (
                    evidence_counter.get(
                        tool_id,
                        0,
                    )
                    + 1
                )

                index = evidence_counter[
                    tool_id
                ]

                evidence_id = (
                    f"{tool_id}:evidence:"
                    f"{index:03d}"
                )

                record = self._evidence_record(
                    evidence_id=evidence_id,
                    tool_id=tool_id,
                    tool_name=tool_name,
                    tool_version=tool_version,
                    source_evidence=(
                        source_evidence
                    ),
                    bundle=bundle,
                )

                if record is None:
                    continue

                bundle.evidence.append(
                    record
                )

        return bundle

    # Metric extraction

    @staticmethod
    def _metric_observation(
        tool_id: str,
        tool_name: str | None,
        tool_version: str | None,
        analysis_status: str,
        metric: Any,
    ) -> MetricObservation | None:

        """Return metric observation.
        
        Args:
            tool_id: Registered tool identifier.
            tool_name: Tool name used by this function.
            tool_version: Tool version used by this function.
            analysis_status: Analysis status used by this function.
            metric: Metric used by this function.
        
        Returns:
            MetricObservation | None returned by this function.
        """
        metric_name = getattr(
            metric,
            "name",
            None,
        )

        if (
            not isinstance(
                metric_name,
                str,
            )
            or not metric_name.strip()
        ):

            return None

        value = getattr(
            metric,
            "value",
            None,
        )

        unit = getattr(
            metric,
            "unit",
            None,
        )

        description = getattr(
            metric,
            "description",
            None,
        )

        metadata = getattr(
            metric,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        return MetricObservation(
            tool_id=tool_id,
            tool_name=(
                str(
                    tool_name
                )
                if tool_name is not None
                else None
            ),
            tool_version=(
                str(
                    tool_version
                )
                if tool_version is not None
                else None
            ),
            metric_name=metric_name.strip(),
            value=value,
            unit=(
                str(
                    unit
                )
                if unit is not None
                else None
            ),
            description=(
                str(
                    description
                )
                if description is not None
                else None
            ),
            analysis_status=analysis_status,
            metadata=dict(
                metadata
            ),
        )

    # Evidence extraction

    def _evidence_record(
        self,
        evidence_id: str,
        tool_id: str,
        tool_name: str | None,
        tool_version: str | None,
        source_evidence: Any,
        bundle: EvidenceBundle,
    ) -> EvidenceRecord | None:

        """Return evidence record.
        
        Args:
            evidence_id: Identifier for evidence.
            tool_id: Registered tool identifier.
            tool_name: Tool name used by this function.
            tool_version: Tool version used by this function.
            source_evidence: Source evidence used by this function.
            bundle: Bundle used by this function.
        
        Returns:
            EvidenceRecord | None returned by this function.
        """
        statement = getattr(
            source_evidence,
            "statement",
            None,
        )

        if (
            not isinstance(
                statement,
                str,
            )
            or not statement.strip()
        ):

            bundle.warnings.append(
                f"Tool '{tool_id}' produced an evidence record "
                "without a usable statement."
            )

            return None

        strength = self._evidence_strength(
            getattr(
                source_evidence,
                "strength",
                None,
            )
        )

        if strength is None:

            bundle.warnings.append(
                f"Tool '{tool_id}' produced evidence with an "
                "unrecognized strength. It was excluded from the "
                "evidence bundle."
            )

            return None

        metric_names = getattr(
            source_evidence,
            "supporting_metrics",
            [],
        )

        if metric_names is None:

            metric_names = []

        if isinstance(
            metric_names,
            str,
        ):

            metric_names = [
                metric_names
            ]

        supporting_references: list[
            MetricReference
        ] = []

        unresolved: list[str] = []

        for metric_name in metric_names:

            if not isinstance(
                metric_name,
                str,
            ):

                continue

            metric_name = metric_name.strip()

            if not metric_name:
                continue

            observation = bundle.metric(
                tool_id=tool_id,
                metric_name=metric_name,
            )

            if observation is None:

                unresolved.append(
                    metric_name
                )

                continue

            supporting_references.append(
                observation.reference
            )

        metadata = getattr(
            source_evidence,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):

            metadata = {}

        if unresolved:

            bundle.warnings.append(
                f"Evidence '{evidence_id}' from tool '{tool_id}' "
                "references metric(s) that were not found in the "
                "successful analysis result: "
                + ", ".join(
                    unresolved
                )
            )

        return EvidenceRecord(
            evidence_id=evidence_id,
            tool_id=tool_id,
            tool_name=(
                str(
                    tool_name
                )
                if tool_name is not None
                else None
            ),
            tool_version=(
                str(
                    tool_version
                )
                if tool_version is not None
                else None
            ),
            statement=statement.strip(),
            strength=strength,
            supporting_metrics=(
                supporting_references
            ),
            unresolved_metric_names=(
                unresolved
            ),
            metadata=dict(
                metadata
            ),
        )

    # Notice extraction

    @classmethod
    def _collect_notices(
        cls,
        analysis: AnalysisResult,
        tool_id: str,
        bundle: EvidenceBundle,
    ) -> None:

        """Collect notices.
        
        Args:
            analysis: Analysis used by this function.
            tool_id: Registered tool identifier.
            bundle: Bundle used by this function.
        
        Returns:
            None.
        """
        for warning in cls._collection(
            analysis,
            "warnings",
        ):

            text = cls._message_text(
                warning
            )

            if text is None:
                continue

            bundle.notices.append(
                EvidenceNotice(
                    tool_id=tool_id,
                    notice_type=(
                        ObservationType.WARNING
                    ),
                    message=text,
                )
            )

        for message in cls._collection(
            analysis,
            "messages",
        ):

            text = cls._message_text(
                message
            )

            if text is None:
                continue

            bundle.notices.append(
                EvidenceNotice(
                    tool_id=tool_id,
                    notice_type=(
                        ObservationType.MESSAGE
                    ),
                    message=text,
                )
            )

    # Compatibility helpers

    @staticmethod
    def _tool_id(
        analysis: AnalysisResult,
    ) -> str:

        """Return tool id.
        
        Args:
            analysis: Analysis used by this function.
        
        Returns:
            Requested text value.
        """
        tool_id = getattr(
            analysis,
            "tool_id",
            None,
        )

        if (
            not isinstance(
                tool_id,
                str,
            )
            or not tool_id.strip()
        ):

            return "unknown_tool"

        return tool_id.strip()

    @staticmethod
    def _status_text(
        status: Any,
    ) -> str:

        """Convert an internal status value to text for display.
        
        Args:
            status: Status value.
        
        Returns:
            Requested text value.
        """
        if hasattr(
            status,
            "value",
        ):

            return str(
                status.value
            )

        if status is None:

            return "unknown"

        return str(
            status
        )

    @staticmethod
    def _evidence_strength(
        value: Any,
    ) -> EvidenceStrength | None:

        """Return evidence strength.
        
        Args:
            value: Value to process.
        
        Returns:
            EvidenceStrength | None returned by this function.
        """
        if isinstance(
            value,
            EvidenceStrength,
        ):

            return value

        if hasattr(
            value,
            "value",
        ):

            value = value.value

        if not isinstance(
            value,
            str,
        ):

            return None

        normalized = (
            value
            .strip()
            .lower()
        )

        for strength in EvidenceStrength:

            if (
                strength.value.lower()
                == normalized
            ):

                return strength

        return None

    @staticmethod
    def _collection(
        source: Any,
        attribute_name: str,
    ) -> list[Any]:

        """Collect ion.
        
        Args:
            source: Input source or source path.
            attribute_name: Attribute name used by this function.
        
        Returns:
            List of result values.
        """
        value = getattr(
            source,
            attribute_name,
            None,
        )

        if value is None:

            return []

        if isinstance(
            value,
            list,
        ):

            return value

        if isinstance(
            value,
            tuple,
        ):

            return list(
                value
            )

        return [
            value
        ]

    @classmethod
    def _direct_evidence_collection(
        cls,
        analysis: AnalysisResult,
    ) -> list[Any]:
        """Prefer the result schema's normal `evidence` field.
        
        The small fallback to `evidences` makes the engine tolerant
        to older development objects without changing the public
        SensorQA schema.
        
        Args:
            analysis: Value for `analysis`.
        
        Returns:
            List of result values.
        """

        evidence = cls._collection(
            analysis,
            "evidence",
        )

        if evidence:

            return evidence

        return cls._collection(
            analysis,
            "evidences",
        )

    @staticmethod
    def _message_text(
        value: Any,
    ) -> str | None:

        """Return message text.
        
        Args:
            value: Value to process.
        
        Returns:
            str | None returned by this function.
        """
        if isinstance(
            value,
            str,
        ):

            cleaned = value.strip()

            return (
                cleaned
                if cleaned
                else None
            )

        message = getattr(
            value,
            "message",
            None,
        )

        if isinstance(
            message,
            str,
        ):

            cleaned = message.strip()

            return (
                cleaned
                if cleaned
                else None
            )

        text = str(
            value
        ).strip()

        return (
            text
            if text
            else None
        )
