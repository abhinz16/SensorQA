#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from sensorqa.core.result_schema import (
    EvidenceStrength,
)
from sensorqa.diagnostics.evidence_engine import (
    EvidenceBundle,
    EvidenceRecord,
    MetricObservation,
    MetricReference,
)


# Hypothesis terminology


class HypothesisState(str, Enum):
    """
    Interpretation state of a diagnostic hypothesis.

    These states describe evidentiary support only.

    They are not probabilities.
    """

    SUPPORTED = "supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class ContributionRole(str, Enum):
    """
    Role played by an observation in a hypothesis.
    """

    SUPPORTING = "supporting"
    CONFLICTING = "conflicting"
    CONTEXT = "context"


class EvidenceSourceType(str, Enum):
    """
    Type of SensorQA evidence referenced by a hypothesis.
    """

    METRIC = "metric"
    EVIDENCE = "evidence"


# Evidence references used by diagnostic rules


@dataclass(frozen=True)
class HypothesisEvidenceReference:
    """
    Reference to evidence already present in EvidenceBundle.

    A diagnostic rule should reference existing SensorQA evidence
    rather than copying values without provenance.
    """

    source_type: EvidenceSourceType

    tool_id: str

    metric_name: str | None = None

    evidence_id: str | None = None

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.tool_id,
            str,
        ) or not self.tool_id.strip():

            raise ValueError(
                "tool_id must be a non-empty string."
            )

        if (
            self.source_type
            == EvidenceSourceType.METRIC
        ):

            if (
                not isinstance(
                    self.metric_name,
                    str,
                )
                or not self.metric_name.strip()
            ):

                raise ValueError(
                    "Metric evidence references require "
                    "metric_name."
                )

        elif (
            self.source_type
            == EvidenceSourceType.EVIDENCE
        ):

            if (
                not isinstance(
                    self.evidence_id,
                    str,
                )
                or not self.evidence_id.strip()
            ):

                raise ValueError(
                    "Evidence-record references require "
                    "evidence_id."
                )


@dataclass
class HypothesisContribution:
    """
    One observation contributing to a hypothesis.

    The rule supplies the rationale explicitly.

    The engine does not infer the physical meaning of the
    referenced observation.
    """

    role: ContributionRole

    reference: HypothesisEvidenceReference

    rationale: str

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.rationale,
            str,
        ) or not self.rationale.strip():

            raise ValueError(
                "Hypothesis contribution rationale must be "
                "a non-empty string."
            )


# Missing evidence


@dataclass
class MissingEvidence:
    """
    Evidence that would improve or disambiguate a hypothesis.

    Examples:

        external vibration reference
        independent thermal cycling experiment
        calibrated orientation fixture
        ground-truth angular-rate measurement

    Missing evidence does not automatically invalidate a
    hypothesis, but should be shown to the user.
    """

    description: str

    importance: EvidenceStrength = (
        EvidenceStrength.MODERATE
    )

    related_tool_id: str | None = None

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.description,
            str,
        ) or not self.description.strip():

            raise ValueError(
                "Missing evidence description must be "
                "a non-empty string."
            )


# Rule proposal


@dataclass
class HypothesisProposal:
    """
    Proposal produced by a diagnostic rule.

    Diagnostic rules, not HypothesisEngine, decide whether their
    explicit evidentiary criteria were met.

    The engine validates and normalizes the proposal.
    """

    hypothesis_id: str

    title: str

    statement: str

    state: HypothesisState

    support_level: EvidenceStrength | None

    supporting: list[
        HypothesisContribution
    ] = field(
        default_factory=list
    )

    conflicting: list[
        HypothesisContribution
    ] = field(
        default_factory=list
    )

    context: list[
        HypothesisContribution
    ] = field(
        default_factory=list
    )

    missing_evidence: list[
        MissingEvidence
    ] = field(
        default_factory=list
    )

    limitations: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        for name, value in (
            (
                "hypothesis_id",
                self.hypothesis_id,
            ),
            (
                "title",
                self.title,
            ),
            (
                "statement",
                self.statement,
            ),
        ):

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):

                raise ValueError(
                    f"{name} must be a non-empty string."
                )

        if (
            self.state
            == HypothesisState.SUPPORTED
            and self.support_level is None
        ):

            raise ValueError(
                "Supported hypotheses require a "
                "support_level."
            )

        if (
            self.state
            != HypothesisState.SUPPORTED
            and self.support_level is not None
        ):

            raise ValueError(
                "Only SUPPORTED hypotheses may define "
                "support_level."
            )


# Final hypothesis


@dataclass
class DiagnosticHypothesis:
    """
    Validated diagnostic hypothesis returned by HypothesisEngine.

    A DiagnosticHypothesis describes an explanation that is
    consistent with available evidence.

    It is not a confirmed root cause unless an appropriately
    designed experiment independently establishes that cause.
    """

    hypothesis_id: str

    rule_id: str

    title: str

    statement: str

    state: HypothesisState

    support_level: EvidenceStrength | None

    supporting: list[
        HypothesisContribution
    ] = field(
        default_factory=list
    )

    conflicting: list[
        HypothesisContribution
    ] = field(
        default_factory=list
    )

    context: list[
        HypothesisContribution
    ] = field(
        default_factory=list
    )

    missing_evidence: list[
        MissingEvidence
    ] = field(
        default_factory=list
    )

    limitations: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    validation_warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "hypothesis_id":
                self.hypothesis_id,

            "rule_id":
                self.rule_id,

            "title":
                self.title,

            "statement":
                self.statement,

            "state":
                self.state.value,

            "support_level": (
                self.support_level.value
                if self.support_level
                is not None
                else None
            ),

            "supporting": [
                self._contribution_to_dict(
                    contribution
                )
                for contribution
                in self.supporting
            ],

            "conflicting": [
                self._contribution_to_dict(
                    contribution
                )
                for contribution
                in self.conflicting
            ],

            "context": [
                self._contribution_to_dict(
                    contribution
                )
                for contribution
                in self.context
            ],

            "missing_evidence": [
                {
                    "description":
                        missing.description,

                    "importance":
                        missing.importance.value,

                    "related_tool_id":
                        missing.related_tool_id,
                }
                for missing
                in self.missing_evidence
            ],

            "limitations":
                list(
                    self.limitations
                ),

            "metadata":
                dict(
                    self.metadata
                ),

            "validation_warnings":
                list(
                    self.validation_warnings
                ),
        }

    @staticmethod
    def _contribution_to_dict(
        contribution: HypothesisContribution,
    ) -> dict[str, Any]:

        """Return contribution to dict.
        
        Args:
            contribution: Contribution used by this function.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "role":
                contribution.role.value,

            "rationale":
                contribution.rationale,

            "reference": {
                "source_type":
                    contribution.reference.source_type.value,

                "tool_id":
                    contribution.reference.tool_id,

                "metric_name":
                    contribution.reference.metric_name,

                "evidence_id":
                    contribution.reference.evidence_id,
            },
        }


# Engine result


@dataclass
class HypothesisEngineResult:
    """
    Collection of diagnostic hypotheses.
    """

    hypotheses: list[
        DiagnosticHypothesis
    ] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    evaluated_rule_ids: list[str] = field(
        default_factory=list
    )

    @property
    def supported_hypotheses(
        self,
    ) -> list[DiagnosticHypothesis]:

        """Return hypotheses that have supporting evidence.
        
        Returns:
            List of result values.
        """
        return [
            hypothesis
            for hypothesis
            in self.hypotheses
            if hypothesis.state
            == HypothesisState.SUPPORTED
        ]

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis
                in self.hypotheses
            ],

            "warnings":
                list(
                    self.warnings
                ),

            "evaluated_rule_ids":
                list(
                    self.evaluated_rule_ids
                ),
        }


# Diagnostic-rule contract


@runtime_checkable
class DiagnosticRule(Protocol):
    """
    Contract implemented by SensorQA diagnostic rules.

    Each rule evaluates one explicit engineering hypothesis.

    Rules should be deterministic and explainable.
    """

    @property
    def rule_id(
        self,
    ) -> str:
        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        ...

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
        ...


# Hypothesis engine


class HypothesisEngine:
    """
    Evaluates explicit diagnostic rules against an EvidenceBundle.

    Important design principle:

        HypothesisEngine does not assign support levels based on
        automatic numerical scoring.

    A rule must explicitly determine what combination of evidence
    justifies WEAK, MODERATE, or STRONG support.

    The engine then:

        - validates references
        - rejects malformed proposals
        - prevents unsupported evidence references
        - preserves conflicting evidence
        - preserves missing evidence
        - provides consistent serialization
    """

    def evaluate(
        self,
        evidence: EvidenceBundle,
        rules: list[
            DiagnosticRule
        ],
    ) -> HypothesisEngineResult:
        """Evaluate diagnostic rules.
        
        Args:
            evidence: Value for `evidence`.
            rules: Value for `rules`.
        
        Returns:
            HypothesisEngineResult returned by the function.
        """

        result = HypothesisEngineResult()

        seen_rule_ids: set[str] = set()

        seen_hypothesis_ids: set[str] = set()

        for rule in rules:

            rule_id = self._rule_id(
                rule
            )

            if rule_id is None:

                result.warnings.append(
                    "A diagnostic rule was skipped because "
                    "it does not define a valid rule_id."
                )

                continue

            if rule_id in seen_rule_ids:

                result.warnings.append(
                    f"Duplicate diagnostic rule ID "
                    f"'{rule_id}' was skipped."
                )

                continue

            seen_rule_ids.add(
                rule_id
            )

            result.evaluated_rule_ids.append(
                rule_id
            )

            try:

                proposal = rule.evaluate(
                    evidence
                )

            except Exception as exc:

                result.warnings.append(
                    f"Diagnostic rule '{rule_id}' failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )

                continue

            # A rule returning None means its preconditions are
            # not relevant to the current dataset.
            if proposal is None:
                continue

            if not isinstance(
                proposal,
                HypothesisProposal,
            ):

                result.warnings.append(
                    f"Diagnostic rule '{rule_id}' returned "
                    "an invalid proposal type."
                )

                continue

            hypothesis_id = (
                proposal.hypothesis_id.strip()
            )

            if hypothesis_id in seen_hypothesis_ids:

                result.warnings.append(
                    f"Duplicate hypothesis ID "
                    f"'{hypothesis_id}' produced by "
                    f"rule '{rule_id}'."
                )

                continue

            seen_hypothesis_ids.add(
                hypothesis_id
            )

            validated = self._validate_proposal(
                rule_id=rule_id,
                proposal=proposal,
                evidence=evidence,
            )

            result.hypotheses.append(
                validated
            )

        return result

    # Proposal validation

    def _validate_proposal(
        self,
        rule_id: str,
        proposal: HypothesisProposal,
        evidence: EvidenceBundle,
    ) -> DiagnosticHypothesis:

        """Validate proposal.
        
        Args:
            rule_id: Identifier for rule.
            proposal: Proposal used by this function.
            evidence: Evidence used by this function.
        
        Returns:
            DiagnosticHypothesis returned by this function.
        """
        validation_warnings: list[str] = []

        supporting = self._validate_contributions(
            contributions=proposal.supporting,
            expected_role=(
                ContributionRole.SUPPORTING
            ),
            evidence=evidence,
            warnings=validation_warnings,
        )

        conflicting = self._validate_contributions(
            contributions=proposal.conflicting,
            expected_role=(
                ContributionRole.CONFLICTING
            ),
            evidence=evidence,
            warnings=validation_warnings,
        )

        context = self._validate_contributions(
            contributions=proposal.context,
            expected_role=(
                ContributionRole.CONTEXT
            ),
            evidence=evidence,
            warnings=validation_warnings,
        )

        state = proposal.state

        support_level = proposal.support_level

        # A rule cannot claim support when every supporting
        # reference is invalid or missing.

        if (
            state
            == HypothesisState.SUPPORTED
            and len(
                supporting
            )
            == 0
        ):

            validation_warnings.append(
                "The diagnostic rule proposed a supported "
                "hypothesis but none of its supporting evidence "
                "references were valid. The hypothesis was "
                "downgraded to insufficient evidence."
            )

            state = (
                HypothesisState.INSUFFICIENT_EVIDENCE
            )

            support_level = None

        # Evidence marked as conflicting must remain visible.
        #
        # The engine does not automatically cancel supporting and
        # conflicting evidence using a numerical score.

        if (
            state
            == HypothesisState.SUPPORTED
            and conflicting
        ):

            validation_warnings.append(
                "The hypothesis contains conflicting evidence. "
                "The support level was preserved because support "
                "criteria are explicitly defined by the diagnostic "
                "rule, but the conflicting observations must remain "
                "visible in interpretation."
            )

        limitations = []

        for limitation in proposal.limitations:

            if not isinstance(
                limitation,
                str,
            ):

                continue

            cleaned = limitation.strip()

            if cleaned:

                limitations.append(
                    cleaned
                )

        # Every physical hypothesis receives a generic causal
        # limitation unless the rule already includes equivalent
        # language.

        causal_limitation = (
            "This hypothesis identifies an explanation that is "
            "consistent with the available evidence. It does not "
            "by itself establish physical root cause."
        )

        if not self._contains_causal_limitation(
            limitations
        ):

            limitations.append(
                causal_limitation
            )

        return DiagnosticHypothesis(
            hypothesis_id=(
                proposal.hypothesis_id.strip()
            ),
            rule_id=rule_id,
            title=proposal.title.strip(),
            statement=proposal.statement.strip(),
            state=state,
            support_level=support_level,
            supporting=supporting,
            conflicting=conflicting,
            context=context,
            missing_evidence=list(
                proposal.missing_evidence
            ),
            limitations=limitations,
            metadata=dict(
                proposal.metadata
            ),
            validation_warnings=(
                validation_warnings
            ),
        )

    # Contribution validation

    def _validate_contributions(
        self,
        contributions: list[
            HypothesisContribution
        ],
        expected_role: ContributionRole,
        evidence: EvidenceBundle,
        warnings: list[str],
    ) -> list[
        HypothesisContribution
    ]:

        """Validate contributions.
        
        Args:
            contributions: Contributions used by this function.
            expected_role: Expected role used by this function.
            evidence: Evidence used by this function.
            warnings: Warnings used by this function.
        
        Returns:
            List of result values.
        """
        output = []

        seen = set()

        for contribution in contributions:

            if not isinstance(
                contribution,
                HypothesisContribution,
            ):

                warnings.append(
                    "A hypothesis contribution was ignored "
                    "because it has an invalid type."
                )

                continue

            if contribution.role != expected_role:

                warnings.append(
                    "A hypothesis contribution was ignored because "
                    f"its role is '{contribution.role.value}' but "
                    f"it appeared in the "
                    f"'{expected_role.value}' collection."
                )

                continue

            reference = contribution.reference

            signature = (
                reference.source_type.value,
                reference.tool_id,
                reference.metric_name,
                reference.evidence_id,
                contribution.role.value,
            )

            if signature in seen:
                continue

            if not self._reference_exists(
                reference=reference,
                evidence=evidence,
            ):

                warnings.append(
                    "Hypothesis evidence reference could not be "
                    "resolved: "
                    f"{self._reference_text(reference)}."
                )

                continue

            seen.add(
                signature
            )

            output.append(
                contribution
            )

        return output

    # Evidence-reference resolution

    @staticmethod
    def _reference_exists(
        reference: HypothesisEvidenceReference,
        evidence: EvidenceBundle,
    ) -> bool:

        """Return reference exists.
        
        Args:
            reference: Reference used by this function.
            evidence: Evidence used by this function.
        
        Returns:
            Boolean result.
        """
        if (
            reference.source_type
            == EvidenceSourceType.METRIC
        ):

            return (
                evidence.metric(
                    tool_id=reference.tool_id,
                    metric_name=(
                        reference.metric_name
                        or ""
                    ),
                )
                is not None
            )

        if (
            reference.source_type
            == EvidenceSourceType.EVIDENCE
        ):

            for record in evidence.evidence:

                if (
                    record.tool_id
                    == reference.tool_id
                    and record.evidence_id
                    == reference.evidence_id
                ):

                    return True

            return False

        return False

    # Convenience constructors

    @staticmethod
    def metric_reference(
        tool_id: str,
        metric_name: str,
    ) -> HypothesisEvidenceReference:
        """Convenience helper for diagnostic-rule authors.
        
        Args:
            tool_id: Registered tool identifier.
            metric_name: Value for `metric_name`.
        
        Returns:
            HypothesisEvidenceReference returned by the function.
        """

        return HypothesisEvidenceReference(
            source_type=(
                EvidenceSourceType.METRIC
            ),
            tool_id=tool_id,
            metric_name=metric_name,
        )

    @staticmethod
    def evidence_reference(
        record: EvidenceRecord,
    ) -> HypothesisEvidenceReference:
        """Build a reference from an EvidenceRecord.
        
        Args:
            record: Value for `record`.
        
        Returns:
            HypothesisEvidenceReference returned by the function.
        """

        return HypothesisEvidenceReference(
            source_type=(
                EvidenceSourceType.EVIDENCE
            ),
            tool_id=record.tool_id,
            evidence_id=record.evidence_id,
        )

    # Lookup helpers for diagnostic rules

    @staticmethod
    def metric_value(
        evidence: EvidenceBundle,
        tool_id: str,
        metric_name: str,
    ) -> Any | None:
        """Return the raw value of one metric.
        
        This helper performs no threshold interpretation.
        
        Args:
            evidence: Value for `evidence`.
            tool_id: Registered tool identifier.
            metric_name: Value for `metric_name`.
        
        Returns:
            Any | None returned by the function.
        """

        observation = evidence.metric(
            tool_id=tool_id,
            metric_name=metric_name,
        )

        if observation is None:
            return None

        return observation.value

    @staticmethod
    def metric_observation(
        evidence: EvidenceBundle,
        tool_id: str,
        metric_name: str,
    ) -> MetricObservation | None:

        """Return a metric observation by tool and metric name.
        
        Args:
            evidence: Evidence used by this function.
            tool_id: Registered tool identifier.
            metric_name: Metric name used by this function.
        
        Returns:
            MetricObservation | None returned by this function.
        """
        return evidence.metric(
            tool_id=tool_id,
            metric_name=metric_name,
        )

    # Helpers

    @staticmethod
    def _rule_id(
        rule: Any,
    ) -> str | None:

        """Return rule id.
        
        Args:
            rule: Rule used by this function.
        
        Returns:
            str | None returned by this function.
        """
        try:

            rule_id = rule.rule_id

        except Exception:

            return None

        if (
            not isinstance(
                rule_id,
                str,
            )
            or not rule_id.strip()
        ):

            return None

        return rule_id.strip()

    @staticmethod
    def _reference_text(
        reference: HypothesisEvidenceReference,
    ) -> str:

        """Return reference text.
        
        Args:
            reference: Reference used by this function.
        
        Returns:
            Requested text value.
        """
        if (
            reference.source_type
            == EvidenceSourceType.METRIC
        ):

            return (
                f"{reference.tool_id} / "
                f"{reference.metric_name}"
            )

        return (
            f"{reference.tool_id} / "
            f"{reference.evidence_id}"
        )

    @staticmethod
    def _contains_causal_limitation(
        limitations: list[str],
    ) -> bool:

        """Return contains causal limitation.
        
        Args:
            limitations: Limitations used by this function.
        
        Returns:
            Boolean result.
        """
        causal_terms = (
            "root cause",
            "causal",
            "causation",
            "does not establish",
        )

        for limitation in limitations:

            normalized = (
                limitation.lower()
            )

            if any(
                term in normalized
                for term in causal_terms
            ):

                return True

        return False
