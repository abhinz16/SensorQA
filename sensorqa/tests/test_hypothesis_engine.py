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
)
from sensorqa.diagnostics.evidence_engine import (
    EvidenceBundle,
    EvidenceRecord,
    MetricObservation,
    MetricReference,
)
from sensorqa.diagnostics.hypothesis_engine import (
    ContributionRole,
    DiagnosticHypothesis,
    EvidenceSourceType,
    HypothesisContribution,
    HypothesisEngine,
    HypothesisEvidenceReference,
    HypothesisProposal,
    HypothesisState,
    MissingEvidence,
)


# Helpers


def metric(
    *,
    tool_id: str = "tool_a",
    name: str = "Metric A",
    value=1.0,
    unit: str | None = None,
) -> MetricObservation:
    """Build one normalized metric observation.
    
    Args:
        tool_id: Registered tool identifier.
        name: Name of the item.
        value: Value to process.
        unit: Engineering unit.
    
    Returns:
        MetricObservation returned by the function.
    """

    return MetricObservation(
        tool_id=tool_id,
        tool_name="Test Tool",
        metric_name=name,
        value=value,
        unit=unit,
        description=None,
        tool_version="1.0.0",
        analysis_status="success",
    )


def evidence_record(
    *,
    evidence_id: str = "tool_a:evidence:001",
    tool_id: str = "tool_a",
    statement: str = "Observation A",
    strength: EvidenceStrength = (
        EvidenceStrength.MODERATE
    ),
) -> EvidenceRecord:
    """Build one normalized direct-evidence record.
    
    Args:
        evidence_id: Identifier for evidence.
        tool_id: Registered tool identifier.
        statement: Value for `statement`.
        strength: Value for `strength`.
    
    Returns:
        EvidenceRecord returned by the function.
    """

    return EvidenceRecord(
        evidence_id=evidence_id,
        tool_id=tool_id,
        tool_name="Test Tool",
        statement=statement,
        strength=strength,
        supporting_metrics=[],
        tool_version="1.0.0",
    )


def metric_reference(
    tool_id: str = "tool_a",
    metric_name: str = "Metric A",
) -> HypothesisEvidenceReference:

    """Run metric reference.
    
    Args:
        tool_id: Registered tool identifier.
        metric_name: Metric name used by this function.
    
    Returns:
        HypothesisEvidenceReference returned by this function.
    """
    return HypothesisEvidenceReference(
        source_type=(
            EvidenceSourceType.METRIC
        ),
        tool_id=tool_id,
        metric_name=metric_name,
    )


def direct_evidence_reference(
    *,
    tool_id: str = "tool_a",
    evidence_id: str = "tool_a:evidence:001",
) -> HypothesisEvidenceReference:

    """Run direct evidence reference.
    
    Args:
        tool_id: Registered tool identifier.
        evidence_id: Identifier for evidence.
    
    Returns:
        HypothesisEvidenceReference returned by this function.
    """
    return HypothesisEvidenceReference(
        source_type=(
            EvidenceSourceType.EVIDENCE
        ),
        tool_id=tool_id,
        evidence_id=evidence_id,
    )


def supporting_contribution(
    *,
    tool_id: str = "tool_a",
    metric_name: str = "Metric A",
    rationale: str = "Metric A supports the hypothesis.",
) -> HypothesisContribution:

    """Run supporting contribution.
    
    Args:
        tool_id: Registered tool identifier.
        metric_name: Metric name used by this function.
        rationale: Rationale used by this function.
    
    Returns:
        HypothesisContribution returned by this function.
    """
    return HypothesisContribution(
        role=ContributionRole.SUPPORTING,
        reference=metric_reference(
            tool_id=tool_id,
            metric_name=metric_name,
        ),
        rationale=rationale,
    )


def conflicting_contribution(
    *,
    tool_id: str = "tool_a",
    metric_name: str = "Metric A",
    rationale: str = "Metric A conflicts with the hypothesis.",
) -> HypothesisContribution:

    """Run conflicting contribution.
    
    Args:
        tool_id: Registered tool identifier.
        metric_name: Metric name used by this function.
        rationale: Rationale used by this function.
    
    Returns:
        HypothesisContribution returned by this function.
    """
    return HypothesisContribution(
        role=ContributionRole.CONFLICTING,
        reference=metric_reference(
            tool_id=tool_id,
            metric_name=metric_name,
        ),
        rationale=rationale,
    )


class FixedRule:
    """
    Diagnostic rule returning a fixed proposal.
    """

    def __init__(
        self,
        rule_id: str,
        proposal: HypothesisProposal | None,
    ) -> None:

        """Initialize the fixed rule.
        
        Args:
            rule_id: Identifier for rule.
            proposal: Proposal used by this function.
        
        Returns:
            None.
        """
        self._rule_id = rule_id
        self.proposal = proposal

    @property
    def rule_id(
        self,
    ) -> str:

        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        return self._rule_id

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
        return self.proposal


class FailingRule:

    """Represents failing rule."""
    @property
    def rule_id(
        self,
    ) -> str:

        """Return the stable identifier for this diagnostic rule.
        
        Returns:
            Requested text value.
        """
        return "failing_rule"

    def evaluate(
        self,
        evidence: EvidenceBundle,
    ):

        """Evaluate evaluate.
        
        Args:
            evidence: Evidence used by this function.
        
        Returns:
            Result of the operation.
        """
        raise RuntimeError(
            "Deliberate test failure"
        )


# Basic proposal evaluation


def test_supported_hypothesis_is_created():
    """Check that supported hypothesis is created.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    proposal = HypothesisProposal(
        hypothesis_id="hypothesis_a",
        title="Hypothesis A",
        statement=(
            "Observed behavior is consistent "
            "with hypothesis A."
        ),
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            supporting_contribution()
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    assert len(
        result.hypotheses
    ) == 1

    hypothesis = result.hypotheses[
        0
    ]

    assert isinstance(
        hypothesis,
        DiagnosticHypothesis,
    )

    assert (
        hypothesis.hypothesis_id
        == "hypothesis_a"
    )

    assert hypothesis.rule_id == "rule_a"

    assert (
        hypothesis.state
        == HypothesisState.SUPPORTED
    )

    assert (
        hypothesis.support_level
        == EvidenceStrength.MODERATE
    )


def test_none_proposal_means_rule_not_relevant():
    """Check that none proposal means rule not relevant.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle()

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                None,
            )
        ],
    )

    assert result.hypotheses == []

    assert (
        result.evaluated_rule_ids
        == [
            "rule_a"
        ]
    )


# Proposal structural validation


def test_supported_proposal_requires_support_level():
    """Check that supported proposal requires support level.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        HypothesisProposal(
            hypothesis_id="hypothesis_a",
            title="Hypothesis A",
            statement="Statement",
            state=(
                HypothesisState.SUPPORTED
            ),
            support_level=None,
        )


def test_non_supported_proposal_rejects_support_level():
    """Check that non supported proposal rejects support level.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        HypothesisProposal(
            hypothesis_id="hypothesis_a",
            title="Hypothesis A",
            statement="Statement",
            state=(
                HypothesisState.INSUFFICIENT_EVIDENCE
            ),
            support_level=(
                EvidenceStrength.WEAK
            ),
        )


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    [
        (
            "hypothesis_id",
            "",
        ),
        (
            "title",
            "   ",
        ),
        (
            "statement",
            "",
        ),
    ],
)
def test_required_proposal_strings_must_not_be_empty(
    field,
    value,
):
    """Check that required proposal strings must not be empty.
    
    Args:
        field: Field used by this function.
        value: Value to process.
    
    Returns:
        None.
    """
    kwargs = {
        "hypothesis_id":
            "hypothesis_a",

        "title":
            "Hypothesis A",

        "statement":
            "Statement",

        "state":
            HypothesisState.INSUFFICIENT_EVIDENCE,

        "support_level":
            None,
    }

    kwargs[
        field
    ] = value

    with pytest.raises(
        ValueError
    ):

        HypothesisProposal(
            **kwargs
        )


# Evidence-reference construction


def test_metric_reference_requires_metric_name():
    """Check that metric reference requires metric name.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        HypothesisEvidenceReference(
            source_type=(
                EvidenceSourceType.METRIC
            ),
            tool_id="tool_a",
            metric_name=None,
        )


def test_direct_evidence_reference_requires_evidence_id():
    """Check that direct evidence reference requires evidence id.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        HypothesisEvidenceReference(
            source_type=(
                EvidenceSourceType.EVIDENCE
            ),
            tool_id="tool_a",
            evidence_id=None,
        )


def test_reference_requires_tool_id():
    """Check that reference requires tool id.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        HypothesisEvidenceReference(
            source_type=(
                EvidenceSourceType.METRIC
            ),
            tool_id="",
            metric_name="Metric A",
        )


def test_contribution_requires_rationale():
    """Check that contribution requires rationale.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        HypothesisContribution(
            role=(
                ContributionRole.SUPPORTING
            ),
            reference=metric_reference(),
            rationale="",
        )


# Metric evidence resolution


def test_valid_metric_reference_is_preserved():
    """Check that valid metric reference is preserved.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric(
                tool_id="tool_a",
                name="Metric A",
            )
        ]
    )

    proposal = HypothesisProposal(
        hypothesis_id="hypothesis_a",
        title="Hypothesis A",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            supporting_contribution(
                tool_id="tool_a",
                metric_name="Metric A",
            )
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    hypothesis = result.hypotheses[
        0
    ]

    assert len(
        hypothesis.supporting
    ) == 1

    assert (
        hypothesis.supporting[
            0
        ].reference.metric_name
        == "Metric A"
    )


def test_missing_metric_reference_is_removed():
    """Check that missing metric reference is removed.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle()

    proposal = HypothesisProposal(
        hypothesis_id="hypothesis_a",
        title="Hypothesis A",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            supporting_contribution(
                metric_name=(
                    "Does Not Exist"
                )
            )
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    hypothesis = result.hypotheses[
        0
    ]

    assert hypothesis.supporting == []

    assert any(
        "could not be resolved"
        in warning.lower()
        for warning
        in hypothesis.validation_warnings
    )


# Critical safeguard:
# supported hypothesis without valid evidence is downgraded


def test_supported_hypothesis_with_no_valid_support_is_downgraded():
    """A buggy custom diagnostic rule must not be able to claim
    SUPPORTED when its evidence references do not exist.
    
    Returns:
        None.
    """

    bundle = EvidenceBundle()

    proposal = HypothesisProposal(
        hypothesis_id="bad_support",
        title="Bad Support",
        statement=(
            "This should not remain supported."
        ),
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.STRONG
        ),
        supporting=[
            supporting_contribution(
                tool_id="missing_tool",
                metric_name="Missing Metric",
            )
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "bad_rule",
                proposal,
            )
        ],
    )

    hypothesis = result.hypotheses[
        0
    ]

    assert (
        hypothesis.state
        == HypothesisState.INSUFFICIENT_EVIDENCE
    )

    assert hypothesis.support_level is None

    assert any(
        "downgraded"
        in warning.lower()
        for warning
        in hypothesis.validation_warnings
    )


# Direct EvidenceRecord references


def test_direct_evidence_record_reference_resolves():
    """Check that direct evidence record reference resolves.
    
    Returns:
        None.
    """
    record = evidence_record()

    bundle = EvidenceBundle(
        evidence=[
            record
        ]
    )

    contribution = HypothesisContribution(
        role=ContributionRole.SUPPORTING,
        reference=direct_evidence_reference(),
        rationale=(
            "The originating analysis explicitly "
            "reported this evidence."
        ),
    )

    proposal = HypothesisProposal(
        hypothesis_id="evidence_based",
        title="Evidence Based",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            contribution
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    assert len(
        result.hypotheses[
            0
        ].supporting
    ) == 1


def test_wrong_evidence_id_does_not_resolve():
    """Check that wrong evidence id does not resolve.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        evidence=[
            evidence_record()
        ]
    )

    contribution = HypothesisContribution(
        role=ContributionRole.SUPPORTING,
        reference=direct_evidence_reference(
            evidence_id=(
                "tool_a:evidence:999"
            )
        ),
        rationale="Incorrect reference.",
    )

    proposal = HypothesisProposal(
        hypothesis_id="bad_reference",
        title="Bad Reference",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            contribution
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    hypothesis = result.hypotheses[
        0
    ]

    assert (
        hypothesis.state
        == HypothesisState.INSUFFICIENT_EVIDENCE
    )


# Contribution roles


def test_wrong_role_in_supporting_collection_is_removed():
    """Check that wrong role in supporting collection is removed.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    wrong_role = HypothesisContribution(
        role=ContributionRole.CONTEXT,
        reference=metric_reference(),
        rationale="Context, not support.",
    )

    proposal = HypothesisProposal(
        hypothesis_id="wrong_role",
        title="Wrong Role",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            wrong_role
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    hypothesis = result.hypotheses[
        0
    ]

    assert hypothesis.supporting == []

    assert (
        hypothesis.state
        == HypothesisState.INSUFFICIENT_EVIDENCE
    )


def test_duplicate_contribution_reference_removed():
    """Check that duplicate contribution reference removed.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    first = supporting_contribution(
        rationale="First rationale."
    )

    second = supporting_contribution(
        rationale="Second rationale."
    )

    proposal = HypothesisProposal(
        hypothesis_id="duplicate_reference",
        title="Duplicate Reference",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            first,
            second,
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    assert len(
        result.hypotheses[
            0
        ].supporting
    ) == 1


# Conflicting evidence


def test_conflicting_evidence_is_preserved():
    """Check that conflicting evidence is preserved.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric(
                name="Supporting Metric",
                value=1.0,
            ),
            metric(
                name="Conflicting Metric",
                value=2.0,
            ),
        ]
    )

    proposal = HypothesisProposal(
        hypothesis_id="conflicted",
        title="Conflicted Hypothesis",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            supporting_contribution(
                metric_name="Supporting Metric",
            )
        ],
        conflicting=[
            conflicting_contribution(
                metric_name="Conflicting Metric",
            )
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    hypothesis = result.hypotheses[
        0
    ]

    assert len(
        hypothesis.supporting
    ) == 1

    assert len(
        hypothesis.conflicting
    ) == 1

    # The engine does not perform arbitrary numerical cancellation.
    assert (
        hypothesis.state
        == HypothesisState.SUPPORTED
    )

    assert (
        hypothesis.support_level
        == EvidenceStrength.MODERATE
    )

    assert any(
        "conflicting evidence"
        in warning.lower()
        for warning
        in hypothesis.validation_warnings
    )


def test_explicit_conflicting_evidence_state_is_preserved():
    """Check that explicit conflicting evidence state is preserved.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric(
                name="Metric A",
            )
        ]
    )

    proposal = HypothesisProposal(
        hypothesis_id="conflict_state",
        title="Conflict",
        statement="Evidence conflicts.",
        state=(
            HypothesisState.CONFLICTING_EVIDENCE
        ),
        support_level=None,
        conflicting=[
            conflicting_contribution()
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    assert (
        result.hypotheses[
            0
        ].state
        == HypothesisState.CONFLICTING_EVIDENCE
    )


# Missing evidence


def test_missing_evidence_is_preserved():
    """Check that missing evidence is preserved.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    missing = MissingEvidence(
        description=(
            "Independent vibration reference."
        ),
        importance=(
            EvidenceStrength.STRONG
        ),
        related_tool_id="imu_psd",
    )

    proposal = HypothesisProposal(
        hypothesis_id="missing_external_reference",
        title="Hypothesis",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            supporting_contribution()
        ],
        missing_evidence=[
            missing
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    preserved = (
        result.hypotheses[
            0
        ].missing_evidence
    )

    assert len(
        preserved
    ) == 1

    assert (
        preserved[
            0
        ].importance
        == EvidenceStrength.STRONG
    )


def test_missing_evidence_requires_description():
    """Check that missing evidence requires description.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        MissingEvidence(
            description=""
        )


# Root-cause limitation


def test_generic_root_cause_limitation_added_automatically():
    """Check that generic root cause limitation added automatically.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    proposal = HypothesisProposal(
        hypothesis_id="hypothesis_a",
        title="Hypothesis A",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            supporting_contribution()
        ],
        limitations=[],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    limitations = (
        result.hypotheses[
            0
        ].limitations
    )

    assert any(
        "root cause"
        in limitation.lower()
        for limitation
        in limitations
    )


def test_existing_causal_limitation_not_duplicated():
    """Check that existing causal limitation not duplicated.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    custom_limitation = (
        "This statistical relationship does not establish "
        "causation."
    )

    proposal = HypothesisProposal(
        hypothesis_id="hypothesis_a",
        title="Hypothesis A",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            supporting_contribution()
        ],
        limitations=[
            custom_limitation
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_a",
                proposal,
            )
        ],
    )

    limitations = (
        result.hypotheses[
            0
        ].limitations
    )

    assert limitations == [
        custom_limitation
    ]


# Rule failures and malformed rules


def test_rule_exception_does_not_crash_engine():
    """Check that rule exception does not crash engine.
    
    Returns:
        None.
    """
    result = HypothesisEngine().evaluate(
        evidence=EvidenceBundle(),
        rules=[
            FailingRule()
        ],
    )

    assert result.hypotheses == []

    assert any(
        "deliberate test failure"
        in warning.lower()
        for warning
        in result.warnings
    )


def test_invalid_rule_id_is_skipped():
    """Check that invalid rule id is skipped.
    
    Returns:
        None.
    """
    class InvalidRule:

        """Represents invalid rule."""
        @property
        def rule_id(
            self,
        ):
            """Return the stable identifier for this diagnostic rule.
            
            Returns:
                Calculated value.
            """
            return ""

        def evaluate(
            self,
            evidence,
        ):
            """Evaluate evaluate.
            
            Args:
                evidence: Evidence used by this function.
            
            Returns:
                Result of the operation.
            """
            raise AssertionError(
                "Should never execute."
            )

    result = HypothesisEngine().evaluate(
        evidence=EvidenceBundle(),
        rules=[
            InvalidRule()
        ],
    )

    assert result.hypotheses == []

    assert result.evaluated_rule_ids == []

    assert any(
        "valid rule_id"
        in warning
        for warning
        in result.warnings
    )


def test_rule_returning_invalid_type_is_skipped():
    """Check that rule returning invalid type is skipped.
    
    Returns:
        None.
    """
    class InvalidReturnRule:

        """Represents invalid return rule."""
        @property
        def rule_id(
            self,
        ):
            """Return the stable identifier for this diagnostic rule.
            
            Returns:
                Calculated value.
            """
            return "invalid_return"

        def evaluate(
            self,
            evidence,
        ):
            """Evaluate evaluate.
            
            Args:
                evidence: Evidence used by this function.
            
            Returns:
                Result of the operation.
            """
            return {
                "not":
                    "a proposal"
            }

    result = HypothesisEngine().evaluate(
        evidence=EvidenceBundle(),
        rules=[
            InvalidReturnRule()
        ],
    )

    assert result.hypotheses == []

    assert any(
        "invalid proposal type"
        in warning.lower()
        for warning
        in result.warnings
    )


# Duplicate rule IDs


def test_duplicate_rule_id_is_skipped():
    """Check that duplicate rule id is skipped.
    
    Returns:
        None.
    """
    first = FixedRule(
        "same_rule",
        None,
    )

    second = FixedRule(
        "same_rule",
        None,
    )

    result = HypothesisEngine().evaluate(
        evidence=EvidenceBundle(),
        rules=[
            first,
            second,
        ],
    )

    assert (
        result.evaluated_rule_ids
        == [
            "same_rule"
        ]
    )

    assert any(
        "duplicate diagnostic rule"
        in warning.lower()
        for warning
        in result.warnings
    )


# Duplicate hypothesis IDs


def test_duplicate_hypothesis_id_is_skipped():
    """Check that duplicate hypothesis id is skipped.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    first_proposal = HypothesisProposal(
        hypothesis_id="same_hypothesis",
        title="First",
        statement="First statement.",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            supporting_contribution()
        ],
    )

    second_proposal = HypothesisProposal(
        hypothesis_id="same_hypothesis",
        title="Second",
        statement="Second statement.",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            supporting_contribution()
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_1",
                first_proposal,
            ),
            FixedRule(
                "rule_2",
                second_proposal,
            ),
        ],
    )

    assert len(
        result.hypotheses
    ) == 1

    assert (
        result.hypotheses[
            0
        ].title
        == "First"
    )

    assert any(
        "duplicate hypothesis"
        in warning.lower()
        for warning
        in result.warnings
    )


# Convenience reference helpers


def test_metric_reference_helper():
    """Check that metric reference helper.
    
    Returns:
        None.
    """
    reference = (
        HypothesisEngine.metric_reference(
            tool_id="imu_psd",
            metric_name="GX Dominant Frequency",
        )
    )

    assert (
        reference.source_type
        == EvidenceSourceType.METRIC
    )

    assert reference.tool_id == "imu_psd"

    assert (
        reference.metric_name
        == "GX Dominant Frequency"
    )


def test_evidence_reference_helper():
    """Check that evidence reference helper.
    
    Returns:
        None.
    """
    record = evidence_record(
        evidence_id=(
            "imu_psd:evidence:003"
        ),
        tool_id="imu_psd",
    )

    reference = (
        HypothesisEngine.evidence_reference(
            record
        )
    )

    assert (
        reference.source_type
        == EvidenceSourceType.EVIDENCE
    )

    assert reference.tool_id == "imu_psd"

    assert (
        reference.evidence_id
        == "imu_psd:evidence:003"
    )


# Metric lookup helpers


def test_metric_value_helper():
    """Check that metric value helper.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric(
                tool_id="tool_a",
                name="Metric A",
                value=42.0,
            )
        ]
    )

    value = (
        HypothesisEngine.metric_value(
            evidence=bundle,
            tool_id="tool_a",
            metric_name="Metric A",
        )
    )

    assert value == pytest.approx(
        42.0
    )


def test_metric_value_helper_missing_returns_none():
    """Check that metric value helper missing returns none.
    
    Returns:
        None.
    """
    value = (
        HypothesisEngine.metric_value(
            evidence=EvidenceBundle(),
            tool_id="tool_a",
            metric_name="missing",
        )
    )

    assert value is None


# Supported hypothesis filtering


def test_supported_hypotheses_property():
    """Check that supported hypotheses property.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric()
        ]
    )

    supported = HypothesisProposal(
        hypothesis_id="supported",
        title="Supported",
        statement="Statement",
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.WEAK
        ),
        supporting=[
            supporting_contribution()
        ],
    )

    insufficient = HypothesisProposal(
        hypothesis_id="insufficient",
        title="Insufficient",
        statement="Statement",
        state=(
            HypothesisState.INSUFFICIENT_EVIDENCE
        ),
        support_level=None,
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "rule_supported",
                supported,
            ),
            FixedRule(
                "rule_insufficient",
                insufficient,
            ),
        ],
    )

    assert [
        hypothesis.hypothesis_id
        for hypothesis
        in result.supported_hypotheses
    ] == [
        "supported"
    ]


# Serialization


def test_hypothesis_serialization_preserves_provenance():
    """Check that hypothesis serialization preserves provenance.
    
    Returns:
        None.
    """
    bundle = EvidenceBundle(
        metrics=[
            metric(
                tool_id="imu_axis_correlation",
                name="GY-GZ Correlation",
                value=0.84,
            )
        ]
    )

    proposal = HypothesisProposal(
        hypothesis_id="cross_axis_behavior",
        title="Cross-Axis Behavior",
        statement=(
            "GY and GZ show substantial shared variation."
        ),
        state=HypothesisState.SUPPORTED,
        support_level=(
            EvidenceStrength.MODERATE
        ),
        supporting=[
            supporting_contribution(
                tool_id="imu_axis_correlation",
                metric_name="GY-GZ Correlation",
            )
        ],
    )

    result = HypothesisEngine().evaluate(
        evidence=bundle,
        rules=[
            FixedRule(
                "cross_axis_rule",
                proposal,
            )
        ],
    )

    data = (
        result.hypotheses[
            0
        ].to_dict()
    )

    reference = (
        data[
            "supporting"
        ][
            0
        ][
            "reference"
        ]
    )

    assert (
        reference[
            "tool_id"
        ]
        == "imu_axis_correlation"
    )

    assert (
        reference[
            "metric_name"
        ]
        == "GY-GZ Correlation"
    )

    assert (
        data[
            "support_level"
        ]
        == EvidenceStrength.MODERATE.value
    )


def test_engine_result_serialization():
    """Check that engine result serialization.
    
    Returns:
        None.
    """
    result = HypothesisEngine().evaluate(
        evidence=EvidenceBundle(),
        rules=[],
    )

    data = result.to_dict()

    assert data == {
        "hypotheses":
            [],

        "warnings":
            [],

        "evaluated_rule_ids":
            [],
    }
