#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public presentation API for SensorQA.

This package contains application view models that adapt SensorQA's
internal scientific and workflow objects into stable structures for the future
desktop application, reports, exports, and other presentation surfaces.

Application code should prefer importing from ``sensorqa.presentation`` rather
than from individual implementation modules. That keeps UI code decoupled from
the internal organization of the presentation layer.
"""

from sensorqa.presentation.workflow_summary import (
    AnalysisSummaryView,
    DatasetIssueView,
    DatasetSummaryView,
    EvidenceView,
    MetricView,
    PlotView,
    QualificationSummaryView,
    RequirementCheckView,
    SamplingSummaryView,
    WorkflowSummaryBuilder,
    WorkflowSummaryView,
    build_workflow_summary,
)


__all__ = [
    "DatasetIssueView",
    "SamplingSummaryView",
    "DatasetSummaryView",
    "MetricView",
    "EvidenceView",
    "PlotView",
    "AnalysisSummaryView",
    "RequirementCheckView",
    "QualificationSummaryView",
    "WorkflowSummaryView",
    "WorkflowSummaryBuilder",
    "build_workflow_summary",
]
