#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import AnalysisResult, ExecutionStatus
from sensorqa.core.tool_contract import BaseAnalysisTool, ParameterType, SensorType, ToolCategory, ToolContext, ToolMetadata, ToolParameter
from sensorqa.tools.generic._utils import numeric_column


class HysteresisTool(BaseAnalysisTool):
    """Compare output at overlapping reference levels for opposite sweep directions."""

    @property
    def metadata(self) -> ToolMetadata:
        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="generic_hysteresis", name="Hysteresis", version="1.0.0",
            description=(
                "Quantifies output differences between two sweep directions at overlapping reference conditions."
            ),
            category=ToolCategory.STABILITY,
            compatible_sensor_types=[SensorType.GENERIC],
            required_columns=["measurement", "reference", "sweep_direction"], optional_columns=[],
            parameters=[
                ToolParameter(name="minimum_overlap_levels", parameter_type=ParameterType.INTEGER, default=3,
                              description="Minimum matched reference levels between sweep directions.", minimum=1, maximum=1_000_000),
                ToolParameter(name="reference_match_tolerance", parameter_type=ParameterType.FLOAT, default=0.0,
                              description="Absolute tolerance used to match reference levels between sweeps.", minimum=0.0, maximum=1.0e100),
            ], dependencies=[], author="SensorQA",
        )

    @staticmethod
    def _two_labels(values: pd.Series):
        """Return two labels.
        
        Args:
            values: Values to process.
        
        Returns:
            Calculated value.
        """
        labels = [value for value in pd.unique(values.dropna())]
        return labels if len(labels) == 2 else None

    def validate(self, data: pd.DataFrame, context: ToolContext):
        """Validate validate.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Validation result.
        """
        validation = super().validate(data, context)
        if not validation.valid:
            return validation
        sweep_col = context.column_mapping.get("sweep_direction")
        if sweep_col is None:
            validation.errors.append("Sweep-direction mapping is unavailable.")
        else:
            labels = self._two_labels(data[sweep_col])
            if labels is None:
                validation.errors.append("Hysteresis analysis requires exactly two non-null sweep-direction labels.")
        if validation.errors:
            validation.valid = False
        return validation

    @staticmethod
    def _level_means(reference: np.ndarray, measurement: np.ndarray):
        """Return level means.
        
        Args:
            reference: Reference used by this function.
            measurement: Measurement used by this function.
        
        Returns:
            Calculated value.
        """
        result = []
        for value in np.unique(reference):
            mask = reference == value
            result.append((float(value), float(np.mean(measurement[mask]))))
        return result

    def run(self, data: pd.DataFrame, context: ToolContext) -> AnalysisResult:
        """Run run.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing the calculated metrics and messages.
        """
        mcol = context.column_mapping["measurement"]
        rcol = context.column_mapping["reference"]
        scol = context.column_mapping["sweep_direction"]
        labels = self._two_labels(data[scol])
        if labels is None:
            raise ValueError("Exactly two sweep directions are required.")
        tolerance = float(context.parameters.get("reference_match_tolerance", 0.0))
        minimum_overlap = int(context.parameters.get("minimum_overlap_levels", 3))
        summaries = {}
        for label in labels:
            subset = data[data[scol] == label]
            ref = numeric_column(subset, rcol)
            meas = numeric_column(subset, mcol)
            mask = np.isfinite(ref) & np.isfinite(meas)
            summaries[label] = self._level_means(ref[mask], meas[mask])
        first, second = labels
        pairs = []
        used_second = set()
        for ref_a, meas_a in summaries[first]:
            candidates = [
                (abs(ref_b - ref_a), idx, ref_b, meas_b)
                for idx, (ref_b, meas_b) in enumerate(summaries[second])
                if idx not in used_second and abs(ref_b - ref_a) <= tolerance
            ]
            if not candidates:
                continue
            _, idx, ref_b, meas_b = min(candidates, key=lambda item: item[0])
            used_second.add(idx)
            pairs.append((0.5 * (ref_a + ref_b), meas_a, meas_b, meas_a - meas_b))
        if len(pairs) < minimum_overlap:
            raise ValueError(
                f"Only {len(pairs)} overlapping reference levels were matched; {minimum_overlap} are required."
            )
        differences = np.asarray([item[3] for item in pairs], dtype=float)
        absolute = np.abs(differences)
        max_hysteresis = float(np.max(absolute))
        mean_abs = float(np.mean(absolute))
        rms = float(np.sqrt(np.mean(differences ** 2)))
        reference_values = np.asarray([item[0] for item in pairs], dtype=float)
        reference_span = float(np.max(reference_values) - np.min(reference_values)) if len(reference_values) > 1 else 0.0
        unit = context.units.get("measurement")

        result = AnalysisResult(self.metadata.tool_id, self.metadata.name, self.metadata.version, ExecutionStatus.SUCCESS)
        result.add_metric("Maximum Absolute Hysteresis", max_hysteresis, unit)
        result.add_metric("Mean Absolute Hysteresis", mean_abs, unit)
        result.add_metric("Hysteresis RMS", rms, unit)
        result.add_metric("Matched Hysteresis Level Count", len(pairs))
        result.add_metric("Matched Reference Span", reference_span, context.units.get("reference"))
        result.metadata.update({
            "sweep_labels": [str(first), str(second)],
            "difference_definition": f"mean({first}) - mean({second})",
            "reference_match_tolerance": tolerance,
            "matched_levels": [
                {"reference": float(ref), "first_mean": float(a), "second_mean": float(b), "difference": float(diff)}
                for ref, a, b, diff in pairs
            ],
        })
        if tolerance > 0.0:
            result.add_warning("Reference levels were matched within a numerical tolerance rather than exact equality.")
        return result
