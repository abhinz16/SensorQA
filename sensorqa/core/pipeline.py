#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sensorqa.core.result_schema import (
    AnalysisResult,
    ExecutionStatus,
)
from sensorqa.core.tool_contract import (
    SensorType,
    ToolContext,
    ToolParameter,
)
from sensorqa.core.tool_registry import ToolRegistry
from sensorqa.ingestion.dataset import SensorQADataset


@dataclass
class ToolExecutionRecord:
    """
    Records the execution of one analysis tool.
    """

    tool_id: str
    tool_name: str

    status: ExecutionStatus

    result: AnalysisResult

    parameters: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class PipelineResult:
    """
    Complete result of one SensorQA analysis pipeline run.
    """

    success: bool

    results: list[AnalysisResult] = field(
        default_factory=list
    )

    execution_records: list[ToolExecutionRecord] = field(
        default_factory=list
    )

    execution_order: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    def get_result(
        self,
        tool_id: str,
    ) -> AnalysisResult | None:
        """Retrieve the result produced by a specific tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        for result in self.results:

            if result.tool_id == tool_id:
                return result

        return None

    @property
    def successful_results(
        self,
    ) -> list[AnalysisResult]:
        """Return analyses that executed successfully.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        return [
            result
            for result in self.results
            if result.status == ExecutionStatus.SUCCESS
        ]

    @property
    def skipped_results(
        self,
    ) -> list[AnalysisResult]:
        """Return analyses that were skipped.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        return [
            result
            for result in self.results
            if result.status == ExecutionStatus.SKIPPED
        ]

    @property
    def error_results(
        self,
    ) -> list[AnalysisResult]:
        """Return analyses that encountered execution errors.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        return [
            result
            for result in self.results
            if result.status == ExecutionStatus.ERROR
        ]


class AnalysisPipeline:
    """
    Executes registered SensorQA analysis tools on a
    SensorQADataset.

    Responsibilities:

        1. Verify the dataset can be analyzed.
        2. Resolve requested tools.
        3. Check tool dependencies.
        4. Determine execution order.
        5. Build ToolContext objects.
        6. Execute each tool safely.
        7. Collect standardized AnalysisResult objects.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:

        """Initialize the analysis pipeline.
        
        Args:
            registry: Tool registry used for lookups.
        
        Returns:
            None.
        """
        self.registry = registry

    def run(
        self,
        dataset: SensorQADataset,
        tool_ids: list[str],
        tool_parameters: dict[
            str,
            dict[str, Any],
        ] | None = None,
        allow_invalid_dataset: bool = False,
    ) -> PipelineResult:
        """Run selected SensorQA tools.
        
        Parameters
        ----------
        dataset:
            Fully constructed SensorQADataset.
        
        tool_ids:
            Tool IDs requested for this analysis.
        
            Example:
        
            [
                "generic_accuracy",
                "generic_drift"
            ]
        
        tool_parameters:
            Optional parameter overrides by tool ID.
        
            Example:
        
            {
                "generic_drift": {
                    "model": "linear"
                }
            }
        
        allow_invalid_dataset:
            If False, error-level dataset validation findings
            prevent analysis execution.
        
        Args:
            dataset: Dataset to process.
            tool_ids: Tool identifiers to process.
            tool_parameters: Value for `tool_parameters`.
            allow_invalid_dataset: Flag controlling allow invalid dataset.
        
        Returns:
            PipelineResult returned by the function.
        """

        warnings: list[str] = []
        errors: list[str] = []

        tool_parameters = (
            tool_parameters
            or {}
        )

        # STEP 1: Basic dataset checks

        if not isinstance(
            dataset,
            SensorQADataset,
        ):
            return PipelineResult(
                success=False,
                errors=[
                    "dataset must be a SensorQADataset object."
                ],
            )

        if dataset.row_count == 0:

            return PipelineResult(
                success=False,
                errors=[
                    "Dataset contains no rows."
                ],
            )

        if (
            dataset.validation is not None
            and not dataset.validation.valid
            and not allow_invalid_dataset
        ):

            return PipelineResult(
                success=False,
                errors=[
                    (
                        "Dataset contains error-level quality "
                        "findings. Resolve them before running "
                        "analysis, or explicitly allow execution "
                        "on an invalid dataset."
                    )
                ],
            )

        if (
            dataset.validation is not None
            and not dataset.validation.valid
            and allow_invalid_dataset
        ):

            warnings.append(
                "Analysis is being run on a dataset containing "
                "error-level data-quality findings."
            )

        # STEP 2: Sensor type

        sensor_type = dataset.sensor_type

        if sensor_type is None:

            return PipelineResult(
                success=False,
                errors=[
                    (
                        "Sensor type is not defined in dataset "
                        "metadata. SensorQA cannot determine tool "
                        "compatibility."
                    )
                ],
            )

        if not isinstance(
            sensor_type,
            SensorType,
        ):

            return PipelineResult(
                success=False,
                errors=[
                    (
                        "Dataset sensor type is invalid. "
                        "Expected a SensorType value."
                    )
                ],
            )

        # STEP 3: Clean requested tool IDs

        requested_tool_ids = (
            self._clean_tool_ids(
                tool_ids
            )
        )

        if not requested_tool_ids:

            return PipelineResult(
                success=False,
                errors=[
                    "No analysis tools were selected."
                ],
            )

        # STEP 4: Verify tools exist

        missing_tools = [
            tool_id
            for tool_id
            in requested_tool_ids
            if not self.registry.contains(
                tool_id
            )
        ]

        if missing_tools:

            return PipelineResult(
                success=False,
                errors=[
                    (
                        "The following requested tools are not "
                        "registered: "
                        + ", ".join(
                            sorted(
                                missing_tools
                            )
                        )
                    )
                ],
            )

        # STEP 5: Verify tool compatibility

        incompatible_tools: list[str] = []

        for tool_id in requested_tool_ids:

            tool = self.registry.require(
                tool_id
            )

            if not tool.metadata.supports_sensor_type(
                sensor_type
            ):

                incompatible_tools.append(
                    tool_id
                )

        if incompatible_tools:

            return PipelineResult(
                success=False,
                errors=[
                    (
                        "The following tools are incompatible with "
                        f"sensor type '{sensor_type.value}': "
                        + ", ".join(
                            sorted(
                                incompatible_tools
                            )
                        )
                    )
                ],
            )

        # STEP 6: Verify dependencies are selected

        dependency_errors = (
            self._check_selected_dependencies(
                requested_tool_ids
            )
        )

        if dependency_errors:

            return PipelineResult(
                success=False,
                errors=dependency_errors,
            )

        # STEP 7: Determine dependency-safe execution order

        try:

            execution_order = (
                self._resolve_execution_order(
                    requested_tool_ids
                )
            )

        except ValueError as exc:

            return PipelineResult(
                success=False,
                errors=[
                    str(exc)
                ],
            )

        # STEP 8: Execute tools

        results: list[AnalysisResult] = []

        execution_records: list[
            ToolExecutionRecord
        ] = []

        completed_tool_ids: set[str] = set()

        for tool_id in execution_order:

            tool = self.registry.require(
                tool_id
            )

            # If a required dependency failed or was skipped,
            # do not execute the dependent tool.

            dependency_failure = (
                self._find_failed_dependency(
                    tool_id=tool_id,
                    results=results,
                )
            )

            if dependency_failure is not None:

                result = AnalysisResult(
                    tool_id=tool.metadata.tool_id,
                    tool_name=tool.metadata.name,
                    tool_version=tool.metadata.version,
                    status=ExecutionStatus.SKIPPED,
                    messages=[
                        (
                            "Analysis was skipped because required "
                            f"dependency '{dependency_failure}' "
                            "did not complete successfully."
                        )
                    ],
                )

                results.append(
                    result
                )

                execution_records.append(
                    ToolExecutionRecord(
                        tool_id=tool_id,
                        tool_name=tool.metadata.name,
                        status=result.status,
                        result=result,
                        parameters={},
                    )
                )

                continue

            # Resolve parameter values

            try:

                parameters = (
                    self._build_tool_parameters(
                        tool_id=tool_id,
                        overrides=tool_parameters.get(
                            tool_id,
                            {},
                        ),
                    )
                )

            except ValueError as exc:

                result = AnalysisResult(
                    tool_id=tool.metadata.tool_id,
                    tool_name=tool.metadata.name,
                    tool_version=tool.metadata.version,
                    status=ExecutionStatus.ERROR,
                    messages=[
                        str(exc)
                    ],
                )

                results.append(
                    result
                )

                execution_records.append(
                    ToolExecutionRecord(
                        tool_id=tool_id,
                        tool_name=tool.metadata.name,
                        status=result.status,
                        result=result,
                        parameters={},
                    )
                )

                continue

            # Build context

            context = self._build_context(
                dataset=dataset,
                sensor_type=sensor_type,
                parameters=parameters,
                completed_tool_ids=completed_tool_ids,
            )

            # Give the tool a defensive copy of the data

            analysis_data = (
                dataset.analysis_view()
            )

            # Execute using BaseAnalysisTool.execute()
            #
            # This performs:
            #     validation
            #     execution
            #     exception handling

            result = tool.execute(
                data=analysis_data,
                context=context,
            )

            results.append(
                result
            )

            execution_records.append(
                ToolExecutionRecord(
                    tool_id=tool_id,
                    tool_name=tool.metadata.name,
                    status=result.status,
                    result=result,
                    parameters=parameters,
                )
            )

            if result.status == ExecutionStatus.SUCCESS:

                completed_tool_ids.add(
                    tool_id
                )

        # STEP 9: Determine overall pipeline state

        error_results = [
            result
            for result in results
            if result.status == ExecutionStatus.ERROR
        ]

        successful_results = [
            result
            for result in results
            if result.status == ExecutionStatus.SUCCESS
        ]

        # Pipeline-level success means execution was able to
        # proceed and at least one requested analysis succeeded.
        #
        # Individual tools may still have been skipped.

        pipeline_success = (
            len(successful_results) > 0
            and len(error_results) == 0
        )

        if error_results:

            warnings.append(
                f"{len(error_results)} analysis tool(s) "
                "encountered execution errors."
            )

        skipped_count = sum(
            result.status
            == ExecutionStatus.SKIPPED
            for result in results
        )

        if skipped_count > 0:

            warnings.append(
                f"{skipped_count} analysis tool(s) were skipped."
            )

        return PipelineResult(
            success=pipeline_success,
            results=results,
            execution_records=execution_records,
            execution_order=execution_order,
            warnings=warnings,
            errors=errors,
        )

    # Context construction

    @staticmethod
    def _build_context(
        dataset: SensorQADataset,
        sensor_type: SensorType,
        parameters: dict[str, Any],
        completed_tool_ids: set[str],
    ) -> ToolContext:
        """Create the ToolContext supplied to one analysis tool.
        
        Because SensorQADataset.data already uses standardized
        SensorQA column names, the tool receives an identity
        column mapping:
        
            measurement -> measurement
            reference   -> reference
            gx          -> gx
            ...
        
        Args:
            dataset: Dataset to process.
            sensor_type: Sensor type used to choose compatible tools.
            parameters: Tool parameter values.
            completed_tool_ids: Value for `completed_tool_ids`.
        
        Returns:
            ToolContext returned by the function.
        """

        column_mapping = {
            str(column): str(column)
            for column in dataset.data.columns
        }

        context_metadata: dict[str, Any] = {
            "dataset_summary":
                dataset.summary(),

            "completed_tool_ids":
                sorted(
                    completed_tool_ids
                ),
        }

        if dataset.metadata is not None:

            context_metadata[
                "sensorqa_metadata"
            ] = dataset.metadata.to_dict()

        if dataset.validation is not None:

            context_metadata[
                "sampling"
            ] = {
                "available":
                    dataset.validation.sampling.available,

                "sample_count":
                    dataset.validation.sampling.sample_count,

                "duration_seconds":
                    dataset.validation.sampling.duration_seconds,

                "estimated_sampling_rate_hz":
                    dataset.validation.sampling
                    .estimated_sampling_rate_hz,

                "median_interval_seconds":
                    dataset.validation.sampling
                    .median_interval_seconds,

                "coefficient_of_variation":
                    dataset.validation.sampling
                    .coefficient_of_variation,

                "large_gap_count":
                    dataset.validation.sampling
                    .large_gap_count,
            }

        return ToolContext(
            sensor_type=sensor_type,
            column_mapping=column_mapping,
            units=dict(
                dataset.units
            ),
            parameters=parameters,
            metadata=context_metadata,
        )

    # Parameter handling

    def _build_tool_parameters(
        self,
        tool_id: str,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine tool-declared default parameters with user
        configuration overrides.
        
        Args:
            tool_id: Registered tool identifier.
            overrides: Value for `overrides`.
        
        Returns:
            Dictionary containing the result values.
        """

        tool = self.registry.require(
            tool_id
        )

        definitions = {
            parameter.name: parameter
            for parameter
            in tool.metadata.parameters
        }

        parameters = {
            parameter.name:
                parameter.default
            for parameter
            in tool.metadata.parameters
        }

        unknown_parameters = [
            parameter_name
            for parameter_name
            in overrides
            if parameter_name
            not in definitions
        ]

        if unknown_parameters:

            raise ValueError(
                f"Tool '{tool_id}' received unknown parameter(s): "
                + ", ".join(
                    sorted(
                        unknown_parameters
                    )
                )
            )

        for parameter_name, value in overrides.items():

            definition = definitions[
                parameter_name
            ]

            self._validate_parameter_override(
                definition=definition,
                value=value,
            )

            parameters[
                parameter_name
            ] = value

        return parameters

    @staticmethod
    def _validate_parameter_override(
        definition: ToolParameter,
        value: Any,
    ) -> None:
        """Validate a user-supplied parameter value against the
        parameter definition.
        
        ToolValidator validates defaults when tools are loaded.
        The pipeline validates runtime overrides.
        
        Args:
            definition: Value for `definition`.
            value: Value to process.
        
        Returns:
            None.
        """

        parameter_type = (
            definition.parameter_type.value
        )

        if parameter_type == "integer":

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
            ):

                raise ValueError(
                    f"Parameter '{definition.name}' "
                    "must be an integer."
                )

        elif parameter_type == "float":

            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    (int, float),
                )
            ):

                raise ValueError(
                    f"Parameter '{definition.name}' "
                    "must be numeric."
                )

        elif parameter_type == "string":

            if not isinstance(
                value,
                str,
            ):

                raise ValueError(
                    f"Parameter '{definition.name}' "
                    "must be a string."
                )

        elif parameter_type == "boolean":

            if not isinstance(
                value,
                bool,
            ):

                raise ValueError(
                    f"Parameter '{definition.name}' "
                    "must be True or False."
                )

        elif parameter_type == "choice":

            if value not in definition.choices:

                raise ValueError(
                    f"Parameter '{definition.name}' must be one "
                    "of: "
                    + ", ".join(
                        str(choice)
                        for choice
                        in definition.choices
                    )
                )

        # Numeric limits

        if isinstance(
            value,
            (int, float),
        ) and not isinstance(
            value,
            bool,
        ):

            if (
                definition.minimum is not None
                and value < definition.minimum
            ):

                raise ValueError(
                    f"Parameter '{definition.name}' must be "
                    f">= {definition.minimum}."
                )

            if (
                definition.maximum is not None
                and value > definition.maximum
            ):

                raise ValueError(
                    f"Parameter '{definition.name}' must be "
                    f"<= {definition.maximum}."
                )

    # Dependency handling

    def _check_selected_dependencies(
        self,
        tool_ids: list[str],
    ) -> list[str]:
        """Verify that every dependency of every selected tool
        is also selected.
        
        SensorQA does not silently add analyses the user did not
        request.
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            List of result values.
        """

        selected = set(
            tool_ids
        )

        errors: list[str] = []

        for tool_id in tool_ids:

            tool = self.registry.require(
                tool_id
            )

            missing = [
                dependency
                for dependency
                in tool.metadata.dependencies
                if dependency
                not in selected
            ]

            if missing:

                errors.append(
                    f"Tool '{tool_id}' requires selected tool(s): "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )

        return errors

    def _resolve_execution_order(
        self,
        tool_ids: list[str],
    ) -> list[str]:
        """Determine execution order using a topological sort.
        
        Dependencies always run before the tools that depend
        on them.
        
        Circular dependencies raise ValueError.
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            List of result values.
        """

        selected = set(
            tool_ids
        )

        dependencies: dict[
            str,
            set[str],
        ] = {}

        for tool_id in tool_ids:

            tool = self.registry.require(
                tool_id
            )

            dependencies[
                tool_id
            ] = {
                dependency
                for dependency
                in tool.metadata.dependencies
                if dependency in selected
            }

        ordered: list[str] = []

        remaining = {
            tool_id: set(deps)
            for tool_id, deps
            in dependencies.items()
        }

        while remaining:

            ready = [
                tool_id
                for tool_id, deps
                in remaining.items()
                if not deps
            ]

            if not ready:

                cycle_tools = sorted(
                    remaining.keys()
                )

                raise ValueError(
                    "Circular analysis-tool dependency detected "
                    "among: "
                    + ", ".join(
                        cycle_tools
                    )
                )

            # Preserve the user's requested order wherever
            # dependency constraints allow it.

            ready.sort(
                key=lambda tool_id:
                    tool_ids.index(
                        tool_id
                    )
            )

            for tool_id in ready:

                ordered.append(
                    tool_id
                )

                del remaining[
                    tool_id
                ]

                for deps in remaining.values():

                    deps.discard(
                        tool_id
                    )

        return ordered

    def _find_failed_dependency(
        self,
        tool_id: str,
        results: list[AnalysisResult],
    ) -> str | None:
        """Check whether one of a tool's required dependencies
        failed or was skipped.
        
        Args:
            tool_id: Registered tool identifier.
            results: Value for `results`.
        
        Returns:
            str | None returned by the function.
        """

        tool = self.registry.require(
            tool_id
        )

        if not tool.metadata.dependencies:
            return None

        result_by_id = {
            result.tool_id: result
            for result in results
        }

        for dependency in tool.metadata.dependencies:

            dependency_result = (
                result_by_id.get(
                    dependency
                )
            )

            if dependency_result is None:
                return dependency

            if (
                dependency_result.status
                != ExecutionStatus.SUCCESS
            ):
                return dependency

        return None

    # General helpers

    @staticmethod
    def _clean_tool_ids(
        tool_ids: list[str],
    ) -> list[str]:
        """Remove duplicates while preserving requested order.
        
        Args:
            tool_ids: Tool identifiers to process.
        
        Returns:
            List of result values.
        """

        cleaned: list[str] = []

        seen: set[str] = set()

        for tool_id in tool_ids:

            if not isinstance(
                tool_id,
                str,
            ):

                raise TypeError(
                    "All tool IDs must be strings."
                )

            tool_id = (
                tool_id.strip()
            )

            if not tool_id:
                continue

            if tool_id in seen:
                continue

            seen.add(
                tool_id
            )

            cleaned.append(
                tool_id
            )

        return cleaned
