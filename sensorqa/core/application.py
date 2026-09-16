#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sensorqa.core.config_loader import (
    ConfigLoadReport,
    ConfigLoader,
    SensorQARuntimeConfig,
)
from sensorqa.core.dependency_manager import (
    DependencyManager,
    DependencyResolution,
)
from sensorqa.core.exceptions import (
    DatasetBuildError,
    PipelineConfigurationError,
    RequirementsConfigurationError,
    RuntimeConfigurationError,
)
from sensorqa.core.pipeline import (
    AnalysisPipeline,
    PipelineResult,
)
from sensorqa.core.requirements_engine import (
    QualificationSummary,
    RequirementSet,
    RequirementsEngine,
)
from sensorqa.core.requirements_loader import (
    RequirementLoadReport,
    RequirementsLoader,
)
from sensorqa.core.result_schema import (
    AnalysisResult,
)
from sensorqa.core.tool_contract import (
    SensorType,
)
from sensorqa.core.tool_registry import (
    ToolRegistry,
)
from sensorqa.ingestion.dataset import (
    SensorQADataset,
)
from sensorqa.ingestion.dataset_builder import (
    DatasetBuilder,
)


# Initialization result


@dataclass
class ApplicationInitializationResult:
    """
    Result of configuring the SensorQA application service.
    """

    success: bool

    runtime_config: SensorQARuntimeConfig | None = None

    requirements: RequirementSet | None = None

    config_report: ConfigLoadReport | None = None

    requirements_report: RequirementLoadReport | None = None

    dependency_resolution: DependencyResolution | None = None

    warnings: list[str] = field(
        default_factory=list
    )


# Analysis workflow result


@dataclass
class SensorQAWorkflowResult:
    """
    Complete result of one SensorQA analysis workflow.

    Contains both descriptive analysis and qualification results.
    """

    success: bool

    dataset: SensorQADataset | None = None

    pipeline_result: PipelineResult | None = None

    qualification: QualificationSummary | None = None

    dependency_resolution: DependencyResolution | None = None

    warnings: list[str] = field(
        default_factory=list
    )

    messages: list[str] = field(
        default_factory=list
    )

    @property
    def analyses(
        self,
    ) -> list[AnalysisResult]:
        """Convenience access to completed AnalysisResult objects.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        if self.pipeline_result is None:

            return []

        results = getattr(
            self.pipeline_result,
            "results",
            None,
        )

        if results is None:

            return []

        return list(
            results
        )


# SensorQA application


class SensorQAApplication:
    """
    High-level SensorQA application service.

    The application coordinates:

        ToolRegistry
            ↓
        ConfigLoader
            ↓
        DependencyManager
            ↓
        DatasetBuilder
            ↓
        AnalysisPipeline
            ↓
        RequirementsLoader
            ↓
        RequirementsEngine

    The GUI, API, CLI, or notebook layer should eventually interact
    with this class rather than manually coordinating all backend
    components.

    Tool discovery and registry construction remain separate from
    this class because the ToolRegistry is also useful before
    application configuration is loaded, particularly for plug-in
    inspection and the future Tool Manager.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        dataset_builder: DatasetBuilder,
        pipeline: AnalysisPipeline,
    ) -> None:

        """Initialize the sensor qaapplication.
        
        Args:
            registry: Tool registry used for lookups.
            dataset_builder: Dataset builder used by this function.
            pipeline: Pipeline used by this function.
        
        Returns:
            None.
        """
        if not isinstance(
            registry,
            ToolRegistry,
        ):

            raise TypeError(
                "registry must be a ToolRegistry."
            )

        if not isinstance(
            dataset_builder,
            DatasetBuilder,
        ):

            raise TypeError(
                "dataset_builder must be a DatasetBuilder."
            )

        if not isinstance(
            pipeline,
            AnalysisPipeline,
        ):

            raise TypeError(
                "pipeline must be an AnalysisPipeline."
            )

        self.registry = registry

        self.dataset_builder = (
            dataset_builder
        )

        self.pipeline = pipeline

        self.config_loader = ConfigLoader(
            registry=registry
        )

        self.requirements_loader = (
            RequirementsLoader()
        )

        self.requirements_engine = (
            RequirementsEngine()
        )

        self.dependency_manager = (
            DependencyManager(
                registry=registry
            )
        )

        self.runtime_config: (
            SensorQARuntimeConfig
            | None
        ) = None

        self.requirement_set: (
            RequirementSet
            | None
        ) = None

        self.config_report: (
            ConfigLoadReport
            | None
        ) = None

        self.requirements_report: (
            RequirementLoadReport
            | None
        ) = None

        self.initialized = False

    # Initialization

    def initialize(
        self,
        runtime_config_path: str | Path,
        requirements_path: str | Path | None = None,
        *,
        auto_include_dependencies: bool = True,
        raise_on_error: bool = False,
    ) -> ApplicationInitializationResult:
        """Load and validate application configuration.
        
        This performs:
        
            1. runtime configuration loading
            2. tool selection
            3. dependency resolution
            4. requirements loading
        
        If requirements_path is None, qualification is disabled,
        but descriptive analysis can still run.
        
        Args:
            runtime_config_path: Path used for runtime config.
            requirements_path: Path used for requirements.
            auto_include_dependencies: Flag controlling auto include dependencies.
            raise_on_error: Flag controlling raise on error.
        
        Returns:
            ApplicationInitializationResult returned by the function.
        """

        warnings: list[str] = []

        # Runtime configuration

        config_report = (
            self.config_loader.load(
                runtime_config_path
            )
        )

        self.config_report = (
            config_report
        )

        if not config_report.success:

            if raise_on_error:

                raise RuntimeConfigurationError(
                    "SensorQA runtime configuration "
                    "is invalid.",
                    details={
                        "file":
                            str(
                                runtime_config_path
                            ),

                        "errors": [
                            {
                                "location":
                                    error.location,

                                "message":
                                    error.message,
                            }
                            for error
                            in config_report.errors
                        ],
                    },
                )

            return ApplicationInitializationResult(
                success=False,
                config_report=config_report,
                warnings=list(
                    config_report.warnings
                ),
            )

        if config_report.config is None:

            if raise_on_error:

                raise RuntimeConfigurationError(
                    "Runtime configuration loader "
                    "reported success but returned "
                    "no configuration."
                )

            return ApplicationInitializationResult(
                success=False,
                config_report=config_report,
            )

        runtime_config = (
            config_report.config
        )

        warnings.extend(
            config_report.warnings
        )

        # Dependency resolution

        dependency_resolution = (
            self.dependency_manager.resolve_selection(
                tool_ids=(
                    runtime_config.enabled_tool_ids
                ),
                auto_include_dependencies=(
                    auto_include_dependencies
                ),
            )
        )

        warnings.extend(
            dependency_resolution.warnings
        )

        if not dependency_resolution.valid:

            if raise_on_error:

                raise PipelineConfigurationError(
                    "Enabled analysis tools contain "
                    "invalid dependencies.",
                    details={
                        "issues": [
                            {
                                "tool_id":
                                    issue.tool_id,

                                "dependency_id":
                                    issue.dependency_id,

                                "issue_type":
                                    issue.issue_type,

                                "message":
                                    issue.message,
                            }
                            for issue
                            in dependency_resolution.issues
                        ]
                    },
                )

            return ApplicationInitializationResult(
                success=False,
                runtime_config=runtime_config,
                config_report=config_report,
                dependency_resolution=(
                    dependency_resolution
                ),
                warnings=warnings,
            )

        # Replace enabled order with dependency-correct execution
        # order.
        #
        # Parameters for automatically included dependencies come
        # from their already validated ToolRuntimeConfig defaults.

        runtime_config.enabled_tool_ids = list(
            dependency_resolution.execution_order
        )

        runtime_config.tool_parameters = {
            tool_id:
                runtime_config.parameters_for(
                    tool_id
                )
            for tool_id
            in runtime_config.enabled_tool_ids
        }

        # Requirements

        requirement_set = None
        requirements_report = None

        if requirements_path is not None:

            requirements_report = (
                self.requirements_loader.load(
                    requirements_path
                )
            )

            self.requirements_report = (
                requirements_report
            )

            warnings.extend(
                requirements_report.warnings
            )

            if not requirements_report.success:

                if raise_on_error:

                    raise RequirementsConfigurationError(
                        "SensorQA requirements "
                        "configuration is invalid.",
                        details={
                            "file":
                                str(
                                    requirements_path
                                ),

                            "errors": [
                                {
                                    "location":
                                        error.location,

                                    "message":
                                        error.message,
                                }
                                for error
                                in requirements_report.errors
                            ],
                        },
                    )

                return ApplicationInitializationResult(
                    success=False,
                    runtime_config=runtime_config,
                    config_report=config_report,
                    requirements_report=(
                        requirements_report
                    ),
                    dependency_resolution=(
                        dependency_resolution
                    ),
                    warnings=warnings,
                )

            requirement_set = (
                requirements_report.requirement_set
            )

        # Commit initialized state only after all requested
        # configuration components have succeeded.

        self.runtime_config = (
            runtime_config
        )

        self.requirement_set = (
            requirement_set
        )

        self.initialized = True

        return ApplicationInitializationResult(
            success=True,
            runtime_config=runtime_config,
            requirements=requirement_set,
            config_report=config_report,
            requirements_report=(
                requirements_report
            ),
            dependency_resolution=(
                dependency_resolution
            ),
            warnings=warnings,
        )

    # Analyze an already-built dataset

    def analyze_dataset(
        self,
        dataset: SensorQADataset,
        *,
        qualify: bool = True,
    ) -> SensorQAWorkflowResult:
        """Run the configured SensorQA analyses on an existing
        SensorQADataset.
        
        Args:
            dataset: Dataset to process.
            qualify: Value for `qualify`.
        
        Returns:
            SensorQAWorkflowResult returned by the function.
        """

        self._require_initialized()

        if not isinstance(
            dataset,
            SensorQADataset,
        ):

            raise TypeError(
                "dataset must be a SensorQADataset."
            )

        runtime_config = (
            self._runtime_config()
        )

        messages = []
        warnings = []

        # Analysis tool selection for this dataset
        #
        # Runtime configuration represents tools that are globally
        # enabled in SensorQA. A single configuration may therefore
        # contain tools for more than one sensor family (for example,
        # both IMU tools and generic measurement/reference tools).
        #
        # AnalysisPipeline deliberately treats an explicitly requested
        # incompatible tool as a configuration error. The application
        # layer is responsible for converting the global enabled set
        # into the subset applicable to the current dataset.
        #
        # This keeps tools available for the UI/tool manager without
        # making a mixed global configuration unusable for a specific
        # dataset type.

        selected_tool_ids: list[str] = []
        incompatible_tool_ids: list[str] = []

        sensor_type = dataset.sensor_type

        if isinstance(
            sensor_type,
            SensorType,
        ):

            for tool_id in runtime_config.enabled_tool_ids:

                tool = self.registry.require(
                    tool_id
                )

                if tool.metadata.supports_sensor_type(
                    sensor_type
                ):

                    selected_tool_ids.append(
                        tool_id
                    )

                else:

                    incompatible_tool_ids.append(
                        tool_id
                    )

            if incompatible_tool_ids:

                warnings.append(
                    "Skipped globally enabled tool(s) that are "
                    f"incompatible with sensor type "
                    f"'{sensor_type.value}': "
                    + ", ".join(
                        incompatible_tool_ids
                    )
                )

        else:

            # Preserve AnalysisPipeline as the authority for malformed
            # or missing sensor metadata. Passing the full enabled set
            # lets the pipeline return its normal, explicit sensor-type
            # validation error instead of failing here while filtering.
            selected_tool_ids = list(
                runtime_config.enabled_tool_ids
            )

        selected_tool_parameters = {
            tool_id:
                runtime_config.tool_parameters.get(
                    tool_id,
                    {},
                )
            for tool_id
            in selected_tool_ids
        }

        # Analysis pipeline

        pipeline_result = (
            self.pipeline.run(
                dataset=dataset,
                tool_ids=selected_tool_ids,
                tool_parameters=(
                    selected_tool_parameters
                ),
                allow_invalid_dataset=(
                    runtime_config.pipeline.get(
                        "allow_invalid_dataset",
                        False,
                    )
                ),
            )
        )

        # Qualification

        qualification = None

        if qualify:

            if self.requirement_set is None:

                messages.append(
                    "Qualification was requested, "
                    "but no requirement profile "
                    "is loaded."
                )

            else:

                analysis_results = getattr(
                    pipeline_result,
                    "results",
                    [],
                )

                qualification = (
                    self.requirements_engine.qualify_results(
                        analyses=list(
                            analysis_results
                        ),
                        requirements=(
                            self.requirement_set
                        ),
                    )
                )

                warnings.extend(
                    qualification.messages
                )

        # Determine workflow success
        #
        # Qualification FAIL does NOT mean the software workflow
        # failed. It means the sensor failed one or more configured
        # engineering requirements.
        #
        # Therefore software success is derived from pipeline
        # execution rather than qualification status.

        workflow_success = self._pipeline_completed(
            pipeline_result
        )

        return SensorQAWorkflowResult(
            success=workflow_success,
            dataset=dataset,
            pipeline_result=pipeline_result,
            qualification=qualification,
            dependency_resolution=(
                self.dependency_manager.resolve_selection(
                    tool_ids=selected_tool_ids,
                    auto_include_dependencies=False,
                )
            ),
            warnings=warnings,
            messages=messages,
        )

    # Build and analyze a CSV dataset

    def analyze_csv(
        self,
        source: str | Path,
        *,
        build_kwargs: dict[str, Any] | None = None,
        qualify: bool = True,
        raise_on_build_error: bool = False,
    ) -> SensorQAWorkflowResult:
        """Build a SensorQADataset from a CSV file and run the complete
        configured workflow.
        
        build_kwargs are forwarded directly to DatasetBuilder.build.
        
        This keeps DatasetBuilder as the single authority for:
        
            - CSV loading
            - column mapping
            - unit normalization
            - metadata
            - validation
            - canonical SensorQADataset creation
        
        Example:
        
            application.analyze_csv(
                "imu_test.csv",
                build_kwargs={
                    "column_mapping": mapping,
                    "units": units,
                    "metadata": metadata,
                },
            )
        
        Exact DatasetBuilder keyword options remain defined by
        DatasetBuilder rather than duplicated here.
        
        Args:
            source: Input source or source path.
            build_kwargs: Value for `build_kwargs`.
            qualify: Value for `qualify`.
            raise_on_build_error: Flag controlling raise on build error.
        
        Returns:
            SensorQAWorkflowResult returned by the function.
        """

        self._require_initialized()

        source_path = Path(
            source
        )

        kwargs = dict(
            build_kwargs
            or {}
        )

        # Keep the ingestion interface centralized.

        try:

            build_result = (
                self.dataset_builder.build(
                    source=source_path,
                    **kwargs,
                )
            )

        except Exception as exc:

            if raise_on_build_error:

                raise DatasetBuildError(
                    "DatasetBuilder could not build "
                    "the requested CSV dataset.",
                    details={
                        "source":
                            str(
                                source_path
                            ),

                        "exception_type":
                            exc.__class__.__name__,

                        "exception":
                            str(
                                exc
                            ),
                    },
                ) from exc

            return SensorQAWorkflowResult(
                success=False,
                messages=[
                    (
                        "Dataset construction failed: "
                        f"{exc}"
                    )
                ],
            )

        # DatasetBuilder returns a structured result.
        #
        # Use its success and dataset attributes rather than
        # assuming that a returned object is automatically
        # analysis-ready.

        build_success = getattr(
            build_result,
            "success",
            False,
        )

        dataset = getattr(
            build_result,
            "dataset",
            None,
        )

        if (
            not build_success
            or dataset is None
        ):

            messages = (
                self._extract_build_messages(
                    build_result
                )
            )

            if raise_on_build_error:

                raise DatasetBuildError(
                    "Dataset could not be built.",
                    details={
                        "source":
                            str(
                                source_path
                            ),

                        "messages":
                            messages,
                    },
                )

            return SensorQAWorkflowResult(
                success=False,
                messages=messages,
            )

        # Some build results may successfully construct a dataset
        # while marking it unsuitable for normal analysis.
        #
        # The pipeline remains the final authority because
        # allow_invalid_dataset may explicitly permit advanced
        # investigation.

        return self.analyze_dataset(
            dataset=dataset,
            qualify=qualify,
        )

    # Runtime tool control

    def enabled_tools(
        self,
    ) -> list[str]:
        """Return dependency-resolved enabled tools.
        
        Returns:
            List of result values.
        """

        self._require_initialized()

        return list(
            self._runtime_config().enabled_tool_ids
        )

    def tool_parameters(
        self,
        tool_id: str,
    ) -> dict[str, Any]:
        """Return validated parameters for a configured tool.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Dictionary containing the result values.
        """

        self._require_initialized()

        return (
            self._runtime_config().parameters_for(
                tool_id
            )
        )

    def dependency_summary(
        self,
        tool_id: str,
    ) -> dict:
        """Return dependency information for the Tool Manager.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Dictionary containing the result values.
        """

        return (
            self.dependency_manager.dependency_summary(
                tool_id
            )
        )

    # Tool enablement preview

    def preview_enable_tool(
        self,
        tool_id: str,
    ) -> DependencyResolution:
        """Determine what would need to be enabled if the user toggled
        on a particular tool.
        
        This does not mutate runtime configuration.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            DependencyResolution returned by the function.
        """

        self._require_initialized()

        current = (
            self._runtime_config().enabled_tool_ids
        )

        requested = list(
            current
        )

        if tool_id not in requested:

            requested.append(
                tool_id
            )

        return (
            self.dependency_manager.resolve_selection(
                tool_ids=requested,
                auto_include_dependencies=True,
            )
        )

    def preview_disable_tool(
        self,
        tool_id: str,
    ) -> dict[str, Any]:
        """Determine whether disabling a tool would break enabled
        dependents.
        
        This does not mutate runtime configuration.
        
        Args:
            tool_id: Registered tool identifier.
        
        Returns:
            Dictionary containing the result values.
        """

        self._require_initialized()

        enabled = (
            self._runtime_config().enabled_tool_ids
        )

        can_disable, blockers = (
            self.dependency_manager.can_disable(
                tool_id=tool_id,
                enabled_tool_ids=enabled,
            )
        )

        return {
            "tool_id":
                tool_id,

            "currently_enabled":
                tool_id in enabled,

            "can_disable":
                can_disable,

            "blocking_dependents":
                blockers,
        }

    # Qualification-only workflow

    def qualify_results(
        self,
        analyses: list[AnalysisResult],
    ) -> QualificationSummary:
        """Apply the loaded requirement profile to existing
        AnalysisResult objects without rerunning analysis.
        
        Args:
            analyses: Value for `analyses`.
        
        Returns:
            QualificationSummary returned by the function.
        """

        self._require_initialized()

        if self.requirement_set is None:

            raise RequirementsConfigurationError(
                "No requirements profile is loaded."
            )

        return (
            self.requirements_engine.qualify_results(
                analyses=analyses,
                requirements=self.requirement_set,
            )
        )

    # Application state

    def configuration_snapshot(
        self,
    ) -> dict[str, Any]:
        """Return a reproducibility-oriented configuration snapshot.
        
        Returns:
            Dictionary containing the result values.
        """

        self._require_initialized()

        runtime_config = (
            self._runtime_config()
        )

        snapshot = {
            "runtime_configuration_file":
                runtime_config.file_path,

            "enabled_tool_ids":
                list(
                    runtime_config.enabled_tool_ids
                ),

            "tool_parameters": {
                tool_id:
                    dict(
                        parameters
                    )
                for tool_id, parameters
                in runtime_config.tool_parameters.items()
            },

            "pipeline":
                dict(
                    runtime_config.pipeline
                ),

            "application":
                dict(
                    runtime_config.application
                ),
        }

        if self.requirement_set is not None:

            snapshot[
                "requirements"
            ] = {
                "name":
                    self.requirement_set.name,

                "version":
                    self.requirement_set.version,

                "enabled_requirement_count":
                    len(
                        self.requirement_set.enabled_requirements()
                    ),
            }

        else:

            snapshot[
                "requirements"
            ] = None

        return snapshot

    # Internal helpers

    def _require_initialized(
        self,
    ) -> None:

        """Return require initialized.
        
        Returns:
            None.
        """
        if not self.initialized:

            raise RuntimeConfigurationError(
                "SensorQAApplication has not been initialized. "
                "Call initialize() before running analysis."
            )

    def _runtime_config(
        self,
    ) -> SensorQARuntimeConfig:

        """Run time config.
        
        Returns:
            SensorQARuntimeConfig returned by this function.
        """
        if self.runtime_config is None:

            raise RuntimeConfigurationError(
                "SensorQA runtime configuration is unavailable."
            )

        return self.runtime_config

    @staticmethod
    def _pipeline_completed(
        pipeline_result: PipelineResult,
    ) -> bool:
        """Determine whether the software workflow itself completed.
        
        Do not confuse this with engineering qualification.
        
        Args:
            pipeline_result: Value for `pipeline_result`.
        
        Returns:
            Boolean result.
        """

        success = getattr(
            pipeline_result,
            "success",
            None,
        )

        if isinstance(
            success,
            bool,
        ):

            return success

        # If PipelineResult does not expose a global success field,
        # reaching this point with a result object means the pipeline
        # completed its orchestration. Individual tools may still
        # contain ERROR or SKIPPED statuses.
        return True

    @staticmethod
    def _extract_build_messages(
        build_result: Any,
    ) -> list[str]:
        """Extract useful messages from DatasetBuilder's structured
        result without coupling the application layer to every
        ingestion report detail.
        
        Args:
            build_result: Value for `build_result`.
        
        Returns:
            List of result values.
        """

        messages: list[str] = []

        for attribute_name in (
            "errors",
            "warnings",
            "messages",
        ):

            values = getattr(
                build_result,
                attribute_name,
                None,
            )

            if not values:
                continue

            try:

                iterator = list(
                    values
                )

            except TypeError:

                iterator = [
                    values
                ]

            for value in iterator:

                if isinstance(
                    value,
                    str,
                ):

                    messages.append(
                        value
                    )

                    continue

                message = getattr(
                    value,
                    "message",
                    None,
                )

                if message is not None:

                    messages.append(
                        str(
                            message
                        )
                    )

                else:

                    messages.append(
                        str(
                            value
                        )
                    )

        if not messages:

            messages.append(
                "DatasetBuilder did not produce an "
                "analysis-ready dataset."
            )

        return messages
