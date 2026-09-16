#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.core.application import (
    SensorQAApplication,
)
from sensorqa.core.pipeline import (
    AnalysisPipeline,
)
from sensorqa.core.result_schema import (
    ExecutionStatus,
)
from sensorqa.core.tool_contract import (
    SensorType,
)
from sensorqa.core.tool_loader import (
    ToolLoader,
)
from sensorqa.core.tool_registry import (
    ToolRegistry,
)
from sensorqa.core.tool_validator import (
    ToolValidator,
)
from sensorqa.ingestion.column_mapper import (
    ColumnMapper,
    ColumnMapping,
)
from sensorqa.ingestion.dataset_builder import (
    DatasetBuilder,
)
from sensorqa.ingestion.dataset_validator import (
    DatasetValidator,
)
from sensorqa.ingestion.metadata import (
    DatasetSourceInformation,
    SensorInformation,
    SensorQAMetadata,
    SensorRange,
    TestInformation as SensorQATestInformation,
    TestMode as SensorQATestMode,
)
from sensorqa.ingestion.unit_manager import (
    UnitManager,
)


# Project paths


def find_project_root() -> Path:
    """Locate the SensorQA project directory.
    
    The expected project directory contains:
    
        configs/sensorqa.toml
        configs/requirements.toml
        tools/
    
    Returns:
        Resolved path.
    """

    current = Path(
        __file__
    ).resolve()

    for candidate in (
        current.parent,
        *current.parents,
    ):

        if (
            (
                candidate
                / "configs"
                / "sensorqa.toml"
            ).is_file()
            and (
                candidate
                / "configs"
                / "requirements.toml"
            ).is_file()
            and (
                candidate
                / "tools"
            ).is_dir()
        ):

            return candidate

    raise RuntimeError(
        "Could not locate SensorQA project root."
    )


PROJECT_ROOT = find_project_root()

TOOLS_ROOT = (
    PROJECT_ROOT
    / "tools"
)

GENERIC_TOOLS_DIR = (
    TOOLS_ROOT
    / "generic"
)

IMU_TOOLS_DIR = (
    TOOLS_ROOT
    / "imu"
)

CUSTOM_TOOLS_DIR = (
    TOOLS_ROOT
    / "custom"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "sensorqa.toml"
)

REQUIREMENTS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "requirements.toml"
)


# Synthetic stationary IMU dataset


def stationary_imu_dataframe(
) -> pd.DataFrame:
    """Generate a deterministic stationary IMU test dataset.
    
    Characteristics
    ---------------
    Sampling:
        100 Hz
    
    Duration:
        60 seconds
    
    Accelerometer:
        Approximately +1 g on Z with small bias/noise.
    
    Gyroscope:
        Small nonzero zero-rate biases with noise and a shared
        periodic component.
    
    Temperature:
        22 to 25 degC.
    
    Orientation:
        +Z.
    
    Motion state:
        stationary.
    
    Returns:
        DataFrame containing the requested data.
    """

    rng = np.random.default_rng(
        20260916
    )

    sample_rate_hz = 100.0

    duration_seconds = 60.0

    number_samples = int(
        sample_rate_hz
        * duration_seconds
    )

    timestamp = (
        np.arange(
            number_samples,
            dtype=float,
        )
        / sample_rate_hz
    )

    temperature = np.linspace(
        22.0,
        25.0,
        number_samples,
    )

    temperature_centered = (
        temperature
        - np.mean(
            temperature
        )
    )

    periodic_component = (
        2.0e-4
        * np.sin(
            2.0
            * np.pi
            * 8.0
            * timestamp
        )
    )

    # Gyroscope

    gx = (
        0.0010
        + 4.0e-5
        * temperature_centered
        + periodic_component
        + rng.normal(
            0.0,
            6.0e-4,
            number_samples,
        )
    )

    gy = (
        -0.0013
        + 3.0e-5
        * temperature_centered
        + 0.85
        * periodic_component
        + rng.normal(
            0.0,
            6.0e-4,
            number_samples,
        )
    )

    gz = (
        0.0007
        + rng.normal(
            0.0,
            6.0e-4,
            number_samples,
        )
    )

    # Accelerometer

    gravity = 9.80665

    ax = (
        0.012
        + rng.normal(
            0.0,
            0.018,
            number_samples,
        )
    )

    ay = (
        -0.009
        + rng.normal(
            0.0,
            0.018,
            number_samples,
        )
    )

    az = (
        gravity
        + 0.020
        + rng.normal(
            0.0,
            0.018,
            number_samples,
        )
    )

    return pd.DataFrame(
        {
            "timestamp":
                timestamp,

            "ax":
                ax,

            "ay":
                ay,

            "az":
                az,

            "gx":
                gx,

            "gy":
                gy,

            "gz":
                gz,

            "temperature":
                temperature,

            "orientation_label":
                np.full(
                    number_samples,
                    "+Z",
                    dtype=object,
                ),

            "motion_state":
                np.full(
                    number_samples,
                    "stationary",
                    dtype=object,
                ),
        }
    )


@pytest.fixture
def imu_csv(
    tmp_path,
) -> Path:
    """Write the deterministic IMU fixture to a real CSV.
    
    Args:
        tmp_path: Pytest temporary-directory fixture.
    
    Returns:
        Resolved path.
    """

    path = (
        tmp_path
        / "stationary_imu.csv"
    )

    stationary_imu_dataframe().to_csv(
        path,
        index=False,
    )

    return path


# Metadata helpers


def construct_dataclass_like(
    target_class,
    candidates: dict,
):
    """Construct one of the real SensorQA metadata dataclasses using only
    constructor fields that actually exist.
    
    This keeps the test tolerant to optional metadata fields while
    preserving all required production fields.
    
    Args:
        target_class: Value for `target_class`.
        candidates: Value for `candidates`.
    
    Returns:
        Result returned by the function.
    """

    signature = inspect.signature(
        target_class
    )

    parameters = (
        signature.parameters
    )

    kwargs = {
        name:
            value
        for name, value
        in candidates.items()
        if name in parameters
    }

    return target_class(
        **kwargs
    )


def make_sensor_range(
    minimum: float,
    maximum: float,
    unit: str,
):
    """Construct the real SensorRange object.
    
    Args:
        minimum: Value for `minimum`.
        maximum: Value for `maximum`.
        unit: Engineering unit.
    
    Returns:
        Result returned by the function.
    """

    candidates = {
        "minimum":
            minimum,

        "maximum":
            maximum,

        "min":
            minimum,

        "max":
            maximum,

        "min_value":
            minimum,

        "max_value":
            maximum,

        "lower":
            minimum,

        "upper":
            maximum,

        "unit":
            unit,
    }

    return construct_dataclass_like(
        SensorRange,
        candidates,
    )


def make_sensor_information(
) -> SensorInformation:
    """Construct IMU sensor metadata.
    
    IMPORTANT:
        sensor_type is SensorType.IMU.
    
    It is intentionally NOT the string "imu".
    
    AnalysisPipeline.run() explicitly requires a SensorType enum.
    
    Returns:
        SensorInformation returned by the function.
    """

    gravity = 9.80665

    accelerometer_range = (
        make_sensor_range(
            minimum=(
                -2.0
                * gravity
            ),
            maximum=(
                2.0
                * gravity
            ),
            unit="m/s^2",
        )
    )

    gyroscope_range = (
        make_sensor_range(
            minimum=-4.0,
            maximum=4.0,
            unit="rad/s",
        )
    )

    candidates = {
        # Critical integration field

        "sensor_type":
            SensorType.IMU,

        # Descriptive metadata

        "name":
            "Synthetic Integration IMU",

        "sensor_name":
            "Synthetic Integration IMU",

        "manufacturer":
            "SensorQA Test Fixture",

        "model":
            "Synthetic-IMU",

        # Sampling rate aliases

        "nominal_sampling_rate_hz":
            100.0,

        "nominal_sample_rate_hz":
            100.0,

        "nominal_rate_hz":
            100.0,

        "sampling_rate_hz":
            100.0,

        # Accelerometer range aliases

        "accelerometer_range":
            accelerometer_range,

        "accel_range":
            accelerometer_range,

        # Gyroscope range aliases

        "gyroscope_range":
            gyroscope_range,

        "gyro_range":
            gyroscope_range,
    }

    return construct_dataclass_like(
        SensorInformation,
        candidates,
    )


def make_test_information(
) -> SensorQATestInformation:
    """Stationary IMU test metadata.
    
    Returns:
        SensorQATestInformation returned by the function.
    """

    candidates = {
        "test_mode":
            SensorQATestMode.IMU_STATIONARY,

        "expected_stationary":
            True,

        "temperature_controlled":
            False,

        "description":
            (
                "Synthetic stationary IMU integration "
                "test dataset."
            ),
    }

    return construct_dataclass_like(
        SensorQATestInformation,
        candidates,
    )


def make_dataset_source_information(
):
    """Construct source metadata when supported.
    
    Returns:
        Result returned by the function.
    """

    candidates = {
        "file_name":
            None,

        "name":
            "stationary_imu.csv",

        "source_name":
            "stationary_imu.csv",

        "description":
            "Synthetic SensorQA integration dataset.",

        "source_type":
            "synthetic",

        "format":
            "csv",

        "file_format":
            "csv",
    }

    return construct_dataclass_like(
        DatasetSourceInformation,
        candidates,
    )


def make_metadata(
) -> SensorQAMetadata:
    """Build the real SensorQAMetadata object.
    
    Returns:
        SensorQAMetadata returned by the function.
    """

    sensor_information = (
        make_sensor_information()
    )

    test_information = (
        make_test_information()
    )

    dataset_source = (
        make_dataset_source_information()
    )

    candidates = {
        "sensor":
            sensor_information,

        "sensor_information":
            sensor_information,

        "test":
            test_information,

        "test_information":
            test_information,

        "dataset_source":
            dataset_source,

        "source":
            dataset_source,

        "source_information":
            dataset_source,
    }

    return construct_dataclass_like(
        SensorQAMetadata,
        candidates,
    )


# Real ToolLoader -> ToolRegistry construction


def build_real_registry(
) -> ToolRegistry:
    """Discover and register actual SensorQA tools using the real
    normal tool-loading path.
    
    No fake registry or adapter is used.
    
    Returns:
        ToolRegistry returned by the function.
    """

    loader = ToolLoader(
        generic_tools_dir=(
            GENERIC_TOOLS_DIR
        ),
        imu_tools_dir=(
            IMU_TOOLS_DIR
        ),
        custom_tools_dir=(
            CUSTOM_TOOLS_DIR
        ),
    )

    load_report = (
        loader.discover_tools()
    )

    validator = ToolValidator()

    registry = ToolRegistry(
        validator=validator
    )

    registry.rebuild(
        load_report
    )

    tool_ids = set(
        registry.tool_ids()
    )

    if not tool_ids:

        pytest.fail(
            "Real ToolRegistry contains no tools after "
            "ToolLoader discovery."
        )

    required_tools = {
        "imu_sampling_integrity",
        "imu_gyro_bias",
        "imu_stationary_noise",
        "imu_psd",
        "imu_saturation",
        "imu_axis_correlation",
        "imu_gravity_error",
    }

    missing = (
        required_tools
        - tool_ids
    )

    if missing:

        pytest.fail(
            "Real ToolRegistry is missing required built-in "
            "IMU tools: "
            f"{sorted(missing)}\n"
            f"Registered tools: {sorted(tool_ids)}\n"
            f"Registry warnings: {registry.warnings()}\n"
            f"Registry load errors: {registry.load_errors()}"
        )

    return registry


# DatasetBuilder construction


def discover_csv_loader_class(
):
    """Find the concrete CSV loader class from the runtime module.
    
    Returns:
        Result returned by the function.
    """

    module = importlib.import_module(
        "sensorqa.ingestion.csv_loader"
    )

    candidates = []

    for name, candidate in (
        inspect.getmembers(
            module,
            inspect.isclass,
        )
    ):

        if (
            candidate.__module__
            != module.__name__
        ):
            continue

        normalized_name = (
            name
            .replace(
                "_",
                ""
            )
            .lower()
        )

        if (
            "csv" in normalized_name
            and "loader" in normalized_name
            and "result" not in normalized_name
            and "metadata" not in normalized_name
        ):

            candidates.append(
                candidate
            )

    if not candidates:

        pytest.fail(
            "Could not discover the concrete CSV loader "
            "class in sensorqa.ingestion.csv_loader."
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.__name__
            .replace(
                "_",
                ""
            )
            .lower()
            != "csvloader",
            candidate.__name__,
        )
    )

    return candidates[
        0
    ]


def instantiate_no_required_arguments(
    target_class,
):
    """Instantiate a class that has no required constructor arguments.
    
    Args:
        target_class: Value for `target_class`.
    
    Returns:
        Result returned by the function.
    """

    signature = inspect.signature(
        target_class
    )

    required = []

    for name, parameter in (
        signature.parameters.items()
    ):

        if name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
        ):

            required.append(
                name
            )

    if required:

        pytest.fail(
            f"{target_class.__name__} unexpectedly requires "
            f"constructor arguments: {required}\n"
            f"Signature: {signature}"
        )

    return target_class()


def construct_component(
    target_class,
    dependencies: dict,
):
    """Construct a backend component using its runtime constructor.
    
    Args:
        target_class: Value for `target_class`.
        dependencies: Value for `dependencies`.
    
    Returns:
        Result returned by the function.
    """

    signature = inspect.signature(
        target_class
    )

    kwargs = {}

    missing = []

    for name, parameter in (
        signature.parameters.items()
    ):

        if name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in dependencies:

            kwargs[
                name
            ] = dependencies[
                name
            ]

            continue

        if (
            parameter.default
            is not inspect.Parameter.empty
        ):
            continue

        missing.append(
            name
        )

    if missing:

        pytest.fail(
            f"Cannot construct {target_class.__name__}.\n"
            f"Constructor: {signature}\n"
            f"Missing dependencies: {missing}\n"
            f"Available dependencies: "
            f"{sorted(dependencies)}"
        )

    return target_class(
        **kwargs
    )


def build_dataset_builder(
) -> DatasetBuilder:
    """Build the real DatasetBuilder and its real collaborators.
    
    Returns:
        DatasetBuilder returned by the function.
    """

    csv_loader_class = (
        discover_csv_loader_class()
    )

    csv_loader = (
        instantiate_no_required_arguments(
            csv_loader_class
        )
    )

    column_mapper = (
        ColumnMapper()
    )

    unit_manager = (
        UnitManager()
    )

    dataset_validator = (
        DatasetValidator()
    )

    dependencies = {
        "csv_loader":
            csv_loader,

        "loader":
            csv_loader,

        "column_mapper":
            column_mapper,

        "mapper":
            column_mapper,

        "unit_manager":
            unit_manager,

        "units":
            unit_manager,

        "dataset_validator":
            dataset_validator,

        "validator":
            dataset_validator,
    }

    return construct_component(
        DatasetBuilder,
        dependencies,
    )


# Application construction


def build_application(
) -> SensorQAApplication:
    """Construct SensorQA using only real backend components.
    
    Returns:
        SensorQAApplication returned by the function.
    """

    registry = (
        build_real_registry()
    )

    dataset_builder = (
        build_dataset_builder()
    )

    pipeline = AnalysisPipeline(
        registry=registry
    )

    return SensorQAApplication(
        registry=registry,
        dataset_builder=dataset_builder,
        pipeline=pipeline,
    )


# Initialization


def initialize_application(
    application: SensorQAApplication,
):
    """Initialize using the exact runtime API.
    
    Args:
        application: Initialized SensorQA application.
    
    Returns:
        Result returned by the function.
    """

    return application.initialize(
        runtime_config_path=CONFIG_PATH,
        requirements_path=REQUIREMENTS_PATH,
        auto_include_dependencies=True,
        raise_on_error=False,
    )


# CSV build arguments


def integration_column_mapping(
) -> ColumnMapping:
    """Explicit identity mapping for the synthetic CSV.
    
    Returns:
        ColumnMapping returned by the function.
    """

    return ColumnMapping(
        mapping={
            "timestamp":
                "timestamp",

            "ax":
                "ax",

            "ay":
                "ay",

            "az":
                "az",

            "gx":
                "gx",

            "gy":
                "gy",

            "gz":
                "gz",

            "temperature":
                "temperature",

            "orientation_label":
                "orientation_label",

            "motion_state":
                "motion_state",
        }
    )


def integration_unit_assignments(
) -> dict[str, str]:
    """Physical units for mapped fields.
    
    Returns:
        Dictionary containing the result values.
    """

    return {
        "timestamp":
            "s",

        "ax":
            "m/s^2",

        "ay":
            "m/s^2",

        "az":
            "m/s^2",

        "gx":
            "rad/s",

        "gy":
            "rad/s",

        "gz":
            "rad/s",

        "temperature":
            "degC",
    }


# Analysis


def analyze_csv(
    application: SensorQAApplication,
    csv_path: Path,
):
    """Analyze CSV using the exact production application API.
    
    Actual API:
    
        analyze_csv(
            source,
            *,
            build_kwargs=None,
            qualify=True,
            raise_on_build_error=False,
        )
    
    Args:
        application: Initialized SensorQA application.
        csv_path: Path used for csv.
    
    Returns:
        Result returned by the function.
    """

    build_kwargs = {
        "column_mapping":
            integration_column_mapping(),

        "unit_assignments":
            integration_unit_assignments(),

        "metadata":
            make_metadata(),
    }

    return application.analyze_csv(
        source=csv_path,
        build_kwargs=build_kwargs,
        qualify=True,
        raise_on_build_error=False,
    )


# Fixtures


@pytest.fixture
def application():

    """Run application.
    
    Returns:
        Calculated value.
    """
    return build_application()


@pytest.fixture
def initialized_application(
    application,
):

    """Run initialized application.
    
    Args:
        application: Initialized SensorQA application.
    
    Returns:
        Calculated value.
    """
    initialization = (
        initialize_application(
            application
        )
    )

    return (
        application,
        initialization,
    )


@pytest.fixture
def application_analysis(
    initialized_application,
    imu_csv,
):

    """Run application analysis.
    
    Args:
        initialized_application: Fixture containing an initialized SensorQA application.
        imu_csv: Imu csv used by this function.
    
    Returns:
        Calculated value.
    """
    application, initialization = (
        initialized_application
    )

    analysis = analyze_csv(
        application,
        imu_csv,
    )

    return {
        "application":
            application,

        "initialization":
            initialization,

        "analysis":
            analysis,
    }


# Result helpers


def pipeline_result_from_analysis(
    analysis,
):

    """Run pipeline result from analysis.
    
    Args:
        analysis: Analysis used by this function.
    
    Returns:
        Calculated value.
    """
    pipeline_result = (
        analysis.pipeline_result
    )

    if pipeline_result is None:

        pytest.fail(
            "Application analysis did not produce a pipeline result.\n"
            f"Workflow success: {analysis.success}\n"
            f"Workflow warnings: {analysis.warnings}\n"
            f"Workflow messages: {analysis.messages}"
        )

    return pipeline_result


def pipeline_results(
    pipeline_result,
):

    """Run pipeline results.
    
    Args:
        pipeline_result: Pipeline result used by this function.
    
    Returns:
        Calculated value.
    """
    return list(
        pipeline_result.results
    )


def pipeline_diagnostics(
    pipeline_result,
) -> str:

    """Run pipeline diagnostics.
    
    Args:
        pipeline_result: Pipeline result used by this function.
    
    Returns:
        Requested text value.
    """
    return (
        f"success={pipeline_result.success}\n"
        f"errors={pipeline_result.errors}\n"
        f"warnings={pipeline_result.warnings}\n"
        f"execution_order="
        f"{pipeline_result.execution_order}\n"
        f"result_count="
        f"{len(pipeline_result.results)}"
    )


# Construction tests


def test_application_constructs(
    application,
):

    """Check that application constructs.
    
    Args:
        application: Initialized SensorQA application.
    
    Returns:
        None.
    """
    assert isinstance(
        application,
        SensorQAApplication,
    )


def test_application_uses_real_tool_registry(
    application,
):

    """Check that application uses real tool registry.
    
    Args:
        application: Initialized SensorQA application.
    
    Returns:
        None.
    """
    assert type(
        application.registry
    ) is ToolRegistry


def test_registry_contains_real_imu_tools(
    application,
):

    """Check that registry contains real imu tools.
    
    Args:
        application: Initialized SensorQA application.
    
    Returns:
        None.
    """
    registered = set(
        application.registry.tool_ids()
    )

    required = {
        "imu_sampling_integrity",
        "imu_gyro_bias",
        "imu_stationary_noise",
        "imu_psd",
        "imu_saturation",
        "imu_axis_correlation",
        "imu_gravity_error",
    }

    missing = (
        required
        - registered
    )

    assert not missing, (
        f"Missing real IMU tools: {sorted(missing)}\n"
        f"Registered: {sorted(registered)}"
    )


# Initialization tests


def test_application_initialization_returns_result(
    initialized_application,
):

    """Check that application initialization returns result.
    
    Args:
        initialized_application: Fixture containing an initialized SensorQA application.
    
    Returns:
        None.
    """
    _, initialization = (
        initialized_application
    )

    assert initialization is not None


def test_application_initialization_succeeds(
    initialized_application,
):

    """Check that application initialization succeeds.
    
    Args:
        initialized_application: Fixture containing an initialized SensorQA application.
    
    Returns:
        None.
    """
    _, initialization = (
        initialized_application
    )

    assert initialization.success, (
        "Application initialization failed.\n"
        f"Errors: "
        f"{getattr(initialization, 'errors', [])}\n"
        f"Warnings: "
        f"{getattr(initialization, 'warnings', [])}\n"
        f"Messages: "
        f"{getattr(initialization, 'messages', [])}"
    )


# Dataset tests


def test_application_analyzes_real_csv(
    application_analysis,
):

    """Check that application analyzes real csv.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    assert (
        application_analysis[
            "analysis"
        ]
        is not None
    )


def test_application_produces_dataset(
    application_analysis,
):

    """Check that application produces dataset.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    assert analysis.dataset is not None, (
        f"Workflow warnings: {analysis.warnings}\n"
        f"Workflow messages: {analysis.messages}"
    )


def test_dataset_sensor_type_is_real_enum(
    application_analysis,
):
    """This specifically guards against the bug where metadata used the
    plain string 'imu'.
    
    AnalysisPipeline requires SensorType.IMU.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """

    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    assert analysis.dataset is not None

    assert isinstance(
        analysis.dataset.sensor_type,
        SensorType,
    ), (
        "dataset.sensor_type is not a SensorType enum.\n"
        f"value={analysis.dataset.sensor_type!r}\n"
        f"type={type(analysis.dataset.sensor_type)}"
    )

    assert (
        analysis.dataset.sensor_type
        == SensorType.IMU
    )


def test_dataset_has_expected_rows(
    application_analysis,
):

    """Check that dataset has expected rows.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    dataset = (
        application_analysis[
            "analysis"
        ].dataset
    )

    assert dataset is not None

    assert (
        dataset.row_count
        == 6000
    )


# Pipeline tests


def test_pipeline_result_is_present(
    application_analysis,
):

    """Check that pipeline result is present.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    assert (
        analysis.pipeline_result
        is not None
    )


def test_pipeline_has_no_pipeline_level_errors(
    application_analysis,
):

    """Check that pipeline has no pipeline level errors.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    pipeline = (
        pipeline_result_from_analysis(
            application_analysis[
                "analysis"
            ]
        )
    )

    assert not pipeline.errors, (
        "Pipeline returned before or during tool execution.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )


def test_pipeline_executes_tools(
    application_analysis,
):

    """Check that pipeline executes tools.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    pipeline = (
        pipeline_result_from_analysis(
            application_analysis[
                "analysis"
            ]
        )
    )

    results = (
        pipeline_results(
            pipeline
        )
    )

    assert results, (
        "Pipeline produced zero analysis results.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )


def test_pipeline_executes_multiple_real_tools(
    application_analysis,
):

    """Check that pipeline executes multiple real tools.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    pipeline = (
        pipeline_result_from_analysis(
            application_analysis[
                "analysis"
            ]
        )
    )

    results = (
        pipeline_results(
            pipeline
        )
    )

    assert len(
        results
    ) >= 3, (
        "Expected multiple real analysis results.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )


def test_sampling_integrity_runs_through_application(
    application_analysis,
):

    """Check that sampling integrity runs through application.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    pipeline = (
        pipeline_result_from_analysis(
            application_analysis[
                "analysis"
            ]
        )
    )

    by_tool = {
        result.tool_id:
            result
        for result
        in pipeline.results
    }

    assert (
        "imu_sampling_integrity"
        in by_tool
    ), (
        "imu_sampling_integrity did not execute.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )

    result = (
        by_tool[
            "imu_sampling_integrity"
        ]
    )

    assert (
        result.status
        == ExecutionStatus.SUCCESS
    ), (
        f"Sampling integrity status: {result.status}\n"
        f"Messages: {result.messages}\n"
        f"Warnings: {result.warnings}"
    )


def test_gyro_bias_runs_through_application(
    application_analysis,
):

    """Check that gyro bias runs through application.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    pipeline = (
        pipeline_result_from_analysis(
            application_analysis[
                "analysis"
            ]
        )
    )

    by_tool = {
        result.tool_id:
            result
        for result
        in pipeline.results
    }

    assert (
        "imu_gyro_bias"
        in by_tool
    ), (
        "imu_gyro_bias did not execute.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )

    result = (
        by_tool[
            "imu_gyro_bias"
        ]
    )

    assert (
        result.status
        == ExecutionStatus.SUCCESS
    ), (
        f"Gyro bias status: {result.status}\n"
        f"Messages: {result.messages}\n"
        f"Warnings: {result.warnings}"
    )


def test_pipeline_emits_real_gyro_bias_metrics(
    application_analysis,
):

    """Check that pipeline emits real gyro bias metrics.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    pipeline = (
        pipeline_result_from_analysis(
            application_analysis[
                "analysis"
            ]
        )
    )

    by_tool = {
        result.tool_id:
            result
        for result
        in pipeline.results
    }

    assert (
        "imu_gyro_bias"
        in by_tool
    )

    gyro_result = (
        by_tool[
            "imu_gyro_bias"
        ]
    )

    metric_names = {
        metric.name
        for metric
        in gyro_result.metrics
    }

    assert (
        "GX Bias"
        in metric_names
    )

    assert (
        "GY Bias"
        in metric_names
    )

    assert (
        "GZ Bias"
        in metric_names
    )

    assert (
        "Gyroscope Bias Vector Magnitude"
        in metric_names
    )


# Qualification tests


def test_application_exposes_qualification_result(
    application_analysis,
):

    """Check that application exposes qualification result.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    assert (
        analysis.qualification
        is not None
    )


def test_qualification_is_separate_from_workflow_success(
    application_analysis,
):
    """PASS/FAIL/NOT_EVALUATED is an engineering qualification result.
    
    It is separate from whether the software workflow successfully
    executed.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """

    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    pipeline = (
        pipeline_result_from_analysis(
            analysis
        )
    )

    assert pipeline.success, (
        "The analysis pipeline itself failed.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )

    assert analysis.success is True, (
        f"Workflow success is false.\n"
        f"Pipeline:\n"
        f"{pipeline_diagnostics(pipeline)}\n"
        f"Workflow warnings: {analysis.warnings}\n"
        f"Workflow messages: {analysis.messages}"
    )

    assert (
        analysis.qualification
        is not None
    )


def test_enabled_requirement_metrics_are_evaluable_when_tool_ran(
    application_analysis,
):
    """If a tool successfully ran, its enabled requirement must not become
    NOT_EVALUATED merely because the configured metric name does not
    match the real metric name.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """

    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    qualification = (
        analysis.qualification
    )

    assert qualification is not None

    pipeline = (
        pipeline_result_from_analysis(
            analysis
        )
    )

    successful_tools = {
        result.tool_id
        for result
        in pipeline.results
        if (
            result.status
            == ExecutionStatus.SUCCESS
        )
    }

    checks = getattr(
        qualification,
        "checks",
        [],
    )

    suspicious = []

    for check in checks:

        tool_id = getattr(
            check,
            "tool_id",
            None,
        )

        if (
            tool_id
            not in successful_tools
        ):
            continue

        status_object = getattr(
            check,
            "status",
            None,
        )

        status_text = (
            str(
                getattr(
                    status_object,
                    "value",
                    status_object,
                )
            )
            .lower()
        )

        if (
            "not_evaluated"
            not in status_text
            and "not evaluated"
            not in status_text
        ):
            continue

        reason = str(
            getattr(
                check,
                "reason",
                getattr(
                    check,
                    "message",
                    "",
                ),
            )
        ).lower()

        metric_name = getattr(
            check,
            "metric_name",
            None,
        )

        metric_missing = (
            "metric" in reason
            and (
                "missing" in reason
                or "not found" in reason
                or "not available" in reason
            )
        )

        if metric_missing:

            suspicious.append(
                (
                    tool_id,
                    metric_name,
                    reason,
                )
            )

    assert not suspicious, (
        "Requirement metric-name mismatch detected "
        "for successfully executed tools:\n"
        + "\n".join(
            (
                f"{tool_id} / "
                f"{metric_name}: "
                f"{reason}"
            )
            for (
                tool_id,
                metric_name,
                reason,
            )
            in suspicious
        )
    )


# Full workflow


def test_application_csv_analysis_workflow_succeeds(
    application_analysis,
):

    """Check that application csv analysis workflow succeeds.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """
    analysis = (
        application_analysis[
            "analysis"
        ]
    )

    pipeline = (
        pipeline_result_from_analysis(
            analysis
        )
    )

    assert pipeline.success, (
        "Pipeline failed.\n"
        + pipeline_diagnostics(
            pipeline
        )
    )

    assert analysis.success, (
        "End-to-end SensorQA workflow failed.\n"
        f"Pipeline:\n"
        f"{pipeline_diagnostics(pipeline)}\n"
        f"Workflow warnings: {analysis.warnings}\n"
        f"Workflow messages: {analysis.messages}"
    )

def test_incompatible_globally_enabled_tool_is_filtered_for_imu_dataset(
    application_analysis,
):
    """Runtime configuration may enable tools for multiple sensor families.
    
    The application must pass only tools compatible with the current
    dataset to AnalysisPipeline. The lower-level pipeline should remain
    strict when an incompatible tool is requested explicitly.
    
    Args:
        application_analysis: Fixture containing a completed application analysis.
    
    Returns:
        None.
    """

    application = application_analysis[
        "application"
    ]

    analysis = application_analysis[
        "analysis"
    ]

    assert (
        "generic_accuracy"
        in application.enabled_tools()
    )

    executed_tool_ids = {
        result.tool_id
        for result
        in analysis.pipeline_result.results
    }

    assert (
        "generic_accuracy"
        not in executed_tool_ids
    )

    assert any(
        (
            "generic_accuracy" in warning
            and "incompatible" in warning.lower()
        )
        for warning
        in analysis.warnings
    )
