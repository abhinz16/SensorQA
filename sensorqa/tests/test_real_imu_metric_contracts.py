#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from sensorqa.core.requirements_loader import (
    RequirementsLoader,
)
from sensorqa.core.result_schema import (
    AnalysisResult,
    ExecutionStatus,
)
from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
    SensorType,
    ToolContext,
)
from sensorqa.ingestion.metadata import (
    TestMode as SensorQATestMode,
)


class AttrDict(dict):
    """
    Dictionary that also supports attribute-style access.

    This is useful for synthetic integration metadata because
    SensorQA components may access metadata through either:

        metadata.get("field")

    or:

        metadata.field
    """

    def __getattr__(
        self,
        name,
    ):

        """Return getattr.
        
        Args:
            name: Name of the item.
        
        Returns:
            Calculated value.
        """
        try:
            return self[
                name
            ]

        except KeyError as exc:

            raise AttributeError(
                name
            ) from exc

# Project discovery


def find_project_root() -> Path:
    """Locate SensorQA repository root independently of pytest's
    current working directory.
    
    Returns:
        Resolved path.
    """

    test_file = Path(
        __file__
    ).resolve()

    for candidate in (
        test_file.parent,
        *test_file.parents,
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

SENSORQA_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "sensorqa.toml"
)

REQUIREMENTS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "requirements.toml"
)


# Real tool discovery


def load_module_from_path(
    path: Path,
):
    """Import one real SensorQA tool module.
    
    Args:
        path: Path to the file or directory.
    
    Returns:
        Result returned by the function.
    """

    module_name = (
        "_sensorqa_metric_contract_"
        + path.parent.name
        + "_"
        + path.stem
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Could not import {path}."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def discover_builtin_tools(
) -> dict[
    str,
    BaseAnalysisTool,
]:
    """Discover actual concrete tools from tools/imu and tools/generic.
    
    Returns:
        Dictionary containing the result values.
    """

    output = {}

    for category in (
        "generic",
        "imu",
    ):

        directory = (
            TOOLS_ROOT
            / category
        )

        if not directory.is_dir():
            continue

        for path in sorted(
            directory.glob(
                "*.py"
            )
        ):

            if path.name.startswith(
                "_"
            ):
                continue

            module = load_module_from_path(
                path
            )

            for _, candidate in inspect.getmembers(
                module,
                inspect.isclass,
            ):

                if candidate is BaseAnalysisTool:
                    continue

                if not issubclass(
                    candidate,
                    BaseAnalysisTool,
                ):
                    continue

                if inspect.isabstract(
                    candidate
                ):
                    continue

                if (
                    candidate.__module__
                    != module.__name__
                ):
                    continue

                instance = candidate()

                tool_id = (
                    instance.metadata.tool_id
                )

                if tool_id in output:

                    raise AssertionError(
                        f"Duplicate tool ID "
                        f"'{tool_id}'."
                    )

                output[
                    tool_id
                ] = instance

    return output


# Real configured parameters


def configured_tool_parameters(
) -> dict[
    str,
    dict,
]:
    """Read the parameter overrides from the shipped sensorqa.toml.
    
    Defaults from ToolMetadata are added later by
    parameters_for_tool().
    
    Returns:
        Dictionary containing the result values.
    """

    with SENSORQA_CONFIG_PATH.open(
        "rb"
    ) as file:

        raw = tomllib.load(
            file
        )

    tools = raw.get(
        "tools",
        {}
    )

    output = {}

    for tool_id, configuration in (
        tools.items()
    ):

        output[
            tool_id
        ] = {
            key:
                value
            for key, value
            in configuration.items()
            if key != "enabled"
        }

    return output


def parameters_for_tool(
    tool: BaseAnalysisTool,
    configured: dict[
        str,
        dict,
    ],
) -> dict:
    """Combine real ToolMetadata defaults with sensorqa.toml values.
    
    Args:
        tool: Analysis tool instance.
        configured: Value for `configured`.
    
    Returns:
        Dictionary containing the result values.
    """

    parameters = {}

    for definition in (
        tool.metadata.parameters
    ):

        if definition.default is not None:

            parameters[
                definition.name
            ] = definition.default

    parameters.update(
        configured.get(
            tool.metadata.tool_id,
            {},
        )
    )

    return parameters


# Metadata helpers


def sensor_range(
    minimum: float,
    maximum: float,
    unit: str,
):
    """Synthetic range metadata supporting both mapping and
    attribute-style access.
    
    Args:
        minimum: Value for `minimum`.
        maximum: Value for `maximum`.
        unit: Engineering unit.
    
    Returns:
        Result returned by the function.
    """

    full_scale = max(
        abs(
            minimum
        ),
        abs(
            maximum
        ),
    )

    return AttrDict(
        minimum=minimum,
        maximum=maximum,

        min=minimum,
        max=maximum,

        lower=minimum,
        upper=maximum,

        min_value=minimum,
        max_value=maximum,

        lower_limit=minimum,
        upper_limit=maximum,

        full_scale=full_scale,
        full_scale_value=full_scale,

        span=maximum - minimum,

        unit=unit,
    )


def synthetic_metadata(
    test_mode: SensorQATestMode,
):
    """Construct synthetic SensorQA metadata using an object that
    supports both dictionary-style and attribute-style access.
    
    Args:
        test_mode: Value for `test_mode`.
    
    Returns:
        Result returned by the function.
    """

    g = 9.80665

    accel_range = sensor_range(
        minimum=-2.0 * g,
        maximum=2.0 * g,
        unit="m/s^2",
    )

    gyro_range = sensor_range(
        minimum=-4.0,
        maximum=4.0,
        unit="rad/s",
    )

    sensor = AttrDict(
        # Sampling-rate aliases.
        nominal_sampling_rate_hz=100.0,
        nominal_sample_rate_hz=100.0,
        nominal_rate_hz=100.0,
        sampling_rate_hz=100.0,
        sample_rate_hz=100.0,

        # Accelerometer range aliases.
        accelerometer_range=accel_range,
        accel_range=accel_range,

        accelerometer_full_scale=(
            2.0 * g
        ),

        accel_full_scale=(
            2.0 * g
        ),

        # Gyroscope range aliases.
        gyroscope_range=gyro_range,
        gyro_range=gyro_range,

        gyroscope_full_scale=4.0,
        gyro_full_scale=4.0,
    )

    test = AttrDict(
        test_mode=test_mode,
        expected_stationary=True,
        temperature_controlled=False,
    )

    source = AttrDict(
        name="synthetic_contract_test",
    )

    return AttrDict(
        # Normal SensorQA nested structures.
        sensor=sensor,
        sensor_information=sensor,

        test=test,
        test_information=test,

        source=source,
        source_information=source,

        # Also expose the important acquisition properties at the
        # metadata root for tools that read them directly.
        nominal_sampling_rate_hz=100.0,
        nominal_sample_rate_hz=100.0,
        nominal_rate_hz=100.0,

        accelerometer_range=accel_range,
        accel_range=accel_range,

        gyroscope_range=gyro_range,
        gyro_range=gyro_range,

        accelerometer_full_scale=(
            2.0 * g
        ),

        accel_full_scale=(
            2.0 * g
        ),

        gyroscope_full_scale=4.0,
        gyro_full_scale=4.0,
    )


def tool_runtime_metadata(
    *,
    test_mode: SensorQATestMode,
    dataframe: pd.DataFrame,
) -> dict:
    """Reproduce the metadata dictionary supplied by AnalysisPipeline
    to ToolContext.
    
    Individual analysis tools access SensorQA dataset metadata
    through:
    
        context.metadata["sensorqa_metadata"]
    
    rather than receiving the metadata object directly.
    
    Args:
        test_mode: Value for `test_mode`.
        dataframe: Value for `dataframe`.
    
    Returns:
        Dictionary containing the result values.
    """

    sensorqa_metadata = synthetic_metadata(
        test_mode
    )

    timestamp = (
        dataframe[
            "timestamp"
        ].to_numpy(
            dtype=float
        )
    )

    if len(
        timestamp
    ) >= 2:

        dt = np.diff(
            timestamp
        )

        positive_dt = dt[
            np.isfinite(
                dt
            )
            & (
                dt > 0.0
            )
        ]

    else:

        positive_dt = np.array(
            [],
            dtype=float,
        )

    if positive_dt.size > 0:

        median_dt = float(
            np.median(
                positive_dt
            )
        )

        sampling_rate_hz = (
            1.0
            / median_dt
        )

    else:

        median_dt = None
        sampling_rate_hz = None

    if len(
        timestamp
    ) >= 2:

        duration_seconds = float(
            timestamp[
                -1
            ]
            - timestamp[
                0
            ]
        )

    else:

        duration_seconds = 0.0

    return {
        "sensorqa_metadata":
            sensorqa_metadata,

        "dataset_summary": {
            "row_count":
                int(
                    len(
                        dataframe
                    )
                ),

            "column_count":
                int(
                    len(
                        dataframe.columns
                    )
                ),

            "columns":
                list(
                    dataframe.columns
                ),

            "duration_seconds":
                duration_seconds,
        },

        "completed_tool_ids":
            [],

        "sampling": {
            "sampling_rate_hz":
                sampling_rate_hz,

            "median_interval_seconds":
                median_dt,

            "duration_seconds":
                duration_seconds,
        },
    }


# Unit and mapping helpers


def identity_mapping(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """Standardized pipeline-style identity mapping.
    
    At this stage the synthetic dataframe already uses SensorQA's
    canonical field names, so no physical-to-standard column
    translation is required.
    
    Args:
        dataframe: Value for `dataframe`.
    
    Returns:
        Dictionary containing the result values.
    """

    return {
        column: column
        for column
        in dataframe.columns
    }


def canonical_units(
) -> dict[str, str]:
    """Canonical units for the synthetic standardized dataframe.
    
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


# Synthetic stationary IMU data


def stationary_imu_dataframe(
) -> pd.DataFrame:
    """Long, regularly sampled stationary IMU dataset.
    
    The signal contains:
    
        - realistic fixed biases
        - white noise for Allan analysis
        - a small shared sinusoidal component
        - temperature variation
        - no clipping
    
    The random seed makes the test reproducible.
    
    Returns:
        DataFrame containing the requested data.
    """

    rng = np.random.default_rng(
        123456
    )

    fs = 100.0

    # Five minutes gives Allan deviation enough observation time
    # while remaining small enough for routine tests.
    duration_seconds = 300.0

    sample_count = int(
        fs
        * duration_seconds
    )

    timestamp = (
        np.arange(
            sample_count,
            dtype=float,
        )
        / fs
    )

    shared_component = (
        0.00025
        * np.sin(
            2.0
            * np.pi
            * 8.0
            * timestamp
        )
    )

    temperature = (
        20.0
        + 10.0
        * timestamp
        / timestamp[
            -1
        ]
    )

    # Small temperature-associated behavior is deliberately added
    # so the temperature tool has a non-degenerate regression.
    temperature_centered = (
        temperature
        - np.mean(
            temperature
        )
    )

    gx = (
        0.0010
        + 0.00005
        * temperature_centered
        + shared_component
        + rng.normal(
            0.0,
            0.0007,
            sample_count,
        )
    )

    gy = (
        -0.0015
        + 0.00004
        * temperature_centered
        + 0.9
        * shared_component
        + rng.normal(
            0.0,
            0.0007,
            sample_count,
        )
    )

    gz = (
        0.0008
        + rng.normal(
            0.0,
            0.0007,
            sample_count,
        )
    )

    g = 9.80665

    ax = (
        0.015
        + rng.normal(
            0.0,
            0.020,
            sample_count,
        )
    )

    ay = (
        -0.010
        + rng.normal(
            0.0,
            0.020,
            sample_count,
        )
    )

    az = (
        g
        + 0.025
        + rng.normal(
            0.0,
            0.020,
            sample_count,
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
                    sample_count,
                    "+Z",
                    dtype=object,
                ),

            "motion_state":
                np.full(
                    sample_count,
                    "stationary",
                    dtype=object,
                ),
        }
    )


# Synthetic six-position data


def six_position_dataframe(
) -> pd.DataFrame:
    """Synthetic six-position accelerometer calibration dataset.
    
    Includes known diagonal sensitivity error and fixed bias so the
    calibration tool has meaningful work to perform.
    
    Returns:
        DataFrame containing the requested data.
    """

    rng = np.random.default_rng(
        7890
    )

    fs = 100.0

    samples_per_orientation = 300

    g = 9.80665

    # Deliberately imperfect sensor.
    bias = np.array(
        [
            0.08,
            -0.05,
            0.04,
        ]
    )

    sensitivity = np.array(
        [
            1.010,
            0.992,
            1.006,
        ]
    )

    orientations = {
        "+X":
            np.array(
                [
                    g,
                    0.0,
                    0.0,
                ]
            ),

        "-X":
            np.array(
                [
                    -g,
                    0.0,
                    0.0,
                ]
            ),

        "+Y":
            np.array(
                [
                    0.0,
                    g,
                    0.0,
                ]
            ),

        "-Y":
            np.array(
                [
                    0.0,
                    -g,
                    0.0,
                ]
            ),

        "+Z":
            np.array(
                [
                    0.0,
                    0.0,
                    g,
                ]
            ),

        "-Z":
            np.array(
                [
                    0.0,
                    0.0,
                    -g,
                ]
            ),
    }

    rows = []

    sample_index = 0

    for label, truth in (
        orientations.items()
    ):

        measured_mean = (
            sensitivity
            * truth
            + bias
        )

        for _ in range(
            samples_per_orientation
        ):

            accel = (
                measured_mean
                + rng.normal(
                    0.0,
                    0.015,
                    3,
                )
            )

            rows.append(
                {
                    "timestamp":
                        sample_index
                        / fs,

                    "ax":
                        accel[
                            0
                        ],

                    "ay":
                        accel[
                            1
                        ],

                    "az":
                        accel[
                            2
                        ],

                    "gx":
                        rng.normal(
                            0.0,
                            0.0005,
                        ),

                    "gy":
                        rng.normal(
                            0.0,
                            0.0005,
                        ),

                    "gz":
                        rng.normal(
                            0.0,
                            0.0005,
                        ),

                    "temperature":
                        25.0,

                    "orientation_label":
                        label,

                    "motion_state":
                        "stationary",
                }
            )

            sample_index += 1

    return pd.DataFrame(
        rows
    )


# Tool execution


def tool_context(
    tool: BaseAnalysisTool,
    dataframe: pd.DataFrame,
    configured_parameters: dict,
    *,
    test_mode: SensorQATestMode,
) -> ToolContext:
    """Construct ToolContext using the same metadata shape supplied by
    the real AnalysisPipeline.
    
    Args:
        tool: Analysis tool instance.
        dataframe: Value for `dataframe`.
        configured_parameters: Value for `configured_parameters`.
        test_mode: Value for `test_mode`.
    
    Returns:
        ToolContext returned by the function.
    """

    parameters = parameters_for_tool(
        tool=tool,
        configured=configured_parameters,
    )

    return ToolContext(
        sensor_type=SensorType.IMU,

        column_mapping=identity_mapping(
            dataframe
        ),

        units=canonical_units(),

        parameters=parameters,

        metadata=tool_runtime_metadata(
            test_mode=test_mode,
            dataframe=dataframe,
        ),
    )


def run_real_tool(
    tool: BaseAnalysisTool,
    dataframe: pd.DataFrame,
    configured_parameters: dict,
    *,
    test_mode: SensorQATestMode,
) -> AnalysisResult:
    """Execute the real tool's analysis implementation.
    
    run() is used rather than execute() here because this test is
    specifically validating emitted metric contracts after the
    standardized-data stage. BaseAnalysisTool input validation is
    tested separately.
    
    Args:
        tool: Analysis tool instance.
        dataframe: Value for `dataframe`.
        configured_parameters: Value for `configured_parameters`.
        test_mode: Value for `test_mode`.
    
    Returns:
        Analysis result containing metrics and messages.
    """

    context = tool_context(
        tool=tool,
        dataframe=dataframe,
        configured_parameters=(
            configured_parameters
        ),
        test_mode=test_mode,
    )

    try:

        result = tool.run(
            dataframe,
            context,
        )

    except Exception as exc:

        pytest.fail(
            f"Real tool "
            f"'{tool.metadata.tool_id}' "
            f"raised {exc.__class__.__name__}: "
            f"{exc}"
        )

    assert isinstance(
        result,
        AnalysisResult
    ), (
        f"Tool '{tool.metadata.tool_id}' returned "
        f"{type(result).__name__}, not AnalysisResult."
    )

    return result


def metric_names(
    result: AnalysisResult,
) -> set[str]:

    """Run metric names.
    
    Args:
        result: Result object to process.
    
    Returns:
        set[str] returned by this function.
    """
    return {
        metric.name
        for metric
        in result.metrics
    }


# Fixtures


@pytest.fixture(
    scope="module"
)
def tools():

    """Run tools.
    
    Returns:
        Calculated value.
    """
    discovered = (
        discover_builtin_tools()
    )

    assert discovered

    return discovered


@pytest.fixture(
    scope="module"
)
def configured_parameters():

    """Run configured parameters.
    
    Returns:
        Calculated value.
    """
    return configured_tool_parameters()


@pytest.fixture(
    scope="module"
)
def stationary_data():

    """Run stationary data.
    
    Returns:
        Calculated value.
    """
    return stationary_imu_dataframe()


@pytest.fixture(
    scope="module"
)
def six_position_data():

    """Run six position data.
    
    Returns:
        Calculated value.
    """
    return six_position_dataframe()


@pytest.fixture(
    scope="module"
)
def stationary_results(
    tools,
    configured_parameters,
    stationary_data,
):
    """Execute the real tools that can use the stationary dataset.
    
    Args:
        tools: Value for `tools`.
        configured_parameters: Value for `configured_parameters`.
        stationary_data: Value for `stationary_data`.
    
    Returns:
        Result returned by the function.
    """

    tool_ids = [
        "imu_sampling_integrity",
        "imu_gyro_bias",
        "imu_accelerometer_bias",
        "imu_stationary_noise",
        "imu_allan_deviation",
        "imu_psd",
        "imu_saturation",
        "imu_temperature_stability",
        "imu_axis_correlation",
        "imu_gravity_error",
    ]

    output = {}

    for tool_id in tool_ids:

        assert tool_id in tools

        output[
            tool_id
        ] = run_real_tool(
            tool=tools[
                tool_id
            ],
            dataframe=stationary_data,
            configured_parameters=(
                configured_parameters
            ),
            test_mode=(
                SensorQATestMode.IMU_STATIONARY
            ),
        )

    return output


@pytest.fixture(
    scope="module"
)
def six_position_result(
    tools,
    configured_parameters,
    six_position_data,
):

    """Run six position result.
    
    Args:
        tools: Tools used by this function.
        configured_parameters: Configured parameters used by this function.
        six_position_data: Six position data used by this function.
    
    Returns:
        Calculated value.
    """
    tool_id = (
        "imu_six_position_calibration"
    )

    assert tool_id in tools

    return run_real_tool(
        tool=tools[
            tool_id
        ],
        dataframe=six_position_data,
        configured_parameters=(
            configured_parameters
        ),
        test_mode=(
            SensorQATestMode.IMU_CONTROLLED_ORIENTATION
        ),
    )


# Synthetic-data sanity


def test_stationary_dataset_has_expected_length(
    stationary_data,
):

    """Check that stationary dataset has expected length.
    
    Args:
        stationary_data: Stationary data used by this function.
    
    Returns:
        None.
    """
    assert len(
        stationary_data
    ) == 30000


def test_stationary_sampling_is_exactly_regular(
    stationary_data,
):

    """Check that stationary sampling is exactly regular.
    
    Args:
        stationary_data: Stationary data used by this function.
    
    Returns:
        None.
    """
    dt = np.diff(
        stationary_data[
            "timestamp"
        ].to_numpy()
    )

    assert np.allclose(
        dt,
        0.01,
    )


def test_six_position_dataset_contains_all_orientations(
    six_position_data,
):

    """Check that six position dataset contains all orientations.
    
    Args:
        six_position_data: Six position data used by this function.
    
    Returns:
        None.
    """
    assert set(
        six_position_data[
            "orientation_label"
        ].unique()
    ) == {
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    }


# Real tool execution sanity


def test_stationary_tools_return_analysis_results(
    stationary_results,
):

    """Check that stationary tools return analysis results.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    assert len(
        stationary_results
    ) == 10

    for result in (
        stationary_results.values()
    ):

        assert isinstance(
            result,
            AnalysisResult
        )


def test_stationary_tools_do_not_return_error_status(
    stationary_results,
):
    """The controlled stationary fixture should not cause a real tool
    execution error.
    
    Args:
        stationary_results: Value for `stationary_results`.
    
    Returns:
        None.
    """

    failures = {
        tool_id:
            result
        for tool_id, result
        in stationary_results.items()
        if result.status
        == ExecutionStatus.ERROR
    }

    if failures:
    
        details = []
    
        for tool_id, result in (
            failures.items()
        ):
    
            details.append(
                f"\n{tool_id}"
            )
    
            details.append(
                f"status = {result.status}"
            )
    
            details.append(
                f"messages = {result.messages}"
            )
    
            details.append(
                f"warnings = {result.warnings}"
            )
    
            details.append(
                f"metadata = {result.metadata}"
            )
    
        pytest.fail(
            "One or more real IMU tools returned ERROR:\n"
            + "\n".join(
                details
            )
        )


def test_six_position_tool_does_not_error(
    six_position_result,
):

    """Check that six position tool does not error.
    
    Args:
        six_position_result: Six position result used by this function.
    
    Returns:
        None.
    """
    assert (
        six_position_result.status
        != ExecutionStatus.ERROR
    )


# Key metric contract checks


def test_sampling_integrity_emits_requirement_metrics(
    stationary_results,
):

    """Check that sampling integrity emits requirement metrics.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    result = stationary_results[
        "imu_sampling_integrity"
    ]

    names = metric_names(
        result
    )

    expected = {
        "Sampling Rate Error",
        "Estimated Capture Fraction",
        "Duplicate Timestamp Count",
        "Backward Timestamp Count",
    }

    assert expected.issubset(
        names
    ), (
        "Sampling-integrity metric contract mismatch.\n"
        f"Status: {result.status}\n"
        f"Messages: {result.messages}\n"
        f"Warnings: {result.warnings}\n"
        f"Missing: {sorted(expected - names)}\n"
        f"Actually emitted: {sorted(names)}"
    )


def test_gyro_bias_emits_requirement_metrics(
    stationary_results,
):

    """Check that gyro bias emits requirement metrics.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    result = stationary_results[
        "imu_gyro_bias"
    ]

    names = metric_names(
        result
    )

    expected = {
        "GX Bias",
        "GY Bias",
        "GZ Bias",
        "Gyroscope Bias Vector Magnitude",
    }

    assert expected.issubset(
        names
    ), (
        "Gyroscope-bias metric contract mismatch.\n"
        f"Status: {result.status}\n"
        f"Messages: {result.messages}\n"
        f"Warnings: {result.warnings}\n"
        f"Missing: {sorted(expected - names)}\n"
        f"Actually emitted: {sorted(names)}"
    )


def test_stationary_noise_emits_requirement_metrics(
    stationary_results,
):

    """Check that stationary noise emits requirement metrics.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    names = metric_names(
        stationary_results[
            "imu_stationary_noise"
        ]
    )

    expected = {
        "GX Noise Standard Deviation",
        "GY Noise Standard Deviation",
        "GZ Noise Standard Deviation",
        "AX Noise Standard Deviation",
        "AY Noise Standard Deviation",
        "AZ Noise Standard Deviation",
    }

    assert expected.issubset(
        names
    ), (
        "Stationary-noise metric contract mismatch. "
        f"Missing: {sorted(expected - names)}"
    )


def test_gravity_error_emits_requirement_metrics(
    stationary_results,
):

    """Check that gravity error emits requirement metrics.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    names = metric_names(
        stationary_results[
            "imu_gravity_error"
        ]
    )

    expected = {
        "Gravity Magnitude MAE",
        "Gravity Magnitude RMSE",
        "Maximum Absolute Gravity Magnitude Error",
    }

    assert expected.issubset(
        names
    ), (
        "Gravity-consistency metric contract mismatch. "
        f"Missing: {sorted(expected - names)}"
    )


def test_saturation_emits_all_axis_exact_limit_counts(
    stationary_results,
):

    """Check that saturation emits all axis exact limit counts.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    result = stationary_results[
        "imu_saturation"
    ]

    names = metric_names(
        result
    )

    expected = {
        "AX Exact Limit Count",
        "AY Exact Limit Count",
        "AZ Exact Limit Count",
        "GX Exact Limit Count",
        "GY Exact Limit Count",
        "GZ Exact Limit Count",
    }

    assert expected.issubset(
        names
    ), (
        "Saturation metric contract mismatch.\n"
        f"Status: {result.status}\n"
        f"Messages: {result.messages}\n"
        f"Warnings: {result.warnings}\n"
        f"Missing: {sorted(expected - names)}\n"
        f"Actually emitted: {sorted(names)}"
    )


def test_temperature_tool_emits_disabled_example_metric(
    stationary_results,
):
    """The shipped requirements profile includes a disabled example
    for GX temperature coefficient. Even disabled examples should
    remain valid metric-name examples.
    
    Args:
        stationary_results: Value for `stationary_results`.
    
    Returns:
        None.
    """

    names = metric_names(
        stationary_results[
            "imu_temperature_stability"
        ]
    )

    assert (
        "GX Temperature Coefficient"
        in names
    )


def test_axis_correlation_emits_disabled_example_metric(
    stationary_results,
):

    """Check that axis correlation emits disabled example metric.
    
    Args:
        stationary_results: Stationary results used by this function.
    
    Returns:
        None.
    """
    names = metric_names(
        stationary_results[
            "imu_axis_correlation"
        ]
    )

    assert (
        "Accelerometer Maximum Absolute Axis Correlation"
        in names
    )


# Six-position metric contract


def test_six_position_calibration_emits_bias_metrics(
    six_position_result,
):

    """Check that six position calibration emits bias metrics.
    
    Args:
        six_position_result: Six position result used by this function.
    
    Returns:
        None.
    """
    names = metric_names(
        six_position_result
    )

    expected = {
        "AX Bias",
        "AY Bias",
        "AZ Bias",
    }

    assert expected.issubset(
        names
    ), (
        "Six-position bias metric contract mismatch. "
        f"Missing: {sorted(expected - names)}"
    )


def test_six_position_calibration_emits_sensitivity_error_metrics(
    six_position_result,
):

    """Check that six position calibration emits sensitivity error metrics.
    
    Args:
        six_position_result: Six position result used by this function.
    
    Returns:
        None.
    """
    names = metric_names(
        six_position_result
    )

    expected = {
        "AX Sensitivity Error Percent",
        "AY Sensitivity Error Percent",
        "AZ Sensitivity Error Percent",
    }

    assert expected.issubset(
        names
    ), (
        "Six-position sensitivity metric contract mismatch. "
        f"Missing: {sorted(expected - names)}"
    )


# Full requirements.toml metric-name contract


def test_enabled_requirements_reference_metrics_emitted_by_real_tools(
    stationary_results,
    six_position_result,
):
    """This is the major integration contract.
    
    Every ENABLED requirement in requirements.toml must refer to a
    metric actually emitted by its real analysis tool under a
    representative supported experiment.
    
    Args:
        stationary_results: Value for `stationary_results`.
        six_position_result: Value for `six_position_result`.
    
    Returns:
        None.
    """

    requirements_report = (
        RequirementsLoader().load(
            REQUIREMENTS_PATH
        )
    )

    assert requirements_report.success
    assert (
        requirements_report.requirement_set
        is not None
    )

    results = dict(
        stationary_results
    )

    results[
        "imu_six_position_calibration"
    ] = six_position_result

    missing = []

    for requirement in (
        requirements_report
        .requirement_set
        .enabled_requirements()
    ):

        tool_id = requirement.tool_id

        if tool_id not in results:

            missing.append(
                (
                    requirement.requirement_id,
                    tool_id,
                    requirement.metric_name,
                    "tool was not executed by synthetic contract test",
                )
            )

            continue

        names = metric_names(
            results[
                tool_id
            ]
        )

        if (
            requirement.metric_name
            not in names
        ):

            missing.append(
                (
                    requirement.requirement_id,
                    tool_id,
                    requirement.metric_name,
                    "metric not emitted",
                )
            )

    assert missing == [], (
        "Enabled requirements.toml metric contract mismatch:\n"
        + "\n".join(
            (
                f"  {requirement_id}: "
                f"{tool_id} / "
                f"{metric_name} "
                f"({reason})"
            )
            for (
                requirement_id,
                tool_id,
                metric_name,
                reason,
            )
            in missing
        )
    )


# Disabled requirement examples


def test_disabled_requirements_reference_known_real_metric_names_when_available(
    stationary_results,
    six_position_result,
):
    """Disabled requirements are examples rather than active
    qualification criteria.
    
    We still check them when their real tool emitted the metric in
    this representative experiment.
    
    Allan-derived metrics are data-behavior dependent, so absence of
    an optional Allan coefficient is not treated as a contract
    failure here.
    
    Args:
        stationary_results: Value for `stationary_results`.
        six_position_result: Value for `six_position_result`.
    
    Returns:
        None.
    """

    requirements_report = (
        RequirementsLoader().load(
            REQUIREMENTS_PATH
        )
    )

    assert requirements_report.success
    assert (
        requirements_report.requirement_set
        is not None
    )

    results = dict(
        stationary_results
    )

    results[
        "imu_six_position_calibration"
    ] = six_position_result

    exceptions = {
        (
            "imu_allan_deviation",
            "GX Angle Random Walk Coefficient",
        ),
        (
            "imu_allan_deviation",
            "GY Angle Random Walk Coefficient",
        ),
        (
            "imu_allan_deviation",
            "GZ Angle Random Walk Coefficient",
        ),
    }

    missing = []

    for requirement in (
        requirements_report
        .requirement_set
        .requirements
    ):

        if requirement.enabled:
            continue

        contract = (
            requirement.tool_id,
            requirement.metric_name,
        )

        if contract in exceptions:
            continue

        result = results.get(
            requirement.tool_id
        )

        if result is None:
            continue

        if (
            requirement.metric_name
            not in metric_names(
                result
            )
        ):

            missing.append(
                (
                    requirement.requirement_id,
                    requirement.tool_id,
                    requirement.metric_name,
                )
            )

    assert missing == [], (
        "Disabled requirement example(s) reference metric names "
        "not emitted by representative real tools:\n"
        + "\n".join(
            (
                f"  {requirement_id}: "
                f"{tool_id} / {metric_name}"
            )
            for (
                requirement_id,
                tool_id,
                metric_name,
            )
            in missing
        )
    )


# Qualification-ready unit contract


def test_requirement_metric_units_match_real_emitted_units(
    stationary_results,
    six_position_result,
):
    """RequirementsEngine deliberately performs strict unit matching.
    
    Therefore every enabled requirement that specifies a unit must
    use the same canonical unit text as the real metric output.
    
    Args:
        stationary_results: Value for `stationary_results`.
        six_position_result: Value for `six_position_result`.
    
    Returns:
        None.
    """

    requirements_report = (
        RequirementsLoader().load(
            REQUIREMENTS_PATH
        )
    )

    assert requirements_report.success
    assert (
        requirements_report.requirement_set
        is not None
    )

    results = dict(
        stationary_results
    )

    results[
        "imu_six_position_calibration"
    ] = six_position_result

    mismatches = []

    for requirement in (
        requirements_report
        .requirement_set
        .enabled_requirements()
    ):

        if requirement.unit is None:
            continue

        result = results.get(
            requirement.tool_id
        )

        if result is None:
            continue

        matching_metrics = [
            metric
            for metric
            in result.metrics
            if metric.name
            == requirement.metric_name
        ]

        if not matching_metrics:
            continue

        actual_unit = (
            matching_metrics[
                0
            ].unit
        )

        # Match the same small text normalization currently used by
        # RequirementsEngine.
        expected_normalized = (
            requirement.unit
            .strip()
            .replace(
                "²",
                "^2",
            )
        )

        actual_normalized = (
            actual_unit
            .strip()
            .replace(
                "²",
                "^2",
            )
            if isinstance(
                actual_unit,
                str,
            )
            else actual_unit
        )

        if (
            actual_normalized
            != expected_normalized
        ):

            mismatches.append(
                (
                    requirement.requirement_id,
                    requirement.metric_name,
                    requirement.unit,
                    actual_unit,
                )
            )

    assert mismatches == [], (
        "Requirement unit contract mismatch:\n"
        + "\n".join(
            (
                f"  {requirement_id}: "
                f"{metric_name}: "
                f"requirement={expected!r}, "
                f"tool={actual!r}"
            )
            for (
                requirement_id,
                metric_name,
                expected,
                actual,
            )
            in mismatches
        )
    )
