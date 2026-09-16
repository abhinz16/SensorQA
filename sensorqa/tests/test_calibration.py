#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.calibration import (
    CalibrationEvaluationConfig,
    CalibrationEvaluator,
    CalibrationFitConfig,
    CalibrationManager,
    CalibrationMethod,
    CalibrationModel,
    CalibrationModelFitter,
    CalibrationRunConfig,
    DataSplitConfig,
    SplitStrategy,
    evaluate_calibration_model,
    fit_calibration_model,
    run_calibration,
    split_calibration_data,
)


# Shared fixtures / helpers


def make_linear_frame(
    row_count: int = 100,
    *,
    gain: float = 1.08,
    offset: float = -0.35,
) -> pd.DataFrame:
    """Deterministic sensor/reference data with an exact affine relationship.
    
    Args:
        row_count: Value for `row_count`.
        gain: Value for `gain`.
        offset: Value for `offset`.
    
    Returns:
        DataFrame containing the requested data.
    """

    sensor = np.linspace(
        -5.0,
        15.0,
        row_count,
    )

    reference = (
        gain * sensor
        + offset
    )

    return pd.DataFrame(
        {
            "sensor_value": sensor,
            "reference_value": reference,
        }
    )


class RecordingFitter(
    CalibrationModelFitter
):
    """
    Test double that records exactly what the manager fits on.
    """

    def __init__(
        self,
    ) -> None:

        """Initialize the recording fitter.
        
        Returns:
            None.
        """
        super().__init__()

        self.received_data: (
            pd.DataFrame
            | None
        ) = None

    def fit(
        self,
        data,
        **kwargs,
    ):

        """Run fit.
        
        Args:
            data: Input data to process.
            **kwargs: Kwargs used by this function.
        
        Returns:
            Calculated value.
        """
        self.received_data = (
            data.copy()
        )

        return super().fit(
            data,
            **kwargs,
        )


class RecordingEvaluator(
    CalibrationEvaluator
):
    """
    Test double that records exactly what the manager validates on.
    """

    def __init__(
        self,
    ) -> None:

        """Initialize the recording evaluator.
        
        Returns:
            None.
        """
        super().__init__()

        self.received_data: (
            pd.DataFrame
            | None
        ) = None

    def evaluate(
        self,
        data,
        **kwargs,
    ):

        """Evaluate evaluate.
        
        Args:
            data: Input data to process.
            **kwargs: Kwargs used by this function.
        
        Returns:
            Result of the operation.
        """
        self.received_data = (
            data.copy()
        )

        return super().evaluate(
            data,
            **kwargs,
        )


# Data splitting


def test_sequential_split_is_held_out_and_complete():

    """Check that sequential split is held out and complete.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        20
    )

    result = split_calibration_data(
        data,
        DataSplitConfig(
            strategy=(
                SplitStrategy.SEQUENTIAL
            ),
            validation_fraction=0.25,
        ),
    )

    assert (
        result.training_positions
        == tuple(
            range(
                15
            )
        )
    )

    assert (
        result.validation_positions
        == tuple(
            range(
                15,
                20,
            )
        )
    )

    assert (
        result.training_row_count
        == 15
    )

    assert (
        result.validation_row_count
        == 5
    )

    assert (
        result.total_row_count
        == 20
    )

    assert not (
        set(
            result.training_positions
        )
        & set(
            result.validation_positions
        )
    )

    assert sorted(
        result.training_positions
        + result.validation_positions
    ) == list(
        range(
            20
        )
    )


def test_split_does_not_modify_source_dataframe():

    """Check that split does not modify source dataframe.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        20
    )

    original = data.copy(
        deep=True
    )

    result = split_calibration_data(
        data
    )

    result.training_data.iloc[
        0,
        0,
    ] = 9999.0

    result.validation_data.iloc[
        0,
        0,
    ] = -9999.0

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_random_split_is_reproducible_with_seed():

    """Check that random split is reproducible with seed.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        50
    )

    config = DataSplitConfig(
        strategy=(
            SplitStrategy.RANDOM
        ),
        validation_fraction=0.20,
        random_seed=12345,
    )

    first = split_calibration_data(
        data,
        config,
    )

    second = split_calibration_data(
        data,
        config,
    )

    assert (
        first.training_positions
        == second.training_positions
    )

    assert (
        first.validation_positions
        == second.validation_positions
    )


def test_grouped_split_prevents_group_leakage():

    """Check that grouped split prevents group leakage.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        24
    )

    data[
        "test_run"
    ] = np.repeat(
        [
            "run_1",
            "run_2",
            "run_3",
            "run_4",
        ],
        6,
    )

    result = split_calibration_data(
        data,
        DataSplitConfig(
            strategy=(
                SplitStrategy.GROUPED
            ),
            validation_fraction=0.25,
            random_seed=11,
            group_column="test_run",
        ),
    )

    training_groups = set(
        result.training_data[
            "test_run"
        ]
    )

    validation_groups = set(
        result.validation_data[
            "test_run"
        ]
    )

    assert training_groups
    assert validation_groups

    assert not (
        training_groups
        & validation_groups
    )

    assert (
        training_groups
        | validation_groups
    ) == {
        "run_1",
        "run_2",
        "run_3",
        "run_4",
    }


def test_duplicate_dataframe_indices_do_not_create_split_overlap():

    """Check that duplicate dataframe indices do not create split overlap.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        12
    )

    data.index = [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    ]

    result = split_calibration_data(
        data,
        DataSplitConfig(
            strategy=(
                SplitStrategy.RANDOM
            ),
            validation_fraction=0.25,
            random_seed=7,
        ),
    )

    assert not (
        set(
            result.training_positions
        )
        & set(
            result.validation_positions
        )
    )

    assert (
        result.training_row_count
        + result.validation_row_count
        == len(
            data
        )
    )


def test_invalid_validation_fraction_is_rejected():

    """Check that invalid validation fraction is rejected.
    
    Returns:
        None.
    """
    with pytest.raises(
        ValueError
    ):

        DataSplitConfig(
            validation_fraction=0.0
        )

    with pytest.raises(
        ValueError
    ):

        DataSplitConfig(
            validation_fraction=1.0
        )


# Model fitting


def test_offset_model_recovers_exact_bias():

    """Check that offset model recovers exact bias.
    
    Returns:
        None.
    """
    sensor = np.linspace(
        0.0,
        10.0,
        30,
    )

    reference = (
        sensor
        + 1.75
    )

    data = pd.DataFrame(
        {
            "sensor":
                sensor,

            "reference":
                reference,
        }
    )

    result = fit_calibration_model(
        data,
        measurement_column="sensor",
        reference_column="reference",
        config=CalibrationFitConfig(
            method=(
                CalibrationMethod.OFFSET
            ),
        ),
    )

    assert (
        result.model.gain
        == pytest.approx(
            1.0
        )
    )

    assert (
        result.model.offset
        == pytest.approx(
            1.75
        )
    )

    assert (
        result.training_rmse
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.training_mae
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.training_bias
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )


def test_affine_model_recovers_exact_gain_and_offset():

    """Check that affine model recovers exact gain and offset.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        60,
        gain=1.08,
        offset=-0.35,
    )

    result = fit_calibration_model(
        data,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
    )

    assert (
        result.model.method
        == CalibrationMethod.AFFINE
    )

    assert (
        result.model.gain
        == pytest.approx(
            1.08,
            abs=1e-12,
        )
    )

    assert (
        result.model.offset
        == pytest.approx(
            -0.35,
            abs=1e-12,
        )
    )

    assert (
        result.training_r2
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )


def test_affine_model_rejects_constant_measurements():

    """Check that affine model rejects constant measurements.
    
    Returns:
        None.
    """
    data = pd.DataFrame(
        {
            "sensor":
                np.ones(
                    10
                ),

            "reference":
                np.linspace(
                    0.0,
                    1.0,
                    10,
                ),
        }
    )

    with pytest.raises(
        ValueError,
        match="variation",
    ):

        fit_calibration_model(
            data,
            measurement_column="sensor",
            reference_column="reference",
        )


def test_model_fitting_can_explicitly_drop_nonfinite_pairs():

    """Check that model fitting can explicitly drop nonfinite pairs.
    
    Returns:
        None.
    """
    data = pd.DataFrame(
        {
            "sensor":
                [
                    0.0,
                    1.0,
                    np.nan,
                    3.0,
                    4.0,
                ],

            "reference":
                [
                    1.0,
                    3.0,
                    5.0,
                    np.inf,
                    9.0,
                ],
        }
    )

    result = fit_calibration_model(
        data,
        measurement_column="sensor",
        reference_column="reference",
        config=CalibrationFitConfig(
            minimum_samples=3,
            drop_nonfinite=True,
        ),
    )

    assert (
        result.rows_received
        == 5
    )

    assert (
        result.rows_used
        == 3
    )

    assert (
        result.rows_excluded
        == 2
    )

    assert result.warnings

    assert (
        result.model.gain
        == pytest.approx(
            2.0
        )
    )

    assert (
        result.model.offset
        == pytest.approx(
            1.0
        )
    )


def test_calibration_model_preserves_series_alignment():

    """Check that calibration model preserves series alignment.
    
    Returns:
        None.
    """
    model = CalibrationModel(
        method=(
            CalibrationMethod.AFFINE
        ),
        gain=2.0,
        offset=1.0,
        measurement_column="sensor",
        reference_column="reference",
        training_samples=10,
    )

    values = pd.Series(
        [
            1.0,
            np.nan,
            3.0,
        ],
        index=[
            10,
            10,
            20,
        ],
        name="sensor",
    )

    corrected = (
        model.correct_series(
            values
        )
    )

    assert (
        corrected.index.tolist()
        == [
            10,
            10,
            20,
        ]
    )

    assert (
        corrected.iloc[
            0
        ]
        == pytest.approx(
            3.0
        )
    )

    assert np.isnan(
        corrected.iloc[
            1
        ]
    )

    assert (
        corrected.iloc[
            2
        ]
        == pytest.approx(
            7.0
        )
    )


# Held-out evaluation


def test_held_out_evaluation_reports_before_and_after_metrics():

    """Check that held out evaluation reports before and after metrics.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        30,
        gain=1.08,
        offset=-0.35,
    )

    model = CalibrationModel(
        method=(
            CalibrationMethod.AFFINE
        ),
        gain=1.08,
        offset=-0.35,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
        training_samples=100,
    )

    result = (
        evaluate_calibration_model(
            data,
            model=model,
        )
    )

    assert (
        result.before.rmse
        > 0.0
    )

    assert (
        result.before.mae
        > 0.0
    )

    assert (
        result.after.rmse
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.after.mae
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.after.bias
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.after.r2
        == pytest.approx(
            1.0,
            abs=1e-12,
        )
    )

    assert (
        result.improvement
        .rmse_reduction_percent
        == pytest.approx(
            100.0
        )
    )

    assert (
        result.improvement
        .mae_reduction_percent
        == pytest.approx(
            100.0
        )
    )


def test_evaluation_can_report_calibration_degradation():

    """Check that evaluation can report calibration degradation.
    
    Returns:
        None.
    """
    actual = np.linspace(
        1.0,
        10.0,
        20,
    )

    data = pd.DataFrame(
        {
            "sensor":
                actual.copy(),

            "reference":
                actual.copy(),
        }
    )

    deliberately_bad_model = (
        CalibrationModel(
            method=(
                CalibrationMethod.OFFSET
            ),
            gain=1.0,
            offset=5.0,
            measurement_column="sensor",
            reference_column="reference",
            training_samples=10,
        )
    )

    result = (
        evaluate_calibration_model(
            data,
            model=(
                deliberately_bad_model
            ),
        )
    )

    assert (
        result.before.rmse
        == pytest.approx(
            0.0
        )
    )

    assert (
        result.after.rmse
        == pytest.approx(
            5.0
        )
    )

    # Percent reduction is intentionally
    # undefined when baseline error is zero.
    assert (
        result.improvement
        .rmse_reduction_percent
        is None
    )

    assert any(
        "RMSE" in warning
        for warning
        in result.warnings
    )


def test_negative_reduction_indicates_worse_validation_performance():

    """Check that negative reduction indicates worse validation performance.
    
    Returns:
        None.
    """
    reference = np.linspace(
        0.0,
        10.0,
        20,
    )

    sensor = (
        reference
        + 1.0
    )

    data = pd.DataFrame(
        {
            "sensor":
                sensor,

            "reference":
                reference,
        }
    )

    model = CalibrationModel(
        method=(
            CalibrationMethod.OFFSET
        ),
        gain=1.0,
        offset=1.0,
        measurement_column="sensor",
        reference_column="reference",
        training_samples=10,
    )

    result = (
        evaluate_calibration_model(
            data,
            model=model,
        )
    )

    assert (
        result.before.rmse
        == pytest.approx(
            1.0
        )
    )

    assert (
        result.after.rmse
        == pytest.approx(
            2.0
        )
    )

    assert (
        result.improvement
        .rmse_reduction_percent
        == pytest.approx(
            -100.0
        )
    )


def test_evaluated_data_contains_row_level_before_after_values():

    """Check that evaluated data contains row level before after values.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        10
    )

    model = CalibrationModel(
        method=(
            CalibrationMethod.AFFINE
        ),
        gain=1.08,
        offset=-0.35,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
        training_samples=8,
    )

    result = (
        evaluate_calibration_model(
            data,
            model=model,
            config=(
                CalibrationEvaluationConfig(
                    minimum_samples=2,
                )
            ),
        )
    )

    assert list(
        result.evaluated_data.columns
    ) == [
        "raw_measurement",
        "reference",
        "corrected_measurement",
        "raw_error",
        "corrected_error",
    ]

    assert (
        len(
            result.evaluated_data
        )
        == len(
            data
        )
    )


# Complete calibration workflow


def test_manager_runs_complete_held_out_calibration_workflow():

    """Check that manager runs complete held out calibration workflow.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        100,
        gain=1.08,
        offset=-0.35,
    )

    result = run_calibration(
        data,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
        config=CalibrationRunConfig(
            split=DataSplitConfig(
                strategy=(
                    SplitStrategy.SEQUENTIAL
                ),
                validation_fraction=0.20,
            ),
        ),
    )

    assert (
        result.split.training_row_count
        == 80
    )

    assert (
        result.split.validation_row_count
        == 20
    )

    assert (
        result.model.gain
        == pytest.approx(
            1.08,
            abs=1e-12,
        )
    )

    assert (
        result.model.offset
        == pytest.approx(
            -0.35,
            abs=1e-12,
        )
    )

    assert (
        result.evaluation.after.rmse
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.evaluation.after.mae
        == pytest.approx(
            0.0,
            abs=1e-12,
        )
    )

    assert (
        result.summary()[
            "evaluation_basis"
        ]
        == "held_out_validation"
    )


def test_manager_never_passes_validation_rows_to_fitter():

    """Check that manager never passes validation rows to fitter.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        40
    )

    fitter = RecordingFitter()

    evaluator = (
        RecordingEvaluator()
    )

    manager = CalibrationManager(
        fitter=fitter,
        evaluator=evaluator,
    )

    result = manager.run(
        data,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
        config=CalibrationRunConfig(
            split=DataSplitConfig(
                strategy=(
                    SplitStrategy.SEQUENTIAL
                ),
                validation_fraction=0.25,
            ),
        ),
    )

    assert (
        fitter.received_data
        is not None
    )

    assert (
        evaluator.received_data
        is not None
    )

    pd.testing.assert_frame_equal(
        fitter.received_data,
        result.split.training_data,
    )

    pd.testing.assert_frame_equal(
        evaluator.received_data,
        result.split.validation_data,
    )

    assert not (
        set(
            result.split
            .training_positions
        )
        & set(
            result.split
            .validation_positions
        )
    )


def test_manager_raises_split_minimums_to_downstream_requirements():

    """Check that manager raises split minimums to downstream requirements.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        20
    )

    result = run_calibration(
        data,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
        config=CalibrationRunConfig(

            split=DataSplitConfig(
                validation_fraction=0.20,
                strategy=(
                    SplitStrategy.SEQUENTIAL
                ),
                minimum_train_rows=1,
                minimum_validation_rows=1,
            ),

            fit=CalibrationFitConfig(
                minimum_samples=5,
            ),

            evaluation=(
                CalibrationEvaluationConfig(
                    minimum_samples=4,
                )
            ),
        ),
    )

    assert (
        result.split.training_row_count
        >= 5
    )

    assert (
        result.split.validation_row_count
        >= 4
    )

    assert any(
        (
            "minimum_train_rows "
            "was increased"
        )
        in warning
        for warning
        in result.warnings
    )

    assert any(
        (
            "minimum_validation_rows "
            "was increased"
        )
        in warning
        for warning
        in result.warnings
    )


def test_manager_rejects_same_measurement_and_reference_column():

    """Check that manager rejects same measurement and reference column.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        20
    )

    with pytest.raises(
        ValueError,
        match="must be different",
    ):

        run_calibration(
            data,
            measurement_column=(
                "sensor_value"
            ),
            reference_column=(
                "sensor_value"
            ),
        )


def test_manager_result_dataframe_properties_are_defensive_copies():

    """Check that manager result dataframe properties are defensive copies.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        30
    )

    result = run_calibration(
        data,
        measurement_column=(
            "sensor_value"
        ),
        reference_column=(
            "reference_value"
        ),
    )

    training_copy = (
        result.training_data
    )

    validation_copy = (
        result.validation_data
    )

    evaluated_copy = (
        result.evaluated_validation_data
    )

    training_copy.iloc[
        0,
        0,
    ] = 9999.0

    validation_copy.iloc[
        0,
        0,
    ] = 9999.0

    evaluated_copy.iloc[
        0,
        0,
    ] = 9999.0

    assert (
        result.split
        .training_data
        .iloc[
            0,
            0,
        ]
        != 9999.0
    )

    assert (
        result.split
        .validation_data
        .iloc[
            0,
            0,
        ]
        != 9999.0
    )

    assert (
        result.evaluation
        .evaluated_data
        .iloc[
            0,
            0,
        ]
        != 9999.0
    )


# Public API / serialization behavior


def test_calibration_public_api_supports_ui_facing_summary():

    """Check that calibration public api supports ui facing summary.
    
    Returns:
        None.
    """
    data = make_linear_frame(
        50
    )

    result = (
        CalibrationManager()
        .run(
            data,
            measurement_column=(
                "sensor_value"
            ),
            reference_column=(
                "reference_value"
            ),
        )
    )

    summary = result.summary()

    assert (
        summary[
            "measurement_column"
        ]
        == "sensor_value"
    )

    assert (
        summary[
            "reference_column"
        ]
        == "reference_value"
    )

    assert (
        summary[
            "evaluation_basis"
        ]
        == "held_out_validation"
    )

    assert (
        "split"
        in summary
    )

    assert (
        "fit"
        in summary
    )

    assert (
        "evaluation"
        in summary
    )

    assert (
        "warnings"
        in summary
    )

    assert (
        summary[
            "fit"
        ][
            "model"
        ][
            "method"
        ]
        == "affine"
    )

    assert (
        summary[
            "evaluation"
        ][
            "before"
        ][
            "sample_count"
        ]
        > 0
    )

    assert (
        summary[
            "evaluation"
        ][
            "after"
        ][
            "sample_count"
        ]
        > 0
    )
