#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public calibration API for SensorQA.

The calibration package separates data
partitioning, model fitting, held-out evaluation, and workflow orchestration.
This package exposes the objects that application code,
reports, tests, and the desktop UI should import.

Typical usage
-------------

    from sensorqa.calibration import (
        CalibrationManager,
        CalibrationRunConfig,
        CalibrationMethod,
        DataSplitConfig,
        SplitStrategy,
    )

    result = CalibrationManager().run(
        data,
        measurement_column="sensor_value",
        reference_column="reference_value",
        config=CalibrationRunConfig(
            split=DataSplitConfig(
                strategy=SplitStrategy.SEQUENTIAL,
                validation_fraction=0.20,
            ),
        ),
    )

Importing through ``sensorqa.calibration`` avoids coupling callers to the
internal file layout of the calibration subsystem.
"""

from sensorqa.calibration.data_split import (
    CalibrationDataSplitter,
    DataSplitConfig,
    DataSplitResult,
    SplitStrategy,
    split_calibration_data,
)
from sensorqa.calibration.evaluation import (
    CalibrationErrorMetrics,
    CalibrationEvaluationConfig,
    CalibrationEvaluationResult,
    CalibrationEvaluator,
    CalibrationImprovement,
    evaluate_calibration_model,
)
from sensorqa.calibration.manager import (
    CalibrationManager,
    CalibrationRunConfig,
    CalibrationRunResult,
    run_calibration,
)
from sensorqa.calibration.model import (
    CalibrationFitConfig,
    CalibrationFitResult,
    CalibrationMethod,
    CalibrationModel,
    CalibrationModelFitter,
    fit_calibration_model,
)


__all__ = [
    # Data splitting
    "SplitStrategy",
    "DataSplitConfig",
    "DataSplitResult",
    "CalibrationDataSplitter",
    "split_calibration_data",

    # Model fitting
    "CalibrationMethod",
    "CalibrationFitConfig",
    "CalibrationModel",
    "CalibrationFitResult",
    "CalibrationModelFitter",
    "fit_calibration_model",

    # Held-out evaluation
    "CalibrationEvaluationConfig",
    "CalibrationErrorMetrics",
    "CalibrationImprovement",
    "CalibrationEvaluationResult",
    "CalibrationEvaluator",
    "evaluate_calibration_model",

    # Complete workflow
    "CalibrationRunConfig",
    "CalibrationRunResult",
    "CalibrationManager",
    "run_calibration",
]
