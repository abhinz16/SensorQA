#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from sensorqa.calibration.data_split import (
    CalibrationDataSplitter,
    DataSplitConfig,
    DataSplitResult,
)
from sensorqa.calibration.evaluation import (
    CalibrationEvaluationConfig,
    CalibrationEvaluationResult,
    CalibrationEvaluator,
)
from sensorqa.calibration.model import (
    CalibrationFitConfig,
    CalibrationFitResult,
    CalibrationModel,
    CalibrationModelFitter,
)
from sensorqa.ingestion.dataset import SensorQADataset


@dataclass(frozen=True)
class CalibrationRunConfig:
    """
    Complete configuration for one SensorQA calibration workflow.

    Keeping split, fit, and evaluation settings together gives the future
    application UI one stable object to construct from user selections while
    preserving separation between the three scientific stages.
    """

    split: DataSplitConfig = field(
        default_factory=DataSplitConfig
    )
    fit: CalibrationFitConfig = field(
        default_factory=CalibrationFitConfig
    )
    evaluation: CalibrationEvaluationConfig = field(
        default_factory=CalibrationEvaluationConfig
    )

    def __post_init__(self) -> None:
        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(self.split, DataSplitConfig):
            raise TypeError(
                "split must be a DataSplitConfig."
            )

        if not isinstance(self.fit, CalibrationFitConfig):
            raise TypeError(
                "fit must be a CalibrationFitConfig."
            )

        if not isinstance(
            self.evaluation,
            CalibrationEvaluationConfig,
        ):
            raise TypeError(
                "evaluation must be a CalibrationEvaluationConfig."
            )


@dataclass
class CalibrationRunResult:
    """
    Consolidated result from a complete held-out calibration workflow.

    The result keeps each stage separate so reports and the UI can show:

        1. how data were split,
        2. how the calibration model was fitted, and
        3. how it performed on untouched validation data.

    No single overall "good/bad" calibration verdict is created here. SensorQA
    reports the measured changes and leaves qualification to explicit
    engineering requirements.
    """

    measurement_column: str
    reference_column: str

    split: DataSplitResult
    fit: CalibrationFitResult
    evaluation: CalibrationEvaluationResult

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def model(self) -> CalibrationModel:
        """Convenience access to the fitted calibration model.
        
        Returns:
            CalibrationModel returned by the function.
        """

        return self.fit.model

    @property
    def training_data(self) -> pd.DataFrame:
        """Defensive copy of the calibration/training subset.
        
        Returns:
            DataFrame containing the requested data.
        """

        return self.split.training_data.copy()

    @property
    def validation_data(self) -> pd.DataFrame:
        """Defensive copy of the untouched validation subset.
        
        Returns:
            DataFrame containing the requested data.
        """

        return self.split.validation_data.copy()

    @property
    def evaluated_validation_data(self) -> pd.DataFrame:
        """Defensive copy of row-level held-out before/after values.
        
        This is the table the UI can use for calibration plots or an
        exported validation CSV.
        
        Returns:
            DataFrame containing the requested data.
        """

        return self.evaluation.evaluated_data.copy()

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary for reports, UI state, or an API response.
        
        Returns:
            Dictionary containing the result values.
        """

        return {
            "measurement_column": self.measurement_column,
            "reference_column": self.reference_column,
            "evaluation_basis": "held_out_validation",
            "split": self.split.summary(),
            "fit": self.fit.summary(),
            "evaluation": self.evaluation.summary(),
            "warnings": list(self.warnings),
        }


class CalibrationManager:
    """
    Orchestrate the complete leakage-resistant calibration workflow.

    The manager coordinates the workflow. The calculations remain in
    dedicated components:

        CalibrationDataSplitter
            partitions calibration/training and held-out validation data.

        CalibrationModelFitter
            fits coefficients using training data only.

        CalibrationEvaluator
            applies the frozen model to validation data only.

    This composition layer is the object that the future SensorQA application
    UI should call instead of manually coordinating those stages.
    """

    def __init__(
        self,
        *,
        splitter: CalibrationDataSplitter | None = None,
        fitter: CalibrationModelFitter | None = None,
        evaluator: CalibrationEvaluator | None = None,
    ) -> None:

        """Initialize the calibration manager.
        
        Args:
            splitter: Splitter used by this function.
            fitter: Fitter used by this function.
            evaluator: Evaluator used by this function.
        
        Returns:
            None.
        """
        self.splitter = (
            splitter
            if splitter is not None
            else CalibrationDataSplitter()
        )

        self.fitter = (
            fitter
            if fitter is not None
            else CalibrationModelFitter()
        )

        self.evaluator = (
            evaluator
            if evaluator is not None
            else CalibrationEvaluator()
        )

        if not isinstance(
            self.splitter,
            CalibrationDataSplitter,
        ):
            raise TypeError(
                "splitter must be a CalibrationDataSplitter."
            )

        if not isinstance(
            self.fitter,
            CalibrationModelFitter,
        ):
            raise TypeError(
                "fitter must be a CalibrationModelFitter."
            )

        if not isinstance(
            self.evaluator,
            CalibrationEvaluator,
        ):
            raise TypeError(
                "evaluator must be a CalibrationEvaluator."
            )

    def run(
        self,
        data: pd.DataFrame | SensorQADataset,
        *,
        measurement_column: str,
        reference_column: str,
        config: CalibrationRunConfig | None = None,
    ) -> CalibrationRunResult:
        """Run split -> fit -> held-out evaluation.
        
        The calibration model is fitted exclusively on ``training_data`` and
        is then passed unchanged to the evaluator. Validation rows are never
        supplied to the fitter.
        
        Args:
            data: Input data to process.
            measurement_column: Column containing sensor measurements.
            reference_column: Column containing reference values.
            config: Configuration for the operation.
        
        Returns:
            CalibrationRunResult returned by the function.
        """

        if not isinstance(
            data,
            (pd.DataFrame, SensorQADataset),
        ):
            raise TypeError(
                "data must be a pandas DataFrame or SensorQADataset."
            )

        if (
            not isinstance(measurement_column, str)
            or not measurement_column.strip()
        ):
            raise ValueError(
                "measurement_column must be a non-empty string."
            )

        if (
            not isinstance(reference_column, str)
            or not reference_column.strip()
        ):
            raise ValueError(
                "reference_column must be a non-empty string."
            )

        if measurement_column == reference_column:
            raise ValueError(
                "measurement_column and reference_column must be different."
            )

        resolved_config = (
            config
            if config is not None
            else CalibrationRunConfig()
        )

        if not isinstance(
            resolved_config,
            CalibrationRunConfig,
        ):
            raise TypeError(
                "config must be a CalibrationRunConfig or None."
            )

        # The splitter's minimum row counts are raised, when necessary, to
        # satisfy the downstream fit/evaluation contracts. This prevents a
        # split that is structurally valid but guaranteed to fail at the next
        # stage. The user's requested values are never lowered.
        effective_split_config = replace(
            resolved_config.split,
            minimum_train_rows=max(
                int(
                    resolved_config.split.minimum_train_rows
                ),
                int(
                    resolved_config.fit.minimum_samples
                ),
            ),
            minimum_validation_rows=max(
                int(
                    resolved_config.split.minimum_validation_rows
                ),
                int(
                    resolved_config.evaluation.minimum_samples
                ),
            ),
        )

        split_result = self.splitter.split(
            data=data,
            config=effective_split_config,
        )

        # Only training rows are passed to the fitter.
        fit_result = self.fitter.fit(
            split_result.training_data,
            measurement_column=measurement_column,
            reference_column=reference_column,
            config=resolved_config.fit,
        )

        # Only held-out validation rows are passed to the evaluator. The model
        # returned above is applied as-is; no refitting occurs here.
        evaluation_result = self.evaluator.evaluate(
            split_result.validation_data,
            model=fit_result.model,
            config=resolved_config.evaluation,
        )

        warnings: list[str] = []

        warnings.extend(
            f"Split: {warning}"
            for warning in split_result.warnings
        )

        warnings.extend(
            f"Fit: {warning}"
            for warning in fit_result.warnings
        )

        warnings.extend(
            f"Validation: {warning}"
            for warning in evaluation_result.warnings
        )

        if (
            effective_split_config.minimum_train_rows
            != resolved_config.split.minimum_train_rows
        ):
            warnings.append(
                "Split: minimum_train_rows was increased to "
                f"{effective_split_config.minimum_train_rows} to satisfy "
                "the calibration fit minimum-sample requirement."
            )

        if (
            effective_split_config.minimum_validation_rows
            != resolved_config.split.minimum_validation_rows
        ):
            warnings.append(
                "Split: minimum_validation_rows was increased to "
                f"{effective_split_config.minimum_validation_rows} to "
                "satisfy the held-out evaluation minimum-sample requirement."
            )

        return CalibrationRunResult(
            measurement_column=measurement_column,
            reference_column=reference_column,
            split=split_result,
            fit=fit_result,
            evaluation=evaluation_result,
            warnings=warnings,
        )


def run_calibration(
    data: pd.DataFrame | SensorQADataset,
    *,
    measurement_column: str,
    reference_column: str,
    config: CalibrationRunConfig | None = None,
) -> CalibrationRunResult:
    """Convenience function for one complete calibration workflow.
    
    Args:
        data: Input data to process.
        measurement_column: Column containing sensor measurements.
        reference_column: Column containing reference values.
        config: Configuration for the operation.
    
    Returns:
        CalibrationRunResult returned by the function.
    """

    return CalibrationManager().run(
        data=data,
        measurement_column=measurement_column,
        reference_column=reference_column,
        config=config,
    )
