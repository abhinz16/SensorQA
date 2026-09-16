#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from sensorqa.calibration.model import CalibrationModel


@dataclass(frozen=True)
class CalibrationEvaluationConfig:
    """
    Configuration for held-out calibration evaluation.

    The evaluator never fits or modifies a calibration model.
    It only applies an already-fitted model to data reserved
    for validation.
    """

    minimum_samples: int = 2
    drop_nonfinite: bool = True

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.minimum_samples,
            Integral,
        ):
            raise TypeError(
                "minimum_samples must be an integer."
            )

        if self.minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be at least 1."
            )

        if not isinstance(
            self.drop_nonfinite,
            bool,
        ):
            raise TypeError(
                "drop_nonfinite must be a boolean."
            )


@dataclass(frozen=True)
class CalibrationErrorMetrics:
    """
    Error metrics for predictions against reference values.

    Error is defined as:

        prediction - reference

    Positive bias means predictions are high on average.
    Negative bias means predictions are low on average.
    """

    sample_count: int

    rmse: float
    mae: float
    bias: float
    max_abs_error: float
    error_std: float

    r2: float | None

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return asdict(
            self
        )


@dataclass(frozen=True)
class CalibrationImprovement:
    """
    Held-out before-versus-after calibration changes.

    Positive reduction percentages indicate lower error
    after calibration.

    Negative values indicate degradation.

    R² is represented as an absolute delta rather than
    a percentage change.
    """

    rmse_reduction_percent: float | None

    mae_reduction_percent: float | None

    absolute_bias_reduction_percent: (
        float | None
    )

    max_abs_error_reduction_percent: (
        float | None
    )

    error_std_reduction_percent: (
        float | None
    )

    r2_delta: float | None

    def to_dict(
        self,
    ) -> dict[str, Any]:

        """Return a plain dictionary representation.
        
        Returns:
            Dictionary containing the result values.
        """
        return asdict(
            self
        )


@dataclass
class CalibrationEvaluationResult:
    """
    Held-out evaluation result for one fitted calibration model.

    before:
        Performance of the raw sensor measurements.

    after:
        Performance after applying the calibration model.

    evaluated_data:
        Row-level data suitable for UI plots and
        report tables.
    """

    model: CalibrationModel

    before: CalibrationErrorMetrics

    after: CalibrationErrorMetrics

    improvement: CalibrationImprovement

    rows_received: int

    rows_used: int

    rows_excluded: int

    evaluated_data: pd.DataFrame

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(
        self,
    ) -> dict[str, Any]:
        """JSON-friendly result summary.
        
        Row-level values are kept out of this
        summary and remain available through evaluated_data.
        
        Returns:
            Dictionary containing the result values.
        """

        return {
            "model":
                self.model.to_dict(),

            "before":
                self.before.to_dict(),

            "after":
                self.after.to_dict(),

            "improvement":
                self.improvement.to_dict(),

            "rows_received":
                self.rows_received,

            "rows_used":
                self.rows_used,

            "rows_excluded":
                self.rows_excluded,

            "warnings":
                list(
                    self.warnings
                ),
        }


class CalibrationEvaluator:
    """
    Evaluate a fitted calibration model on untouched
    validation observations.

    This class does not fit, refit, or modify model
    coefficients.
    """

    def evaluate(
        self,
        data: pd.DataFrame,
        *,
        model: CalibrationModel,
        config: (
            CalibrationEvaluationConfig
            | None
        ) = None,
    ) -> CalibrationEvaluationResult:
        """Compare raw and calibrated measurements against
        the same validation reference values.
        
        Args:
            data: Input data to process.
            model: Value for `model`.
            config: Configuration for the operation.
        
        Returns:
            CalibrationEvaluationResult returned by the function.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if not isinstance(
            model,
            CalibrationModel,
        ):
            raise TypeError(
                "model must be a CalibrationModel."
            )

        resolved_config = (
            config
            if config is not None
            else CalibrationEvaluationConfig()
        )

        measurement_column = (
            model.measurement_column
        )

        reference_column = (
            model.reference_column
        )

        missing_columns = [
            column
            for column
            in (
                measurement_column,
                reference_column,
            )
            if column
            not in data.columns
        ]

        if missing_columns:

            raise KeyError(
                "Validation data is missing required "
                "column(s): "
                + ", ".join(
                    missing_columns
                )
            )

        rows_received = len(
            data
        )

        measurement = pd.to_numeric(
            data[
                measurement_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            copy=True,
        )

        reference = pd.to_numeric(
            data[
                reference_column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float,
            copy=True,
        )

        finite_mask = (
            np.isfinite(
                measurement
            )
            & np.isfinite(
                reference
            )
        )

        rows_used = int(
            finite_mask.sum()
        )

        rows_excluded = (
            rows_received
            - rows_used
        )

        if (
            rows_excluded
            and not resolved_config.drop_nonfinite
        ):

            raise ValueError(
                "Validation data contains non-finite "
                "or non-numeric measurement/reference "
                "pairs and drop_nonfinite=False."
            )

        if (
            rows_used
            < resolved_config.minimum_samples
        ):

            raise ValueError(
                "Not enough valid paired observations "
                "for held-out calibration evaluation: "
                f"{rows_used} available, "
                f"{resolved_config.minimum_samples} required."
            )

        raw = measurement[
            finite_mask
        ]

        actual = reference[
            finite_mask
        ]

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):

            corrected = (
                float(
                    model.gain
                )
                * raw
                + float(
                    model.offset
                )
            )

        if not np.all(
            np.isfinite(
                corrected
            )
        ):

            raise ValueError(
                "Calibration model produced non-finite "
                "corrected values on otherwise valid "
                "validation observations."
            )

        before = (
            self._calculate_metrics(
                actual=actual,
                predicted=raw,
            )
        )

        after = (
            self._calculate_metrics(
                actual=actual,
                predicted=corrected,
            )
        )

        warnings: list[str] = []

        if rows_excluded:

            warnings.append(
                f"Excluded {rows_excluded} row(s) "
                "containing non-finite or non-numeric "
                "validation pairs."
            )

        if before.r2 is None:

            warnings.append(
                "Before-calibration validation R² is "
                "undefined because the reference values "
                "have effectively zero variance."
            )

        if after.r2 is None:

            warnings.append(
                "After-calibration validation R² is "
                "undefined because the reference values "
                "have effectively zero variance."
            )

        improvement = (
            CalibrationImprovement(

                rmse_reduction_percent=(
                    self._reduction_percent(
                        before=before.rmse,
                        after=after.rmse,
                    )
                ),

                mae_reduction_percent=(
                    self._reduction_percent(
                        before=before.mae,
                        after=after.mae,
                    )
                ),

                absolute_bias_reduction_percent=(
                    self._reduction_percent(
                        before=abs(
                            before.bias
                        ),
                        after=abs(
                            after.bias
                        ),
                    )
                ),

                max_abs_error_reduction_percent=(
                    self._reduction_percent(
                        before=(
                            before.max_abs_error
                        ),
                        after=(
                            after.max_abs_error
                        ),
                    )
                ),

                error_std_reduction_percent=(
                    self._reduction_percent(
                        before=(
                            before.error_std
                        ),
                        after=(
                            after.error_std
                        ),
                    )
                ),

                r2_delta=(
                    None
                    if (
                        before.r2 is None
                        or after.r2 is None
                    )
                    else float(
                        after.r2
                        - before.r2
                    )
                ),
            )
        )

        undefined_reductions = [
            name
            for (
                name,
                value,
            )
            in (
                (
                    "RMSE",
                    improvement
                    .rmse_reduction_percent,
                ),
                (
                    "MAE",
                    improvement
                    .mae_reduction_percent,
                ),
                (
                    "absolute bias",
                    improvement
                    .absolute_bias_reduction_percent,
                ),
                (
                    "maximum absolute error",
                    improvement
                    .max_abs_error_reduction_percent,
                ),
                (
                    "error standard deviation",
                    improvement
                    .error_std_reduction_percent,
                ),
            )
            if value is None
        ]

        if undefined_reductions:

            warnings.append(
                "Percent reduction is undefined for "
                "zero before-calibration metric(s): "
                + ", ".join(
                    undefined_reductions
                )
                + "."
            )

        used_positions = (
            np.flatnonzero(
                finite_mask
            )
        )

        used_index = (
            data.index.take(
                used_positions
            )
        )

        evaluated_data = pd.DataFrame(
            {
                "raw_measurement":
                    raw,

                "reference":
                    actual,

                "corrected_measurement":
                    corrected,

                "raw_error":
                    raw
                    - actual,

                "corrected_error":
                    corrected
                    - actual,
            },
            index=used_index,
        )

        evaluated_data.index.name = (
            data.index.name
        )

        return CalibrationEvaluationResult(
            model=model,

            before=before,

            after=after,

            improvement=improvement,

            rows_received=(
                rows_received
            ),

            rows_used=(
                rows_used
            ),

            rows_excluded=(
                rows_excluded
            ),

            evaluated_data=(
                evaluated_data
            ),

            warnings=warnings,
        )

    @staticmethod
    def _calculate_metrics(
        *,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> CalibrationErrorMetrics:
        """Calculate a common error-metric set.
        
        Args:
            actual: Reference or observed values.
            predicted: Predicted values.
        
        Returns:
            CalibrationErrorMetrics returned by the function.
        """

        error = (
            predicted
            - actual
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    error
                )
            )
        )

        bias = float(
            np.mean(
                error
            )
        )

        max_abs_error = float(
            np.max(
                np.abs(
                    error
                )
            )
        )

        error_std = float(
            np.std(
                error,
                ddof=0,
            )
        )

        r2 = (
            CalibrationEvaluator
            ._calculate_r2(
                actual=actual,
                predicted=predicted,
            )
        )

        return CalibrationErrorMetrics(
            sample_count=len(
                actual
            ),

            rmse=rmse,

            mae=mae,

            bias=bias,

            max_abs_error=(
                max_abs_error
            ),

            error_std=(
                error_std
            ),

            r2=r2,
        )

    @staticmethod
    def _calculate_r2(
        *,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float | None:
        """Calculate R² when the reference values have
        nonzero variance.
        
        Args:
            actual: Reference or observed values.
            predicted: Predicted values.
        
        Returns:
            Numeric result.
        """

        residual_sum_squares = float(
            np.sum(
                (
                    actual
                    - predicted
                ) ** 2
            )
        )

        centered = (
            actual
            - np.mean(
                actual
            )
        )

        total_sum_squares = float(
            np.sum(
                centered ** 2
            )
        )

        if np.isclose(
            total_sum_squares,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ):

            return None

        return float(
            1.0
            - (
                residual_sum_squares
                / total_sum_squares
            )
        )

    @staticmethod
    def _reduction_percent(
        *,
        before: float,
        after: float,
    ) -> float | None:
        """Calculate percent reduction in an error metric.
        
        Positive:
            error decreased.
        
        Negative:
            error increased.
        
        None:
            the before-calibration metric was zero,
            so percentage reduction is undefined.
        
        Args:
            before: Value for `before`.
            after: Value for `after`.
        
        Returns:
            Numeric result.
        """

        if np.isclose(
            before,
            0.0,
            rtol=0.0,
            atol=1e-15,
        ):

            return None

        return float(
            100.0
            * (
                before
                - after
            )
            / before
        )


def evaluate_calibration_model(
    data: pd.DataFrame,
    *,
    model: CalibrationModel,
    config: (
        CalibrationEvaluationConfig
        | None
    ) = None,
) -> CalibrationEvaluationResult:
    """Convenience function for one held-out
    calibration evaluation.
    
    Args:
        data: Input data to process.
        model: Value for `model`.
        config: Configuration for the operation.
    
    Returns:
        CalibrationEvaluationResult returned by the function.
    """

    return (
        CalibrationEvaluator()
        .evaluate(
            data=data,
            model=model,
            config=config,
        )
    )
