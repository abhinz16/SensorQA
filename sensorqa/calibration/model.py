#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


class CalibrationMethod(str, Enum):
    """
    Empirical calibration models supported by SensorQA.

    OFFSET
        Corrects only a constant additive bias:

            corrected = measurement + offset

    AFFINE
        Corrects a linear scale-and-offset relationship:

            corrected = gain * measurement + offset

    These models describe empirical correction relationships.
    They do not, by themselves, establish the physical cause of
    a sensor error.
    """

    OFFSET = "offset"
    AFFINE = "affine"


@dataclass(frozen=True)
class CalibrationFitConfig:
    """
    Configuration used when fitting a calibration model.

    minimum_samples protects against fitting a model to an
    inadequately small calibration subset.

    Invalid/non-finite measurement-reference pairs can either
    be rejected or explicitly omitted.
    """

    method: CalibrationMethod = CalibrationMethod.AFFINE

    minimum_samples: int = 3

    drop_nonfinite: bool = True

    def __post_init__(
        self,
    ) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.method,
            CalibrationMethod,
        ):
            raise TypeError(
                "method must be a CalibrationMethod."
            )

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

        if (
            self.method
            == CalibrationMethod.AFFINE
            and self.minimum_samples < 2
        ):
            raise ValueError(
                "AFFINE calibration requires "
                "minimum_samples >= 2."
            )

        if not isinstance(
            self.drop_nonfinite,
            bool,
        ):
            raise TypeError(
                "drop_nonfinite must be a boolean."
            )


@dataclass(frozen=True)
class CalibrationModel:
    """
    Fitted empirical correction model.

    The model maps a raw sensor measurement to a corrected
    estimate of the reference quantity:

        corrected = gain * measurement + offset

    For OFFSET calibration, gain is always 1.0.
    """

    method: CalibrationMethod

    gain: float
    offset: float

    measurement_column: str
    reference_column: str

    training_samples: int

    def correct_value(
        self,
        measurement: float | int,
    ) -> float:
        """Correct one finite numeric measurement.
        
        Args:
            measurement: Value for `measurement`.
        
        Returns:
            Numeric result.
        """

        value = float(
            measurement
        )

        if not np.isfinite(
            value
        ):
            raise ValueError(
                "measurement must be finite."
            )

        return (
            float(
                self.gain
            )
            * value
            + float(
                self.offset
            )
        )

    def correct_series(
        self,
        measurements: pd.Series,
        *,
        name: str | None = None,
    ) -> pd.Series:
        """Apply the fitted calibration to a pandas Series.
        
        NaN and infinite values are returned as NaN so row
        alignment is preserved.
        
        Args:
            measurements: Value for `measurements`.
            name: Name of the item.
        
        Returns:
            pd.Series returned by the function.
        """

        if not isinstance(
            measurements,
            pd.Series,
        ):
            raise TypeError(
                "measurements must be a pandas Series."
            )

        numeric = pd.to_numeric(
            measurements,
            errors="coerce",
        ).astype(
            float
        )

        finite_mask = np.isfinite(
            numeric.to_numpy(
                dtype=float,
                copy=False,
            )
        )

        corrected = pd.Series(
            np.nan,
            index=measurements.index,
            dtype=float,
            name=(
                name
                if name is not None
                else measurements.name
            ),
        )

        corrected.loc[
            finite_mask
        ] = (
            float(
                self.gain
            )
            * numeric.loc[
                finite_mask
            ]
            + float(
                self.offset
            )
        )

        return corrected

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """JSON-friendly representation for reports and UI.
        
        Returns:
            Dictionary containing the result values.
        """

        data = asdict(
            self
        )

        data[
            "method"
        ] = (
            self.method.value
        )

        return data


@dataclass
class CalibrationFitResult:
    """
    Result of fitting a calibration model.

    These metrics describe only the training/calibration data.

    They must NOT be interpreted as held-out validation
    performance.
    """

    model: CalibrationModel

    training_rmse: float
    training_mae: float
    training_bias: float
    training_r2: float | None

    rows_received: int
    rows_used: int
    rows_excluded: int

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(
        self,
    ) -> dict[str, Any]:
        """Serializable result for reports and the UI.
        
        Returns:
            Dictionary containing the result values.
        """

        return {
            "model":
                self.model.to_dict(),

            "training_rmse":
                self.training_rmse,

            "training_mae":
                self.training_mae,

            "training_bias":
                self.training_bias,

            "training_r2":
                self.training_r2,

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


class CalibrationModelFitter:
    """
    Fit empirical calibration models using only calibration
    or training observations.

    This class only fits calibration models.

    Validation is handled separately so SensorQA cannot
    accidentally evaluate calibration performance on the same
    data used to fit the correction.
    """

    def fit(
        self,
        data: pd.DataFrame,
        *,
        measurement_column: str,
        reference_column: str,
        config: (
            CalibrationFitConfig
            | None
        ) = None,
    ) -> CalibrationFitResult:
        """Fit a calibration model from paired sensor and
        reference measurements.
        
        Args:
            data: Input data to process.
            measurement_column: Column containing sensor measurements.
            reference_column: Column containing reference values.
            config: Configuration for the operation.
        
        Returns:
            CalibrationFitResult returned by the function.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if (
            not isinstance(
                measurement_column,
                str,
            )
            or not measurement_column
        ):
            raise ValueError(
                "measurement_column must be "
                "a non-empty string."
            )

        if (
            not isinstance(
                reference_column,
                str,
            )
            or not reference_column
        ):
            raise ValueError(
                "reference_column must be "
                "a non-empty string."
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
                "Calibration data is missing required "
                "column(s): "
                + ", ".join(
                    missing_columns
                )
            )

        resolved_config = (
            config
            if config is not None
            else CalibrationFitConfig()
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

        rows_received = len(
            data
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
                "Calibration data contains non-finite "
                "or non-numeric measurement/reference "
                "pairs and drop_nonfinite=False."
            )

        if (
            rows_used
            < resolved_config.minimum_samples
        ):
            raise ValueError(
                "Not enough valid paired observations "
                "to fit calibration: "
                f"{rows_used} available, "
                f"{resolved_config.minimum_samples} "
                "required."
            )

        x = measurement[
            finite_mask
        ]

        y = reference[
            finite_mask
        ]

        warnings: list[str] = []

        if rows_excluded:

            warnings.append(
                f"Excluded {rows_excluded} row(s) "
                "containing non-finite or non-numeric "
                "calibration pairs."
            )

        # Offset-only calibration

        if (
            resolved_config.method
            == CalibrationMethod.OFFSET
        ):

            gain = 1.0

            offset = float(
                np.mean(
                    y - x
                )
            )

        # Gain + offset calibration

        elif (
            resolved_config.method
            == CalibrationMethod.AFFINE
        ):

            x_variance = float(
                np.var(
                    x
                )
            )

            if (
                not np.isfinite(
                    x_variance
                )
                or np.isclose(
                    x_variance,
                    0.0,
                    rtol=0.0,
                    atol=1e-15,
                )
            ):
                raise ValueError(
                    "AFFINE calibration requires "
                    "variation in the measurement column; "
                    "all usable measurements are "
                    "effectively constant."
                )

            design = np.column_stack(
                (
                    x,
                    np.ones_like(
                        x
                    ),
                )
            )

            (
                coefficients,
                _,
                rank,
                _,
            ) = np.linalg.lstsq(
                design,
                y,
                rcond=None,
            )

            if rank < 2:
                raise ValueError(
                    "AFFINE calibration could not "
                    "determine independent gain and "
                    "offset coefficients."
                )

            gain = float(
                coefficients[
                    0
                ]
            )

            offset = float(
                coefficients[
                    1
                ]
            )

        else:
            raise ValueError(
                "Unsupported calibration method: "
                f"{resolved_config.method!r}"
            )

        if not (
            np.isfinite(
                gain
            )
            and np.isfinite(
                offset
            )
        ):
            raise ValueError(
                "Calibration fitting produced "
                "non-finite coefficients."
            )

        predicted = (
            gain * x
            + offset
        )

        residual = (
            predicted
            - y
        )

        training_rmse = float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        )

        training_mae = float(
            np.mean(
                np.abs(
                    residual
                )
            )
        )

        training_bias = float(
            np.mean(
                residual
            )
        )

        training_r2 = (
            self._calculate_r2(
                actual=y,
                predicted=predicted,
            )
        )

        if training_r2 is None:

            warnings.append(
                "Training R² is undefined because "
                "the reference values have effectively "
                "zero variance."
            )

        model = CalibrationModel(
            method=(
                resolved_config.method
            ),

            gain=gain,

            offset=offset,

            measurement_column=(
                measurement_column
            ),

            reference_column=(
                reference_column
            ),

            training_samples=(
                rows_used
            ),
        )

        return CalibrationFitResult(
            model=model,

            training_rmse=(
                training_rmse
            ),

            training_mae=(
                training_mae
            ),

            training_bias=(
                training_bias
            ),

            training_r2=(
                training_r2
            ),

            rows_received=(
                rows_received
            ),

            rows_used=(
                rows_used
            ),

            rows_excluded=(
                rows_excluded
            ),

            warnings=(
                warnings
            ),
        )

    @staticmethod
    def _calculate_r2(
        *,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float | None:
        """Calculate R² when mathematically defined.
        
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


def fit_calibration_model(
    data: pd.DataFrame,
    *,
    measurement_column: str,
    reference_column: str,
    config: (
        CalibrationFitConfig
        | None
    ) = None,
) -> CalibrationFitResult:
    """Convenience function for fitting one calibration model.
    
    Args:
        data: Input data to process.
        measurement_column: Column containing sensor measurements.
        reference_column: Column containing reference values.
        config: Configuration for the operation.
    
    Returns:
        CalibrationFitResult returned by the function.
    """

    return (
        CalibrationModelFitter()
        .fit(
            data=data,

            measurement_column=(
                measurement_column
            ),

            reference_column=(
                reference_column
            ),

            config=config,
        )
    )
