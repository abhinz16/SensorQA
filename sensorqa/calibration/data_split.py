#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from sensorqa.ingestion.dataset import SensorQADataset


class SplitStrategy(str, Enum):
    """
    Strategies supported for held-out calibration evaluation.

    RANDOM
        Randomly assigns individual rows to training or validation.
        This is appropriate only when rows can reasonably be treated as
        independent observations.

    SEQUENTIAL
        Uses the earlier rows for training and the final rows for
        validation. This is usually safer for time-ordered sensor data
        because future observations cannot leak into calibration fitting.

    GROUPED
        Keeps all rows sharing one group value entirely in either the
        training or validation subset. This is useful for repeated runs,
        devices, samples, subjects, operating points, or experiments.
    """

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    GROUPED = "grouped"


@dataclass(frozen=True)
class DataSplitConfig:
    """
    Configuration for a held-out calibration split.

    SensorQA separates calibration fitting data from
    validation data. A calibration model must never be evaluated on the
    same observations used to fit it.
    """

    validation_fraction: float = 0.20
    strategy: SplitStrategy = SplitStrategy.SEQUENTIAL
    random_seed: int | None = 42
    group_column: str | None = None

    minimum_train_rows: int = 2
    minimum_validation_rows: int = 1

    def __post_init__(self) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.validation_fraction,
            Real,
        ):
            raise TypeError(
                "validation_fraction must be a real number."
            )

        if not (
            0.0
            < float(self.validation_fraction)
            < 1.0
        ):
            raise ValueError(
                "validation_fraction must be strictly between 0 and 1."
            )

        if not isinstance(
            self.strategy,
            SplitStrategy,
        ):
            raise TypeError(
                "strategy must be a SplitStrategy."
            )

        if (
            self.random_seed is not None
            and not isinstance(
                self.random_seed,
                Integral,
            )
        ):
            raise TypeError(
                "random_seed must be an integer or None."
            )

        if not isinstance(
            self.minimum_train_rows,
            Integral,
        ):
            raise TypeError(
                "minimum_train_rows must be an integer."
            )

        if not isinstance(
            self.minimum_validation_rows,
            Integral,
        ):
            raise TypeError(
                "minimum_validation_rows must be an integer."
            )

        if self.minimum_train_rows < 1:
            raise ValueError(
                "minimum_train_rows must be at least 1."
            )

        if self.minimum_validation_rows < 1:
            raise ValueError(
                "minimum_validation_rows must be at least 1."
            )

        if (
            self.strategy
            == SplitStrategy.GROUPED
        ):

            if (
                not isinstance(
                    self.group_column,
                    str,
                )
                or not self.group_column.strip()
            ):
                raise ValueError(
                    "group_column is required for GROUPED splitting."
                )

        elif self.group_column is not None:

            raise ValueError(
                "group_column may only be supplied for GROUPED splitting."
            )


@dataclass
class DataSplitResult:
    """
    Result of one held-out calibration split.

    Positions refer to zero-based row positions in the original input,
    rather than DataFrame index labels. This avoids ambiguity when an
    uploaded dataset contains duplicate index values.
    """

    training_data: pd.DataFrame
    validation_data: pd.DataFrame

    training_positions: tuple[int, ...]
    validation_positions: tuple[int, ...]

    strategy: SplitStrategy

    requested_validation_fraction: float

    random_seed: int | None = None
    group_column: str | None = None

    training_groups: tuple[Any, ...] = field(
        default_factory=tuple
    )

    validation_groups: tuple[Any, ...] = field(
        default_factory=tuple
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:

        """Validate fields after initialization.
        
        Returns:
            None.
        """
        if not isinstance(
            self.training_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "training_data must be a pandas DataFrame."
            )

        if not isinstance(
            self.validation_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "validation_data must be a pandas DataFrame."
            )

        train_positions = set(
            self.training_positions
        )

        validation_positions = set(
            self.validation_positions
        )

        if (
            train_positions
            & validation_positions
        ):
            raise ValueError(
                "Training and validation positions must not overlap."
            )

        if (
            len(self.training_positions)
            != len(self.training_data)
        ):
            raise ValueError(
                "training_positions length must match "
                "training_data rows."
            )

        if (
            len(self.validation_positions)
            != len(self.validation_data)
        ):
            raise ValueError(
                "validation_positions length must match "
                "validation_data rows."
            )

        if (
            list(
                self.training_data.columns
            )
            != list(
                self.validation_data.columns
            )
        ):
            raise ValueError(
                "Training and validation data must contain "
                "identical columns."
            )

    @property
    def training_row_count(
        self,
    ) -> int:

        """Return training row count.
        
        Returns:
            Calculated value.
        """
        return len(
            self.training_data
        )

    @property
    def validation_row_count(
        self,
    ) -> int:

        """Return validation row count.
        
        Returns:
            Calculated value.
        """
        return len(
            self.validation_data
        )

    @property
    def total_row_count(
        self,
    ) -> int:

        """Return total row count.
        
        Returns:
            Calculated value.
        """
        return (
            self.training_row_count
            + self.validation_row_count
        )

    @property
    def actual_validation_fraction(
        self,
    ) -> float:

        """Return actual validation fraction.
        
        Returns:
            Calculated value.
        """
        if self.total_row_count == 0:
            return 0.0

        return (
            self.validation_row_count
            / self.total_row_count
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        """Lightweight serializable summary for the UI and reports.
        
        Returns:
            Dictionary containing the result values.
        """

        return {
            "strategy":
                self.strategy.value,

            "training_rows":
                self.training_row_count,

            "validation_rows":
                self.validation_row_count,

            "total_rows":
                self.total_row_count,

            "requested_validation_fraction":
                self.requested_validation_fraction,

            "actual_validation_fraction":
                self.actual_validation_fraction,

            "random_seed":
                self.random_seed,

            "group_column":
                self.group_column,

            "training_group_count":
                len(
                    self.training_groups
                ),

            "validation_group_count":
                len(
                    self.validation_groups
                ),

            "warnings":
                list(
                    self.warnings
                ),
        }


class CalibrationDataSplitter:
    """
    Creates leakage-resistant train/validation splits for calibration.

    This class performs only data partitioning.

    It does not:
        - fit calibration models
        - calculate coefficients
        - evaluate calibration accuracy

    Those responsibilities will remain separate so the future SensorQA
    application can clearly show:

        calibration data
        validation data
        fitted model
        held-out performance
    """

    def split(
        self,
        data: (
            pd.DataFrame
            | SensorQADataset
        ),
        config: (
            DataSplitConfig
            | None
        ) = None,
    ) -> DataSplitResult:
        """Split a DataFrame or SensorQADataset.
        
        The original input is never modified.
        
        Args:
            data: Input data to process.
            config: Configuration for the operation.
        
        Returns:
            DataSplitResult returned by the function.
        """

        resolved_config = (
            config
            if config is not None
            else DataSplitConfig()
        )

        frame = self._extract_frame(
            data
        )

        self._validate_frame(
            frame=frame,
            config=resolved_config,
        )

        if (
            resolved_config.strategy
            == SplitStrategy.SEQUENTIAL
        ):

            return self._split_sequential(
                frame=frame,
                config=resolved_config,
            )

        if (
            resolved_config.strategy
            == SplitStrategy.RANDOM
        ):

            return self._split_random(
                frame=frame,
                config=resolved_config,
            )

        if (
            resolved_config.strategy
            == SplitStrategy.GROUPED
        ):

            return self._split_grouped(
                frame=frame,
                config=resolved_config,
            )

        raise ValueError(
            "Unsupported split strategy: "
            f"{resolved_config.strategy!r}"
        )

    @staticmethod
    def _extract_frame(
        data: (
            pd.DataFrame
            | SensorQADataset
        ),
    ) -> pd.DataFrame:

        """Return extract frame.
        
        Args:
            data: Input data to process.
        
        Returns:
            DataFrame containing the requested data.
        """
        if isinstance(
            data,
            SensorQADataset,
        ):

            return (
                data.analysis_view()
            )

        if isinstance(
            data,
            pd.DataFrame,
        ):

            return data.copy()

        raise TypeError(
            "data must be a pandas DataFrame "
            "or SensorQADataset."
        )

    @staticmethod
    def _validate_frame(
        frame: pd.DataFrame,
        config: DataSplitConfig,
    ) -> None:

        """Validate frame.
        
        Args:
            frame: DataFrame to process.
            config: Configuration for the operation.
        
        Returns:
            None.
        """
        if frame.empty:

            raise ValueError(
                "Calibration splitting requires "
                "at least one data row."
            )

        minimum_required = (
            int(
                config.minimum_train_rows
            )
            + int(
                config.minimum_validation_rows
            )
        )

        if (
            len(frame)
            < minimum_required
        ):

            raise ValueError(
                "Dataset is too small for the requested "
                "calibration split: "
                f"{len(frame)} rows available, but at least "
                f"{minimum_required} are required."
            )

        if (
            config.strategy
            == SplitStrategy.GROUPED
        ):

            assert (
                config.group_column
                is not None
            )

            if (
                config.group_column
                not in frame.columns
            ):

                raise KeyError(
                    "Grouped calibration split requires "
                    f"column '{config.group_column}', "
                    "but it is not present."
                )

            if (
                frame[
                    config.group_column
                ]
                .isna()
                .any()
            ):

                raise ValueError(
                    "Grouped calibration split does not "
                    "allow missing values in group column "
                    f"'{config.group_column}'."
                )

            group_count = (
                frame[
                    config.group_column
                ]
                .nunique(
                    dropna=False
                )
            )

            if group_count < 2:

                raise ValueError(
                    "Grouped calibration splitting "
                    "requires at least two distinct groups."
                )

    @staticmethod
    def _desired_validation_rows(
        row_count: int,
        config: DataSplitConfig,
    ) -> int:

        """Return desired validation rows.
        
        Args:
            row_count: Row count used by this function.
            config: Configuration for the operation.
        
        Returns:
            Calculated value.
        """
        requested = int(
            round(
                row_count
                * float(
                    config.validation_fraction
                )
            )
        )

        validation_rows = max(
            requested,
            int(
                config.minimum_validation_rows
            ),
        )

        maximum_validation_rows = (
            row_count
            - int(
                config.minimum_train_rows
            )
        )

        validation_rows = min(
            validation_rows,
            maximum_validation_rows,
        )

        if (
            validation_rows
            < config.minimum_validation_rows
        ):

            raise ValueError(
                "Unable to create a validation subset "
                "satisfying the configured minimum "
                "row count."
            )

        return validation_rows

    def _split_sequential(
        self,
        frame: pd.DataFrame,
        config: DataSplitConfig,
    ) -> DataSplitResult:

        """Return split sequential.
        
        Args:
            frame: DataFrame to process.
            config: Configuration for the operation.
        
        Returns:
            DataSplitResult returned by this function.
        """
        validation_rows = (
            self._desired_validation_rows(
                row_count=len(
                    frame
                ),
                config=config,
            )
        )

        split_position = (
            len(frame)
            - validation_rows
        )

        training_positions = tuple(
            range(
                0,
                split_position,
            )
        )

        validation_positions = tuple(
            range(
                split_position,
                len(frame),
            )
        )

        return self._build_result(
            frame=frame,
            training_positions=(
                training_positions
            ),
            validation_positions=(
                validation_positions
            ),
            config=config,
        )

    def _split_random(
        self,
        frame: pd.DataFrame,
        config: DataSplitConfig,
    ) -> DataSplitResult:

        """Return split random.
        
        Args:
            frame: DataFrame to process.
            config: Configuration for the operation.
        
        Returns:
            DataSplitResult returned by this function.
        """
        validation_rows = (
            self._desired_validation_rows(
                row_count=len(
                    frame
                ),
                config=config,
            )
        )

        rng = (
            np.random.default_rng(
                config.random_seed
            )
        )

        shuffled_positions = (
            rng.permutation(
                len(frame)
            )
        )

        validation_positions = tuple(
            sorted(
                int(position)
                for position
                in shuffled_positions[
                    :validation_rows
                ]
            )
        )

        validation_position_set = set(
            validation_positions
        )

        training_positions = tuple(
            position
            for position
            in range(
                len(frame)
            )
            if (
                position
                not in validation_position_set
            )
        )

        return self._build_result(
            frame=frame,
            training_positions=(
                training_positions
            ),
            validation_positions=(
                validation_positions
            ),
            config=config,
        )

    def _split_grouped(
        self,
        frame: pd.DataFrame,
        config: DataSplitConfig,
    ) -> DataSplitResult:

        """Return split grouped.
        
        Args:
            frame: DataFrame to process.
            config: Configuration for the operation.
        
        Returns:
            DataSplitResult returned by this function.
        """
        assert (
            config.group_column
            is not None
        )

        target_validation_rows = (
            self._desired_validation_rows(
                row_count=len(
                    frame
                ),
                config=config,
            )
        )

        grouped_positions: dict[
            Any,
            list[int],
        ] = {}

        for (
            position,
            group_value,
        ) in enumerate(
            frame[
                config.group_column
            ].tolist()
        ):

            grouped_positions.setdefault(
                group_value,
                [],
            ).append(
                position
            )

        group_values = list(
            grouped_positions.keys()
        )

        rng = (
            np.random.default_rng(
                config.random_seed
            )
        )

        shuffled_indices = (
            rng.permutation(
                len(
                    group_values
                )
            )
        )

        candidate_groups = [
            group_values[
                int(index)
            ]
            for index
            in shuffled_indices
        ]

        selected_validation_groups: list[
            Any
        ] = []

        selected_validation_rows = 0

        minimum_train_rows = int(
            config.minimum_train_rows
        )

        for (
            group_value
        ) in candidate_groups:

            group_size = len(
                grouped_positions[
                    group_value
                ]
            )

            rows_after_selection = (
                len(frame)
                - (
                    selected_validation_rows
                    + group_size
                )
            )

            # Never consume so many rows that
            # training becomes invalid.
            if (
                rows_after_selection
                < minimum_train_rows
            ):
                continue

            selected_validation_groups.append(
                group_value
            )

            selected_validation_rows += (
                group_size
            )

            if (
                selected_validation_rows
                >= target_validation_rows
            ):
                break

        if (
            selected_validation_rows
            < config.minimum_validation_rows
        ):

            raise ValueError(
                "Unable to create a grouped validation "
                "subset while preserving the configured "
                "minimum training rows."
            )

        validation_group_set = set(
            selected_validation_groups
        )

        validation_positions = tuple(
            position
            for (
                position,
                group_value,
            ) in enumerate(
                frame[
                    config.group_column
                ].tolist()
            )
            if (
                group_value
                in validation_group_set
            )
        )

        validation_position_set = set(
            validation_positions
        )

        training_positions = tuple(
            position
            for position
            in range(
                len(frame)
            )
            if (
                position
                not in validation_position_set
            )
        )

        training_groups = tuple(
            group_value
            for group_value
            in group_values
            if (
                group_value
                not in validation_group_set
            )
        )

        validation_groups = tuple(
            group_value
            for group_value
            in group_values
            if (
                group_value
                in validation_group_set
            )
        )

        warnings: list[str] = []

        actual_fraction = (
            len(
                validation_positions
            )
            / len(frame)
        )

        requested_fraction = float(
            config.validation_fraction
        )

        # Group integrity has priority over matching
        # the requested fraction exactly.
        if (
            abs(
                actual_fraction
                - requested_fraction
            )
            > 0.05
        ):

            warnings.append(
                "Grouped splitting could not match the "
                "requested validation fraction closely "
                "because complete groups must remain together."
            )

        return self._build_result(
            frame=frame,
            training_positions=(
                training_positions
            ),
            validation_positions=(
                validation_positions
            ),
            config=config,
            training_groups=(
                training_groups
            ),
            validation_groups=(
                validation_groups
            ),
            warnings=warnings,
        )

    @staticmethod
    def _build_result(
        frame: pd.DataFrame,
        training_positions: tuple[
            int,
            ...,
        ],
        validation_positions: tuple[
            int,
            ...,
        ],
        config: DataSplitConfig,
        training_groups: tuple[
            Any,
            ...,
        ] = (),
        validation_groups: tuple[
            Any,
            ...,
        ] = (),
        warnings: (
            list[str]
            | None
        ) = None,
    ) -> DataSplitResult:

        """Build result.
        
        Args:
            frame: DataFrame to process.
            training_positions: Training positions used by this function.
            validation_positions: Validation positions used by this function.
            config: Configuration for the operation.
            training_groups: Training groups used by this function.
            validation_groups: Validation groups used by this function.
            warnings: Warnings used by this function.
        
        Returns:
            DataSplitResult returned by this function.
        """
        if (
            len(
                training_positions
            )
            < config.minimum_train_rows
        ):

            raise ValueError(
                "Training subset does not satisfy "
                "minimum_train_rows."
            )

        if (
            len(
                validation_positions
            )
            < config.minimum_validation_rows
        ):

            raise ValueError(
                "Validation subset does not satisfy "
                "minimum_validation_rows."
            )

        if (
            set(
                training_positions
            )
            & set(
                validation_positions
            )
        ):

            raise RuntimeError(
                "Internal split error: training "
                "and validation overlap."
            )

        if (
            len(
                training_positions
            )
            + len(
                validation_positions
            )
            != len(frame)
        ):

            raise RuntimeError(
                "Internal split error: not every "
                "source row was assigned."
            )

        training_data = (
            frame.iloc[
                list(
                    training_positions
                )
            ]
            .copy()
        )

        validation_data = (
            frame.iloc[
                list(
                    validation_positions
                )
            ]
            .copy()
        )

        return DataSplitResult(
            training_data=(
                training_data
            ),

            validation_data=(
                validation_data
            ),

            training_positions=(
                training_positions
            ),

            validation_positions=(
                validation_positions
            ),

            strategy=(
                config.strategy
            ),

            requested_validation_fraction=(
                float(
                    config.validation_fraction
                )
            ),

            random_seed=(
                config.random_seed
            ),

            group_column=(
                config.group_column
            ),

            training_groups=(
                training_groups
            ),

            validation_groups=(
                validation_groups
            ),

            warnings=(
                list(
                    warnings
                )
                if warnings is not None
                else []
            ),
        )


def split_calibration_data(
    data: (
        pd.DataFrame
        | SensorQADataset
    ),
    config: (
        DataSplitConfig
        | None
    ) = None,
) -> DataSplitResult:
    """Convenience function for one held-out calibration split.
    
    Args:
        data: Input data to process.
        config: Configuration for the operation.
    
    Returns:
        DataSplitResult returned by the function.
    """

    return (
        CalibrationDataSplitter()
        .split(
            data=data,
            config=config,
        )
    )
