#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd

from sensorqa.core.result_schema import (
    AnalysisResult,
    EvidenceStrength,
    ExecutionStatus,
)
from sensorqa.core.tool_contract import (
    BaseAnalysisTool,
    ParameterType,
    SensorType,
    ToolCategory,
    ToolContext,
    ToolMetadata,
    ToolParameter,
)


class GravityErrorTool(BaseAnalysisTool):
    """
    Characterizes static accelerometer consistency with gravity.

    The tool can operate in two modes:

    1. Unlabeled stationary data
       Calculates gravity-magnitude consistency without assuming
       a known sensor orientation.

    2. Labeled static orientations
       In addition to magnitude statistics, compares the mean
       acceleration vector for each recognized orientation with
       its expected gravity vector.

    Recognized orientation labels:

        +X
        -X
        +Y
        -Y
        +Z
        -Z

    Important:
        Angular disagreement between a measured and expected
        gravity vector is reported as an angular deviation.

        SensorQA does NOT automatically interpret that value as:
            - mechanical misalignment
            - sensor nonorthogonality
            - cross-axis sensitivity
    """

    STANDARD_GRAVITY = 9.80665

    @property
    def metadata(self) -> ToolMetadata:

        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="imu_gravity_error",
            name="Gravity Consistency",
            version="1.0.0",
            description=(
                "Evaluates static accelerometer gravity-magnitude "
                "consistency and, when static orientation labels are "
                "available, compares measured mean acceleration "
                "vectors with expected gravity vectors."
            ),
            category=ToolCategory.IMU,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.ACCELEROMETER,
            ],
            required_columns=[
                "ax",
                "ay",
                "az",
            ],
            optional_columns=[
                "orientation_label",
            ],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=100,
                    description=(
                        "Minimum simultaneous valid XYZ samples "
                        "required for overall gravity analysis."
                    ),
                    minimum=10,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_samples_per_orientation",
                    parameter_type=ParameterType.INTEGER,
                    default=50,
                    description=(
                        "Minimum simultaneous valid XYZ samples "
                        "required before a labeled orientation is "
                        "analyzed separately."
                    ),
                    minimum=5,
                    maximum=10_000_000,
                ),
                ToolParameter(
                    name="gravity_magnitude",
                    parameter_type=ParameterType.FLOAT,
                    default=self.STANDARD_GRAVITY,
                    description=(
                        "Expected gravity magnitude."
                    ),
                    unit="m/s^2",
                    minimum=1.0,
                    maximum=20.0,
                ),
                ToolParameter(
                    name="magnitude_warning_percent",
                    parameter_type=ParameterType.FLOAT,
                    default=2.0,
                    description=(
                        "Highlight gravity-magnitude disagreement "
                        "larger than this percentage."
                    ),
                    unit="%",
                    minimum=0.0,
                    maximum=100.0,
                ),
                ToolParameter(
                    name="orientation_angle_warning_deg",
                    parameter_type=ParameterType.FLOAT,
                    default=5.0,
                    description=(
                        "Highlight labeled static orientations whose "
                        "measured mean acceleration vector differs "
                        "from the expected vector by more than this "
                        "angle."
                    ),
                    unit="deg",
                    minimum=0.0,
                    maximum=180.0,
                ),
            ],
            dependencies=[],
            author="SensorQA",
        )

    def validate(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ):
        """Verify that static gravity characterization is supported.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Result returned by the function.
        """

        validation = super().validate(
            data=data,
            context=context,
        )

        if not validation.valid:
            return validation

        minimum_samples = int(
            context.parameters.get(
                "minimum_samples",
                100,
            )
        )

        matrix, _ = self._valid_acceleration_data(
            data=data,
            context=context,
        )

        if len(matrix) < minimum_samples:

            validation.errors.append(
                "Only "
                f"{len(matrix)} simultaneous valid AX/AY/AZ "
                f"samples are available. At least "
                f"{minimum_samples} are required."
            )

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is False:

            validation.errors.append(
                "Gravity-consistency analysis requires stationary "
                "or controlled static-orientation data, but metadata "
                "identifies this experiment as dynamic."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm from metadata that the "
                "accelerometer was stationary. Gravity-based results "
                "should therefore be interpreted cautiously."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform static gravity-consistency analysis.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Analysis result containing metrics and messages.
        """

        result = AnalysisResult(
            tool_id=self.metadata.tool_id,
            tool_name=self.metadata.name,
            tool_version=self.metadata.version,
            status=ExecutionStatus.SUCCESS,
        )

        gravity = float(
            context.parameters[
                "gravity_magnitude"
            ]
        )

        magnitude_warning_percent = float(
            context.parameters[
                "magnitude_warning_percent"
            ]
        )

        angle_warning_deg = float(
            context.parameters[
                "orientation_angle_warning_deg"
            ]
        )

        minimum_samples_per_orientation = int(
            context.parameters[
                "minimum_samples_per_orientation"
            ]
        )

        acceleration_unit = context.units.get(
            "ax",
            "m/s^2",
        )

        acceleration_matrix, valid_row_mask = (
            self._valid_acceleration_data(
                data=data,
                context=context,
            )
        )

        # Per-sample gravity magnitude
        #
        # This remains meaningful even when the dataset contains
        # several different static orientations.

        magnitudes = np.linalg.norm(
            acceleration_matrix,
            axis=1,
        )

        magnitude_error = (
            magnitudes
            - gravity
        )

        absolute_magnitude_error = np.abs(
            magnitude_error
        )

        mean_magnitude = float(
            np.mean(
                magnitudes
            )
        )

        magnitude_std = float(
            np.std(
                magnitudes,
                ddof=1,
            )
        )

        mean_signed_error = float(
            np.mean(
                magnitude_error
            )
        )

        mean_absolute_error = float(
            np.mean(
                absolute_magnitude_error
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    magnitude_error ** 2
                )
            )
        )

        maximum_absolute_error = float(
            np.max(
                absolute_magnitude_error
            )
        )

        mean_error_percent = (
            mean_signed_error
            / gravity
            * 100.0
        )

        mean_absolute_error_percent = (
            mean_absolute_error
            / gravity
            * 100.0
        )

        rmse_percent = (
            rmse
            / gravity
            * 100.0
        )

        maximum_absolute_error_percent = (
            maximum_absolute_error
            / gravity
            * 100.0
        )

        # Global metrics

        result.add_metric(
            name="Gravity Sample Count",
            value=len(
                magnitudes
            ),
            unit=None,
        )

        result.add_metric(
            name="Mean Acceleration Magnitude",
            value=mean_magnitude,
            unit=acceleration_unit,
            description=(
                "Mean magnitude of the simultaneous three-axis "
                "acceleration measurements."
            ),
        )

        result.add_metric(
            name="Acceleration Magnitude Standard Deviation",
            value=magnitude_std,
            unit=acceleration_unit,
            description=(
                "Variation of measured acceleration-vector "
                "magnitude during the static experiment."
            ),
        )

        result.add_metric(
            name="Mean Gravity Magnitude Error",
            value=mean_signed_error,
            unit=acceleration_unit,
            description=(
                "Mean signed difference between measured "
                "acceleration magnitude and configured gravity."
            ),
        )

        result.add_metric(
            name="Mean Gravity Magnitude Error Percent",
            value=mean_error_percent,
            unit="%",
        )

        result.add_metric(
            name="Gravity Magnitude MAE",
            value=mean_absolute_error,
            unit=acceleration_unit,
        )

        result.add_metric(
            name="Gravity Magnitude MAE Percent",
            value=mean_absolute_error_percent,
            unit="%",
        )

        result.add_metric(
            name="Gravity Magnitude RMSE",
            value=rmse,
            unit=acceleration_unit,
        )

        result.add_metric(
            name="Gravity Magnitude RMSE Percent",
            value=rmse_percent,
            unit="%",
        )

        result.add_metric(
            name="Maximum Absolute Gravity Magnitude Error",
            value=maximum_absolute_error,
            unit=acceleration_unit,
        )

        result.add_metric(
            name=(
                "Maximum Absolute Gravity Magnitude Error Percent"
            ),
            value=maximum_absolute_error_percent,
            unit="%",
        )

        if (
            mean_absolute_error_percent
            > magnitude_warning_percent
        ):

            result.add_warning(
                "Mean absolute gravity-magnitude error exceeds "
                "the configured analysis threshold. "
                f"Observed = {mean_absolute_error_percent:.3f}%, "
                f"threshold = {magnitude_warning_percent:.3f}%."
            )

            result.add_evidence(
                statement=(
                    "Static acceleration-vector magnitude differs "
                    "materially from the configured gravity "
                    "magnitude over the observed data."
                ),
                strength=EvidenceStrength.MODERATE,
                supporting_metrics=[
                    "Gravity Magnitude MAE Percent",
                    "Gravity Magnitude RMSE Percent",
                ],
            )

        # Orientation labels

        orientation_labels = self._orientation_labels(
            data=data,
            context=context,
            valid_row_mask=valid_row_mask,
        )

        orientation_results = {}

        recognized_orientation_count = 0
        analyzed_orientation_count = 0

        if orientation_labels is not None:

            unique_labels = list(
                pd.unique(
                    orientation_labels
                )
            )

            for raw_label in unique_labels:

                if raw_label is None:
                    continue

                label = str(
                    raw_label
                ).strip()

                if not label:
                    continue

                label_mask = (
                    orientation_labels
                    == raw_label
                )

                orientation_data = (
                    acceleration_matrix[
                        label_mask
                    ]
                )

                sample_count = len(
                    orientation_data
                )

                if (
                    sample_count
                    < minimum_samples_per_orientation
                ):

                    result.add_warning(
                        f"Orientation '{label}' contains only "
                        f"{sample_count} valid samples and was "
                        "not analyzed separately."
                    )

                    continue

                analyzed_orientation_count += 1

                expected_vector = (
                    self._expected_gravity_vector(
                        orientation_label=label,
                        gravity=gravity,
                    )
                )

                mean_vector = np.mean(
                    orientation_data,
                    axis=0,
                )

                mean_vector_magnitude = float(
                    np.linalg.norm(
                        mean_vector
                    )
                )

                orientation_magnitudes = np.linalg.norm(
                    orientation_data,
                    axis=1,
                )

                mean_sample_magnitude = float(
                    np.mean(
                        orientation_magnitudes
                    )
                )

                orientation_magnitude_error = (
                    mean_vector_magnitude
                    - gravity
                )

                orientation_magnitude_error_percent = (
                    orientation_magnitude_error
                    / gravity
                    * 100.0
                )

                orientation_entry = {
                    "label":
                        label,

                    "sample_count":
                        sample_count,

                    "mean_vector": {
                        "ax":
                            float(
                                mean_vector[
                                    0
                                ]
                            ),

                        "ay":
                            float(
                                mean_vector[
                                    1
                                ]
                            ),

                        "az":
                            float(
                                mean_vector[
                                    2
                                ]
                            ),
                    },

                    "mean_vector_magnitude":
                        mean_vector_magnitude,

                    "mean_sample_magnitude":
                        mean_sample_magnitude,

                    "magnitude_error":
                        orientation_magnitude_error,

                    "magnitude_error_percent":
                        orientation_magnitude_error_percent,

                    "recognized":
                        expected_vector is not None,
                }

                display_label = self._display_label(
                    label
                )

                # Magnitude metrics remain valid even if the label
                # itself is not recognized.

                result.add_metric(
                    name=(
                        f"{display_label} Mean Vector Magnitude"
                    ),
                    value=mean_vector_magnitude,
                    unit=acceleration_unit,
                )

                result.add_metric(
                    name=(
                        f"{display_label} Gravity Magnitude "
                        "Error Percent"
                    ),
                    value=orientation_magnitude_error_percent,
                    unit="%",
                )

                # Expected-vector comparison

                if expected_vector is not None:

                    recognized_orientation_count += 1

                    vector_error = (
                        mean_vector
                        - expected_vector
                    )

                    vector_error_magnitude = float(
                        np.linalg.norm(
                            vector_error
                        )
                    )

                    angle_error_deg = (
                        self._vector_angle_degrees(
                            measured=mean_vector,
                            expected=expected_vector,
                        )
                    )

                    orientation_entry[
                        "expected_vector"
                    ] = {
                        "ax":
                            float(
                                expected_vector[
                                    0
                                ]
                            ),

                        "ay":
                            float(
                                expected_vector[
                                    1
                                ]
                            ),

                        "az":
                            float(
                                expected_vector[
                                    2
                                ]
                            ),
                    }

                    orientation_entry[
                        "vector_error"
                    ] = {
                        "ax":
                            float(
                                vector_error[
                                    0
                                ]
                            ),

                        "ay":
                            float(
                                vector_error[
                                    1
                                ]
                            ),

                        "az":
                            float(
                                vector_error[
                                    2
                                ]
                            ),
                    }

                    orientation_entry[
                        "vector_error_magnitude"
                    ] = (
                        vector_error_magnitude
                    )

                    orientation_entry[
                        "angular_deviation_deg"
                    ] = (
                        angle_error_deg
                    )

                    result.add_metric(
                        name=(
                            f"{display_label} Gravity Vector "
                            "Error Magnitude"
                        ),
                        value=vector_error_magnitude,
                        unit=acceleration_unit,
                        description=(
                            "Magnitude of the difference between "
                            "the measured mean acceleration vector "
                            "and the expected static gravity vector."
                        ),
                    )

                    if angle_error_deg is not None:

                        result.add_metric(
                            name=(
                                f"{display_label} Gravity Vector "
                                "Angular Deviation"
                            ),
                            value=angle_error_deg,
                            unit="deg",
                            description=(
                                "Angle between the measured mean "
                                "acceleration vector and the "
                                "expected gravity vector. This is "
                                "not automatically interpreted as "
                                "mechanical misalignment."
                            ),
                        )

                        if (
                            angle_error_deg
                            > angle_warning_deg
                        ):

                            result.add_warning(
                                f"Orientation '{label}' has a "
                                "measured gravity-vector angular "
                                f"deviation of "
                                f"{angle_error_deg:.3f} deg."
                            )

                            result.add_evidence(
                                statement=(
                                    f"Static orientation '{label}' "
                                    "shows a measurable angular "
                                    "difference between the "
                                    "observed mean acceleration "
                                    "vector and its expected gravity "
                                    "vector."
                                ),
                                strength=EvidenceStrength.MODERATE,
                                supporting_metrics=[
                                    (
                                        f"{display_label} "
                                        "Gravity Vector Angular "
                                        "Deviation"
                                    ),
                                    (
                                        f"{display_label} "
                                        "Gravity Vector Error "
                                        "Magnitude"
                                    ),
                                ],
                            )

                else:

                    result.add_warning(
                        f"Orientation label '{label}' is not one of "
                        "the recognized SensorQA static labels "
                        "(+X, -X, +Y, -Y, +Z, -Z). "
                        "Magnitude statistics were retained, but "
                        "expected-vector comparison was skipped."
                    )

                orientation_results[
                    label
                ] = orientation_entry

        else:

            result.add_message(
                "No orientation labels are available. SensorQA "
                "performed gravity-magnitude analysis only."
            )

        # Orientation coverage

        result.add_metric(
            name="Analyzed Static Orientation Count",
            value=analyzed_orientation_count,
            unit=None,
        )

        result.add_metric(
            name="Recognized Static Orientation Count",
            value=recognized_orientation_count,
            unit=None,
        )

        recognized_labels = {
            self._canonical_orientation_label(
                label
            )
            for label, entry
            in orientation_results.items()
            if entry.get(
                "recognized"
            )
        }

        recognized_labels.discard(
            None
        )

        canonical_six = {
            "+X",
            "-X",
            "+Y",
            "-Y",
            "+Z",
            "-Z",
        }

        missing_static_orientations = sorted(
            canonical_six
            - recognized_labels
        )

        if recognized_labels:

            result.add_message(
                "Recognized static orientations: "
                + ", ".join(
                    sorted(
                        recognized_labels
                    )
                )
            )

        if (
            recognized_orientation_count > 0
            and missing_static_orientations
        ):

            result.add_message(
                "Static orientations not represented in this "
                "dataset: "
                + ", ".join(
                    missing_static_orientations
                )
            )

        if recognized_labels == canonical_six:

            result.add_evidence(
                statement=(
                    "The dataset contains all six canonical static "
                    "accelerometer orientations, providing suitable "
                    "orientation coverage for a later dedicated "
                    "six-position calibration analysis."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[
                    "Recognized Static Orientation Count",
                ],
            )

        # Stationarity

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is True:

            result.add_evidence(
                statement=(
                    "Experiment metadata supports stationary/static "
                    "interpretation of the gravity measurements."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[
                    "Gravity Magnitude MAE Percent",
                    "Gravity Magnitude RMSE Percent",
                ],
            )

        else:

            result.add_warning(
                "Stationarity was not confirmed from metadata. "
                "Gravity-consistency results should therefore "
                "be interpreted cautiously."
            )

        # Reproducibility metadata

        result.metadata.update(
            {
                "gravity_magnitude":
                    gravity,

                "acceleration_unit":
                    acceleration_unit,

                "minimum_samples_per_orientation":
                    minimum_samples_per_orientation,

                "magnitude_warning_percent":
                    magnitude_warning_percent,

                "orientation_angle_warning_deg":
                    angle_warning_deg,

                "stationary_confirmed":
                    stationary_state,

                "orientation_results":
                    orientation_results,

                "recognized_orientation_count":
                    recognized_orientation_count,

                "recognized_orientations":
                    sorted(
                        recognized_labels
                    ),

                "missing_canonical_orientations":
                    missing_static_orientations,

                "mechanical_misalignment_inferred":
                    False,
            }
        )

        return result

    # Simultaneous XYZ data

    @staticmethod
    def _valid_acceleration_data(
        data: pd.DataFrame,
        context: ToolContext,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """Return:
            N x 3 simultaneous valid acceleration samples
            original-row Boolean mask
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Tuple containing the result values.
        """

        arrays = []

        for axis in (
            "ax",
            "ay",
            "az",
        ):

            column = context.column_mapping[
                axis
            ]

            arrays.append(
                pd.to_numeric(
                    data[column],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
            )

        complete = np.column_stack(
            arrays
        )

        valid_mask = np.all(
            np.isfinite(
                complete
            ),
            axis=1,
        )

        return (
            complete[
                valid_mask
            ],
            valid_mask,
        )

    # Orientation labels

    @staticmethod
    def _orientation_labels(
        data: pd.DataFrame,
        context: ToolContext,
        valid_row_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Return orientation labels corresponding exactly to the
        simultaneous valid XYZ rows.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
            valid_row_mask: Value for `valid_row_mask`.
        
        Returns:
            np.ndarray | None returned by the function.
        """

        column = context.column_mapping.get(
            "orientation_label"
        )

        if (
            column is None
            or column not in data.columns
        ):

            return None

        labels = data[
            column
        ].to_numpy(
            dtype=object
        )

        return labels[
            valid_row_mask
        ]

    # Gravity-vector definitions

    @classmethod
    def _expected_gravity_vector(
        cls,
        orientation_label: str,
        gravity: float,
    ) -> np.ndarray | None:

        """Calculate expected gravity vector.
        
        Args:
            orientation_label: Orientation label used by this function.
            gravity: Gravity used by this function.
        
        Returns:
            np.ndarray | None returned by this function.
        """
        canonical = (
            cls._canonical_orientation_label(
                orientation_label
            )
        )

        if canonical == "+X":

            return np.array(
                [
                    gravity,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        if canonical == "-X":

            return np.array(
                [
                    -gravity,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )

        if canonical == "+Y":

            return np.array(
                [
                    0.0,
                    gravity,
                    0.0,
                ],
                dtype=float,
            )

        if canonical == "-Y":

            return np.array(
                [
                    0.0,
                    -gravity,
                    0.0,
                ],
                dtype=float,
            )

        if canonical == "+Z":

            return np.array(
                [
                    0.0,
                    0.0,
                    gravity,
                ],
                dtype=float,
            )

        if canonical == "-Z":

            return np.array(
                [
                    0.0,
                    0.0,
                    -gravity,
                ],
                dtype=float,
            )

        return None

    @staticmethod
    def _canonical_orientation_label(
        label: str,
    ) -> str | None:
        """Convert common explicit label variants into one of:
        
            +X, -X, +Y, -Y, +Z, -Z
        
        Args:
            label: Label shown to the user.
        
        Returns:
            str | None returned by the function.
        """

        normalized = (
            str(
                label
            )
            .strip()
            .lower()
            .replace(
                " ",
                "",
            )
            .replace(
                "_",
                "",
            )
        )

        aliases = {
            "+x": "+X",
            "x+": "+X",
            "plusx": "+X",
            "positivex": "+X",

            "-x": "-X",
            "x-": "-X",
            "minusx": "-X",
            "negativex": "-X",

            "+y": "+Y",
            "y+": "+Y",
            "plusy": "+Y",
            "positivey": "+Y",

            "-y": "-Y",
            "y-": "-Y",
            "minusy": "-Y",
            "negativey": "-Y",

            "+z": "+Z",
            "z+": "+Z",
            "plusz": "+Z",
            "positivez": "+Z",

            "-z": "-Z",
            "z-": "-Z",
            "minusz": "-Z",
            "negativez": "-Z",
        }

        return aliases.get(
            normalized
        )

    @classmethod
    def _display_label(
        cls,
        label: str,
    ) -> str:
        """Produce a stable label for metric names.
        
        Args:
            label: Label shown to the user.
        
        Returns:
            String representation.
        """

        canonical = (
            cls._canonical_orientation_label(
                label
            )
        )

        if canonical is not None:
            return canonical

        cleaned = (
            str(
                label
            )
            .strip()
        )

        return (
            cleaned
            if cleaned
            else "Unknown Orientation"
        )

    # Vector-angle calculation

    @staticmethod
    def _vector_angle_degrees(
        measured: np.ndarray,
        expected: np.ndarray,
    ) -> float | None:
        """Calculate angle between two vectors.
        
        Returns None if either vector has zero magnitude.
        
        Args:
            measured: Value for `measured`.
            expected: Value for `expected`.
        
        Returns:
            Numeric result.
        """

        measured_norm = float(
            np.linalg.norm(
                measured
            )
        )

        expected_norm = float(
            np.linalg.norm(
                expected
            )
        )

        if (
            measured_norm <= 0.0
            or expected_norm <= 0.0
        ):

            return None

        cosine = float(
            np.dot(
                measured,
                expected,
            )
            / (
                measured_norm
                * expected_norm
            )
        )

        cosine = float(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )

        return float(
            np.degrees(
                np.arccos(
                    cosine
                )
            )
        )

    # Stationarity metadata

    @staticmethod
    def _stationary_state(
        context: ToolContext,
    ) -> bool | None:

        """Read the stationary-state metadata used by this analysis.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            bool | None returned by this function.
        """
        metadata = context.metadata.get(
            "sensorqa_metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        test_metadata = metadata.get(
            "test"
        )

        if not isinstance(
            test_metadata,
            dict,
        ):
            return None

        expected_stationary = (
            test_metadata.get(
                "expected_stationary"
            )
        )

        if expected_stationary is True:
            return True

        if expected_stationary is False:
            return False

        test_mode = test_metadata.get(
            "test_mode"
        )

        if hasattr(
            test_mode,
            "value",
        ):
            test_mode = (
                test_mode.value
            )

        if test_mode in {
            "imu_stationary",
            "imu_controlled_orientation",
        }:
            return True

        if test_mode == "imu_dynamic":
            return False

        return None
