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


class SixPositionCalibrationTool(BaseAnalysisTool):
    """
    Estimates a diagonal accelerometer calibration model from
    six static orientations:

        +X
        -X
        +Y
        -Y
        +Z
        -Z

    The model is:

        a_corrected = S @ (a_measured - b)

    where:

        b = [bx, by, bz]

    and S is diagonal:

        [sx  0   0 ]
        [0   sy  0 ]
        [0   0   sz]

    This V1 model estimates:

        - per-axis bias
        - raw per-axis sensitivity
        - sensitivity error
        - correction scale factor

    It deliberately does NOT estimate:

        - full cross-axis sensitivity matrix
        - nonorthogonality matrix
        - mechanical mounting angles
        - complete 3x3 affine calibration matrix

    Reported corrected residuals use the same six-position
    calibration dataset used to estimate the model. They are
    calibration-fit residuals, NOT independent validation results.
    """

    STANDARD_GRAVITY = 9.80665

    REQUIRED_ORIENTATIONS = (
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    )

    AXIS_CONFIGURATION = {
        "x": {
            "index": 0,
            "plus": "+X",
            "minus": "-X",
            "field": "ax",
            "label": "AX",
        },
        "y": {
            "index": 1,
            "plus": "+Y",
            "minus": "-Y",
            "field": "ay",
            "label": "AY",
        },
        "z": {
            "index": 2,
            "plus": "+Z",
            "minus": "-Z",
            "field": "az",
            "label": "AZ",
        },
    }

    @property
    def metadata(self) -> ToolMetadata:

        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="imu_six_position_calibration",
            name="Six-Position Accelerometer Calibration",
            version="1.0.0",
            description=(
                "Estimates per-axis accelerometer bias and diagonal "
                "scale correction using six labeled static gravity "
                "orientations."
            ),
            category=ToolCategory.CALIBRATION,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.ACCELEROMETER,
            ],
            required_columns=[
                "ax",
                "ay",
                "az",
                "orientation_label",
            ],
            optional_columns=[
                "temperature",
            ],
            parameters=[
                ToolParameter(
                    name="minimum_samples_per_orientation",
                    parameter_type=ParameterType.INTEGER,
                    default=50,
                    description=(
                        "Minimum number of simultaneous valid XYZ "
                        "samples required for each of the six "
                        "static orientations."
                    ),
                    minimum=5,
                    maximum=10_000_000,
                ),
                ToolParameter(
                    name="gravity_magnitude",
                    parameter_type=ParameterType.FLOAT,
                    default=self.STANDARD_GRAVITY,
                    description=(
                        "Reference gravity magnitude used for "
                        "six-position calibration."
                    ),
                    unit="m/s^2",
                    minimum=1.0,
                    maximum=20.0,
                ),
                ToolParameter(
                    name="minimum_primary_axis_span_fraction",
                    parameter_type=ParameterType.FLOAT,
                    default=0.50,
                    description=(
                        "Warn when the measured +axis to -axis span "
                        "is smaller than this fraction of the ideal "
                        "2g span."
                    ),
                    minimum=0.01,
                    maximum=1.0,
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
        """Verify that all six required static orientations are present
        with sufficient simultaneous valid XYZ samples.
        
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

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is False:

            validation.errors.append(
                "Six-position accelerometer calibration requires "
                "static orientation data, but experiment metadata "
                "identifies this dataset as dynamic."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm from metadata that this "
                "is a controlled static-orientation experiment."
            )

        orientation_groups = self._orientation_groups(
            data=data,
            context=context,
        )

        minimum_samples = int(
            context.parameters.get(
                "minimum_samples_per_orientation",
                50,
            )
        )

        missing_orientations = [
            orientation
            for orientation in self.REQUIRED_ORIENTATIONS
            if orientation not in orientation_groups
        ]

        if missing_orientations:

            validation.errors.append(
                "Six-position calibration requires all six "
                "canonical orientations. Missing: "
                + ", ".join(
                    missing_orientations
                )
            )

        for orientation in self.REQUIRED_ORIENTATIONS:

            if orientation not in orientation_groups:
                continue

            sample_count = len(
                orientation_groups[
                    orientation
                ]
            )

            if sample_count < minimum_samples:

                validation.errors.append(
                    f"Orientation '{orientation}' contains only "
                    f"{sample_count} simultaneous valid XYZ "
                    f"samples. At least {minimum_samples} are "
                    "required."
                )

        if validation.errors:

            validation.valid = False

            return validation

        # Ensure each +/- pair contains sufficient measurable span
        # to form a finite scale estimate.

        orientation_means = {
            orientation:
                np.mean(
                    orientation_groups[
                        orientation
                    ],
                    axis=0,
                )
            for orientation
            in self.REQUIRED_ORIENTATIONS
        }

        for configuration in self.AXIS_CONFIGURATION.values():

            index = configuration[
                "index"
            ]

            plus_value = float(
                orientation_means[
                    configuration[
                        "plus"
                    ]
                ][
                    index
                ]
            )

            minus_value = float(
                orientation_means[
                    configuration[
                        "minus"
                    ]
                ][
                    index
                ]
            )

            separation = (
                plus_value
                - minus_value
            )

            if abs(
                separation
            ) <= np.finfo(
                float
            ).eps:

                validation.errors.append(
                    f"{configuration['label']} +/− orientation "
                    "means have essentially zero separation, so "
                    "a finite scale factor cannot be estimated."
                )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Estimate diagonal bias and scale calibration.
        
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

        minimum_span_fraction = float(
            context.parameters[
                "minimum_primary_axis_span_fraction"
            ]
        )

        acceleration_unit = context.units.get(
            "ax",
            "m/s^2",
        )

        orientation_groups = self._orientation_groups(
            data=data,
            context=context,
        )

        # Orientation statistics

        orientation_statistics = {}

        orientation_means = {}

        for orientation in self.REQUIRED_ORIENTATIONS:

            samples = orientation_groups[
                orientation
            ]

            mean_vector = np.mean(
                samples,
                axis=0,
            )

            std_vector = np.std(
                samples,
                axis=0,
                ddof=1,
            )

            orientation_means[
                orientation
            ] = mean_vector

            orientation_statistics[
                orientation
            ] = {
                "sample_count":
                    len(
                        samples
                    ),

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

                "standard_deviation": {
                    "ax":
                        float(
                            std_vector[
                                0
                            ]
                        ),

                    "ay":
                        float(
                            std_vector[
                                1
                            ]
                        ),

                    "az":
                        float(
                            std_vector[
                                2
                            ]
                        ),
                },
            }

        # Bias and sensitivity estimation

        bias_vector = np.zeros(
            3,
            dtype=float,
        )

        raw_sensitivity = np.zeros(
            3,
            dtype=float,
        )

        correction_scale = np.zeros(
            3,
            dtype=float,
        )

        for axis_key, configuration in (
            self.AXIS_CONFIGURATION.items()
        ):

            index = configuration[
                "index"
            ]

            plus_orientation = configuration[
                "plus"
            ]

            minus_orientation = configuration[
                "minus"
            ]

            axis_label = configuration[
                "label"
            ]

            plus_mean = float(
                orientation_means[
                    plus_orientation
                ][
                    index
                ]
            )

            minus_mean = float(
                orientation_means[
                    minus_orientation
                ][
                    index
                ]
            )

            # Standard six-position equations
            #
            # bias:
            #
            #     b = (A+ + A-) / 2
            #
            # raw sensitivity relative to ideal:
            #
            #     k = (A+ - A-) / (2g)
            #
            # correction factor:
            #
            #     s = 1 / k
            #
            # corrected value:
            #
            #     A_corr = s * (A_raw - b)

            bias = (
                plus_mean
                + minus_mean
            ) / 2.0

            measured_half_span = (
                plus_mean
                - minus_mean
            ) / 2.0

            sensitivity = (
                measured_half_span
                / gravity
            )

            if abs(
                sensitivity
            ) <= np.finfo(
                float
            ).eps:

                result.status = ExecutionStatus.ERROR

                result.add_message(
                    f"{axis_label} sensitivity is effectively "
                    "zero. Calibration cannot be calculated."
                )

                return result

            scale_correction = (
                1.0
                / sensitivity
            )

            bias_vector[
                index
            ] = bias

            raw_sensitivity[
                index
            ] = sensitivity

            correction_scale[
                index
            ] = scale_correction

            sensitivity_error_percent = (
                sensitivity
                - 1.0
            ) * 100.0

            measured_span = (
                plus_mean
                - minus_mean
            )

            ideal_span = (
                2.0
                * gravity
            )

            span_fraction = (
                abs(
                    measured_span
                )
                / ideal_span
            )

            # Metrics

            result.add_metric(
                name=f"{axis_label} Bias",
                value=bias,
                unit=acceleration_unit,
                description=(
                    f"Bias estimated as the midpoint between "
                    f"{plus_orientation} and {minus_orientation} "
                    "primary-axis means."
                ),
            )

            result.add_metric(
                name=(
                    f"{axis_label} Raw Sensitivity Ratio"
                ),
                value=sensitivity,
                unit=None,
                description=(
                    "Measured primary-axis sensitivity divided "
                    "by ideal sensitivity. Ideal value is 1."
                ),
            )

            result.add_metric(
                name=(
                    f"{axis_label} Sensitivity Error Percent"
                ),
                value=sensitivity_error_percent,
                unit="%",
                description=(
                    "Difference between measured and ideal "
                    "per-axis sensitivity."
                ),
            )

            result.add_metric(
                name=(
                    f"{axis_label} Correction Scale Factor"
                ),
                value=scale_correction,
                unit=None,
                description=(
                    "Multiplicative factor applied after bias "
                    "subtraction in the diagonal calibration model."
                ),
            )

            result.add_metric(
                name=(
                    f"{axis_label} Measured Plus-Minus Span"
                ),
                value=measured_span,
                unit=acceleration_unit,
            )

            if (
                span_fraction
                < minimum_span_fraction
            ):

                result.add_warning(
                    f"{axis_label} +/− orientation span is only "
                    f"{span_fraction * 100.0:.1f}% of the ideal "
                    "2g span. Check orientation labeling, axis "
                    "mapping, units, or sensor configuration."
                )

            if scale_correction < 0.0:

                result.add_warning(
                    f"{axis_label} requires a negative correction "
                    "scale factor. This indicates that the measured "
                    "axis polarity is opposite to the expected "
                    "orientation-label convention and should be "
                    "reviewed."
                )

        # Calibration matrices

        correction_matrix = np.diag(
            correction_scale
        )

        result.add_metric(
            name="Accelerometer Bias Vector Magnitude",
            value=float(
                np.linalg.norm(
                    bias_vector
                )
            ),
            unit=acceleration_unit,
        )

        # Evaluate calibration-set orientation means
        #
        # IMPORTANT:
        # These are NOT held-out validation metrics.

        raw_expected_residuals = []
        corrected_expected_residuals = []

        raw_magnitude_errors = []
        corrected_magnitude_errors = []

        raw_angle_errors = []
        corrected_angle_errors = []

        calibration_fit_results = {}

        for orientation in self.REQUIRED_ORIENTATIONS:

            raw_mean = orientation_means[
                orientation
            ]

            expected = self._expected_gravity_vector(
                orientation=orientation,
                gravity=gravity,
            )

            corrected_mean = (
                correction_scale
                * (
                    raw_mean
                    - bias_vector
                )
            )

            raw_error = (
                raw_mean
                - expected
            )

            corrected_error = (
                corrected_mean
                - expected
            )

            raw_error_magnitude = float(
                np.linalg.norm(
                    raw_error
                )
            )

            corrected_error_magnitude = float(
                np.linalg.norm(
                    corrected_error
                )
            )

            raw_magnitude = float(
                np.linalg.norm(
                    raw_mean
                )
            )

            corrected_magnitude = float(
                np.linalg.norm(
                    corrected_mean
                )
            )

            raw_magnitude_error = (
                raw_magnitude
                - gravity
            )

            corrected_magnitude_error = (
                corrected_magnitude
                - gravity
            )

            raw_angle = self._vector_angle_degrees(
                measured=raw_mean,
                expected=expected,
            )

            corrected_angle = (
                self._vector_angle_degrees(
                    measured=corrected_mean,
                    expected=expected,
                )
            )

            raw_expected_residuals.extend(
                raw_error.tolist()
            )

            corrected_expected_residuals.extend(
                corrected_error.tolist()
            )

            raw_magnitude_errors.append(
                raw_magnitude_error
            )

            corrected_magnitude_errors.append(
                corrected_magnitude_error
            )

            if raw_angle is not None:

                raw_angle_errors.append(
                    raw_angle
                )

            if corrected_angle is not None:

                corrected_angle_errors.append(
                    corrected_angle
                )

            result.add_metric(
                name=(
                    f"{orientation} Raw Calibration-Set "
                    "Vector Error"
                ),
                value=raw_error_magnitude,
                unit=acceleration_unit,
            )

            result.add_metric(
                name=(
                    f"{orientation} Corrected Calibration-Set "
                    "Vector Error"
                ),
                value=corrected_error_magnitude,
                unit=acceleration_unit,
                description=(
                    "Residual vector error after applying the "
                    "estimated diagonal model to the same "
                    "orientation data used to derive calibration. "
                    "This is not independent validation."
                ),
            )

            if raw_angle is not None:

                result.add_metric(
                    name=(
                        f"{orientation} Raw Calibration-Set "
                        "Angular Deviation"
                    ),
                    value=raw_angle,
                    unit="deg",
                )

            if corrected_angle is not None:

                result.add_metric(
                    name=(
                        f"{orientation} Corrected Calibration-Set "
                        "Angular Deviation"
                    ),
                    value=corrected_angle,
                    unit="deg",
                )

            calibration_fit_results[
                orientation
            ] = {
                "raw_mean_vector":
                    self._vector_dict(
                        raw_mean
                    ),

                "expected_vector":
                    self._vector_dict(
                        expected
                    ),

                "corrected_mean_vector":
                    self._vector_dict(
                        corrected_mean
                    ),

                "raw_vector_error":
                    self._vector_dict(
                        raw_error
                    ),

                "corrected_vector_error":
                    self._vector_dict(
                        corrected_error
                    ),

                "raw_vector_error_magnitude":
                    raw_error_magnitude,

                "corrected_vector_error_magnitude":
                    corrected_error_magnitude,

                "raw_magnitude_error":
                    raw_magnitude_error,

                "corrected_magnitude_error":
                    corrected_magnitude_error,

                "raw_angular_deviation_deg":
                    raw_angle,

                "corrected_angular_deviation_deg":
                    corrected_angle,
            }

        # Aggregate calibration-fit residual metrics

        raw_expected_residuals = np.asarray(
            raw_expected_residuals,
            dtype=float,
        )

        corrected_expected_residuals = np.asarray(
            corrected_expected_residuals,
            dtype=float,
        )

        calibration_set_raw_rmse = float(
            np.sqrt(
                np.mean(
                    raw_expected_residuals ** 2
                )
            )
        )

        calibration_set_corrected_rmse = float(
            np.sqrt(
                np.mean(
                    corrected_expected_residuals ** 2
                )
            )
        )

        raw_magnitude_mae = float(
            np.mean(
                np.abs(
                    raw_magnitude_errors
                )
            )
        )

        corrected_magnitude_mae = float(
            np.mean(
                np.abs(
                    corrected_magnitude_errors
                )
            )
        )

        result.add_metric(
            name="Raw Calibration-Set Component RMSE",
            value=calibration_set_raw_rmse,
            unit=acceleration_unit,
            description=(
                "Component-level RMSE across the six orientation "
                "mean vectors before calibration."
            ),
        )

        result.add_metric(
            name="Corrected Calibration-Set Component RMSE",
            value=calibration_set_corrected_rmse,
            unit=acceleration_unit,
            description=(
                "Component-level RMSE across the same six "
                "orientation means after diagonal calibration. "
                "This is calibration-fit performance, not "
                "held-out validation performance."
            ),
        )

        result.add_metric(
            name="Raw Calibration-Set Gravity Magnitude MAE",
            value=raw_magnitude_mae,
            unit=acceleration_unit,
        )

        result.add_metric(
            name=(
                "Corrected Calibration-Set Gravity Magnitude MAE"
            ),
            value=corrected_magnitude_mae,
            unit=acceleration_unit,
        )

        if raw_angle_errors:

            result.add_metric(
                name="Raw Calibration-Set Mean Angular Deviation",
                value=float(
                    np.mean(
                        raw_angle_errors
                    )
                ),
                unit="deg",
            )

        if corrected_angle_errors:

            result.add_metric(
                name=(
                    "Corrected Calibration-Set Mean "
                    "Angular Deviation"
                ),
                value=float(
                    np.mean(
                        corrected_angle_errors
                    )
                ),
                unit="deg",
            )

        # Evidence and limitations

        result.add_evidence(
            statement=(
                "All six canonical static orientations are "
                "available with sufficient samples, supporting "
                "estimation of a diagonal accelerometer bias and "
                "scale calibration model."
            ),
            strength=EvidenceStrength.STRONG,
            supporting_metrics=[
                "AX Bias",
                "AY Bias",
                "AZ Bias",
                "AX Correction Scale Factor",
                "AY Correction Scale Factor",
                "AZ Correction Scale Factor",
            ],
        )

        result.add_warning(
            "Corrected residual metrics in this tool are calculated "
            "using the same six-position dataset used to estimate "
            "the calibration parameters. They describe calibration "
            "fit only and must not be interpreted as independent "
            "validation performance."
        )

        result.add_warning(
            "This V1 calibration model is diagonal. Remaining "
            "off-axis residuals are not automatically interpreted "
            "as axis misalignment, nonorthogonality, or cross-axis "
            "sensitivity."
        )

        # Reproducibility metadata

        result.metadata.update(
            {
                "model":
                    "diagonal_bias_and_scale",

                "calibration_equation":
                    (
                        "a_corrected = S * "
                        "(a_measured - b)"
                    ),

                "gravity_magnitude":
                    gravity,

                "acceleration_unit":
                    acceleration_unit,

                "bias_vector":
                    self._vector_dict(
                        bias_vector
                    ),

                "raw_sensitivity_ratio": {
                    "ax":
                        float(
                            raw_sensitivity[
                                0
                            ]
                        ),

                    "ay":
                        float(
                            raw_sensitivity[
                                1
                            ]
                        ),

                    "az":
                        float(
                            raw_sensitivity[
                                2
                            ]
                        ),
                },

                "correction_scale_factors": {
                    "ax":
                        float(
                            correction_scale[
                                0
                            ]
                        ),

                    "ay":
                        float(
                            correction_scale[
                                1
                            ]
                        ),

                    "az":
                        float(
                            correction_scale[
                                2
                            ]
                        ),
                },

                "correction_matrix":
                    correction_matrix.tolist(),

                "orientation_statistics":
                    orientation_statistics,

                "calibration_fit_results":
                    calibration_fit_results,

                "evaluation_mode":
                    "calibration_fit_only",

                "independent_validation_performed":
                    False,

                "full_3x3_misalignment_matrix_estimated":
                    False,

                "cross_axis_sensitivity_estimated":
                    False,
            }
        )

        return result

    # Orientation grouping

    @classmethod
    def _orientation_groups(
        cls,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> dict[
        str,
        np.ndarray,
    ]:
        """Return simultaneous valid XYZ measurements grouped by
        canonical six-position label.
        
        Unknown orientation labels are ignored.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Dictionary containing the result values.
        """

        ax = pd.to_numeric(
            data[
                context.column_mapping[
                    "ax"
                ]
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        ay = pd.to_numeric(
            data[
                context.column_mapping[
                    "ay"
                ]
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        az = pd.to_numeric(
            data[
                context.column_mapping[
                    "az"
                ]
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        orientation_column = (
            context.column_mapping[
                "orientation_label"
            ]
        )

        labels = data[
            orientation_column
        ].to_numpy(
            dtype=object
        )

        grouped_lists = {
            orientation: []
            for orientation
            in cls.REQUIRED_ORIENTATIONS
        }

        for index in range(
            len(data)
        ):

            vector = np.array(
                [
                    ax[
                        index
                    ],
                    ay[
                        index
                    ],
                    az[
                        index
                    ],
                ],
                dtype=float,
            )

            if not np.all(
                np.isfinite(
                    vector
                )
            ):

                continue

            canonical = (
                cls._canonical_orientation_label(
                    labels[
                        index
                    ]
                )
            )

            if canonical is None:
                continue

            grouped_lists[
                canonical
            ].append(
                vector
            )

        output = {}

        for orientation, vectors in (
            grouped_lists.items()
        ):

            if not vectors:
                continue

            output[
                orientation
            ] = np.vstack(
                vectors
            )

        return output

    # Orientation definitions

    @staticmethod
    def _canonical_orientation_label(
        label,
    ) -> str | None:

        """Return canonical orientation label.
        
        Args:
            label: Label shown to the user.
        
        Returns:
            str | None returned by this function.
        """
        if label is None:

            return None

        try:

            if pd.isna(
                label
            ):

                return None

        except (
            TypeError,
            ValueError,
        ):
            pass

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

    @staticmethod
    def _expected_gravity_vector(
        orientation: str,
        gravity: float,
    ) -> np.ndarray:

        """Calculate expected gravity vector.
        
        Args:
            orientation: Orientation used by this function.
            gravity: Gravity used by this function.
        
        Returns:
            np.ndarray returned by this function.
        """
        vectors = {
            "+X": np.array(
                [
                    gravity,
                    0.0,
                    0.0,
                ]
            ),

            "-X": np.array(
                [
                    -gravity,
                    0.0,
                    0.0,
                ]
            ),

            "+Y": np.array(
                [
                    0.0,
                    gravity,
                    0.0,
                ]
            ),

            "-Y": np.array(
                [
                    0.0,
                    -gravity,
                    0.0,
                ]
            ),

            "+Z": np.array(
                [
                    0.0,
                    0.0,
                    gravity,
                ]
            ),

            "-Z": np.array(
                [
                    0.0,
                    0.0,
                    -gravity,
                ]
            ),
        }

        return vectors[
            orientation
        ].astype(
            float
        )

    # Vector helpers

    @staticmethod
    def _vector_angle_degrees(
        measured: np.ndarray,
        expected: np.ndarray,
    ) -> float | None:

        """Calculate vector angle degrees.
        
        Args:
            measured: Measured used by this function.
            expected: Expected used by this function.
        
        Returns:
            Calculated value.
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

    @staticmethod
    def _vector_dict(
        vector: np.ndarray,
    ) -> dict[
        str,
        float,
    ]:

        """Calculate vector dict.
        
        Args:
            vector: Vector used by this function.
        
        Returns:
            Dictionary containing the result values.
        """
        return {
            "ax":
                float(
                    vector[
                        0
                    ]
                ),

            "ay":
                float(
                    vector[
                        1
                    ]
                ),

            "az":
                float(
                    vector[
                        2
                    ]
                ),
        }

    # Test metadata

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
