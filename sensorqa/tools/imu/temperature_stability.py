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


class TemperatureStabilityTool(BaseAnalysisTool):
    """
    Characterizes statistical relationships between IMU output
    and temperature during stationary testing.

    For each available accelerometer or gyroscope axis, the tool
    calculates:

        - temperature span
        - linear temperature coefficient
        - linear regression R^2
        - Pearson correlation coefficient
        - predicted output change across observed temperature span
        - regression residual standard deviation
        - temperature-binned mean and standard deviation

    Important:
        A relationship between sensor output and temperature is
        statistical evidence of dependence.

        It does not by itself prove that temperature caused the
        observed change because temperature may covary with:

            - elapsed time
            - sensor warm-up
            - supply voltage
            - mechanical conditions
            - other environmental variables
    """

    @property
    def metadata(self) -> ToolMetadata:

        """Return the metadata used to register this analysis tool.
        
        Returns:
            ToolMetadata returned by this function.
        """
        return ToolMetadata(
            tool_id="imu_temperature_stability",
            name="Temperature Stability",
            version="1.0.0",
            description=(
                "Evaluates statistical relationships between "
                "stationary IMU output and temperature using linear "
                "regression, correlation, temperature coefficients, "
                "and temperature-binned behavior."
            ),
            category=ToolCategory.ENVIRONMENTAL,
            compatible_sensor_types=[
                SensorType.IMU,
                SensorType.ACCELEROMETER,
                SensorType.GYROSCOPE,
            ],
            required_columns=[
                "temperature",
            ],
            optional_columns=[],
            parameters=[
                ToolParameter(
                    name="minimum_samples",
                    parameter_type=ParameterType.INTEGER,
                    default=500,
                    description=(
                        "Minimum number of valid paired temperature "
                        "and sensor samples required per axis."
                    ),
                    minimum=20,
                    maximum=100_000_000,
                ),
                ToolParameter(
                    name="minimum_temperature_span_degc",
                    parameter_type=ParameterType.FLOAT,
                    default=2.0,
                    description=(
                        "Minimum observed temperature span required "
                        "for temperature-dependence analysis."
                    ),
                    unit="degC",
                    minimum=0.1,
                    maximum=500.0,
                ),
                ToolParameter(
                    name="temperature_bins",
                    parameter_type=ParameterType.INTEGER,
                    default=10,
                    description=(
                        "Number of equal-width temperature bins "
                        "used for binned summary statistics."
                    ),
                    minimum=3,
                    maximum=100,
                ),
                ToolParameter(
                    name="minimum_samples_per_bin",
                    parameter_type=ParameterType.INTEGER,
                    default=20,
                    description=(
                        "Minimum samples required before a "
                        "temperature bin is retained."
                    ),
                    minimum=2,
                    maximum=1_000_000,
                ),
                ToolParameter(
                    name="association_r2_threshold",
                    parameter_type=ParameterType.FLOAT,
                    default=0.50,
                    description=(
                        "R^2 threshold above which SensorQA reports "
                        "moderate evidence of a strong statistical "
                        "association with temperature."
                    ),
                    minimum=0.0,
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
        """Verify that temperature variation and appropriate stationary
        sensor channels are available.
        
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

        # Static/stationary experiment requirement

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is False:

            validation.errors.append(
                "Temperature-stability analysis requires stationary "
                "or static data. Metadata identifies this experiment "
                "as dynamic."
            )

        elif stationary_state is None:

            validation.warnings.append(
                "SensorQA could not confirm that the recording was "
                "stationary. Apparent temperature dependence may "
                "therefore include real sensor motion."
            )

        # Sensor-channel availability

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        if not available_groups:

            validation.errors.append(
                "No complete accelerometer or gyroscope XYZ group "
                "is available for temperature-stability analysis."
            )

        # Temperature quality

        temperature_column = context.column_mapping.get(
            "temperature"
        )

        if temperature_column is None:

            validation.errors.append(
                "Temperature mapping is unavailable."
            )

            validation.valid = False
            return validation

        temperature = pd.to_numeric(
            data[temperature_column],
            errors="coerce",
        )

        valid_temperature = temperature.dropna()

        if len(valid_temperature) < 2:

            validation.errors.append(
                "At least two valid temperature measurements "
                "are required."
            )

            validation.valid = False
            return validation

        temperature_span = float(
            valid_temperature.max()
            - valid_temperature.min()
        )

        minimum_span = float(
            context.parameters.get(
                "minimum_temperature_span_degc",
                2.0,
            )
        )

        if temperature_span < minimum_span:

            validation.errors.append(
                "Temperature variation is insufficient for "
                "temperature-stability analysis. "
                f"Observed span = {temperature_span:.3f} degC, "
                f"required span = {minimum_span:.3f} degC."
            )

        if validation.errors:
            validation.valid = False

        return validation

    def run(
        self,
        data: pd.DataFrame,
        context: ToolContext,
    ) -> AnalysisResult:
        """Perform temperature-dependence analysis.
        
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

        temperature_column = context.column_mapping[
            "temperature"
        ]

        temperature = pd.to_numeric(
            data[temperature_column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        temperature_unit = context.units.get(
            "temperature",
            "degC",
        )

        finite_temperature = temperature[
            np.isfinite(
                temperature
            )
        ]

        global_temperature_minimum = float(
            np.min(
                finite_temperature
            )
        )

        global_temperature_maximum = float(
            np.max(
                finite_temperature
            )
        )

        global_temperature_span = (
            global_temperature_maximum
            - global_temperature_minimum
        )

        result.add_metric(
            name="Temperature Minimum",
            value=global_temperature_minimum,
            unit=temperature_unit,
        )

        result.add_metric(
            name="Temperature Maximum",
            value=global_temperature_maximum,
            unit=temperature_unit,
        )

        result.add_metric(
            name="Temperature Span",
            value=global_temperature_span,
            unit=temperature_unit,
            description=(
                "Observed temperature range during the experiment."
            ),
        )

        result.add_metric(
            name="Valid Temperature Sample Count",
            value=len(
                finite_temperature
            ),
            unit=None,
        )

        available_groups = self._available_sensor_groups(
            data=data,
            context=context,
        )

        minimum_samples = int(
            context.parameters[
                "minimum_samples"
            ]
        )

        minimum_span = float(
            context.parameters[
                "minimum_temperature_span_degc"
            ]
        )

        number_of_bins = int(
            context.parameters[
                "temperature_bins"
            ]
        )

        minimum_samples_per_bin = int(
            context.parameters[
                "minimum_samples_per_bin"
            ]
        )

        association_threshold = float(
            context.parameters[
                "association_r2_threshold"
            ]
        )

        temperature_results = {}

        processed_axis_count = 0

        # Process available IMU groups

        for group_name, axes in available_groups.items():

            group_results = {}

            for axis in axes:

                axis_column = context.column_mapping[
                    axis
                ]

                axis_values = pd.to_numeric(
                    data[axis_column],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                valid_mask = (
                    np.isfinite(
                        temperature
                    )
                    & np.isfinite(
                        axis_values
                    )
                )

                x = temperature[
                    valid_mask
                ]

                y = axis_values[
                    valid_mask
                ]

                sample_count = len(
                    x
                )

                axis_label = axis.upper()

                if sample_count < minimum_samples:

                    result.add_warning(
                        f"{axis_label} has only {sample_count} valid "
                        "temperature/output pairs. At least "
                        f"{minimum_samples} are required."
                    )

                    continue

                axis_temperature_minimum = float(
                    np.min(
                        x
                    )
                )

                axis_temperature_maximum = float(
                    np.max(
                        x
                    )
                )

                axis_temperature_span = (
                    axis_temperature_maximum
                    - axis_temperature_minimum
                )

                if axis_temperature_span < minimum_span:

                    result.add_warning(
                        f"{axis_label} paired data span only "
                        f"{axis_temperature_span:.3f} degC, which "
                        "is insufficient for the configured "
                        "temperature analysis."
                    )

                    continue

                # Linear regression
                #
                # y = intercept + slope * T

                regression = self._linear_regression(
                    x=x,
                    y=y,
                )

                slope = regression[
                    "slope"
                ]

                intercept = regression[
                    "intercept"
                ]

                r_squared = regression[
                    "r_squared"
                ]

                pearson_r = regression[
                    "pearson_r"
                ]

                residual_std = regression[
                    "residual_std"
                ]

                predicted_change = (
                    slope
                    * axis_temperature_span
                )

                axis_unit = context.units.get(
                    axis
                )

                coefficient_unit = (
                    f"{axis_unit}/{temperature_unit}"
                    if axis_unit
                    else None
                )

                # Core metrics

                result.add_metric(
                    name=(
                        f"{axis_label} Temperature Coefficient"
                    ),
                    value=slope,
                    unit=coefficient_unit,
                    description=(
                        "Slope from linear regression of sensor "
                        "output against temperature."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Temperature Regression R2"
                    ),
                    value=r_squared,
                    unit=None,
                    description=(
                        "Fraction of output variance described by "
                        "the linear temperature regression. This is "
                        "an association metric and does not establish "
                        "causality."
                    ),
                )

                if pearson_r is not None:

                    result.add_metric(
                        name=(
                            f"{axis_label} Temperature "
                            "Correlation"
                        ),
                        value=pearson_r,
                        unit=None,
                        description=(
                            "Pearson correlation coefficient between "
                            "temperature and sensor output."
                        ),
                    )

                result.add_metric(
                    name=(
                        f"{axis_label} Predicted Change Across "
                        "Temperature Span"
                    ),
                    value=predicted_change,
                    unit=axis_unit,
                    description=(
                        "Linear-model output change across the "
                        "observed temperature span."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Temperature Regression "
                        "Residual Standard Deviation"
                    ),
                    value=residual_std,
                    unit=axis_unit,
                    description=(
                        "Standard deviation of residual output after "
                        "subtracting the fitted linear temperature "
                        "relationship."
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Temperature Sample Count"
                    ),
                    value=sample_count,
                    unit=None,
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Temperature Span"
                    ),
                    value=axis_temperature_span,
                    unit=temperature_unit,
                )

                # Temperature-binned behavior

                binned = self._temperature_bins(
                    temperature=x,
                    values=y,
                    number_of_bins=number_of_bins,
                    minimum_samples_per_bin=(
                        minimum_samples_per_bin
                    ),
                )

                result.add_metric(
                    name=(
                        f"{axis_label} Valid Temperature Bin Count"
                    ),
                    value=len(
                        binned
                    ),
                    unit=None,
                )

                # Evidence

                if (
                    r_squared is not None
                    and r_squared
                    >= association_threshold
                ):

                    result.add_evidence(
                        statement=(
                            f"{axis_label} output shows a substantial "
                            "linear statistical association with "
                            "temperature over the observed test range."
                        ),
                        strength=EvidenceStrength.MODERATE,
                        supporting_metrics=[
                            (
                                f"{axis_label} Temperature "
                                "Coefficient"
                            ),
                            (
                                f"{axis_label} Temperature "
                                "Regression R2"
                            ),
                            (
                                f"{axis_label} Predicted Change "
                                "Across Temperature Span"
                            ),
                        ],
                    )

                    result.add_warning(
                        f"{axis_label} shows a relatively strong "
                        "statistical relationship with temperature "
                        f"(R2 = {r_squared:.3f}). This does not by "
                        "itself establish temperature as the causal "
                        "source because temperature may covary with "
                        "elapsed time or other test conditions."
                    )

                # Save detailed data for later plots

                group_results[
                    axis
                ] = {
                    "sample_count":
                        sample_count,

                    "temperature_minimum":
                        axis_temperature_minimum,

                    "temperature_maximum":
                        axis_temperature_maximum,

                    "temperature_span":
                        axis_temperature_span,

                    "slope":
                        slope,

                    "intercept":
                        intercept,

                    "r_squared":
                        r_squared,

                    "pearson_r":
                        pearson_r,

                    "residual_std":
                        residual_std,

                    "predicted_change":
                        predicted_change,

                    "bins":
                        binned,
                }

                processed_axis_count += 1

            temperature_results[
                group_name
            ] = group_results

        # Overall result

        if processed_axis_count == 0:

            result.status = ExecutionStatus.ERROR

            result.add_message(
                "No IMU axis contained enough valid paired "
                "temperature data for temperature-stability analysis."
            )

            return result

        stationary_state = self._stationary_state(
            context
        )

        if stationary_state is True:

            result.add_evidence(
                statement=(
                    "Experiment metadata identifies this recording "
                    "as stationary or static, reducing motion as a "
                    "confounding source in temperature/output "
                    "relationships."
                ),
                strength=EvidenceStrength.STRONG,
                supporting_metrics=[],
            )

        else:

            result.add_warning(
                "Stationarity was not confirmed. Apparent "
                "temperature relationships may include genuine "
                "motion-related changes."
            )

        result.metadata.update(
            {
                "temperature_unit":
                    temperature_unit,

                "temperature_minimum":
                    global_temperature_minimum,

                "temperature_maximum":
                    global_temperature_maximum,

                "temperature_span":
                    global_temperature_span,

                "minimum_samples":
                    minimum_samples,

                "minimum_temperature_span_degc":
                    minimum_span,

                "temperature_bins":
                    number_of_bins,

                "minimum_samples_per_bin":
                    minimum_samples_per_bin,

                "association_r2_threshold":
                    association_threshold,

                "causal_interpretation":
                    False,

                "temperature_results":
                    temperature_results,
            }
        )

        return result

    # Linear regression

    @staticmethod
    def _linear_regression(
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[
        str,
        float | None,
    ]:
        """Ordinary least-squares linear regression.
        
        Returns:
            slope
            intercept
            R^2
            Pearson r
            residual standard deviation
        
        Args:
            x: Independent-variable values.
            y: Dependent-variable values.
        """

        x = np.asarray(
            x,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        x_mean = float(
            np.mean(
                x
            )
        )

        y_mean = float(
            np.mean(
                y
            )
        )

        dx = (
            x - x_mean
        )

        dy = (
            y - y_mean
        )

        sxx = float(
            np.sum(
                dx ** 2
            )
        )

        syy = float(
            np.sum(
                dy ** 2
            )
        )

        if sxx <= 0.0:

            raise ValueError(
                "Temperature has no measurable variation."
            )

        slope = float(
            np.sum(
                dx * dy
            )
            / sxx
        )

        intercept = float(
            y_mean
            - slope
            * x_mean
        )

        predicted = (
            intercept
            + slope
            * x
        )

        residual = (
            y - predicted
        )

        residual_sum_squares = float(
            np.sum(
                residual ** 2
            )
        )

        if syy > 0.0:

            r_squared = float(
                1.0
                - (
                    residual_sum_squares
                    / syy
                )
            )

            pearson_r = float(
                np.sum(
                    dx * dy
                )
                / np.sqrt(
                    sxx * syy
                )
            )

            pearson_r = float(
                np.clip(
                    pearson_r,
                    -1.0,
                    1.0,
                )
            )

        else:

            # Constant output means R^2 and correlation are not
            # meaningful even though the fitted slope is zero.
            r_squared = None
            pearson_r = None

        if len(
            residual
        ) > 2:

            residual_std = float(
                np.sqrt(
                    residual_sum_squares
                    / (
                        len(residual)
                        - 2
                    )
                )
            )

        else:

            residual_std = float(
                np.sqrt(
                    np.mean(
                        residual ** 2
                    )
                )
            )

        return {
            "slope":
                slope,

            "intercept":
                intercept,

            "r_squared":
                r_squared,

            "pearson_r":
                pearson_r,

            "residual_std":
                residual_std,
        }

    # Temperature binning

    @staticmethod
    def _temperature_bins(
        temperature: np.ndarray,
        values: np.ndarray,
        number_of_bins: int,
        minimum_samples_per_bin: int,
    ) -> list[
        dict[str, float | int]
    ]:
        """Build equal-width temperature bins for later plotting
        and nonparametric trend inspection.
        
        Args:
            temperature: Value for `temperature`.
            values: Values to process.
            number_of_bins: Value for `number_of_bins`.
            minimum_samples_per_bin: Minimum accepted samples per bin.
        
        Returns:
            Dictionary containing the result values.
        """

        temperature = np.asarray(
            temperature,
            dtype=float,
        )

        values = np.asarray(
            values,
            dtype=float,
        )

        minimum_temperature = float(
            np.min(
                temperature
            )
        )

        maximum_temperature = float(
            np.max(
                temperature
            )
        )

        if maximum_temperature <= minimum_temperature:

            return []

        edges = np.linspace(
            minimum_temperature,
            maximum_temperature,
            number_of_bins + 1,
        )

        # digitize returns indices 1...number_of_bins+1.
        # Values exactly equal to the final edge are clipped into
        # the last valid bin.
        indices = np.digitize(
            temperature,
            edges[
                1:-1
            ],
            right=False,
        )

        output = []

        for bin_index in range(
            number_of_bins
        ):

            mask = (
                indices
                == bin_index
            )

            count = int(
                np.sum(
                    mask
                )
            )

            if count < minimum_samples_per_bin:
                continue

            bin_temperature = temperature[
                mask
            ]

            bin_values = values[
                mask
            ]

            output.append(
                {
                    "bin_index":
                        bin_index,

                    "sample_count":
                        count,

                    "temperature_mean":
                        float(
                            np.mean(
                                bin_temperature
                            )
                        ),

                    "temperature_minimum":
                        float(
                            np.min(
                                bin_temperature
                            )
                        ),

                    "temperature_maximum":
                        float(
                            np.max(
                                bin_temperature
                            )
                        ),

                    "output_mean":
                        float(
                            np.mean(
                                bin_values
                            )
                        ),

                    "output_standard_deviation":
                        float(
                            np.std(
                                bin_values,
                                ddof=1,
                            )
                        )
                        if count > 1
                        else 0.0,
                }
            )

        return output

    # Sensor-group discovery

    @staticmethod
    def _available_sensor_groups(
        data: pd.DataFrame,
        context: ToolContext,
    ) -> dict[
        str,
        tuple[str, str, str],
    ]:

        """Return sensor groups that contain all required axis columns.
        
        Args:
            data: Input data to process.
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            Dictionary containing the result values.
        """
        groups = {
            "accelerometer": (
                "ax",
                "ay",
                "az",
            ),
            "gyroscope": (
                "gx",
                "gy",
                "gz",
            ),
        }

        available = {}

        for group_name, axes in groups.items():

            if all(
                (
                    context.column_mapping.get(
                        axis
                    )
                    in data.columns
                )
                for axis in axes
            ):

                available[
                    group_name
                ] = axes

        return available

    # Stationarity metadata

    @staticmethod
    def _stationary_state(
        context: ToolContext,
    ) -> bool | None:
        """Determine whether metadata supports stationary/static
        interpretation.
        
        Args:
            context: Tool context containing column mappings, units, and settings.
        
        Returns:
            bool | None returned by the function.
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
