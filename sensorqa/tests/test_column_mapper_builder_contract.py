#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPOSITORY_ROOT))

from sensorqa.ingestion.column_mapper import (
    ColumnMapper,
    ColumnMapping,
    StandardField,
)


def make_dataframe():

    """Run make dataframe.
    
    Returns:
        Constructed object.
    """
    return pd.DataFrame(
        {
            "timestamp": [
                0.00,
                0.01,
                0.02,
            ],

            "ax": [
                0.0,
                0.1,
                0.2,
            ],

            "ay": [
                0.0,
                0.0,
                0.0,
            ],

            "az": [
                9.81,
                9.80,
                9.82,
            ],

            "gx": [
                0.001,
                0.001,
                0.001,
            ],

            "gy": [
                0.0,
                0.0,
                0.0,
            ],

            "gz": [
                0.0,
                0.0,
                0.0,
            ],

            "custom_channel": [
                10.0,
                11.0,
                12.0,
            ],
        }
    )


def make_mapping():

    """Run make mapping.
    
    Returns:
        Constructed object.
    """
    return ColumnMapping(
        mapping={
            "timestamp":
                "timestamp",

            "ax":
                "ax",

            "ay":
                "ay",

            "az":
                "az",

            "gx":
                "gx",

            "gy":
                "gy",

            "gz":
                "gz",
        }
    )


def test_dataset_builder_validate_call_style():

    """Check that dataset builder validate call style.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    result = mapper.validate(
        data=make_dataframe(),
        mapping=make_mapping(),
    )

    assert result.valid


def test_validation_has_errors():

    """Check that validation has errors.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    result = mapper.validate(
        data=make_dataframe(),
        mapping=make_mapping(),
    )

    assert hasattr(
        result,
        "errors",
    )


def test_validation_has_warnings():

    """Check that validation has warnings.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    result = mapper.validate(
        data=make_dataframe(),
        mapping=make_mapping(),
    )

    assert hasattr(
        result,
        "warnings",
    )


def test_validation_has_duplicated_dataset_columns():

    """Check that validation has duplicated dataset columns.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    result = mapper.validate(
        data=make_dataframe(),
        mapping=make_mapping(),
    )

    assert hasattr(
        result,
        "duplicated_dataset_columns",
    )

    assert (
        result.duplicated_dataset_columns
        == {}
    )


def test_duplicate_dataset_column_is_detected():

    """Check that duplicate dataset column is detected.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    mapping = ColumnMapping(
        mapping={
            "ax":
                "ax",

            "ay":
                "ax",
        }
    )

    result = mapper.validate(
        data=make_dataframe(),
        mapping=mapping,
    )

    assert (
        "ax"
        in result.duplicated_dataset_columns
    )

    assert set(
        result
        .duplicated_dataset_columns[
            "ax"
        ]
    ) == {
        "ax",
        "ay",
    }


def test_dataset_builder_create_mapped_view_call_style():

    """Check that dataset builder create mapped view call style.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    output = mapper.create_mapped_view(
        data=make_dataframe(),
        mapping=make_mapping(),
        keep_unmapped_columns=True,
    )

    assert isinstance(
        output,
        pd.DataFrame,
    )


def test_keep_unmapped_columns_true():

    """Check that keep unmapped columns true.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    output = mapper.create_mapped_view(
        data=make_dataframe(),
        mapping=make_mapping(),
        keep_unmapped_columns=True,
    )

    assert (
        "custom_channel"
        in output.columns
    )


def test_keep_unmapped_columns_false():

    """Check that keep unmapped columns false.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    output = mapper.create_mapped_view(
        data=make_dataframe(),
        mapping=make_mapping(),
        keep_unmapped_columns=False,
    )

    assert (
        "custom_channel"
        not in output.columns
    )


def test_identity_mapping_preserves_standard_names():

    """Check that identity mapping preserves standard names.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    output = mapper.create_mapped_view(
        data=make_dataframe(),
        mapping=make_mapping(),
        keep_unmapped_columns=True,
    )

    assert "timestamp" in output.columns
    assert "ax" in output.columns
    assert "ay" in output.columns
    assert "az" in output.columns
    assert "gx" in output.columns
    assert "gy" in output.columns
    assert "gz" in output.columns


def test_source_to_canonical_dictionary_supported():

    """Check that source to canonical dictionary supported.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    dataframe = pd.DataFrame(
        {
            "Time_s": [
                0.0,
                0.01,
            ],

            "Accel_X": [
                0.1,
                0.2,
            ],
        }
    )

    output = mapper.create_mapped_view(
        data=dataframe,

        mapping={
            "Time_s":
                "timestamp",

            "Accel_X":
                "ax",
        },

        keep_unmapped_columns=True,
    )

    assert (
        "timestamp"
        in output.columns
    )

    assert (
        "ax"
        in output.columns
    )


def test_canonical_to_source_dictionary_supported():

    """Check that canonical to source dictionary supported.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    dataframe = pd.DataFrame(
        {
            "Time_s": [
                0.0,
                0.01,
            ],

            "Accel_X": [
                0.1,
                0.2,
            ],
        }
    )

    output = mapper.create_mapped_view(
        data=dataframe,

        mapping={
            "timestamp":
                "Time_s",

            "ax":
                "Accel_X",
        },

        keep_unmapped_columns=True,
    )

    assert (
        "timestamp"
        in output.columns
    )

    assert (
        "ax"
        in output.columns
    )


def test_missing_dataset_column_is_invalid():

    """Check that missing dataset column is invalid.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    result = mapper.validate(
        data=make_dataframe(),

        mapping=ColumnMapping(
            mapping={
                "timestamp":
                    "does_not_exist",
            }
        ),
    )

    assert not result.valid

    assert result.errors


def test_required_fields_are_checked():

    """Check that required fields are checked.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    result = mapper.validate(
        data=make_dataframe(),

        mapping=ColumnMapping(
            mapping={
                "timestamp":
                    "timestamp",
            }
        ),

        required_fields=[
            "timestamp",
            "ax",
        ],
    )

    assert not result.valid

    assert (
        "ax"
        in result.missing_required_fields
    )


def test_common_imu_alias_inference():

    """Check that common imu alias inference.
    
    Returns:
        None.
    """
    mapper = ColumnMapper()

    mapping = mapper.infer_mapping(
        [
            "Time_s",
            "Accel_X",
            "Accel_Y",
            "Accel_Z",
            "Gyro_X",
            "Gyro_Y",
            "Gyro_Z",
        ]
    )

    assert (
        mapping.source_for(
            StandardField.TIMESTAMP
        )
        == "Time_s"
    )

    assert (
        mapping.source_for(
            StandardField.AX
        )
        == "Accel_X"
    )

    assert (
        mapping.source_for(
            StandardField.GZ
        )
        == "Gyro_Z"
    )


def test_column_mapping_to_dict_exists():

    """Check that column mapping to dict exists.
    
    Returns:
        None.
    """
    mapping = make_mapping()

    serialized = (
        mapping.to_dict()
    )

    assert isinstance(
        serialized,
        dict,
    )

    assert (
        "mapping"
        in serialized
    )
