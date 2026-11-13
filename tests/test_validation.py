"""Contract-validation edge cases for the tas(time,lat,lon) NetCDF schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.compute import make_synthetic_dataset
from app.config import Settings
from app.netcdf_io import DatasetError, validate_dataset


def _write(ds: xr.Dataset, path: Path) -> Path:
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()
    return path


def test_extra_data_variable_rejected(settings: Settings):
    times = pd.date_range("2021-01-01", periods=2, freq="MS")
    ds = make_synthetic_dataset(times, 4, 5, seed=1)
    ds["extra"] = (("lat", "lon"), np.zeros((4, 5), np.float32))
    path = _write(ds, settings.samples_dir / "extra.nc")
    with pytest.raises(DatasetError, match="only the 'tas'"):
        validate_dataset(path)


def test_wrong_dimension_order_rejected(settings: Settings):
    times = pd.date_range("2021-01-01", periods=3, freq="MS")
    lat = np.linspace(-30, 30, 4)
    lon = np.linspace(0, 90, 5)
    data = np.zeros((4, 3, 5), np.float32) + 280.0
    ds = xr.Dataset(
        {"tas": (("lat", "time", "lon"), data, {"units": "K"})},
        coords={
            "time": ("time", times.values),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    path = _write(ds, settings.samples_dir / "dimorder.nc")
    with pytest.raises(DatasetError, match="dimensions exactly"):
        validate_dataset(path)


def test_non_monotonic_lat_rejected(settings: Settings):
    times = pd.date_range("2021-01-01", periods=2, freq="MS")
    ds = make_synthetic_dataset(times, 4, 5, seed=2)
    ds = ds.assign_coords(lat=("lat", np.array([10.0, -10.0, 30.0, -30.0])))
    path = _write(ds, settings.samples_dir / "badlat.nc")
    with pytest.raises(DatasetError, match="monotonic"):
        validate_dataset(path)


def test_non_gregorian_calendar_rejected(settings: Settings):
    lat = np.linspace(-30, 30, 4)
    lon = np.linspace(0, 90, 5)
    data = np.zeros((3, 4, 5), np.float32) + 280.0
    ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), data, {"units": "K"})},
        coords={
            "time": (
                "time",
                np.arange(3, dtype=np.float64),
                {
                    "units": "days since 2000-01-01",
                    "calendar": "noleap",
                },
            ),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    path = _write(ds, settings.samples_dir / "noleap.nc")
    with pytest.raises(DatasetError, match="Gregorian"):
        validate_dataset(path)


def test_missing_tas_variable_rejected(settings: Settings):
    lat = np.linspace(-30, 30, 4)
    lon = np.linspace(0, 90, 5)
    ds = xr.Dataset(
        {"temperature": (("lat", "lon"), np.zeros((4, 5), np.float32))},
        coords={"lat": ("lat", lat), "lon": ("lon", lon)},
    )
    path = _write(ds, settings.samples_dir / "notas.nc")
    with pytest.raises(DatasetError, match="missing required variable"):
        validate_dataset(path)


def test_unreadable_file_rejected(settings: Settings):
    path = settings.samples_dir / "garbage.nc"
    path.write_bytes(b"this is definitely not an HDF5 file")
    with pytest.raises(DatasetError, match="cannot read"):
        validate_dataset(path)


def test_integer_tas_rejected(settings: Settings):
    times = pd.date_range("2021-01-01", periods=2, freq="MS")
    lat = np.linspace(-30, 30, 4)
    lon = np.linspace(0, 90, 5)
    ds = xr.Dataset(
        {"tas": (
            ("time", "lat", "lon"),
            np.zeros((2, 4, 5), np.int32) + 280,
            {"units": "K"},
        )},
        coords={
            "time": ("time", times.values),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    path = _write(ds, settings.samples_dir / "int.nc")
    with pytest.raises(DatasetError, match="floating point"):
        validate_dataset(path)
