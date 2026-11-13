"""Validation and grid identity for the ``tas(time, lat, lon)`` NetCDF files.

Every dataset handled by the service must obey the same contract:

* a single data variable ``tas`` with dimensions exactly ``(time, lat, lon)``
* ``time`` holds Gregorian first-of-month timestamps with no duplicate values
* ``lat`` / ``lon`` are one-dimensional, strictly monotonic numeric coordinates
* ``tas`` stores Kelvin temperatures

The grid signature (sizes plus exact coordinate values) is what the service
compares between the fixed historical baseline and every uploaded target:
targets whose grid is not byte-for-byte identical in coordinates are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REQUIRED_VARIABLE = "tas"
REQUIRED_DIMS = ("time", "lat", "lon")
KELVIN_UNITS = frozenset({"k", "kelvin"})


class DatasetError(ValueError):
    """Raised when an uploaded or configured dataset violates the contract."""


@dataclass(frozen=True)
class GridSignature:
    """Full identity of a (lat, lon) grid."""

    lat: np.ndarray
    lon: np.ndarray

    def matches(self, other: "GridSignature") -> bool:
        if self.lat.shape != other.lat.shape or self.lon.shape != other.lon.shape:
            return False
        return bool(
            np.array_equal(self.lat, other.lat, equal_nan=True)
            and np.array_equal(self.lon, other.lon, equal_nan=True)
        )

    def as_dict(self) -> dict:
        return {
            "lat_size": int(self.lat.size),
            "lon_size": int(self.lon.size),
            "lat_min": float(self.lat.min()),
            "lat_max": float(self.lat.max()),
            "lon_min": float(self.lon.min()),
            "lon_max": float(self.lon.max()),
        }


def _check_monotonic(values: np.ndarray, name: str) -> None:
    if values.ndim != 1:
        raise DatasetError(f"coordinate '{name}' must be one-dimensional")
    if values.size < 1:
        raise DatasetError(f"coordinate '{name}' must not be empty")
    diffs = np.diff(values.astype(np.float64, copy=False))
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise DatasetError(f"coordinate '{name}' must be strictly monotonic")
    if np.any(np.isnan(values.astype(np.float64, copy=False))):
        raise DatasetError(f"coordinate '{name}' must not contain NaN")


def _check_time(time: xr.DataArray) -> None:
    if time.ndim != 1 or time.size == 0:
        raise DatasetError("coordinate 'time' must be a non-empty 1-D array")
    # With default decoding, standard Gregorian calendars become datetime64
    # while 360-day / noleap calendars yield cftime objects, which are
    # rejected because the contract requires the proleptic Gregorian calendar.
    values = np.asarray(time.values)
    if not np.issubdtype(values.dtype, np.datetime64):
        raise DatasetError(
            "time must use a standard Gregorian calendar (datetime64 values)"
        )
    if np.any(np.isnan(values.astype("datetime64[ns]").astype("float64"))):
        raise DatasetError("time must not contain NaT values")
    if len(np.unique(values)) != len(values):
        raise DatasetError("time coordinate must not contain duplicate values")
    stamps = pd.DatetimeIndex(values)
    if np.any(stamps.day != 1):
        raise DatasetError("every time value must be the first day of a month")


def _check_kelvin(ds: xr.Dataset) -> None:
    units = ds[REQUIRED_VARIABLE].attrs.get("units", "")
    if not isinstance(units, str) or units.strip().lower() not in KELVIN_UNITS:
        raise DatasetError(
            f"variable '{REQUIRED_VARIABLE}' must declare Kelvin units "
            f"(got units={units!r})"
        )


def validate_dataset(path: str | Path) -> xr.Dataset:
    """Open and fully validate a user-supplied NetCDF file.

    The returned dataset is open and must be closed by the caller.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"file not found: {path}")
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
    except Exception as exc:  # h5py/HDF5 raises a variety of exception types
        raise DatasetError(f"cannot read NetCDF/HDF5 file: {exc}") from exc

    try:
        if REQUIRED_VARIABLE not in ds.data_vars:
            raise DatasetError("missing required variable 'tas'")
        extra_vars = [
            name for name in ds.data_vars if name != REQUIRED_VARIABLE
        ]
        if extra_vars:
            raise DatasetError(
                "only the 'tas' data variable is allowed; found: "
                + ", ".join(sorted(extra_vars))
            )
        tas = ds[REQUIRED_VARIABLE]
        if tuple(tas.dims) != REQUIRED_DIMS:
            raise DatasetError(
                "'tas' must have dimensions exactly (time, lat, lon), got "
                f"{tas.dims}"
            )
        if not np.issubdtype(tas.dtype, np.floating):
            raise DatasetError(
                f"'tas' must be floating point Kelvin values, got {tas.dtype}"
            )
        for coord in REQUIRED_DIMS:
            if coord not in ds.coords:
                raise DatasetError(f"missing coordinate '{coord}'")
        extra_coord_dims = [
            str(name)
            for name, coord in ds.coords.items()
            if set(coord.dims) - set(REQUIRED_DIMS)
        ]
        if extra_coord_dims:
            raise DatasetError(
                "unexpected auxiliary coordinates: "
                + ", ".join(sorted(extra_coord_dims))
            )
        _check_kelvin(ds)
        _check_time(ds["time"])
        _check_monotonic(ds["lat"].values, "lat")
        _check_monotonic(ds["lon"].values, "lon")
        # Shape consistency between variable and coordinates.
        expected = (ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"])
        if tas.shape != expected:
            raise DatasetError(
                f"'tas' shape {tas.shape} does not match coordinate sizes "
                f"{expected}"
            )
    except DatasetError:
        ds.close()
        raise
    except Exception as exc:
        ds.close()
        raise DatasetError(f"invalid dataset: {exc}") from exc
    return ds


def grid_signature(ds: xr.Dataset) -> GridSignature:
    return GridSignature(
        lat=np.array(ds["lat"].values, dtype=np.float64),
        lon=np.array(ds["lon"].values, dtype=np.float64),
    )


def load_grid_signature(path: str | Path) -> GridSignature:
    ds = validate_dataset(path)
    try:
        return grid_signature(ds)
    finally:
        ds.close()


def validate_target(path: str | Path, baseline_grid: GridSignature) -> xr.Dataset:
    """Validate an upload and guarantee its grid matches the baseline grid."""
    ds = validate_dataset(path)
    try:
        if not grid_signature(ds).matches(baseline_grid):
            raise DatasetError(
                "target grid (lat/lon coordinates) must be identical to the "
                "fixed historical baseline grid"
            )
    except DatasetError:
        ds.close()
        raise
    return ds


def month_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Return a 1..12 calendar-month label for every timestamp."""
    return times.month.to_numpy(dtype=np.int64)


__all__ = [
    "DatasetError",
    "GridSignature",
    "load_grid_signature",
    "month_index",
    "validate_dataset",
    "validate_target",
    "grid_signature",
]


# Re-exported for callers that only want math helpers without importing pandas.
def is_finite_number(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
