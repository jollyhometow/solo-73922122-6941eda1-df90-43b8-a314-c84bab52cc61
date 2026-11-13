"""Pure computation: monthly climatology, anomalies and synthetic samples.

Both heavy entry points accept an :class:`Interrupter` so the worker process
can react to SIGTERM *between* every small work slice.  The interruption
points also bracket each phase transition: a signal arriving while a phase
finishes is observed before the next phase starts, guaranteeing that a
cancelled worker never publishes a late result.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .netcdf_io import month_index, validate_dataset

# NetCDF4 / HDF5 rename is not guaranteed atomic, so output goes to a sibling
# temp file and os.replace() promotes it only after a successful flush.
_TMP_SUFFIX = ".part"


class Cancelled(RuntimeError):
    """Raised inside a worker when an interrupt was requested."""


class Interrupter:
    """Callable cancellation flag with sleeping that responds immediately."""

    def __init__(self, is_set: Callable[[], bool] | None = None) -> None:
        self._is_set = is_set or (lambda: False)

    def check(self) -> None:
        if self._is_set():
            raise Cancelled("interrupted")

    def sleep(self, seconds: float) -> None:
        """Sleep in short slices, raising Cancelled when flagged."""
        deadline = seconds
        slept = 0.0
        slice_len = min(0.05, max(seconds / 10.0, 0.005))
        while slept < deadline:
            self.check()
            step = min(slice_len, deadline - slept)
            # Busy-free interruptible sleep without signal-handler coupling.
            import time

            time.sleep(step)
            slept += step
        self.check()


def _atomic_write(ds: xr.Dataset, out_path: Path) -> None:
    out_path = Path(out_path)
    tmp_path = out_path.with_name(out_path.name + _TMP_SUFFIX)
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_netcdf(tmp_path, engine="h5netcdf")
    os.replace(tmp_path, out_path)


def compute_monthly_climatology(
    input_path: str | Path,
    output_path: str | Path,
    interrupter: Interrupter,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Mean of ``tas`` per calendar month (1..12), ignoring NaN cells.

    Output variable ``tas`` has dimensions ``(month, lat, lon)``.  Months with
    no valid observations at a cell are NaN; anomaly subtraction then yields
    missing values for those cells, exactly as required.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    ds = validate_dataset(input_path)
    try:
        tas = ds["tas"]
        times = pd.DatetimeIndex(ds["time"].values)
        months = month_index(times)
        n_lat, n_lon = ds.sizes["lat"], ds.sizes["lon"]
        clim = np.full((12, n_lat, n_lon), np.nan, dtype=np.float32)

        for month in range(1, 13):
            interrupter.check()
            selection = tas.loc[months == month, :, :]
            if selection.sizes["time"] > 0:
                with np.errstate(invalid="ignore"), warnings_suppress():
                    clim[month - 1] = np.nanmean(
                        selection.values, axis=0
                    ).astype(np.float32)
            if on_progress is not None:
                on_progress("climatology", month, 12)

        # Phase boundary: never publish when cancellation raced with the last
        # slice of the computation.
        interrupter.check()
        out = xr.Dataset(
            {
                "tas": (
                    ("month", "lat", "lon"),
                    clim,
                    {"units": "K", "long_name": "monthly mean temperature"},
                )
            },
            coords={
                "month": (
                    "month",
                    np.arange(1, 13, dtype=np.int32),
                    {"long_name": "calendar month", "units": "month"},
                ),
                "lat": ds["lat"].copy(),
                "lon": ds["lon"].copy(),
            },
            attrs={
                "title": "monthly temperature climatology",
                "source_file": input_path.name,
            },
        )
        interrupter.check()
        _atomic_write(out, output_path)
    finally:
        ds.close()


def compute_anomalies(
    target_path: str | Path,
    climatology_path: str | Path,
    output_path: str | Path,
    interrupter: Interrupter,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """target minus same-calendar-month historical mean, per grid cell.

    Coordinates of the target are preserved verbatim.  Where the climatology
    has no value the anomaly is NaN.
    """
    target_path = Path(target_path)
    climatology_path = Path(climatology_path)
    output_path = Path(output_path)

    target = validate_dataset(target_path)
    clim_ds = xr.open_dataset(climatology_path, engine="h5netcdf")
    try:
        if set(clim_ds["tas"].dims) != {"month", "lat", "lon"}:
            raise ValueError("climatology file must contain tas(month,lat,lon)")
        clim = clim_ds["tas"].values  # shape (12, lat, lon)
        times = pd.DatetimeIndex(target["time"].values)
        months = month_index(times)
        n_time = target.sizes["time"]

        result = np.full(
            (n_time, target.sizes["lat"], target.sizes["lon"]),
            np.nan,
            dtype=np.float32,
        )
        # Process one time step at a time so the interrupt window stays small.
        values = target["tas"].values
        for t in range(n_time):
            interrupter.check()
            month_clim = clim[months[t] - 1]
            with np.errstate(invalid="ignore"):
                result[t] = (values[t] - month_clim).astype(np.float32)
            if on_progress is not None:
                on_progress("anomaly", t + 1, n_time)

        # Phase boundary: cancellation racing completion suppresses publish.
        interrupter.check()
        out = xr.Dataset(
            {
                "tas": (
                    ("time", "lat", "lon"),
                    result,
                    {
                        "units": "K",
                        "long_name": "temperature anomaly",
                        "cell_methods": "time: mean (target minus climatology)",
                    },
                )
            },
            coords={
                "time": target["time"].copy(),
                "lat": target["lat"].copy(),
                "lon": target["lon"].copy(),
            },
            attrs={
                "title": "monthly temperature anomalies",
                "source_file": target_path.name,
                "climatology_file": climatology_path.name,
            },
        )
        interrupter.check()
        _atomic_write(out, output_path)
    finally:
        clim_ds.close()
        target.close()


class warnings_suppress:
    """Tiny context manager avoiding a module-level import cycle surprise."""

    def __enter__(self) -> "warnings_suppress":
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        return self

    def __exit__(self, *exc: object) -> None:
        self._ctx.__exit__(*exc)


# ---------------------------------------------------------------------------
# Synthetic sample data
# ---------------------------------------------------------------------------


def make_synthetic_dataset(
    times: pd.DatetimeIndex,
    n_lat: int,
    n_lon: int,
    seed: int,
    nan_fraction: float = 0.02,
) -> xr.Dataset:
    """Build a physically plausible-looking monthly Kelvin temperature field."""
    rng = np.random.default_rng(seed)
    lat = np.linspace(-60.0, 60.0, n_lat, dtype=np.float64)
    lon = np.linspace(0.0, 360.0 - 360.0 / n_lon, n_lon, dtype=np.float64)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")

    n_time = len(times)
    data = np.empty((n_time, n_lat, n_lon), dtype=np.float32)
    doy = np.array([t.timetuple().tm_yday for t in times], dtype=np.float64)
    seasonal = np.cos(2.0 * np.pi * (doy - 15.0) / 365.25)
    for t, stamp in enumerate(times):
        base = 288.15 - 0.45 * np.abs(lat_g)  # colder towards the poles
        season = 9.0 * seasonal[t] * np.cos(np.deg2rad(lat_g))
        wave = 2.5 * np.sin(np.deg2rad(lon_g) + stamp.month)
        noise = rng.normal(0.0, 0.6, size=lat_g.shape)
        data[t] = (base + season + wave + noise).astype(np.float32)

    if nan_fraction > 0.0:
        mask = rng.random(data.shape) < nan_fraction
        data[mask] = np.nan

    return xr.Dataset(
        {
            "tas": (
                ("time", "lat", "lon"),
                data,
                {"units": "K", "long_name": "near-surface air temperature"},
            )
        },
        coords={
            "time": (
                "time",
                pd.DatetimeIndex(times).values,
                {"standard_name": "time"},
            ),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.8", "title": "synthetic monthly tas sample"},
    )


def write_sample_files(
    baseline_path: str | Path,
    target_paths: dict[str, Path],
    n_lat: int = 12,
    n_lon: int = 16,
    baseline_years: int = 5,
) -> None:
    """Generate the fixed historical baseline and demo target NetCDF files."""
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_times = pd.date_range(
        "2011-01-01",
        periods=12 * baseline_years,
        freq="MS",
    )
    baseline = make_synthetic_dataset(
        baseline_times, n_lat, n_lon, seed=20240101
    )
    _atomic_write(baseline, baseline_path)
    baseline.close()

    target_times = pd.date_range("2021-01-01", periods=6, freq="MS")
    for index, (label, path) in enumerate(target_paths.items()):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = make_synthetic_dataset(
            target_times, n_lat, n_lon, seed=99000 + index
        )
        # A gentle warming signal so anomalies in the demo are visibly nonzero.
        target["tas"].values[:] = target["tas"].values + 1.7 + 0.05 * index
        _atomic_write(target, path)
        target.close()
