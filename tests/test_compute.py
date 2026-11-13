"""Tests for validation, climatology math and anomaly subtraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from app.compute import (
    Cancelled,
    Interrupter,
    compute_anomalies,
    compute_monthly_climatology,
    make_synthetic_dataset,
)
from app.config import Settings
from app.netcdf_io import (
    DatasetError,
    load_grid_signature,
    validate_dataset,
    validate_target,
)


def test_valid_synthetic_dataset_roundtrip(target_file: Path):
    ds = validate_dataset(target_file)
    try:
        assert tuple(ds["tas"].dims) == ("time", "lat", "lon")
        assert ds["tas"].attrs["units"].lower() == "k"
        times = pd.DatetimeIndex(ds["time"].values)
        assert (times.day == 1).all()
    finally:
        ds.close()


def test_duplicate_time_rejected(settings: Settings, baseline_file: Path):
    bad = settings.samples_dir / "dup.nc"
    times = pd.DatetimeIndex(
        ["2021-01-01", "2021-01-01", "2021-03-01"], dtype="datetime64[ns]"
    )
    ds = make_synthetic_dataset(times, 6, 8, seed=3)
    ds.to_netcdf(bad, engine="h5netcdf")
    ds.close()
    with pytest.raises(DatasetError, match="duplicate"):
        validate_dataset(bad)


def test_non_first_of_month_rejected(settings: Settings):
    bad = settings.samples_dir / "mid.nc"
    times = pd.DatetimeIndex(["2021-01-15"], dtype="datetime64[ns]")
    ds = make_synthetic_dataset(times, 6, 8, seed=4)
    ds.to_netcdf(bad, engine="h5netcdf")
    ds.close()
    with pytest.raises(DatasetError, match="first day"):
        validate_dataset(bad)


def test_celsius_units_rejected(settings: Settings):
    bad = settings.samples_dir / "celsius.nc"
    times = pd.date_range("2021-01-01", periods=2, freq="MS")
    ds = make_synthetic_dataset(times, 6, 8, seed=5)
    ds["tas"].attrs["units"] = "degC"
    ds.to_netcdf(bad, engine="h5netcdf")
    ds.close()
    with pytest.raises(DatasetError, match="Kelvin"):
        validate_dataset(bad)


def test_grid_mismatch_rejected(settings: Settings, baseline_file: Path):
    times = pd.date_range("2021-01-01", periods=2, freq="MS")
    other = make_synthetic_dataset(times, n_lat=5, n_lon=8, seed=6)
    path = settings.samples_dir / "other-grid.nc"
    other.to_netcdf(path, engine="h5netcdf")
    other.close()
    grid = load_grid_signature(baseline_file)
    with pytest.raises(DatasetError, match="identical"):
        validate_target(path, grid)


def test_climatology_ignores_nan_and_anomaly_math(
    settings: Settings, baseline_file: Path, target_file: Path, tmp_path: Path
):
    clim_path = tmp_path / "clim.nc"
    out_path = tmp_path / "anom.nc"
    compute_monthly_climatology(
        baseline_file, clim_path, Interrupter()
    )
    with xr.open_dataset(clim_path, engine="h5netcdf") as clim:
        assert tuple(clim["tas"].dims) == ("month", "lat", "lon")
        assert clim.sizes["month"] == 12
        # Not every cell can be NaN in the synthetic data.
        assert np.isfinite(clim["tas"].values).any()

    compute_anomalies(target_file, clim_path, out_path, Interrupter())
    with xr.open_dataset(target_file, engine="h5netcdf") as tgt, xr.open_dataset(
        out_path, engine="h5netcdf"
    ) as out:
        # Target coordinates preserved verbatim.
        np.testing.assert_array_equal(
            out["time"].values, tgt["time"].values
        )
        np.testing.assert_array_equal(out["lat"].values, tgt["lat"].values)
        np.testing.assert_array_equal(out["lon"].values, tgt["lon"].values)
        months = pd.DatetimeIndex(tgt["time"].values).month
        expected = tgt["tas"].values - clim["tas"].values[months - 1]
        np.testing.assert_allclose(
            out["tas"].values, expected, rtol=1e-5, atol=1e-5, equal_nan=True
        )
        assert out["tas"].dtype == np.float32


def test_missing_baseline_month_yields_nan(
    settings: Settings, tmp_path: Path
):
    # Baseline containing only January/February: other months have no mean.
    base_times = pd.date_range("2011-01-01", periods=2, freq="MS")
    base = make_synthetic_dataset(base_times, 4, 5, seed=11)
    base_path = tmp_path / "base.nc"
    base.to_netcdf(base_path, engine="h5netcdf")
    base.close()

    clim_path = tmp_path / "clim.nc"
    compute_monthly_climatology(base_path, clim_path, Interrupter())

    tgt_times = pd.date_range("2021-01-01", periods=3, freq="MS")  # Jan..Mar
    tgt = make_synthetic_dataset(tgt_times, 4, 5, seed=12)
    tgt_path = tmp_path / "tgt.nc"
    tgt.to_netcdf(tgt_path, engine="h5netcdf")
    tgt.close()

    out_path = tmp_path / "anom.nc"
    compute_anomalies(tgt_path, clim_path, out_path, Interrupter())
    with xr.open_dataset(out_path, engine="h5netcdf") as out:
        values = out["tas"].values
        # January/February have a climatology (some cells may be NaN because
        # the synthetic input itself contains missing values).
        assert np.isfinite(values[0]).any()
        assert np.isfinite(values[1]).any()
        assert np.isnan(values[2]).all()     # March: no baseline value -> NaN


def test_interrupter_raises_and_sleeps_respond():
    flag = {"set": False}
    interrupter = Interrupter(lambda: flag["set"])
    flag["set"] = True
    with pytest.raises(Cancelled):
        interrupter.check()
    flag["set"] = False
    interrupter.sleep(0.01)  # must not raise
    flag["set"] = True
    with pytest.raises(Cancelled):
        interrupter.sleep(5.0)


def test_interrupted_climatology_leaves_no_output(
    baseline_file: Path, tmp_path: Path
):
    out = tmp_path / "clim.nc"
    interrupter = Interrupter(lambda: True)
    with pytest.raises(Cancelled):
        compute_monthly_climatology(baseline_file, out, interrupter)
    assert not out.exists()
    assert not tmp_path.joinpath("clim.nc.part").exists()


def test_interrupted_anomaly_leaves_no_output(
    baseline_file: Path, target_file: Path, tmp_path: Path
):
    clim = tmp_path / "clim.nc"
    compute_monthly_climatology(baseline_file, clim, Interrupter())
    out = tmp_path / "anom.nc"
    with pytest.raises(Cancelled):
        compute_anomalies(
            target_file, clim, out, Interrupter(lambda: True)
        )
    assert not out.exists()
