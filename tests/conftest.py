"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path

import pandas as pd
import pytest

from app.compute import make_synthetic_dataset
from app.config import Settings


@pytest.fixture()
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("SLOW_STEP_DELAY", "0.01")
    monkeypatch.setenv("SLOW_STEPS", "2")
    monkeypatch.setenv("WORKER_TERM_TIMEOUT", "4")
    monkeypatch.setenv("POLL_INTERVAL", "0.02")
    settings = Settings.from_env()
    settings.ensure_dirs()
    return settings


@pytest.fixture()
def baseline_file(settings: Settings) -> Path:
    settings.ensure_dirs()
    times = pd.date_range("2011-01-01", periods=36, freq="MS")
    ds = make_synthetic_dataset(times, n_lat=6, n_lon=8, seed=42)
    path = settings.baseline_input_path
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()
    return path


@pytest.fixture()
def target_file(settings: Settings, baseline_file: Path) -> Path:
    times = pd.date_range("2021-01-01", periods=4, freq="MS")
    ds = make_synthetic_dataset(times, n_lat=6, n_lon=8, seed=7)
    ds["tas"].values[:] = ds["tas"].values + 2.0
    path = settings.samples_dir / "target.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()
    return path


def run(coro):
    """Run a coroutine on a fresh loop (Python 3.11 has no asyncio.run issue)."""
    return asyncio.run(coro)


async def wait_until(predicate, timeout: float = 20.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(interval)


def assert_state(job, state) -> None:
    assert job.state is state, f"expected {state}, got {job.state}: {job.message}"


def round_trip_bytes(path: Path) -> bytes:
    return path.read_bytes()
