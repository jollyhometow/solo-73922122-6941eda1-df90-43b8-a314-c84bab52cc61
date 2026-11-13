"""Integration tests for the JobManager rules."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from app.manager import JobManager, LimitExceededError
from app.models import BaselineStatus, JobState
from app.netcdf_io import DatasetError
from tests.conftest import wait_until


def _payload(path: Path) -> bytes:
    return path.read_bytes()


async def _start(settings) -> JobManager:
    manager = JobManager(settings)
    await manager.startup()
    return manager


async def _submit(manager: JobManager, target_file: Path, name: str = "t.nc"):
    return await manager.submit(name, _payload(target_file))


async def _states(manager: JobManager):
    return {jid: job.state for jid, job in manager._jobs.items()}


# --------------------------------------------------------------------- happy
async def test_end_to_end_done_and_downloadable(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        job = await _submit(manager, target_file)
        await wait_until(lambda: job.terminal, timeout=40)
        assert job.state is JobState.DONE
        assert job.output_path.is_file()
        with xr.open_dataset(job.output_path, engine="h5netcdf") as out:
            assert tuple(out["tas"].dims) == ("time", "lat", "lon")
            assert out.sizes["time"] == 4
    finally:
        await manager.shutdown()


async def test_climatology_is_reused_by_second_job(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        first = await _submit(manager, target_file, "a.nc")
        await wait_until(lambda: first.state is JobState.DONE, timeout=40)
        assert settings.baseline_work_path.is_file()

        second = await _submit(manager, target_file, "b.nc")
        # No active baseline run should be needed anymore.
        await wait_until(lambda: second.terminal, timeout=40)
        assert second.state is JobState.DONE
        assert manager._baseline_run is None or (
            manager._baseline_run.status is BaselineStatus.DONE
        )
    finally:
        await manager.shutdown()


# ------------------------------------------------------------- concurrency
async def test_max_two_active_jobs_rejected_with_409(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        jobs = []
        jobs.append(await _submit(manager, target_file))
        jobs.append(await _submit(manager, target_file))
        with pytest.raises(LimitExceededError):
            await _submit(manager, target_file)

        # Terminal jobs free capacity again.
        await wait_until(
            lambda: all(j.state is JobState.DONE for j in jobs), timeout=60
        )
        third = await _submit(manager, target_file)
        await wait_until(lambda: third.terminal, timeout=40)
        assert third.state is JobState.DONE
    finally:
        await manager.shutdown()


async def test_invalid_grid_upload_is_rejected_and_not_counted(
    settings, baseline_file
):
    import pandas as pd

    from app.compute import make_synthetic_dataset

    manager = await _start(settings)
    try:
        times = pd.date_range("2021-01-01", periods=2, freq="MS")
        bad = make_synthetic_dataset(times, 5, 8, seed=99)
        bad_path = settings.samples_dir / "bad-grid.nc"
        bad.to_netcdf(bad_path, engine="h5netcdf")
        bad.close()
        with pytest.raises(DatasetError):
            await manager.submit("bad.nc", bad_path.read_bytes())
        assert manager._jobs == {}
    finally:
        await manager.shutdown()


# ----------------------------------------------------------- shared baseline
async def test_two_jobs_share_one_baseline_run(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")

        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        run = manager._baseline_run
        assert run.waiters == {a.id, b.id}
        assert run.proc is not None and run.proc.pid  # exactly one process

        await wait_until(
            lambda: a.state is JobState.DONE and b.state is JobState.DONE,
            timeout=60,
        )
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancel_one_waiter_does_not_affect_other(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        run = manager._baseline_run

        await manager.cancel(a.id)
        await wait_until(lambda: a.state is JobState.CANCELLED, timeout=10)

        # The shared run survives because b is still waiting.
        assert run.status is BaselineStatus.RUNNING
        assert run.waiters == {b.id}
        assert run.proc.returncode is None
        assert not a.target_path.parent.exists()  # cancelled files cleaned

        await wait_until(lambda: b.state is JobState.DONE, timeout=60)
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancel_last_waiter_terminates_baseline(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        run = manager._baseline_run
        pid = run.proc.pid

        await manager.cancel(a.id)
        await wait_until(lambda: a.state is JobState.CANCELLED, timeout=10)
        await manager.cancel(b.id)

        # Process must be reaped, run terminated, artifacts cleaned.
        await wait_until(
            lambda: run.status is BaselineStatus.TERMINATED, timeout=15
        )
        assert run.proc.returncode is not None
        assert run.proc.pid == pid
        assert not settings.baseline_work_path.exists()
        assert not run.part_path.exists()
        assert b.state is JobState.CANCELLED

        # A later submission must NOT attach to the terminated/draining run.
        # It may briefly observe the draining slot before spawning a fresh
        # run, so assert on the outcome: the job completes and never joins the
        # old run's waiter set.
        old_run = run
        c = await _submit(manager, target_file, "c.nc")
        assert c.id not in old_run.waiters

        async def _fresh_run_or_done():
            current = manager._baseline_run
            return c.state is JobState.DONE or (
                current is not None
                and current is not old_run
                and current.status
                in (BaselineStatus.STARTING, BaselineStatus.RUNNING)
            )

        await wait_until(_fresh_run_or_done, timeout=15)
        await wait_until(lambda: c.state is JobState.DONE, timeout=60)
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancel_queued_job_before_worker_starts(
    settings, baseline_file, target_file
):
    # Fill both slots first, then cancel a queued one (it has no process).
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None, timeout=10
        )
        await manager.cancel(b.id)
        await wait_until(lambda: b.state is JobState.CANCELLED, timeout=10)
        assert b.proc is None
        await wait_until(lambda: a.state is JobState.DONE, timeout=60)
    finally:
        await manager.shutdown()


# ------------------------------------------------------------- own-process
async def test_cancel_running_anomaly_reaps_and_reports_cancelled(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        seed = await _submit(manager, target_file, "seed.nc")
        await wait_until(lambda: seed.state is JobState.DONE, timeout=40)

        job = await _submit(manager, target_file, "late.nc")
        await wait_until(lambda: job.proc is not None, timeout=15)
        proc = job.proc
        await manager.cancel(job.id)
        assert job.state is JobState.CANCELLING  # not cancelled yet
        await wait_until(lambda: job.state is JobState.CANCELLED, timeout=20)
        assert proc.returncode is not None  # actually reaped
        assert not job.target_path.parent.exists()
        assert not job.output_path.exists()
    finally:
        await manager.shutdown()


async def test_cancel_racing_completion_never_publishes_late_result(
    settings, baseline_file, target_file
):
    # Hammer cancellation while the anomaly worker is on its way out.
    manager = await _start(settings)
    try:
        seed = await _submit(manager, target_file, "seed.nc")
        await wait_until(lambda: seed.state is JobState.DONE, timeout=40)

        for attempt in range(3):
            job = await _submit(manager, target_file, f"race{attempt}.nc")
            await wait_until(lambda: job.proc is not None, timeout=15)
            # Cancel immediately: either the worker exits first or publish is
            # suppressed; both outcomes must remain internally consistent.
            await manager.cancel(job.id)
            await wait_until(lambda: job.terminal, timeout=30)
            if job.state is JobState.DONE:
                assert job.output_path.is_file()
            else:
                assert job.state is JobState.CANCELLED
                assert not job.output_path.exists()
    finally:
        await manager.shutdown()


async def test_baseline_worker_failure_marks_waiters_error(
    settings, target_file, monkeypatch
):
    # Break the baseline input after manager startup validation by pointing the
    # worker at an unreadable file: simulate via a corrupt replacement.
    manager = await _start(settings)
    try:
        good = settings.baseline_input_path
        backup = good.with_suffix(".nc.bak")
        good.rename(backup)
        good.write_bytes(b"not an hdf5 file at all")
        job = await _submit(manager, target_file, "x.nc")
        await wait_until(lambda: job.terminal, timeout=20)
        assert job.state is JobState.ERROR
        assert "baseline" in job.message.lower()
        good.unlink()
        backup.rename(good)
    finally:
        await manager.shutdown()


async def test_cancel_is_idempotent_and_terminal_jobs_unchanged(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        job = await _submit(manager, target_file, "a.nc")
        await wait_until(lambda: job.terminal, timeout=40)
        before = job.state
        again = await manager.cancel(job.id)
        assert again.state is before
    finally:
        await manager.shutdown()


async def test_shutdown_terminates_all_workers(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    a = await _submit(manager, target_file, "a.nc")
    b = await _submit(manager, target_file, "b.nc")
    await wait_until(
        lambda: manager._baseline_run is not None
        and manager._baseline_run.proc is not None,
        timeout=10,
    )
    base_proc = manager._baseline_run.proc
    await manager.shutdown()
    assert base_proc.returncode is not None
    for job in (a, b):
        assert job.proc is None or job.proc.returncode is not None
