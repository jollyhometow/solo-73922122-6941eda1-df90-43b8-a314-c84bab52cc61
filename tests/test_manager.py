"""Integration tests for the JobManager rules."""

from __future__ import annotations

import asyncio
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
        # The last waiter stays cancelling until the shared process is reaped.
        assert b.state is JobState.CANCELLING

        # Process must be reaped, run terminated, artifacts cleaned.
        await wait_until(
            lambda: run.status is BaselineStatus.TERMINATED, timeout=15
        )
        assert run.proc.returncode is not None
        assert run.proc.pid == pid
        assert not settings.baseline_work_path.exists()
        assert not run.part_path.exists()
        await wait_until(lambda: b.state is JobState.CANCELLED, timeout=10)
        assert not b.target_path.parent.exists()

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


async def test_last_waiter_stays_cancelling_until_baseline_reaped(
    settings, baseline_file, target_file
):
    # Regression: the last waiting job used to show ``cancelled`` (directory
    # removed, slot released) while the shared baseline process was still alive.
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

        # a still has another waiter behind it: shared computation survives.
        await manager.cancel(a.id)
        await wait_until(lambda: a.state is JobState.CANCELLED, timeout=10)
        assert run.status is BaselineStatus.RUNNING
        assert run.proc.returncode is None

        # Last waiter: must be parked in cancelling and keep occupying a slot
        # while the shared worker is still alive.
        await manager.cancel(b.id)
        assert b.state is JobState.CANCELLING
        assert b.active  # slot still held
        active = sum(1 for j in manager._jobs.values() if j.active)
        assert active == 1
        # Directory must not be removed before the run is reaped.
        assert b.target_path.parent.exists()
        assert run.status is BaselineStatus.DRAINING

        # Repeated cancels must be no-ops and cannot finish the teardown early.
        again = await manager.cancel(b.id)
        assert again.state is JobState.CANCELLING
        assert again is b
        assert run.status is BaselineStatus.DRAINING
        assert b.target_path.parent.exists()

        # Only after exit + reap + cleanup may the job become cancelled.
        await wait_until(lambda: run.status is BaselineStatus.TERMINATED, timeout=15)
        assert run.closed.is_set()
        assert run.proc.returncode is not None
        await wait_until(lambda: b.state is JobState.CANCELLED, timeout=10)
        assert not b.target_path.parent.exists()
        assert not run.part_path.exists()
        assert not settings.baseline_work_path.exists()
    finally:
        await manager.shutdown()


async def test_new_job_during_drain_starts_own_baseline_and_is_not_clobbered(
    settings, baseline_file, target_file
):
    # While the last-waiter teardown is still in flight, the remaining slot
    # accepts a new job that starts a fresh baseline run; the old teardown must
    # never delete or overwrite the new job's files.
    manager = await _start(settings)
    try:
        a = await _submit(manager, target_file, "a.nc")
        b = await _submit(manager, target_file, "b.nc")
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        old_run = manager._baseline_run

        await manager.cancel(a.id)
        await wait_until(lambda: a.state is JobState.CANCELLED, timeout=10)
        await manager.cancel(b.id)
        assert b.state is JobState.CANCELLING

        # One slot is still free: a fresh submission is accepted.
        c = await _submit(manager, target_file, "c.nc")
        assert c.id not in old_run.waiters
        c_dir = c.target_path.parent
        assert c_dir.exists()

        async def _fresh_run():
            current = manager._baseline_run
            return current is not None and current is not old_run

        await wait_until(_fresh_run, timeout=15)
        new_run = manager._baseline_run
        assert new_run is not old_run
        assert new_run.status in (
            BaselineStatus.STARTING,
            BaselineStatus.RUNNING,
        )
        assert old_run.part_path != new_run.part_path
        assert old_run.cancel_file != new_run.cancel_file

        # Repeated cancel of the old job while both runs coexist.
        await manager.cancel(b.id)
        assert b.state is JobState.CANCELLING

        await wait_until(
            lambda: old_run.status is BaselineStatus.TERMINATED, timeout=15
        )
        await wait_until(lambda: b.state is JobState.CANCELLED, timeout=10)

        # Old teardown left the new job's directory and artifacts untouched.
        assert c_dir.exists()
        assert c.target_path.is_file()
        if new_run.proc is not None and new_run.proc.returncode is None:
            pass  # new baseline still legitimately running

        await wait_until(lambda: c.state is JobState.DONE, timeout=60)
        assert c.output_path.is_file()
        assert settings.baseline_work_path.is_file()
    finally:
        await manager.shutdown()


async def test_cancelled_job_never_enters_anomaly_or_serves_result(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        job = await _submit(manager, target_file, "a.nc")
        await wait_until(
            lambda: manager._baseline_run is not None,
            timeout=10,
        )
        await manager.cancel(job.id)
        await wait_until(lambda: job.terminal, timeout=20)
        assert job.state is JobState.CANCELLED
        assert job.proc is None  # never launched an anomaly worker
        assert not job.output_path.exists()
        assert not job.target_path.parent.exists()
        assert manager.baseline_climatology_ready() is False
    finally:
        await manager.shutdown()


async def test_repeated_cancel_hammer_during_drain_never_finalizes_early(
    settings, baseline_file, target_file
):
    manager = await _start(settings)
    try:
        jobs = [await _submit(manager, target_file, f"{i}.nc") for i in range(2)]
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run.status is BaselineStatus.RUNNING,
            timeout=10,
        )
        old_run = manager._baseline_run
        for job in jobs:
            await manager.cancel(job.id)

        last = jobs[-1]
        assert last.state is JobState.CANCELLING

        # Hammer cancel while the shared worker is still being reaped.  None of
        # these calls may finalize the job or release its slot.
        for _ in range(5):
            again = await manager.cancel(last.id)
            assert again.state is JobState.CANCELLING
            assert last.target_path.parent.exists()
            await asyncio.sleep(0)

        # The remaining slot stays usable throughout the teardown.
        new_job = await _submit(manager, target_file, "new.nc")
        await wait_until(
            lambda: manager._baseline_run is not None
            and manager._baseline_run is not old_run,
            timeout=15,
        )

        await wait_until(lambda: old_run.closed.is_set(), timeout=15)
        await wait_until(lambda: last.state is JobState.CANCELLED, timeout=10)
        assert not last.target_path.parent.exists()

        # The interleaved job must survive the old teardown untouched.
        assert new_job.target_path.is_file()
        await wait_until(lambda: new_job.state is JobState.DONE, timeout=60)
        assert new_job.output_path.is_file()
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
