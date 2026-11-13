"""Async task manager: shared baseline, two-job limit, strict cancellation.

Concurrency model
-----------------
A single :class:`asyncio.Lock` guards every state transition.  Subprocesses are
awaited outside the lock; the lock is only re-acquired at handoff points,
where cancellation flags are re-checked before publishing results or starting
a new phase.

Shared baseline
---------------
All queued jobs attach to one :class:`BaselineRun`.  When the *last* attached
job cancels, the run enters ``draining``: its worker is terminated and reaped,
partial files are removed, and no later submission may attach to it — a new
submission starts a fresh run instead.  Once a climatology file exists every
later job skips the baseline entirely.

Cancellation
------------
A job is only reported ``cancelled`` after the process it depends on has
exited and been reaped and its files have been cleaned up.  Every phase
handoff re-checks the flag under the lock, so a cancel racing with completion
neither starts the anomaly phase nor publishes a late result.

For the *last* waiter of a shared baseline run the cancellation is two-stage:
the job becomes ``cancelling`` (and keeps occupying its slot) while the shared
worker is terminated, reaped and its artifacts removed; only when the run is
fully closed does the job become ``cancelled``.  Repeated cancel requests while
this teardown is in flight are no-ops and can never finalize the job early.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .models import (
    BaselineStatus,
    JobStage,
    JobState,
    TERMINAL_STATES,
)
from .netcdf_io import DatasetError, GridSignature, load_grid_signature, validate_target

LOG = logging.getLogger("manager")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    id: str
    upload_name: str
    target_path: Path
    output_path: Path
    created_at: datetime
    updated_at: datetime
    state: JobState = JobState.QUEUED
    stage: JobStage = JobStage.WAITING_BASELINE
    message: str = "accepted; waiting for baseline climatology"
    proc: asyncio.subprocess.Process | None = None
    baseline_active: bool = False
    # The shared baseline run this job is currently waiting on (if any).  When
    # the last attached waiter cancels, this is the run whose exit/reap/cleanup
    # must complete before the job may become ``cancelled``.
    baseline_run: "BaselineRun | None" = None
    supervisor: asyncio.Task | None = None
    cancel_file: Path | None = None
    baseline_attempts: int = 0

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def active(self) -> bool:
        return not self.terminal


@dataclass
class BaselineRun:
    id: str
    output_path: Path
    part_path: Path
    log_path: Path
    cancel_file: Path
    waiters: set[str] = field(default_factory=set)
    status: BaselineStatus = BaselineStatus.STARTING
    proc: asyncio.subprocess.Process | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)
    log_handle: object = None
    message: str = ""
    supervisor: asyncio.Task | None = None
    # Jobs parked in ``cancelling`` until this run is fully closed (worker
    # exited and reaped, partial files and cancel flags removed).
    gated_cancels: set[str] = field(default_factory=set)
    # Set once, and only once, the teardown/publish path has finished.  This is
    # distinct from ``event`` which may already be set while the worker is still
    # alive (e.g. immediately when draining starts).
    closed: asyncio.Event = field(default_factory=asyncio.Event)


class LimitExceededError(RuntimeError):
    pass


class JobNotFoundError(KeyError):
    pass


class JobManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._baseline_run: BaselineRun | None = None
        self._baseline_grid: GridSignature | None = None
        self._shutdown = False

    # ------------------------------------------------------------------ setup
    async def startup(self) -> None:
        self.settings.ensure_dirs()
        baseline_input = self.settings.baseline_input_path
        if not baseline_input.is_file():
            raise FileNotFoundError(
                f"fixed historical baseline NetCDF not found: {baseline_input}"
            )
        # Validate the configured baseline eagerly so misconfiguration fails
        # the service instead of the first request.
        self._baseline_grid = load_grid_signature(baseline_input)
        if self.baseline_climatology_ready():
            LOG.info("using existing climatology %s", self.settings.baseline_work_path)
        self._clean_stale_jobs()
        LOG.info(
            "manager ready; baseline grid %s, max active jobs %d",
            self._baseline_grid.as_dict(),
            self.settings.max_active_jobs,
        )

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutdown = True
            jobs = list(self._jobs.values())
            run = self._baseline_run

        terminators = [
            self._terminate(job.proc, job.cancel_file, job.id[:8])
            for job in jobs
            if job.proc is not None and job.proc.returncode is None
        ]
        if (
            run is not None
            and run.proc is not None
            and run.proc.returncode is None
        ):
            terminators.append(
                self._terminate(run.proc, run.cancel_file, run.id[:8])
            )
        if terminators:
            await asyncio.gather(*terminators, return_exceptions=True)

        # Let supervisors run their reaping/cleanup to completion within a
        # bounded window rather than abandoning them mid-handoff.
        supervisors = [
            job.supervisor for job in jobs if job.supervisor is not None
        ]
        if run is not None and run.supervisor is not None:
            supervisors.append(run.supervisor)
        if supervisors:
            await asyncio.wait(
                [asyncio.shield(t) for t in supervisors],
                timeout=self.settings.worker_term_timeout + 2.0,
            )
        for task in supervisors:
            if not task.done():
                task.cancel()

        # Safety net: every process has been terminated above, so any job still
        # parked in ``cancelling`` can be finalized now instead of lingering.
        async with self._lock:
            for job in jobs:
                if job.state is JobState.CANCELLING:
                    gated = job.baseline_run
                    if gated is not None and job.id in gated.gated_cancels:
                        gated.gated_cancels.discard(job.id)
                    self._finalize_cancel_locked(job)

    def _clean_stale_jobs(self) -> None:
        jobs_dir = self.settings.jobs_dir
        if not jobs_dir.exists():
            return
        for child in jobs_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    # --------------------------------------------------------------- queries
    def baseline_climatology_ready(self) -> bool:
        return self.settings.baseline_work_path.is_file()

    def list_jobs(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def get_job(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def baseline_snapshot(self) -> dict:
        run = self._baseline_run
        ready = self.baseline_climatology_ready()
        snapshot = {
            "configured_file": self.settings.baseline_input_path.name,
            "climatology_ready": ready,
            "active_run": None,
        }
        if self._baseline_grid is not None:
            snapshot["grid"] = self._baseline_grid.as_dict()
        if run is not None:
            snapshot["active_run"] = {
                "run_id": run.id,
                "status": run.status.value,
                "waiters": sorted(run.waiters),
                "message": run.message,
            }
        return snapshot

    # ---------------------------------------------------------------- submit
    async def submit(self, filename: str, payload: bytes) -> JobRecord:
        async with self._lock:
            if self._shutdown:
                raise RuntimeError("service is shutting down")
            active = sum(1 for job in self._jobs.values() if job.active)
            if active >= self.settings.max_active_jobs:
                raise LimitExceededError(
                    f"at most {self.settings.max_active_jobs} unfinished "
                    "jobs are accepted"
                )

            job_id = uuid.uuid4().hex
            job_dir = self.settings.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            target_path = job_dir / "target.nc"
            target_path.write_bytes(payload)

            # Full contract + identical-grid validation before acceptance.
            try:
                ds = validate_target(target_path, self._baseline_grid)
            except DatasetError:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise
            else:
                ds.close()

            now = _utcnow()
            job = JobRecord(
                id=job_id,
                upload_name=filename,
                target_path=target_path,
                output_path=job_dir / "anomaly.nc",
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = job
            job.supervisor = asyncio.create_task(
                self._supervise_job(job_id), name=f"job-{job_id[:8]}"
            )
            LOG.info("accepted job %s (upload=%s)", job_id, filename)
            return job

    # ---------------------------------------------------------------- cancel
    async def cancel(self, job_id: str) -> JobRecord:
        async with self._lock:
            job = self._jobs[job_id]
            if job.terminal or job.state is JobState.CANCELLING:
                # Terminal job: idempotent.  Cancelling job: teardown is
                # already in flight; a repeated request must not shortcut it.
                return job
            return await self._cancel_locked(job)

    async def _cancel_locked(self, job: JobRecord) -> JobRecord:
        """Mark/perform cancellation. Caller must hold the lock."""
        if job.terminal or job.state is JobState.CANCELLING:
            return job

        # Detach from any shared baseline run first.
        run = job.baseline_run
        if run is not None and job.id in run.waiters:
            run.waiters.discard(job.id)
            job.baseline_active = False

            if not run.waiters and run.status in (
                BaselineStatus.STARTING,
                BaselineStatus.RUNNING,
            ):
                # Last waiter leaves: the shared worker must be terminated and
                # reaped and its artifacts cleaned before the job may become
                # cancelled.  Park the job in ``cancelling`` and keep its slot.
                job.state = JobState.CANCELLING
                job.stage = JobStage.WAITING_BASELINE
                job.message = (
                    "cancellation requested; waiting for shared baseline "
                    "process exit, reaping and cleanup"
                )
                job.updated_at = _utcnow()
                run.gated_cancels.add(job.id)
                await self._drain_baseline_run_locked(run)
                return job

        if job.proc is None:
            # Queued job with no own process and no shared run to wait for:
            # nothing has to be reaped for this job, finalize immediately.
            self._finalize_cancel_locked(job)
            return job

        job.state = JobState.CANCELLING
        job.stage = JobStage.ANOMALY
        job.message = "cancellation requested; waiting for worker exit"
        job.updated_at = _utcnow()
        if job.proc.returncode is None:
            asyncio.create_task(
                self._terminate(job.proc, job.cancel_file, job.id[:8]),
                name=f"kill-{job.id[:8]}",
            )
        return job

    async def _drain_baseline_run_locked(self, run: BaselineRun) -> None:
        """Begin the last-waiter teardown of a shared baseline run.

        Only marks the run as draining and arms termination; the actual
        reaping/cleanup and ``closed`` event are produced by the baseline
        supervisor, so parked cancelling jobs cannot be finalized early.
        ``event`` is deliberately *not* set here: attached job supervisors must
        wait for the completion event rather than a drain announcement.
        """
        run.status = BaselineStatus.DRAINING
        run.message = "last waiter cancelled; terminating shared computation"
        if run.proc is not None and run.proc.returncode is None:
            asyncio.create_task(
                self._terminate(run.proc, run.cancel_file, run.id[:8]),
                name=f"kill-base-{run.id[:8]}",
            )
        LOG.warning("baseline run %s draining", run.id)

    async def _terminate(
        self,
        proc: asyncio.subprocess.Process,
        cancel_file: Path | None,
        label: str,
    ) -> None:
        """Terminate a worker and guarantee it is reaped.

        SIGTERM is the primary mechanism.  On hosts whose seccomp policy
        refuses kill(2) (EPERM) the worker's cooperative cancel file is used
        instead; escalation to SIGKILL is attempted but may equally be denied,
        in which case the cancel file still guarantees an orderly exit.
        """
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except (ProcessLookupError, PermissionError) as exc:
            LOG.warning(
                "SIGTERM unavailable for %s (%s); using cooperative cancel file",
                label,
                exc,
            )
            if cancel_file is not None:
                try:
                    cancel_file.touch()
                except OSError:
                    LOG.exception("cannot write cancel file %s", cancel_file)
        try:
            await asyncio.wait_for(
                proc.wait(), timeout=self.settings.worker_term_timeout
            )
            return
        except asyncio.TimeoutError:
            LOG.warning("worker %s ignored graceful stop, escalating", label)
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError) as exc:
            LOG.warning("SIGKILL unavailable for %s (%s)", label, exc)
            if cancel_file is not None:
                cancel_file.touch(mode=0o644, exist_ok=True)
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            LOG.exception("reap failed for worker %s", label)

    def _finalize_cancel_locked(self, job: JobRecord) -> bool:
        """Finalize a cancellation when, and only when, it is safe.

        Safe means: no own worker is alive and the shared baseline run the
        cancellation was gated on (if any) has finished its exit/reap/cleanup.
        Returns ``True`` when the job reached ``cancelled`` and ``False`` when
        teardown is still in progress (caller must keep waiting).
        """
        if job.state == JobState.CANCELLED:
            return True
        if job.proc is not None and job.proc.returncode is None:
            return False
        run = job.baseline_run
        if (
            run is not None
            and job.id in run.gated_cancels
            and not run.closed.is_set()
        ):
            # The shared worker this job is bound to has not been reaped and
            # cleaned up yet; keep the job ``cancelling`` and its slot held.
            return False
        shutil.rmtree(job.target_path.parent, ignore_errors=True)
        job.state = JobState.CANCELLED
        job.message = "cancelled after worker exit and cleanup"
        job.baseline_active = False
        job.updated_at = _utcnow()
        LOG.info("job %s cancelled and cleaned up", job.id)
        return True

    def _release_gated_cancels_locked(self, run: BaselineRun) -> None:
        """Finalize parked cancelling jobs once the run is fully closed."""
        for job_id in list(run.gated_cancels):
            job = self._jobs.get(job_id)
            if job is None or job.terminal:
                continue
            if not self._finalize_cancel_locked(job):
                # Should not happen: closed was just set; keep the parked job
                # until a later handoff finalizes it.
                continue
        run.gated_cancels.clear()

    # --------------------------------------------------------- job lifecycle
    async def _supervise_job(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            ready = await self._ensure_baseline(job)
            if not ready:
                return  # baseline failure or cancellation finalized the job

            # Handoff gate #1: baseline -> anomaly.
            proc = await self._launch_anomaly(job)
            if proc is None:
                return  # cancelled while acquiring the lock

            rc = await proc.wait()

            # Handoff gate #2: worker exit -> result publication.
            async with self._lock:
                if job.state == JobState.CANCELLING:
                    self._finalize_cancel_locked(job)
                    return
                if rc == 0 and job.output_path.is_file():
                    job.state = JobState.DONE
                    job.message = "anomaly NetCDF ready"
                    job.updated_at = _utcnow()
                    LOG.info("job %s done", job.id)
                    return
                job.state = JobState.ERROR
                job.message = f"worker exited with code {rc}"
                job.updated_at = _utcnow()
                self._remove_partial(job.output_path)
                LOG.error("job %s failed rc=%s", job_id, rc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOG.exception("supervisor error for %s", job_id)
            async with self._lock:
                if job.state is JobState.CANCELLING:
                    # Never turn a parked cancellation into an error; finalize
                    # if no shared run teardown is pending anymore.
                    self._finalize_cancel_locked(job)
                elif not job.terminal:
                    job.state = JobState.ERROR
                    job.message = f"internal error: {exc}"
                    job.updated_at = _utcnow()
                    self._remove_partial(job.output_path)

    async def _ensure_baseline(self, job: JobRecord) -> bool:
        """Block until climatology is usable. False => job is finalized.

        A job only ever consumes a climatology that was actually published.
        Waking on a draining/terminated/errored run loops back and starts (or
        joins) a fresh run instead of inheriting the stale outcome.  Baseline
        failures are retried a bounded number of times before erroring the job,
        so a persistently broken baseline cannot spin forever.

        A job parked in ``cancelling`` by a last-waiter drain stays parked until
        that run's ``closed`` event fires (worker reaped + artifacts cleaned);
        only then may it become ``cancelled``.
        """
        max_attempts = 2
        attached_run: BaselineRun | None = None
        while True:
            async with self._lock:
                if job.terminal:
                    return False
                if job.state is JobState.CANCELLING:
                    parked_run = job.baseline_run
                    if (
                        parked_run is not None
                        and job.id in parked_run.gated_cancels
                        and not parked_run.closed.is_set()
                    ):
                        # Teardown still in progress: do not finalize, do not
                        # attach to any other run, keep the slot occupied.
                        closed_event = parked_run.closed
                    else:
                        self._finalize_cancel_locked(job)
                        return False
                elif self.baseline_climatology_ready():
                    job.stage = JobStage.ANOMALY
                    job.baseline_active = False
                    if job.baseline_run is not None:
                        job.baseline_run.waiters.discard(job.id)
                        job.baseline_run = None
                    job.message = "baseline climatology available"
                    job.updated_at = _utcnow()
                    return True
                else:
                    run = self._baseline_run
                    stale = (
                        run is not None
                        and run.status
                        not in (
                            BaselineStatus.STARTING,
                            BaselineStatus.RUNNING,
                        )
                    )
                    if run is None or stale or run is attached_run:
                        # The run we were attached to ended without publishing,
                        # or no usable run exists: never join a draining/old run.
                        if stale:
                            self._baseline_run = None
                        if attached_run is not None:
                            attached_run.waiters.discard(job.id)
                        job.baseline_attempts += 1
                        run = self._spawn_baseline_run_locked()
                    if job.id not in run.waiters:
                        run.waiters.add(job.id)
                        job.baseline_run = run
                        job.baseline_active = True
                        job.message = (
                            f"shared baseline run {run.id[:8]} in progress"
                        )
                        job.updated_at = _utcnow()
                    attached_run = run
                    closed_event = None

            if closed_event is not None:
                await closed_event.wait()
                continue

            await attached_run.event.wait()

            async with self._lock:
                if job.terminal:
                    return False
                if job.state is JobState.CANCELLING:
                    # Loop back: the top of the loop either parks until the run
                    # is closed or finalizes once the teardown completed.  Never
                    # finalize a parked last-waiter cancel here.
                    continue
                if self.baseline_climatology_ready():
                    job.stage = JobStage.ANOMALY
                    job.baseline_active = False
                    if job.baseline_run is not None:
                        job.baseline_run.waiters.discard(job.id)
                        job.baseline_run = None
                    job.message = "baseline climatology available"
                    job.updated_at = _utcnow()
                    return True
                ended_run = self._baseline_run
                if ended_run is not None and (
                    ended_run.status is BaselineStatus.ERROR
                    or job.baseline_attempts >= max_attempts
                ):
                    job.state = JobState.ERROR
                    job.message = (
                        "baseline computation failed: "
                        + (ended_run.message if ended_run else "unknown")
                    )
                    job.updated_at = _utcnow()
                    LOG.error(
                        "job %s aborted after %d attempt(s): %s",
                        job.id,
                        job.baseline_attempts,
                        job.message,
                    )
                    return False
                LOG.info(
                    "job %s retrying baseline after run %s (%s)",
                    job.id,
                    attached_run.id[:8] if attached_run else "?",
                    ended_run.status.value if ended_run else "none",
                )

    async def _launch_anomaly(
        self, job: JobRecord
    ) -> asyncio.subprocess.Process | None:
        async with self._lock:
            if job.state in (JobState.CANCELLING, JobState.CANCELLED):
                # A cancelled job may never enter the anomaly phase.
                if not job.terminal:
                    self._finalize_cancel_locked(job)
                return None
            log_handle = open(job.target_path.parent / "worker.log", "wb")
            cancel_file = job.target_path.parent / "cancel.flag"
            cmd = [
                sys.executable,
                "-I",
                "-m",
                "app.worker",
                "--step-delay",
                str(self.settings.slow_step_delay),
                "--steps",
                str(self.settings.slow_steps),
                "--cancel-file",
                str(cancel_file),
                "anomaly",
                "--target",
                str(job.target_path),
                "--climatology",
                str(self.settings.baseline_work_path),
                "--output",
                str(job.output_path),
                "--name",
                job.id[:8],
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=log_handle,
            )
            job.proc = proc
            job.cancel_file = cancel_file
            job.state = JobState.RUNNING
            job.stage = JobStage.ANOMALY
            job.message = "anomaly worker running in isolated process"
            job.updated_at = _utcnow()
            LOG.info("launched anomaly worker pid=%s for %s", proc.pid, job.id)

            async def _close_log() -> None:
                await proc.wait()
                log_handle.close()

            asyncio.create_task(_close_log(), name=f"log-{job.id[:8]}")
            return proc

    # ------------------------------------------------------ baseline process
    def _spawn_baseline_run_locked(self) -> BaselineRun:
        run_id = uuid.uuid4().hex
        base_dir = self.settings.data_dir / "baseline"
        base_dir.mkdir(parents=True, exist_ok=True)
        output = self.settings.baseline_work_path
        part = base_dir / f"climatology.nc.part-{run_id}"
        log_path = base_dir / f"worker-{run_id[:8]}.log"
        cancel_file = base_dir / f"cancel-{run_id[:8]}.flag"
        log_handle = open(log_path, "wb")
        cmd = [
            sys.executable,
            "-I",
            "-m",
            "app.worker",
            "--step-delay",
            str(self.settings.slow_step_delay),
            "--steps",
            str(self.settings.slow_steps),
            "--cancel-file",
            str(cancel_file),
            "baseline",
            "--input",
            str(self.settings.baseline_input_path),
            "--output",
            str(part),
        ]

        async def _spawn() -> asyncio.subprocess.Process:
            return await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=log_handle,
            )

        run = BaselineRun(
            id=run_id,
            output_path=output,
            part_path=part,
            log_path=log_path,
            cancel_file=cancel_file,
            log_handle=log_handle,
        )
        # The subprocess is created lazily by the supervisor to keep this
        # method synchronous; the run is STARTING in the meantime.
        run.supervisor = asyncio.create_task(
            self._supervise_baseline(run, _spawn), name=f"base-{run_id[:8]}"
        )
        self._baseline_run = run
        LOG.info("created baseline run %s", run_id)
        return run

    async def _supervise_baseline(
        self,
        run: BaselineRun,
        spawn: "callable",
    ) -> None:
        try:
            async with self._lock:
                if self._shutdown or run.status == BaselineStatus.DRAINING:
                    # Last-waiter cancel raced the spawn: never start the
                    # actual computation, just finalize the empty run.
                    self._close_baseline_run_locked(
                        run,
                        BaselineStatus.TERMINATED,
                        "terminated before worker start",
                        clean_artifacts=False,
                    )
                    return
                proc = await spawn()
                run.proc = proc
                if run.status == BaselineStatus.DRAINING:
                    # The cancel landed while the subprocess was being created.
                    # Arm termination immediately; the reap path below closes
                    # the run and releases the parked cancelling waiters.
                    run.message = "shared worker spawned while draining"
                    if proc.returncode is None:
                        asyncio.create_task(
                            self._terminate(
                                proc, run.cancel_file, run.id[:8]
                            ),
                            name=f"kill-base-{run.id[:8]}",
                        )
                else:
                    run.status = BaselineStatus.RUNNING
                    run.message = "shared climatology worker running"
        except Exception as exc:  # noqa: BLE001
            LOG.exception("baseline spawn failed")
            async with self._lock:
                if run.status == BaselineStatus.DRAINING:
                    self._close_baseline_run_locked(
                        run,
                        BaselineStatus.TERMINATED,
                        "terminated during spawn",
                        clean_artifacts=False,
                    )
                else:
                    self._close_baseline_run_locked(
                        run,
                        BaselineStatus.ERROR,
                        f"spawn failed: {exc}",
                        clean_artifacts=True,
                    )
            return

        rc = await proc.wait()

        async with self._lock:
            if run.status == BaselineStatus.DRAINING:
                # Cancellation won the race: no publish, clean artifacts. The
                # run closes only here -- after the worker has exited and been
                # reaped -- so parked cancelling jobs become ``cancelled`` only
                # at this point.
                self._close_baseline_run_locked(
                    run,
                    BaselineStatus.TERMINATED,
                    "shared computation terminated on last cancel",
                    clean_artifacts=True,
                )
                LOG.warning("baseline run %s terminated (rc=%s)", run.id, rc)
                return
            if rc == 0 and run.part_path.is_file():
                # Atomic publication; until this replace happens, jobs keep
                # waiting, so a late file can never be consumed.
                run.part_path.replace(run.output_path)
                self._close_baseline_run_locked(
                    run,
                    BaselineStatus.DONE,
                    "climatology published",
                    clean_artifacts=False,
                )
                LOG.info("baseline run %s published climatology", run.id)
                return
            self._close_baseline_run_locked(
                run,
                BaselineStatus.ERROR,
                f"baseline worker exited with code {rc}",
                clean_artifacts=True,
            )
            LOG.error("baseline run %s failed rc=%s", run.id, rc)

    def _close_baseline_run_locked(
        self,
        run: BaselineRun,
        status: BaselineStatus,
        message: str,
        clean_artifacts: bool,
    ) -> None:
        """Single terminal transition for a baseline run.

        Closes the log, optionally removes the partial output and cooperative
        cancel flag, records the terminal state and then -- and only then --
        marks the run closed and releases parked last-waiter cancellations.
        The part/cancel file names carry the run id, so an older run's teardown
        can never touch a newer run's files or another job's directory.
        """
        self._safe_close(run.log_handle)
        if clean_artifacts:
            run.part_path.unlink(missing_ok=True)
        # The cancel flag belongs to this run only; keep cleanup scoped to it.
        run.cancel_file.unlink(missing_ok=True)
        run.status = status
        run.message = message
        run.event.set()
        run.closed.set()
        self._release_gated_cancels_locked(run)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _safe_close(handle: object) -> None:
        close = getattr(handle, "close", None)
        if callable(close):
            try:
                close()
            except OSError:
                pass

    @staticmethod
    def _remove_partial(path: Path) -> None:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".part").unlink(missing_ok=True)
