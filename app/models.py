"""Job / baseline state vocabulary.

States are deliberately small and explicit so the cancellation rules can be
audited against the requirements:

* ``queued``     -- accepted, no own worker process running yet (may be waiting
                    on the shared baseline computation)
* ``running``    -- a worker subprocess owned by this job is executing
* ``cancelling`` -- cancellation requested; the worker must exit and files be
                    cleaned up before the job is allowed to become ``cancelled``
* ``done`` / ``error`` / ``cancelled`` -- terminal states
"""

from __future__ import annotations

import enum


class JobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {JobState.DONE, JobState.ERROR, JobState.CANCELLED}
)


class JobStage(str, enum.Enum):
    WAITING_BASELINE = "waiting_baseline"
    ANOMALY = "anomaly"


class BaselineStatus(str, enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    # Last attached waiter cancelled: worker is being terminated/reaped and no
    # new job may attach to this run anymore.
    DRAINING = "draining"
    DONE = "done"
    ERROR = "error"
    TERMINATED = "terminated"


# Worker process exit code convention (128 + SIGTERM(15)).
EXIT_CANCELLED = 143
EXIT_ERROR = 1
EXIT_OK = 0
