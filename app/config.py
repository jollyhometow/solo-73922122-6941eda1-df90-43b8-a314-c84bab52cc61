"""Runtime configuration.

All knobs are overridable through environment variables so the same image can
be used both for fast unit tests and for the deliberately slow, browser-demo
deployment driven by Docker Compose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    baseline_filename: str = "baseline.nc"
    max_active_jobs: int = 2
    # Artificial per-step delay inside the worker subprocesses.  The work is
    # split into ``slow_steps`` interruptible slices so SIGTERM is handled
    # promptly and cancellation can be observed from the browser.
    slow_step_delay: float = 0.25
    slow_steps: int = 12
    # Seconds to wait for a graceful worker exit before SIGKILL.
    worker_term_timeout: float = 8.0
    poll_interval: float = 0.15

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            max_active_jobs=_env_int("MAX_ACTIVE_JOBS", 2),
            slow_step_delay=_env_float("SLOW_STEP_DELAY", 0.25),
            slow_steps=_env_int("SLOW_STEPS", 12),
            worker_term_timeout=_env_float("WORKER_TERM_TIMEOUT", 8.0),
            poll_interval=_env_float("POLL_INTERVAL", 0.15),
        )

    @property
    def baseline_input_path(self) -> Path:
        return self.data_dir / "baseline" / self.baseline_filename

    @property
    def baseline_work_path(self) -> Path:
        # Output of the shared baseline subprocess (monthly climatology).
        return self.data_dir / "baseline" / "climatology.nc"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.data_dir / "baseline",
            self.jobs_dir,
            self.samples_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
