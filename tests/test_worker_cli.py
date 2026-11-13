"""Black-box tests for the isolated worker CLI (run with -I).

The environment may forbid kill(2) via seccomp.  When signals work we assert
the SIGTERM/143 contract directly; otherwise we exercise the equivalent
cooperative cancel-file channel that the manager falls back to.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import xarray as xr


def _cmd(*extra: str) -> list[str]:
    return [sys.executable, "-I", "-m", "app.worker", *extra]


def _signals_allowed() -> bool:
    probe = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    time.sleep(0.15)
    try:
        os.kill(probe.pid, signal.SIGTERM)
        probe.wait(timeout=5)
        return True
    except (PermissionError, ProcessLookupError, OSError):
        probe.kill() if False else None
        try:
            probe.wait(timeout=5)
        except Exception:
            pass
        return False


SIGNALS_OK = _signals_allowed()


def test_worker_baseline_and_anomaly_cli(
    baseline_file: Path, target_file: Path, tmp_path: Path
):
    clim = tmp_path / "clim.nc"
    proc = subprocess.run(
        _cmd("baseline", "--input", str(baseline_file), "--output", str(clim)),
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert clim.is_file()

    out = tmp_path / "anom.nc"
    proc = subprocess.run(
        _cmd(
            "--step-delay", "0.01", "--steps", "2",
            "anomaly",
            "--target", str(target_file),
            "--climatology", str(clim),
            "--output", str(out),
        ),
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    with xr.open_dataset(out, engine="h5netcdf") as ds:
        assert tuple(ds["tas"].dims) == ("time", "lat", "lon")


def test_worker_cancel_baseline_cleans_output(
    baseline_file: Path, tmp_path: Path
):
    out = tmp_path / "slow-clim.nc"
    cancel_file = tmp_path / "cancel.flag"
    proc = subprocess.Popen(
        _cmd(
            "--step-delay", "1.0", "--steps", "20",
            "--cancel-file", str(cancel_file),
            "baseline",
            "--input", str(baseline_file),
            "--output", str(out),
        ),
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.6)
    if SIGNALS_OK:
        proc.terminate()
        expected_rc = 143
    else:
        cancel_file.touch()
        expected_rc = 143
    _, stderr = proc.communicate(timeout=15)
    assert proc.returncode == expected_rc, stderr
    assert not out.exists()
    assert not Path(str(out) + ".part").exists()
    assert "cancelled" in stderr.lower()


def test_worker_cancel_anomaly_publishes_nothing(
    baseline_file: Path, target_file: Path, tmp_path: Path
):
    clim = tmp_path / "clim.nc"
    subprocess.run(
        _cmd("baseline", "--input", str(baseline_file), "--output", str(clim)),
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
    )
    out = tmp_path / "slow-anom.nc"
    cancel_file = tmp_path / "cancel.flag"
    proc = subprocess.Popen(
        _cmd(
            "--step-delay", "1.0", "--steps", "20",
            "--cancel-file", str(cancel_file),
            "anomaly",
            "--target", str(target_file),
            "--climatology", str(clim),
            "--output", str(out),
        ),
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.6)
    if SIGNALS_OK:
        proc.terminate()
    else:
        cancel_file.touch()
    _, stderr = proc.communicate(timeout=15)
    assert proc.returncode == 143, stderr
    assert not out.exists()


def test_cancel_file_checked_during_startup_window(
    baseline_file: Path, tmp_path: Path
):
    # Flag present before the worker even starts: no phase may begin.
    out = tmp_path / "never.nc"
    cancel_file = tmp_path / "cancel.flag"
    cancel_file.touch()
    proc = subprocess.run(
        _cmd(
            "--step-delay", "0.2", "--steps", "4",
            "--cancel-file", str(cancel_file),
            "baseline",
            "--input", str(baseline_file),
            "--output", str(out),
        ),
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 143
    assert not out.exists()
