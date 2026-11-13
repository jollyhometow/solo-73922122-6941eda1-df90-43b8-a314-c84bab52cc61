"""Independent-process worker CLI.

Spawned by the manager with ``python -I -m app.worker ...`` so all heavy work
runs in an isolated process.  Cancellation flips a flag through two channels:

1. SIGTERM/SIGINT (the normal mechanism in a real container), and
2. a ``--cancel-file`` whose appearance a tiny watcher thread polls.  This is
   used by the manager only when the host refuses signal delivery (restricted
   seccomp), and keeps the same interruptible semantics either way.

The :class:`~app.compute.Interrupter` observes the flag at every slice and
every phase boundary.  A worker therefore:

* never starts a later phase after cancellation was requested,
* never leaves a published output file behind when cancelled,
* exits with code 143 and lets the manager reap it before the job is reported
  as cancelled.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .compute import (
    Cancelled,
    Interrupter,
    compute_anomalies,
    compute_monthly_climatology,
)
from .models import EXIT_CANCELLED, EXIT_ERROR, EXIT_OK

LOG = logging.getLogger("worker")
_CANCEL_POLL = 0.05


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tas anomaly worker")
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="seconds of artificial delay per slow step",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="number of artificial slow steps (delay spread over the work)",
    )
    parser.add_argument(
        "--cancel-file",
        default=None,
        help="when this path appears the worker cancels cooperatively",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    base = sub.add_parser("baseline", help="monthly historical climatology")
    base.add_argument("--input", required=True)
    base.add_argument("--output", required=True)

    anom = sub.add_parser("anomaly", help="target minus climatology")
    anom.add_argument("--target", required=True)
    anom.add_argument("--climatology", required=True)
    anom.add_argument("--output", required=True)
    anom.add_argument("--name", default="job")
    return parser


def _remove_outputs(output: Path) -> None:
    for candidate in (output, output.with_name(output.name + ".part")):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            LOG.warning("could not remove %s", candidate)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s worker[%(process)d] %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args(argv)
    output = Path(args.output)

    cancel_flag = threading.Event()
    cancel_file = Path(args.cancel_file) if args.cancel_file else None

    def _handle_signal(signum: int, _frame: object) -> None:
        LOG.warning("received signal %s, requesting cancellation", signum)
        cancel_flag.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    stop_watcher = threading.Event()

    def _watch_cancel_file() -> None:
        # Cooperative fallback for sandboxes that block kill(2).  The flag is
        # identical to the signal path, so phase gates behave the same.
        if cancel_file is None:
            return
        while not stop_watcher.is_set():
            if cancel_file.exists():
                LOG.warning(
                    "cancel file %s appeared, requesting cancellation",
                    cancel_file,
                )
                cancel_flag.set()
                return
            time.sleep(_CANCEL_POLL)

    watcher = threading.Thread(
        target=_watch_cancel_file, name="cancel-watcher", daemon=True
    )
    watcher.start()
    interrupter = Interrupter(cancel_flag.is_set)

    total_delay = max(0.0, args.step_delay) * max(1, args.steps)

    def progress_sleeper(ticks: int):
        per_tick = total_delay / max(ticks, 1) if total_delay else 0.0

        def _cb(stage: str, done: int, total: int) -> None:
            LOG.info("%s progress %d/%d", stage, done, total)
            if per_tick:
                interrupter.sleep(per_tick)

        return _cb

    try:
        # Phase 0: startup / input acquisition window — cancellation here means
        # no computation starts at all.
        LOG.info("starting mode=%s pid=%d", args.mode, os.getpid())
        if total_delay:
            interrupter.sleep(total_delay * 0.15)

        if args.mode == "baseline":
            # Phase 1: compute. Phase 2 (publish) is gated again inside.
            compute_monthly_climatology(
                args.input,
                args.output,
                interrupter,
                on_progress=progress_sleeper(12),
            )
        else:
            compute_anomalies(
                args.target,
                args.climatology,
                args.output,
                interrupter,
                on_progress=progress_sleeper(6),
            )

        # Final gate: a signal arriving while the write was in flight still
        # causes the late output to be discarded and the exit to read cancel.
        interrupter.check()
    except Cancelled:
        LOG.warning("cancelled; discarding any partial output %s", output)
        _remove_outputs(output)
        stop_watcher.set()
        return EXIT_CANCELLED
    except Exception as exc:  # noqa: BLE001 - worker must report any failure
        LOG.exception("worker failed: %s", exc)
        _remove_outputs(output)
        stop_watcher.set()
        return EXIT_ERROR

    stop_watcher.set()
    LOG.info("finished mode=%s output=%s", args.mode, output)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
