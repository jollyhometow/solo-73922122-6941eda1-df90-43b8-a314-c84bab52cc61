#!/usr/bin/env python3
"""Generate synthetic baseline / target NetCDF files for the demo.

Usage::

    python -I -m scripts.generate_samples --data-dir /data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.compute import write_sample_files
from app.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="data root (defaults to the DATA_DIR env var or /data)",
    )
    parser.add_argument("--n-lat", type=int, default=12)
    parser.add_argument("--n-lon", type=int, default=16)
    parser.add_argument("--baseline-years", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings.from_env()
    if args.data_dir:
        object.__setattr__(settings, "data_dir", Path(args.data_dir))
    settings.ensure_dirs()

    target_paths = {
        "a": settings.samples_dir / "target_sample_a.nc",
        "b": settings.samples_dir / "target_sample_b.nc",
    }
    write_sample_files(
        settings.baseline_input_path,
        target_paths,
        n_lat=args.n_lat,
        n_lon=args.n_lon,
        baseline_years=args.baseline_years,
    )
    print(f"baseline -> {settings.baseline_input_path}")
    for label, path in target_paths.items():
        print(f"target {label} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
