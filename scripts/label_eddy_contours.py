#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from netCDF4 import Dataset, num2date


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_EDDY_TRACKER_SRC = REPO_ROOT / "py-eddy-tracker" / "src"
if str(PY_EDDY_TRACKER_SRC) not in sys.path:
    sys.path.insert(0, str(PY_EDDY_TRACKER_SRC))

from py_eddy_tracker.appli.grid import identification  # noqa: E402


@dataclass(frozen=True)
class TimeStep:
    dataset_path: Path
    time_index: int
    date: datetime


@dataclass(frozen=True)
class DetectionConfig:
    output_dir: str
    zarr: bool
    lon_name: str
    lat_name: str
    time_name: str
    adt_name: str
    ugos_name: str
    vgos_name: str
    cut_wavelength: float
    filter_order: int
    isoline_step: float
    shape_error: float
    pixel_min: int
    pixel_max: int
    nb_step_min: int
    sampling: int
    sampling_method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use py-eddy-tracker to label eddy contours from CMEMS META4.0 DT "
            "ADT/UGOS/VGOS daily NetCDF files."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="data/*.nc",
        help="Glob pattern for input NetCDF files. Default: data/*.nc",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/meta40_dt_contours",
        help="Directory for anticyclonic/cyclonic contour files and summary.csv",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--limit-days",
        type=int,
        help="Stop after processing this many daily fields.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing contour files instead of skipping them.",
    )
    parser.add_argument(
        "--zarr",
        action="store_true",
        help="Store outputs as zarr instead of NetCDF.",
    )
    parser.add_argument(
        "--lon-name",
        default="longitude",
        help="Longitude coordinate variable name.",
    )
    parser.add_argument(
        "--lat-name",
        default="latitude",
        help="Latitude coordinate variable name.",
    )
    parser.add_argument(
        "--time-name",
        default="time",
        help="Time coordinate variable name.",
    )
    parser.add_argument(
        "--adt-name",
        default="adt",
        help="ADT variable name used for contour detection.",
    )
    parser.add_argument(
        "--ugos-name",
        default="ugos",
        help="Eastward geostrophic velocity variable name.",
    )
    parser.add_argument(
        "--vgos-name",
        default="vgos",
        help="Northward geostrophic velocity variable name.",
    )
    parser.add_argument(
        "--cut-wavelength",
        type=float,
        default=800.0,
        help="High-pass filter cutoff wavelength in km. META4.0 DT default: 800.",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=1,
        help="Bessel filter order. META4.0 DT default: 1.",
    )
    parser.add_argument(
        "--isoline-step",
        type=float,
        default=0.002,
        help="Contour interval in meters. META4.0 DT default: 0.002.",
    )
    parser.add_argument(
        "--shape-error",
        type=float,
        default=55.0,
        help="Maximum accepted contour shape error in percent.",
    )
    parser.add_argument(
        "--pixel-min",
        type=int,
        default=5,
        help="Minimum pixel count inside a valid contour.",
    )
    parser.add_argument(
        "--pixel-max",
        type=int,
        default=2000,
        help="Maximum pixel count inside a valid contour.",
    )
    parser.add_argument(
        "--nb-step-min",
        type=int,
        default=2,
        help="Minimum amplitude threshold in contour steps.",
    )
    parser.add_argument(
        "--sampling",
        type=int,
        default=50,
        help="Stored contour sample count.",
    )
    parser.add_argument(
        "--sampling-method",
        choices=("visvalingam", "uniform"),
        default="visvalingam",
        help="Contour resampling method.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for daily contour detection. Default: 1",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def normalize_date(value) -> datetime:
    return datetime(value.year, value.month, value.day)


def list_time_steps(
    dataset_paths: list[Path],
    time_name: str,
    start_date: datetime | None,
    end_date: datetime | None,
    limit_days: int | None,
) -> list[TimeStep]:
    steps: list[TimeStep] = []
    for dataset_path in dataset_paths:
        with Dataset(dataset_path) as ds:
            time_var = ds.variables[time_name]
            values = num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        for time_index, value in enumerate(values):
            date = normalize_date(value)
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            steps.append(TimeStep(dataset_path=dataset_path, time_index=time_index, date=date))
            if limit_days and len(steps) >= limit_days:
                return steps
    return steps


def output_paths(output_dir: Path, date: datetime, zarr: bool) -> tuple[Path, Path]:
    suffix = ".zarr" if zarr else ".nc"
    anti = output_dir / f"Anticyclonic_{date:%Y%m%dT%H%M%S}{suffix}"
    cyclo = output_dir / f"Cyclonic_{date:%Y%m%dT%H%M%S}{suffix}"
    return anti, cyclo


def output_template(date: datetime) -> str:
    return f"%(path)s/%(sign_type)s_{date:%Y%m%dT%H%M%S}.nc"


def build_summary_row(
    step: TimeStep,
    anti_path: Path,
    cyclo_path: Path,
    status: str,
    anticyclonic_count: int | str,
    cyclonic_count: int | str,
) -> dict[str, object]:
    return dict(
        date=step.date.strftime("%Y-%m-%d"),
        dataset=str(step.dataset_path.relative_to(REPO_ROOT)),
        time_index=step.time_index,
        anticyclonic_count=anticyclonic_count,
        cyclonic_count=cyclonic_count,
        anticyclonic_file=str(anti_path.relative_to(REPO_ROOT)),
        cyclonic_file=str(cyclo_path.relative_to(REPO_ROOT)),
        status=status,
    )


def detect_one_day(step: TimeStep, config: DetectionConfig) -> dict[str, object]:
    output_dir = Path(config.output_dir)
    anti_path, cyclo_path = output_paths(output_dir, step.date, config.zarr)
    anticyclonic, cyclonic = identification(
        filename=str(step.dataset_path),
        lon=config.lon_name,
        lat=config.lat_name,
        date=step.date,
        h=config.adt_name,
        u=config.ugos_name,
        v=config.vgos_name,
        cut_wavelength=config.cut_wavelength,
        filter_order=config.filter_order,
        indexs={config.time_name: step.time_index},
        step=config.isoline_step,
        shape_error=config.shape_error,
        pixel_limit=(config.pixel_min, config.pixel_max),
        nb_step_min=config.nb_step_min,
        sampling=config.sampling,
        sampling_method=config.sampling_method,
    )
    anticyclonic.write_file(
        path=str(output_dir),
        filename=output_template(step.date),
        zarr_flag=config.zarr,
    )
    cyclonic.write_file(
        path=str(output_dir),
        filename=output_template(step.date),
        zarr_flag=config.zarr,
    )
    return build_summary_row(
        step=step,
        anti_path=anti_path,
        cyclo_path=cyclo_path,
        status="written",
        anticyclonic_count=len(anticyclonic),
        cyclonic_count=len(cyclonic),
    )


def run_serial_detection(
    pending_steps: list[TimeStep],
    config: DetectionConfig,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    total = len(pending_steps)
    for index, step in enumerate(pending_steps, start=1):
        print(
            f"[{index}/{total}] detect {step.date:%Y-%m-%d} "
            f"from {step.dataset_path.name} time={step.time_index}"
        )
        summary_rows.append(detect_one_day(step, config))
    return summary_rows


def run_parallel_detection(
    pending_steps: list[TimeStep],
    config: DetectionConfig,
    workers: int,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    total = len(pending_steps)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_step = {
            executor.submit(detect_one_day, step, config): step for step in pending_steps
        }
        while future_to_step:
            done, _ = wait(future_to_step, return_when=FIRST_COMPLETED)
            for future in done:
                step = future_to_step.pop(future)
                row = future.result()
                completed += 1
                summary_rows.append(row)
                print(
                    f"[{completed}/{total}] done {step.date:%Y-%m-%d} "
                    f"AE={row['anticyclonic_count']} CE={row['cyclonic_count']}"
                )
    return summary_rows


def main() -> int:
    args = parse_args()
    dataset_paths = sorted(REPO_ROOT.glob(args.input_glob))
    if not dataset_paths:
        print(f"No NetCDF files matched: {args.input_glob}", file=sys.stderr)
        return 1

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = list_time_steps(
        dataset_paths=dataset_paths,
        time_name=args.time_name,
        start_date=args.start_date,
        end_date=args.end_date,
        limit_days=args.limit_days,
    )
    if not steps:
        print("No daily fields matched the requested date range.", file=sys.stderr)
        return 1

    detect_config = DetectionConfig(
        output_dir=str(output_dir),
        zarr=args.zarr,
        lon_name=args.lon_name,
        lat_name=args.lat_name,
        time_name=args.time_name,
        adt_name=args.adt_name,
        ugos_name=args.ugos_name,
        vgos_name=args.vgos_name,
        cut_wavelength=args.cut_wavelength,
        filter_order=args.filter_order,
        isoline_step=args.isoline_step,
        shape_error=args.shape_error,
        pixel_min=args.pixel_min,
        pixel_max=args.pixel_max,
        nb_step_min=args.nb_step_min,
        sampling=args.sampling,
        sampling_method=args.sampling_method,
    )

    summary_rows: list[dict[str, object]] = []
    pending_steps: list[TimeStep] = []
    total = len(steps)
    for index, step in enumerate(steps, start=1):
        anti_path, cyclo_path = output_paths(output_dir, step.date, args.zarr)
        if not args.overwrite and anti_path.exists() and cyclo_path.exists():
            print(f"[{index}/{total}] skip {step.date:%Y-%m-%d} existing outputs")
            summary_rows.append(
                build_summary_row(
                    step=step,
                    anti_path=anti_path,
                    cyclo_path=cyclo_path,
                    status="skipped_existing",
                    anticyclonic_count="",
                    cyclonic_count="",
                )
            )
            continue
        pending_steps.append(step)

    if pending_steps:
        workers = max(1, min(args.workers, os.cpu_count() or 1, len(pending_steps)))
        print(f"Running detection for {len(pending_steps)} day(s) with workers={workers}")
        if workers == 1:
            summary_rows.extend(run_serial_detection(pending_steps, detect_config))
        else:
            summary_rows.extend(run_parallel_detection(pending_steps, detect_config, workers))

    summary_rows.sort(key=lambda row: (str(row["date"]), int(row["time_index"])))

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "dataset",
                "time_index",
                "anticyclonic_count",
                "cyclonic_count",
                "anticyclonic_file",
                "cyclonic_file",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    written = sum(row["status"] == "written" for row in summary_rows)
    skipped = sum(row["status"] == "skipped_existing" for row in summary_rows)
    print(f"Finished. written={written} skipped={skipped} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
