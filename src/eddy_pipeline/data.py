from __future__ import annotations

import csv
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from matplotlib.path import Path as MplPath
from torch.utils.data import Dataset

from .config import DataConfig, ExperimentConfig


@dataclass(frozen=True)
class DayRecord:
    date: str
    dataset_path: str
    time_index: int


@dataclass(frozen=True)
class LabelBuildConfig:
    contour_dir: str
    label_dir: str
    ignore_index: int
    background_index: int
    cyclonic_index: int
    anticyclonic_index: int


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _date_str(value: Any) -> str:
    return np.datetime_as_string(np.datetime64(value), unit="D")


def _open_prepared_dataset(path: str, interp_limit_days: int) -> xr.Dataset:
    del interp_limit_days
    return xr.open_dataset(path)


def build_split_index(data_config: DataConfig, split: str) -> list[DayRecord]:
    split_cfg = data_config.splits[split]
    records: list[DayRecord] = []
    raw_dir = Path(data_config.raw_dir)
    start = np.datetime64(split_cfg.start_date)
    end = np.datetime64(split_cfg.end_date)
    for file_name in split_cfg.files:
        dataset_path = raw_dir / file_name
        ds = xr.open_dataset(dataset_path)
        times = ds["time"].values
        for time_index, value in enumerate(times):
            date = np.datetime64(value, "D")
            if date < start or date > end:
                continue
            records.append(
                DayRecord(
                    date=_date_str(date),
                    dataset_path=str(dataset_path),
                    time_index=time_index,
                )
            )
        ds.close()
    return records


def load_reference_grid(data_config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_file = Path(data_config.raw_dir) / data_config.splits["train"].files[0]
    ds = xr.open_dataset(first_file)
    lat = ds["latitude"].values.astype(np.float32)
    lon = ds["longitude"].values.astype(np.float32)
    mask = np.isfinite(ds[data_config.variables[0]].isel(time=0).values)
    ds.close()
    return lat, lon, mask


def ensure_train_stats(config: ExperimentConfig, overwrite: bool = False) -> dict[str, dict[str, float]]:
    stats_path = Path(config.data.stats_path)
    if stats_path.exists() and not overwrite:
        with stats_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    _ensure_dir(stats_path.parent)
    accum: dict[str, dict[str, float]] = {
        var: {"sum": 0.0, "sum_sq": 0.0, "count": 0.0} for var in config.data.variables
    }
    for file_name in config.data.splits["train"].files:
        ds = _open_prepared_dataset(str(Path(config.data.raw_dir) / file_name), config.data.interp_limit_days)
        for var in config.data.variables:
            values = ds[var].values.astype(np.float64)
            finite = np.isfinite(values)
            selected = values[finite]
            accum[var]["sum"] += float(selected.sum())
            accum[var]["sum_sq"] += float(np.square(selected).sum())
            accum[var]["count"] += float(selected.size)
        ds.close()

    stats: dict[str, dict[str, float]] = {}
    for var, values in accum.items():
        mean = values["sum"] / values["count"]
        variance = values["sum_sq"] / values["count"] - mean * mean
        stats[var] = {
            "mean": mean,
            "std": max(float(np.sqrt(max(variance, 1e-12))), 1e-6),
        }

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats


def _fill_missing_frame(ds: xr.Dataset, var_name: str, time_index: int, frame: np.ndarray, max_gap: int) -> np.ndarray:
    missing = ~np.isfinite(frame)
    if not missing.any():
        return frame

    filled = frame.copy()
    total_steps = int(ds.sizes["time"])
    unresolved = missing.copy()
    for offset in range(1, max_gap + 1):
        prev_frame = None
        next_frame = None
        prev_idx = time_index - offset
        next_idx = time_index + offset

        if prev_idx >= 0:
            prev_frame = ds[var_name].isel(time=prev_idx).values.astype(np.float32)
        if next_idx < total_steps:
            next_frame = ds[var_name].isel(time=next_idx).values.astype(np.float32)

        if prev_frame is not None and next_frame is not None:
            both = unresolved & np.isfinite(prev_frame) & np.isfinite(next_frame)
            filled[both] = 0.5 * (prev_frame[both] + next_frame[both])
            unresolved[both] = False

        if prev_frame is not None:
            only_prev = unresolved & np.isfinite(prev_frame)
            filled[only_prev] = prev_frame[only_prev]
            unresolved[only_prev] = False

        if next_frame is not None:
            only_next = unresolved & np.isfinite(next_frame)
            filled[only_next] = next_frame[only_next]
            unresolved[only_next] = False

        if not unresolved.any():
            break
    return filled


def _contour_file(contour_dir: Path, sign: str, date: str) -> Path:
    return contour_dir / f"{sign}_{date.replace('-', '')}T000000.nc"


def _polygon_to_mask(
    polygon_lon: np.ndarray,
    polygon_lat: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(polygon_lon) & np.isfinite(polygon_lat)
    polygon_lon = polygon_lon[valid]
    polygon_lat = polygon_lat[valid]
    if polygon_lon.size < 3:
        return np.zeros((lat_grid.size, lon_grid.size), dtype=bool)

    lon_min = float(polygon_lon.min())
    lon_max = float(polygon_lon.max())
    lat_min = float(polygon_lat.min())
    lat_max = float(polygon_lat.max())

    lon_mask = (lon_grid >= lon_min) & (lon_grid <= lon_max)
    lat_mask = (lat_grid >= lat_min) & (lat_grid <= lat_max)
    if not lon_mask.any() or not lat_mask.any():
        return np.zeros((lat_grid.size, lon_grid.size), dtype=bool)

    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    sub_lon = lon_grid[lon_idx]
    sub_lat = lat_grid[lat_idx]
    mesh_lon, mesh_lat = np.meshgrid(sub_lon, sub_lat)
    points = np.column_stack([mesh_lon.ravel(), mesh_lat.ravel()])
    path = MplPath(np.column_stack([polygon_lon, polygon_lat]), closed=True)
    inside = path.contains_points(points, radius=1e-9).reshape(mesh_lat.shape)

    out = np.zeros((lat_grid.size, lon_grid.size), dtype=bool)
    out[np.ix_(lat_idx, lon_idx)] = inside
    return out


def _build_single_label(
    record: DayRecord,
    build_config: LabelBuildConfig,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
) -> dict[str, Any]:
    contour_dir = Path(build_config.contour_dir)
    label_dir = Path(build_config.label_dir)
    out_path = label_dir / f"{record.date}.npz"
    anti_path = _contour_file(contour_dir, "Anticyclonic", record.date)
    cyc_path = _contour_file(contour_dir, "Cyclonic", record.date)

    mask = np.full(ocean_mask.shape, build_config.ignore_index, dtype=np.uint8)
    mask[ocean_mask] = build_config.background_index
    conflicts = 0
    layers: list[tuple[int, float, np.ndarray]] = []

    for source_path, class_value in (
        (cyc_path, build_config.cyclonic_index),
        (anti_path, build_config.anticyclonic_index),
    ):
        if not source_path.exists():
            continue
        ds = xr.open_dataset(source_path)
        count = int(ds.sizes.get("obs", 0))
        for obs_index in range(count):
            polygon_lat = ds["effective_contour_latitude"].isel(obs=obs_index).values
            polygon_lon = ds["effective_contour_longitude"].isel(obs=obs_index).values
            amplitude = float(ds["amplitude"].isel(obs=obs_index).item())
            layers.append((class_value, amplitude, _polygon_to_mask(polygon_lon, polygon_lat, lon, lat)))
        ds.close()

    layers.sort(key=lambda item: item[1])
    for class_value, _, polygon_mask in layers:
        if not polygon_mask.any():
            continue
        overlap = polygon_mask & ocean_mask & (mask != build_config.background_index) & (mask != class_value)
        conflicts += int(overlap.sum())
        mask[polygon_mask & ocean_mask] = class_value

    anticyclonic_count = 0
    cyclonic_count = 0
    if anti_path.exists():
        anti_ds = xr.open_dataset(anti_path)
        anticyclonic_count = int(anti_ds.sizes.get("obs", 0))
        anti_ds.close()
    if cyc_path.exists():
        cyc_ds = xr.open_dataset(cyc_path)
        cyclonic_count = int(cyc_ds.sizes.get("obs", 0))
        cyc_ds.close()

    np.savez_compressed(
        out_path,
        mask=mask,
        ocean_mask=ocean_mask.astype(np.uint8),
        cyclonic_count=np.array(cyclonic_count, dtype=np.int32),
        anticyclonic_count=np.array(anticyclonic_count, dtype=np.int32),
        overlap_count=np.array(conflicts, dtype=np.int32),
    )

    return {
        "date": record.date,
        "cyclonic_count": cyclonic_count,
        "anticyclonic_count": anticyclonic_count,
        "overlap_count": conflicts,
        "mask_file": out_path.name,
    }


def build_label_store(
    config: ExperimentConfig,
    splits: list[str] | None = None,
    overwrite: bool = False,
    workers: int | None = None,
) -> Path:
    # 作用：
    # 1. 读取 py-eddy-tracker 生成的每日气旋/反气旋轮廓
    # 2. 将轮廓内部填充成栅格像素
    # 3. 生成训练用标签：
    #    0=背景海洋, 1=气旋, 2=反气旋, 255=陆地/忽略
    label_dir = _ensure_dir(config.data.label_store_dir)
    contour_dir = Path(config.data.contour_dir)
    lat, lon, ocean_mask = load_reference_grid(config.data)
    splits = splits or list(config.data.splits.keys())
    workers = workers or config.data.label_workers
    build_config = LabelBuildConfig(
        contour_dir=str(contour_dir),
        label_dir=str(label_dir),
        ignore_index=config.data.ignore_index,
        background_index=config.data.background_index,
        cyclonic_index=config.data.cyclonic_index,
        anticyclonic_index=config.data.anticyclonic_index,
    )
    summary_rows: list[dict[str, Any]] = []

    for split in splits:
        records = build_split_index(config.data, split)
        total = len(records)
        written = 0
        skipped = 0
        missing = 0
        pending_records: list[DayRecord] = []
        print(
            f"[label_store] split={split} total_days={total} "
            f"overwrite={overwrite} workers={workers}"
        )
        for index, record in enumerate(records, start=1):
            out_path = label_dir / f"{record.date}.npz"
            if out_path.exists() and not overwrite:
                skipped += 1
                if index == 1 or index % 100 == 0 or index == total:
                    print(
                        f"[label_store] split={split} progress={index}/{total} "
                        f"written={written} skipped={skipped} missing={missing}"
                    )
                continue

            anti_path = _contour_file(contour_dir, "Anticyclonic", record.date)
            cyc_path = _contour_file(contour_dir, "Cyclonic", record.date)
            if not anti_path.exists() and not cyc_path.exists():
                missing += 1
                if index == 1 or index % 100 == 0 or index == total:
                    print(
                        f"[label_store] split={split} progress={index}/{total} "
                        f"written={written} skipped={skipped} missing={missing}"
                    )
                continue
            pending_records.append(record)

        if pending_records:
            actual_workers = max(1, min(workers, os.cpu_count() or 1, len(pending_records)))
            if actual_workers == 1:
                for pending_index, record in enumerate(pending_records, start=1):
                    row = _build_single_label(record, build_config, lat, lon, ocean_mask)
                    summary_rows.append({"split": split, **row})
                    written += 1
                    if pending_index == 1 or pending_index % 25 == 0 or pending_index == len(pending_records):
                        print(
                            f"[label_store] split={split} progress={pending_index}/{len(pending_records)} "
                            f"written={written} skipped={skipped} missing={missing} "
                            f"date={record.date} ce={row['cyclonic_count']} "
                            f"ae={row['anticyclonic_count']} overlap={row['overlap_count']}"
                        )
            else:
                print(f"[label_store] split={split} parallel_build={len(pending_records)} workers={actual_workers}")
                completed = 0
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    future_to_record = {
                        executor.submit(_build_single_label, record, build_config, lat, lon, ocean_mask): record
                        for record in pending_records
                    }
                    while future_to_record:
                        done, _ = wait(future_to_record, return_when=FIRST_COMPLETED)
                        for future in done:
                            record = future_to_record.pop(future)
                            row = future.result()
                            summary_rows.append({"split": split, **row})
                            written += 1
                            completed += 1
                            if completed == 1 or completed % 25 == 0 or completed == len(pending_records):
                                print(
                                    f"[label_store] split={split} progress={completed}/{len(pending_records)} "
                                    f"written={written} skipped={skipped} missing={missing} "
                                    f"date={record.date} ce={row['cyclonic_count']} "
                                    f"ae={row['anticyclonic_count']} overlap={row['overlap_count']}"
                                )

        print(
            f"[label_store] split={split} done total_days={total} "
            f"written={written} skipped={skipped} missing={missing}"
        )

    summary_path = label_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "split",
                "cyclonic_count",
                "anticyclonic_count",
                "overlap_count",
                "mask_file",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[label_store] summary written to {summary_path}")
    return label_dir


class DailySequenceDataset(Dataset):
    def __init__(
        self,
        config: ExperimentConfig,
        split: str,
        channels: list[str] | None = None,
        seq_len: int | None = None,
        include_labels: bool = True,
    ) -> None:
        self.config = config
        self.split = split
        self.channels = channels or config.data.variables
        self.seq_len = seq_len or config.data.seq_len
        self.include_labels = include_labels
        self.records = build_split_index(config.data, split)
        self.stats = ensure_train_stats(config)
        self.dataset_cache: dict[str, xr.Dataset] = {}
        self.label_dir = Path(config.data.label_store_dir)

    def __len__(self) -> int:
        return len(self.records)

    def _get_dataset(self, path: str) -> xr.Dataset:
        dataset = self.dataset_cache.get(path)
        if dataset is None:
            dataset = _open_prepared_dataset(path, self.config.data.interp_limit_days)
            self.dataset_cache[path] = dataset
        return dataset

    def _load_frame(self, record: DayRecord) -> tuple[np.ndarray, np.ndarray]:
        ds = self._get_dataset(record.dataset_path)
        channels: list[np.ndarray] = []
        ocean_mask = None
        for var in self.channels:
            frame = ds[var].isel(time=record.time_index).values.astype(np.float32)
            frame = _fill_missing_frame(ds, var, record.time_index, frame, self.config.data.interp_limit_days)
            if ocean_mask is None:
                ocean_mask = np.isfinite(frame)
            normalized = np.nan_to_num(
                (frame - self.stats[var]["mean"]) / self.stats[var]["std"],
                nan=0.0,
            )
            channels.append(normalized)
        assert ocean_mask is not None
        return np.stack(channels, axis=0), ocean_mask.astype(np.uint8)

    def _load_label(self, date: str) -> dict[str, np.ndarray]:
        label_path = self.label_dir / f"{date}.npz"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label mask for {date}: {label_path}")
        with np.load(label_path) as data:
            return {key: data[key] for key in data.files}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_frames: list[np.ndarray] = []
        current_idx = index
        for offset in range(self.seq_len - 1, -1, -1):
            source_idx = max(0, current_idx - offset)
            record = self.records[source_idx]
            frame, _ = self._load_frame(record)
            sequence_frames.append(frame)

        current_record = self.records[index]
        current_frame, ocean_mask = self._load_frame(current_record)
        sample: dict[str, Any] = {
            "x": torch.from_numpy(np.stack(sequence_frames, axis=0)),
            "ocean_mask": torch.from_numpy(ocean_mask),
            "date": current_record.date,
            "current_x": torch.from_numpy(current_frame),
        }

        if self.include_labels:
            label_data = self._load_label(current_record.date)
            sample["y"] = torch.from_numpy(label_data["mask"].astype(np.int64))
            sample["label_meta"] = {
                "cyclonic_count": int(label_data["cyclonic_count"]),
                "anticyclonic_count": int(label_data["anticyclonic_count"]),
                "overlap_count": int(label_data["overlap_count"]),
            }
        return sample


def build_daily_dataset(
    config: ExperimentConfig,
    split: str,
    seq_len: int | None = None,
    channels: list[str] | None = None,
    include_labels: bool = True,
) -> DailySequenceDataset:
    return DailySequenceDataset(
        config=config,
        split=split,
        channels=channels,
        seq_len=seq_len,
        include_labels=include_labels,
    )
