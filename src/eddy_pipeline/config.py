from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SplitConfig:
    name: str
    files: list[str]
    start_date: str
    end_date: str


@dataclass
class DataConfig:
    raw_dir: str
    variables: list[str]
    seq_len: int
    contour_dir: str
    label_store_dir: str
    processed_dir: str
    stats_path: str
    interp_limit_days: int
    label_workers: int
    ignore_index: int
    background_index: int
    cyclonic_index: int
    anticyclonic_index: int
    splits: dict[str, SplitConfig]


@dataclass
class ModelConfig:
    channels: list[int]
    kernel_size: int
    dropout: float
    num_classes: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    grad_accum_steps: int
    amp: bool
    early_stopping_patience: int
    save_dir: str
    seed: int


@dataclass
class PredictionConfig:
    batch_size: int
    num_workers: int
    output_dir: str
    checkpoint_path: str = ""


@dataclass
class EvaluationConfig:
    output_dir: str


@dataclass
class ExperimentConfig:
    project_root: Path
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    prediction: PredictionConfig
    evaluation: EvaluationConfig

    @property
    def device(self) -> str:
        return "cuda"


def _resolve_path(project_root: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def _parse_split_configs(raw_splits: dict[str, Any]) -> dict[str, SplitConfig]:
    splits: dict[str, SplitConfig] = {}
    for name, values in raw_splits.items():
        splits[name] = SplitConfig(name=name, **values)
    return splits


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path).resolve()
    project_root = path.parents[1]

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw_data = raw["data"]
    raw_data["splits"] = _parse_split_configs(raw_data["splits"])
    for key in ("raw_dir", "contour_dir", "label_store_dir", "processed_dir", "stats_path"):
        raw_data[key] = _resolve_path(project_root, raw_data[key])

    training = raw["training"]
    training["save_dir"] = _resolve_path(project_root, training["save_dir"])

    prediction = raw["prediction"]
    prediction["output_dir"] = _resolve_path(project_root, prediction["output_dir"])
    prediction["checkpoint_path"] = _resolve_path(project_root, prediction["checkpoint_path"])

    evaluation = raw["evaluation"]
    evaluation["output_dir"] = _resolve_path(project_root, evaluation["output_dir"])

    return ExperimentConfig(
        project_root=project_root,
        data=DataConfig(**raw_data),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**training),
        prediction=PredictionConfig(**prediction),
        evaluation=EvaluationConfig(**evaluation),
    )
