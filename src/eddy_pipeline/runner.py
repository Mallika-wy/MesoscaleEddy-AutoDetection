from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []

from .config import ExperimentConfig
from .data import build_daily_dataset, build_label_store, ensure_train_stats
from .metrics import confusion_matrix, summarize_metrics
from .model import DualAttentionConvLSTMUNet


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(config: ExperimentConfig) -> DualAttentionConvLSTMUNet:
    return DualAttentionConvLSTMUNet.from_config(
        config.model,
        in_channels=len(config.data.variables),
        seq_len=config.data.seq_len,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _iter_with_progress(iterable, desc: str, total: int):
    return tqdm(iterable, desc=desc, total=total, leave=False)


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
    grad_accum_steps: int = 1,
    stage_name: str = "",
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_steps = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    autocast_enabled = amp and device.type == "cuda"
    num_batches = len(loader)
    if stage_name:
        print(f"[{stage_name}] start batches={num_batches}")
    iterator = _iter_with_progress(loader, desc=stage_name or ("train" if is_train else "eval"), total=num_batches)
    for step, batch in enumerate(iterator, start=1):
        batch = _move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            logits = model(batch["x"])
            loss = criterion(logits, batch["y"])
            scaled_loss = loss / grad_accum_steps

        if is_train and optimizer is not None and scaler is not None:
            scaler.scale(scaled_loss).backward()
            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        preds = torch.argmax(logits, dim=1)
        conf_mat += confusion_matrix(preds.cpu(), batch["y"].cpu(), num_classes, ignore_index)
        total_loss += float(loss.item())
        total_steps += 1
        iterator.set_postfix(loss=f"{loss.item():.4f}")
        if step == 1 or step % 20 == 0 or step == num_batches:
            print(f"[{stage_name}] progress {step}/{num_batches} loss={loss.item():.4f}")

    if is_train and optimizer is not None and scaler is not None and total_steps % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    metrics = summarize_metrics(conf_mat)
    if stage_name:
        print(
            f"[{stage_name}] done avg_loss={total_loss / max(total_steps, 1):.4f} "
            f"acc={metrics['overall_accuracy']:.4f} wmdc={metrics['weighted_mean_dice']:.4f}"
        )
    return total_loss / max(total_steps, 1), metrics


def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def train(config: ExperimentConfig) -> Path:
    _set_seed(config.training.seed)
    print("[train] preparing train statistics")
    ensure_train_stats(config)
    print("[train] preparing label store")
    build_label_store(config)

    device = _resolve_device()
    save_dir = _ensure_dir(config.training.save_dir)
    history_path = save_dir / "history.csv"

    train_dataset = build_daily_dataset(config, "train")
    val_dataset = build_daily_dataset(config, "val")
    print(
        f"[train] device={device} train_samples={len(train_dataset)} "
        f"val_samples={len(val_dataset)} batch_size={config.training.batch_size}"
    )
    train_loader = _build_loader(train_dataset, config.training.batch_size, config.training.num_workers, True)
    val_loader = _build_loader(val_dataset, config.training.batch_size, config.training.num_workers, False)

    model = _build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.amp and device.type == "cuda")

    best_path = save_dir / "best.pt"
    best_score = -float("inf")
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, config.training.epochs + 1):
        print(f"[train] epoch {epoch}/{config.training.epochs}")
        train_loss, train_metrics = _epoch_pass(
            model,
            train_loader,
            criterion,
            device,
            config.model.num_classes,
            config.data.ignore_index,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.training.amp,
            grad_accum_steps=config.training.grad_accum_steps,
            stage_name=f"train epoch {epoch}",
        )
        with torch.no_grad():
            val_loss, val_metrics = _epoch_pass(
                model,
                val_loader,
                criterion,
                device,
                config.model.num_classes,
                config.data.ignore_index,
                optimizer=None,
                scaler=None,
                amp=config.training.amp,
                stage_name=f"val epoch {epoch}",
            )

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(row)
        print(
            f"[train] epoch {epoch} summary "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_wmdc={train_metrics['weighted_mean_dice']:.4f} "
            f"val_wmdc={val_metrics['weighted_mean_dice']:.4f}"
        )

        current_score = val_metrics["weighted_mean_dice"]
        if current_score > best_score:
            best_score = current_score
            epochs_without_improvement = 0
            _save_checkpoint(best_path, model, optimizer, epoch, val_metrics)
            print(f"[train] new best checkpoint saved: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"[train] no improvement for {epochs_without_improvement} epoch(s); "
                f"best_wmdc={best_score:.4f}"
            )
            if epochs_without_improvement >= config.training.early_stopping_patience:
                print("[train] early stopping triggered")
                break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"[train] history written to {history_path}")
    return best_path


def predict(
    config: ExperimentConfig,
    split: str,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
) -> Path:
    device = _resolve_device()
    ckpt_path = Path(checkpoint_path or config.prediction.checkpoint_path or Path(config.training.save_dir) / "best.pt")
    out_dir = _ensure_dir(output_dir or Path(config.prediction.output_dir) / split)

    dataset = build_daily_dataset(config, split, include_labels=False)
    loader = _build_loader(dataset, config.prediction.batch_size, config.prediction.num_workers, False)
    print(
        f"[predict] split={split} device={device} samples={len(dataset)} "
        f"batch_size={config.prediction.batch_size}"
    )
    print(f"[predict] checkpoint={ckpt_path}")
    print(f"[predict] output_dir={out_dir}")
    model = _build_model(config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    processed = 0
    with torch.no_grad():
        total_batches = len(loader)
        for batch_index, batch in enumerate(_iter_with_progress(loader, desc=f"predict:{split}", total=total_batches), start=1):
            batch = _move_batch(batch, device)
            logits = model(batch["x"])
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            preds = np.argmax(probs, axis=1).astype(np.uint8)
            ocean_masks = batch["ocean_mask"].cpu().numpy().astype(np.uint8)
            for index, date in enumerate(batch["date"]):
                np.savez_compressed(
                    out_dir / f"{date}.npz",
                    prediction=preds[index],
                    probabilities=probs[index],
                    ocean_mask=ocean_masks[index],
                )
                processed += 1
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == total_batches:
                print(f"[predict] progress batches={batch_index}/{total_batches} samples={processed}/{len(dataset)}")
    print(f"[predict] finished split={split} samples={processed}")
    return out_dir


def evaluate(
    config: ExperimentConfig,
    split: str,
    pred_dir: str | None = None,
) -> Path:
    prediction_dir = Path(pred_dir or Path(config.prediction.output_dir) / split)
    output_dir = _ensure_dir(Path(config.evaluation.output_dir) / split)
    dataset = build_daily_dataset(config, split, include_labels=True)
    print(f"[evaluate] split={split} prediction_dir={prediction_dir}")
    print(f"[evaluate] samples={len(dataset)} output_dir={output_dir}")

    conf_mat = torch.zeros((config.model.num_classes, config.model.num_classes), dtype=torch.float64)
    daily_rows: list[dict[str, Any]] = []
    total = len(dataset)
    for index, sample in enumerate(_iter_with_progress(dataset, desc=f"evaluate:{split}", total=total), start=1):
        pred_path = prediction_dir / f"{sample['date']}.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        with np.load(pred_path) as prediction_data:
            prediction = torch.from_numpy(prediction_data["prediction"].astype(np.int64))
        target = sample["y"]
        conf_mat += confusion_matrix(prediction, target, config.model.num_classes, config.data.ignore_index)

        gt_np = target.numpy()
        pred_np = prediction.numpy()
        gt_cyc, _ = ndimage.label(gt_np == config.data.cyclonic_index)
        gt_anti, _ = ndimage.label(gt_np == config.data.anticyclonic_index)
        pred_cyc, pred_cyc_count = ndimage.label(pred_np == config.data.cyclonic_index)
        pred_anti, pred_anti_count = ndimage.label(pred_np == config.data.anticyclonic_index)
        daily_rows.append(
            {
                "date": sample["date"],
                "pred_cyclonic_components": int(pred_cyc_count),
                "pred_anticyclonic_components": int(pred_anti_count),
                "gt_cyclonic_components": int(sample["label_meta"]["cyclonic_count"]),
                "gt_anticyclonic_components": int(sample["label_meta"]["anticyclonic_count"]),
                "gt_overlap_count": int(sample["label_meta"]["overlap_count"]),
            }
        )
        del gt_cyc, gt_anti, pred_cyc, pred_anti
        if index == 1 or index % 100 == 0 or index == total:
            print(f"[evaluate] progress {index}/{total} date={sample['date']}")

    metrics = summarize_metrics(conf_mat)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    daily_path = output_dir / "daily_component_summary.csv"
    with daily_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily_rows[0].keys()))
        writer.writeheader()
        writer.writerows(daily_rows)
    print(
        f"[evaluate] done acc={metrics['overall_accuracy']:.4f} "
        f"wmdc={metrics['weighted_mean_dice']:.4f} metrics={metrics_path}"
    )
    return metrics_path
