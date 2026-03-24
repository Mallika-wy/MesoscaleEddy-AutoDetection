from __future__ import annotations

from typing import Any

import numpy as np
import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)
    indices = target * num_classes + pred
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.float64)


def dice_per_class(conf_mat: torch.Tensor) -> list[float]:
    num_classes = conf_mat.shape[0]
    scores: list[float] = []
    for class_index in range(num_classes):
        tp = conf_mat[class_index, class_index]
        fp = conf_mat[:, class_index].sum() - tp
        fn = conf_mat[class_index, :].sum() - tp
        denom = 2 * tp + fp + fn
        score = (2 * tp / denom).item() if denom > 0 else 0.0
        scores.append(score)
    return scores


def weighted_mean_dice(conf_mat: torch.Tensor) -> float:
    weights = conf_mat.sum(dim=1)
    total = weights.sum()
    if total <= 0:
        return 0.0
    dice = torch.tensor(dice_per_class(conf_mat), dtype=torch.float64, device=conf_mat.device)
    return float((dice * (weights / total)).sum().item())


def overall_accuracy(conf_mat: torch.Tensor) -> float:
    total = conf_mat.sum()
    return float((conf_mat.diag().sum() / total).item()) if total > 0 else 0.0


def summarize_metrics(conf_mat: torch.Tensor) -> dict[str, Any]:
    per_class = dice_per_class(conf_mat)
    return {
        "overall_accuracy": overall_accuracy(conf_mat),
        "weighted_mean_dice": weighted_mean_dice(conf_mat),
        "dice_background": per_class[0],
        "dice_cyclonic": per_class[1] if len(per_class) > 1 else 0.0,
        "dice_anticyclonic": per_class[2] if len(per_class) > 2 else 0.0,
    }


def merge_metric_dicts(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}
