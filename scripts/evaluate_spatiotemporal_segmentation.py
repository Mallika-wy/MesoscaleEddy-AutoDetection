#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eddy_pipeline import evaluate, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions for the JTECH-adapted dual-attention ConvLSTM model.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "jtech_adapt.yaml"),
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        required=True,
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--pred-dir",
        default="",
        help="Optional prediction directory override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    metrics_path = evaluate(config, split=args.split, pred_dir=args.pred_dir or None)
    print(f"Evaluation metrics saved to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
