#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eddy_pipeline import load_config, predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the JTECH-adapted dual-attention ConvLSTM model.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "jtech_adapt.yaml"),
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        required=True,
        help="Dataset split to predict.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional checkpoint override.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional prediction output override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_dir = predict(
        config,
        split=args.split,
        checkpoint_path=args.checkpoint or None,
        output_dir=args.output_dir or None,
    )
    print(f"Predictions saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
