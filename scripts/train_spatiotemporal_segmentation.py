#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eddy_pipeline import load_config, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the JTECH-adapted dual-attention ConvLSTM eddy segmentation model.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "jtech_adapt.yaml"),
        help="Path to the experiment config YAML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = train(config)
    print(f"Training complete. Best checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
