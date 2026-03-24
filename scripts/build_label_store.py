#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# 仓库根目录与源码目录。
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eddy_pipeline import build_label_store, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    # 这个脚本不负责检测涡旋。
    # 它只负责把 py-eddy-tracker 已经产出的轮廓文件，
    # 转成训练真正需要的像素级标签掩膜。
    parser = argparse.ArgumentParser(description="Rasterize py-eddy-tracker contour files into daily training masks.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "jtech_adapt.yaml"),
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        help="Optional split filter. May be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行进程数。0 表示使用配置文件中的 data.label_workers。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    print("Starting label-store build...")
    print(f"Config: {args.config}")
    print(f"Contour input dir: {config.data.contour_dir}")
    print(f"Label output dir: {config.data.label_store_dir}")
    print(f"Overwrite existing files: {args.overwrite}")
    print(f"Workers: {args.workers or config.data.label_workers}")
    if args.split:
        print(f"Target splits: {', '.join(args.split)}")
    else:
        print("Target splits: train, val, test")
    output_dir = build_label_store(
        config,
        splits=args.split,
        overwrite=args.overwrite,
        workers=args.workers or None,
    )
    print(f"Label store ready: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
