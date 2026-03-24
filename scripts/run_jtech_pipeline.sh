#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

ENV_NAME=${ENV_NAME:-ocean-swinlstm}
CONFIG_PATH=${CONFIG_PATH:-"$REPO_ROOT/configs/jtech_adapt.yaml"}
LABEL_WORKERS_DEFAULT=${LABEL_WORKERS_DEFAULT:-2}
CONTOUR_WORKERS_DEFAULT=${CONTOUR_WORKERS_DEFAULT:-8}

run_python() {
  conda run -n "$ENV_NAME" python "$@"
}

usage() {
  cat <<EOF
Usage:
  sh scripts/run_jtech_pipeline.sh contours <start_date> <end_date> [workers] [output_dir]
  sh scripts/run_jtech_pipeline.sh labels [split] [workers]
  sh scripts/run_jtech_pipeline.sh train
  sh scripts/run_jtech_pipeline.sh predict <split> [checkpoint] [output_dir]
  sh scripts/run_jtech_pipeline.sh evaluate <split> [pred_dir]
  sh scripts/run_jtech_pipeline.sh train_eval [split]
  sh scripts/run_jtech_pipeline.sh full <start_date> <end_date> [split]

Environment variables:
  ENV_NAME                Conda environment name. Default: ocean-swinlstm
  CONFIG_PATH             Config file path. Default: configs/jtech_adapt.yaml
  LABEL_WORKERS_DEFAULT   Default workers for build_label_store.py
  CONTOUR_WORKERS_DEFAULT Default workers for label_eddy_contours.py

Examples:
  sh scripts/run_jtech_pipeline.sh train
  sh scripts/run_jtech_pipeline.sh predict test
  sh scripts/run_jtech_pipeline.sh evaluate test
  sh scripts/run_jtech_pipeline.sh train_eval test
  sh scripts/run_jtech_pipeline.sh full 1993-01-01 2024-12-31 test
EOF
}

run_contours() {
  if [ $# -lt 2 ]; then
    echo "contours requires <start_date> <end_date>" >&2
    exit 1
  fi
  START_DATE=$1
  END_DATE=$2
  WORKERS=${3:-$CONTOUR_WORKERS_DEFAULT}
  OUTPUT_DIR=${4:-outputs/meta40_dt_contours}

  echo "[pipeline] step=contours env=$ENV_NAME config=$CONFIG_PATH"
  echo "[pipeline] start_date=$START_DATE end_date=$END_DATE workers=$WORKERS output_dir=$OUTPUT_DIR"
  run_python "$REPO_ROOT/scripts/label_eddy_contours.py" \
    --start-date "$START_DATE" \
    --end-date "$END_DATE" \
    --output-dir "$OUTPUT_DIR" \
    --workers "$WORKERS"
}

run_labels() {
  SPLIT=${1:-}
  WORKERS=${2:-$LABEL_WORKERS_DEFAULT}

  echo "[pipeline] step=labels env=$ENV_NAME config=$CONFIG_PATH workers=$WORKERS"
  if [ -n "$SPLIT" ]; then
    run_python "$REPO_ROOT/scripts/build_label_store.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --workers "$WORKERS"
  else
    run_python "$REPO_ROOT/scripts/build_label_store.py" \
      --config "$CONFIG_PATH" \
      --workers "$WORKERS"
  fi
}

run_train() {
  echo "[pipeline] step=train env=$ENV_NAME config=$CONFIG_PATH"
  run_python "$REPO_ROOT/scripts/train_spatiotemporal_segmentation.py" --config "$CONFIG_PATH"
}

run_predict() {
  if [ $# -lt 1 ]; then
    echo "predict requires <split>" >&2
    exit 1
  fi
  SPLIT=$1
  CHECKPOINT=${2:-}
  OUTPUT_DIR=${3:-}

  echo "[pipeline] step=predict split=$SPLIT env=$ENV_NAME config=$CONFIG_PATH"
  if [ -n "$CHECKPOINT" ] && [ -n "$OUTPUT_DIR" ]; then
    run_python "$REPO_ROOT/scripts/predict_spatiotemporal_segmentation.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$OUTPUT_DIR"
  elif [ -n "$CHECKPOINT" ]; then
    run_python "$REPO_ROOT/scripts/predict_spatiotemporal_segmentation.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --checkpoint "$CHECKPOINT"
  elif [ -n "$OUTPUT_DIR" ]; then
    run_python "$REPO_ROOT/scripts/predict_spatiotemporal_segmentation.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --output-dir "$OUTPUT_DIR"
  else
    run_python "$REPO_ROOT/scripts/predict_spatiotemporal_segmentation.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT"
  fi
}

run_evaluate() {
  if [ $# -lt 1 ]; then
    echo "evaluate requires <split>" >&2
    exit 1
  fi
  SPLIT=$1
  PRED_DIR=${2:-}

  echo "[pipeline] step=evaluate split=$SPLIT env=$ENV_NAME config=$CONFIG_PATH"
  if [ -n "$PRED_DIR" ]; then
    run_python "$REPO_ROOT/scripts/evaluate_spatiotemporal_segmentation.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT" \
      --pred-dir "$PRED_DIR"
  else
    run_python "$REPO_ROOT/scripts/evaluate_spatiotemporal_segmentation.py" \
      --config "$CONFIG_PATH" \
      --split "$SPLIT"
  fi
}

run_train_eval() {
  SPLIT=${1:-test}
  run_train
  run_predict "$SPLIT"
  run_evaluate "$SPLIT"
}

run_full() {
  if [ $# -lt 2 ]; then
    echo "full requires <start_date> <end_date>" >&2
    exit 1
  fi
  START_DATE=$1
  END_DATE=$2
  SPLIT=${3:-test}

  run_contours "$START_DATE" "$END_DATE"
  run_labels
  run_train
  run_predict "$SPLIT"
  run_evaluate "$SPLIT"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

COMMAND=$1
shift

case "$COMMAND" in
  contours)
    run_contours "$@"
    ;;
  labels)
    run_labels "$@"
    ;;
  train)
    run_train
    ;;
  predict)
    run_predict "$@"
    ;;
  evaluate)
    run_evaluate "$@"
    ;;
  train_eval)
    run_train_eval "$@"
    ;;
  full)
    run_full "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage
    exit 1
    ;;
esac
