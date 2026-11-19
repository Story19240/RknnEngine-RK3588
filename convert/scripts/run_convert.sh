#!/bin/bash
# One-shot YOLOv8 slim-head -> RKNN conversion inside the container.
#
#   run_convert.sh --weights best.pt --calib /calib --size 640 --platform rk3588
#   run_convert.sh --weights best.pt --quantized FP16 --optimization-level 3
set -euo pipefail

WEIGHTS=""
CALIB=""
SIZE=640
PLATFORM=rk3588
OUT=/work
NUM=200
QUANT=INT8
HEAD_MODE=slim
OPT_LEVEL=3
OPSET=12
PER_CLASS=""
SEED=42

usage() {
  cat <<EOF
Usage: run_convert.sh --weights <pt> [options]
  --weights PATH    (required) yolov8 .pt to convert
  --calib PATH       image dir for INT8 calibration (required if --quantized INT8)
  --size N           input size (default 640)
  --platform NAME    rk3566|rk3568|rk3588|rk3562 (default rk3588)
  --out PATH         output dir (default /work)
  --num N            #calibration images (default 200)
  --quantized INT8|FP16  (default INT8)
  --head-mode MODE   slim|fused (default slim)
  --optimization-level N  1|2|3 (default 3)
  --opset N          ONNX opset (default 12)
  --per-class N      calibration images per class (default: --num / class count)
  --seed N           calibration sampling seed (default 42)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --weights)          WEIGHTS="$2";   shift 2;;
    --calib)            CALIB="$2";     shift 2;;
    --size)             SIZE="$2";      shift 2;;
    --platform)         PLATFORM="$2";  shift 2;;
    --out)              OUT="$2";       shift 2;;
    --num)              NUM="$2";       shift 2;;
    --quantized)        QUANT="$2";     shift 2;;
    --head-mode)        HEAD_MODE="$2"; shift 2;;
    --optimization-level) OPT_LEVEL="$2"; shift 2;;
    --opset)            OPSET="$2";     shift 2;;
    --per-class)        PER_CLASS="$2"; shift 2;;
    --seed)             SEED="$2";      shift 2;;
    -h|--help)   usage; exit 0;;
    *) echo "unknown arg: $1"; usage; exit 1;;
  esac
done

[ -n "$WEIGHTS" ] || { echo "ERROR: --weights is required"; usage; exit 1; }
[ -f "$WEIGHTS" ] || { echo "ERROR: weights not found: $WEIGHTS"; exit 1; }
case "$HEAD_MODE" in
  slim|fused) ;;
  *) echo "ERROR: --head-mode must be slim or fused"; exit 1;;
esac
case "$OPT_LEVEL" in
  1|2|3) ;;
  *) echo "ERROR: --optimization-level must be 1, 2, or 3"; exit 1;;
esac
case "$QUANT" in
  INT8|FP16) ;;
  *) echo "ERROR: --quantized must be INT8 or FP16"; exit 1;;
esac
mkdir -p "$OUT"

base=$(basename "$WEIGHTS" .pt)
ONNX="$OUT/${base}_${HEAD_MODE}.onnx"
ONNX_META="$OUT/${base}_${HEAD_MODE}.onnx.json"
if [ "$QUANT" = "INT8" ]; then
  QUANT_SUFFIX=int8
else
  QUANT_SUFFIX=16bit
fi
RKNN="$OUT/${base}_${HEAD_MODE}_${QUANT_SUFFIX}_opt${OPT_LEVEL}.rknn"
RKNN_META="$OUT/${base}_${HEAD_MODE}_${QUANT_SUFFIX}_opt${OPT_LEVEL}.rknn.json"

echo "==> [1/3] export ONNX ($WEIGHTS -> $ONNX, mode=$HEAD_MODE, imgsz=$SIZE)"
python3 /opt/convert/export_onnx.py \
  --weights "$WEIGHTS" \
  --imgsz "$SIZE" \
  --opset "$OPSET" \
  --head-mode "$HEAD_MODE" \
  --out "$ONNX" \
  --metadata-out "$ONNX_META"

if [ "$QUANT" = "INT8" ]; then
  [ -n "$CALIB" ] || { echo "ERROR: INT8 needs --calib <dir>"; usage; exit 1; }
  DATASET="$OUT/${base}_${HEAD_MODE}_dataset.txt"
  DATASET_META="$OUT/${base}_${HEAD_MODE}_calibration.json"
  echo "==> [2/3] build balanced calibration list (src=$CALIB, num=$NUM)"
  PREPARE_ARGS=(
    --src "$CALIB"
    --num "$NUM"
    --out "$DATASET"
    --manifest "$DATASET_META"
    --seed "$SEED"
  )
  if [ -n "$PER_CLASS" ]; then
    PREPARE_ARGS+=(--per-class "$PER_CLASS")
  fi
  python3 /opt/convert/prepare_quant_dataset.py "${PREPARE_ARGS[@]}"
  echo "==> [3/3] onnx -> rknn (INT8, platform=$PLATFORM, opt=$OPT_LEVEL)"
  python3 /opt/convert/to_rknn.py \
    --onnx "$ONNX" \
    --dataset "$DATASET" \
    --platform "$PLATFORM" \
    --optimization-level "$OPT_LEVEL" \
    --out "$RKNN" \
    --manifest "$RKNN_META" \
    --quantized INT8
else
  echo "==> [2/2] onnx -> rknn (FP16, no calibration, platform=$PLATFORM)"
  python3 /opt/convert/to_rknn.py \
    --onnx "$ONNX" \
    --platform "$PLATFORM" \
    --optimization-level "$OPT_LEVEL" \
    --out "$RKNN" \
    --manifest "$RKNN_META" \
    --quantized FP16
fi

echo "================================================"
echo "[OK] RKNN -> $RKNN"
echo "    metadata -> $RKNN_META"
echo "    Make sure librknnrt.so on the board matches this rknn-toolkit2 version."
ls -lh "$RKNN"
