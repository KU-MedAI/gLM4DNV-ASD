#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/path/to/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN not found: ${PYTHON_BIN}"
  exit 1
fi

THIS_SCRIPT="$(readlink -f "$0" 2>/dev/null || echo "$0")"
echo "============================================================"
echo "[RUNNING SCRIPT] ${THIS_SCRIPT}"
echo "============================================================"
echo

# RUN ROOT
: "${RUN_ROOT:=/path/to/experiments/hyenadna/bend}"

: "${POOL_SCRIPT_DIR:=/fine-tuned/HyenaDNA/variant-pooling}"
POOL_PY="${POOL_SCRIPT_DIR}/ft_hyenadna_variant_pooling.py"

FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
BEST_FILE="${FT_OUTPUT_ROOT}/best_run.txt"

BP=200
VARIANT_LIST="/path/to/data/variant_list.feather"
DNV="/path/to/data/variant.feather"
BASE="LongSafari/hyenadna-large-1m-seqlen-hf"

MERGE_LORA=0
PAD_IDX=4

REVERSE_MODE="reverse"
POOLING="max"

OUT_DIR="${RUN_ROOT}/mut_pool/output_best"
LOG_DIR="${RUN_ROOT}/mut_pool/logs_best"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

GPUS=(1)
NUM_SPLITS="${#GPUS[@]}"
BATCH_SIZE=5000

for f in "${POOL_PY}" "${VARIANT_LIST}" "${DNV}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] Missing file: ${f}"
    exit 1
  fi
done

if [[ ! -f "${BEST_FILE}" ]]; then
  echo "[ERROR] best_run.txt not found: ${BEST_FILE}"
  exit 1
fi

BEST_RUN_DIR_RAW="$(cat "${BEST_FILE}")"
BEST_RUN_DIR_RAW="$(echo "${BEST_RUN_DIR_RAW}" | sed 's/[>]+$//' | tr -d '\r')"
BEST_RUN_DIR="$(echo "${BEST_RUN_DIR_RAW}" | awk '{print $NF}')"

echo "[INFO] BEST_RUN_DIR_RAW = ${BEST_RUN_DIR_RAW}"
echo "[INFO] BEST_RUN_DIR     = ${BEST_RUN_DIR}"
echo

if [[ "${BEST_RUN_DIR}" != /* ]]; then
  BEST_RUN_DIR="${FT_OUTPUT_ROOT}/${BEST_RUN_DIR}"
fi

if [[ ! -d "${BEST_RUN_DIR}" ]]; then
  echo "[ERROR] BEST_RUN_DIR not found: ${BEST_RUN_DIR}"
  exit 1
fi

RUN_NAME="$(basename "${BEST_RUN_DIR}")"
ADIR="${BEST_RUN_DIR}/adapter_best"
PTH="${BEST_RUN_DIR}/best.pth"

echo "[INFO] RUN_NAME = ${RUN_NAME}"
echo "[INFO] ADIR     = ${ADIR}"
echo "[INFO] PTH      = ${PTH}"
echo

FINAL_OUT="${OUT_DIR}/${RUN_NAME}_bp${BP}_mutmax_fwdrev_concat_split${NUM_SPLITS}.feather"
TEMP_DIR="${OUT_DIR}/temp_${RUN_NAME}"
MERGE_LOG="${LOG_DIR}/merge_${RUN_NAME}.log"

rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

PIDS=()

for i in "${!GPUS[@]}"; do
  GPU_ID="${GPUS[$i]}"
  OUT_PART="${TEMP_DIR}/part_${i}.feather"
  LOG_PART="${LOG_DIR}/pool_gpu${GPU_ID}_split${i}.log"

  echo "[LAUNCH] split ${i} GPU ${GPU_ID}"

  ARGS=(
    --bp "${BP}"
    --variant_list_path "${VARIANT_LIST}"
    --dnv_path "${DNV}"
    --base_ckpt "${BASE}"
    --device "cuda:0"
    --batch_size "${BATCH_SIZE}"
    --pad_idx "${PAD_IDX}"
    --reverse_mode "${REVERSE_MODE}"
    --pooling "${POOLING}"
    --out_path "${OUT_PART}"
    --data_split "${i}"
    --num_splits "${NUM_SPLITS}"
  )

  if [[ -d "${ADIR}" ]]; then
    ARGS+=( --lora_adapter_dir "${ADIR}" )
  elif [[ -f "${PTH}" ]]; then
    ARGS+=( --ft_pth_path "${PTH}" )
  else
    echo "[ERROR] No adapter or pth in ${BEST_RUN_DIR}"
    exit 1
  fi

  (
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    "${PYTHON_BIN}" "${POOL_PY}" "${ARGS[@]}"
  ) >> "${LOG_PART}" 2>&1 &

  PIDS+=("$!")
done

echo "[WAIT] splits..."
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

"${PYTHON_BIN}" - <<EOF > "${MERGE_LOG}" 2>&1
import os, pandas as pd
temp_dir="${TEMP_DIR}"
final_out="${FINAL_OUT}"
parts=[os.path.join(temp_dir,f) for f in os.listdir(temp_dir)]
dfs=[pd.read_feather(p) for p in parts]
merged=pd.concat(dfs,ignore_index=True)
merged.to_feather(final_out)
print("Saved:",final_out,"shape=",merged.shape)
EOF

rm -rf "${TEMP_DIR}"

echo
echo "============================================================"
echo "DONE"
echo " Best run : ${BEST_RUN_DIR}"
echo " Output   : ${FINAL_OUT}"
echo "============================================================"