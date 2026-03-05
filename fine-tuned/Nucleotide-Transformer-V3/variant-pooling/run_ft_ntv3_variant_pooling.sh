#!/usr/bin/env bash
set -euo pipefail

# **Varies depending on the experiment**
# RUN ROOT
: "${RUN_ROOT:=/path/to/experiments/ntv3/bend}"

# scripts
: "${POOL_SCRIPT_DIR:=/fine-tuned/Nucleotide-Transformer-V3/variant-pooling}"
POOL_PY="${POOL_SCRIPT_DIR}/ft_ntv3_variant_pooling.py"

# where FT outputs are
FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
BEST_FILE="${FT_OUTPUT_ROOT}/best_run.txt"

# mutation pooling inputs
: "${POOL_BP:=1024}"
: "${VARIANT_LIST:=/path/to/data/variant_list.feather}"
: "${DNV:=/path/to/data/variant.feather}"
: "${CHECKPOINT:=InstaDeepAI/NTv3_650M_post}"
: "${SPECIES:=human}"
: "${MERGE_LORA:=0}"

# output/logs
OUT_DIR="${RUN_ROOT}/mut_pool/output_best"
LOG_DIR="${RUN_ROOT}/mut_pool/logs_best"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# GPU 
if [[ -z "${POOL_GPUS+x}" ]]; then
    POOL_GPUS=(0)
fi
NUM_GPUS=${#POOL_GPUS[@]}

# INFO
echo "============================================================"
echo "[INFO] NTv3 Fine-tuned Mutation Pooling (GPU Parallel)"
echo "============================================================"
echo "[INFO] RUN_ROOT        = ${RUN_ROOT}"
echo "[INFO] FT_OUTPUT_ROOT  = ${FT_OUTPUT_ROOT}"
echo "[INFO] GPUs            = ${POOL_GPUS[@]}"
echo "[INFO] NUM_GPUS        = ${NUM_GPUS}"
echo "[INFO] BP              = ${POOL_BP}"
echo "[INFO] SPECIES         = ${SPECIES}"
echo "[INFO] CHECKPOINT      = ${CHECKPOINT}"
echo "============================================================"
echo


# 1) Load BEST run
if [[ ! -f "${BEST_FILE}" ]]; then
  echo "[ERROR] best_run.txt not found: ${BEST_FILE}"
  echo "You must run FT sweep + pick_best.py before pooling."
  exit 1
fi

BEST_RUN_DIR="$(cat "${BEST_FILE}")"
BEST_RUN_DIR="$(echo "${BEST_RUN_DIR}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

# relative path safety
if [[ "${BEST_RUN_DIR}" != /* ]]; then
  BEST_RUN_DIR="${FT_OUTPUT_ROOT}/${BEST_RUN_DIR}"
fi

if [[ ! -d "${BEST_RUN_DIR}" ]]; then
  echo "[ERROR] BEST_RUN_DIR not found: ${BEST_RUN_DIR}"
  exit 1
fi

echo "[INFO] BEST_RUN_DIR = ${BEST_RUN_DIR}"
echo


# 2) Setup paths
RUN_NAME="$(basename "${BEST_RUN_DIR}")"
ADIR="${BEST_RUN_DIR}/adapter_best"


if [[ ! -d "${ADIR}" ]]; then
    echo "[ERROR] adapter_best not found: ${ADIR}"
    echo "NTv3 pooling requires LoRA adapter directory"
    exit 1
fi

echo "[INFO] Using LoRA adapter: ${ADIR}"
echo

TEMP_DIR="${OUT_DIR}/temp_splits"
mkdir -p "${TEMP_DIR}"

echo "============================================================"
echo "[INFO] Starting parallel GPU pooling with ${NUM_GPUS} GPUs"
echo "============================================================"

PIDS=()

for i in "${!POOL_GPUS[@]}"; do
    GPU_ID=${POOL_GPUS[$i]}
    
    OUT_PART="${TEMP_DIR}/${RUN_NAME}_bp${POOL_BP}_gpu${GPU_ID}_part${i}.feather"
    LOG_PART="${LOG_DIR}/pool_${RUN_NAME}_gpu${GPU_ID}_part${i}.log"
    
    echo "[GPU ${GPU_ID}] Starting process ${i}/${NUM_GPUS}..."
    
    POOL_ARGS=(
        --bp "${POOL_BP}"
        --variant_list_path "${VARIANT_LIST}"
        --dnv_path "${DNV}"
        --base_ckpt "${CHECKPOINT}"
        --lora_adapter_dir "${ADIR}"
        --species "${SPECIES}"
        --device "cuda:0"
        --batch_size 800
        --use_amp
        --out_path "${OUT_PART}"
        --data_split "${i}"
        --num_splits "${NUM_GPUS}"
    )
    
    if [[ "${MERGE_LORA}" == "1" ]]; then
        POOL_ARGS+=( --merge_lora )
    fi
    
    if [[ $i -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES=${GPU_ID} python "${POOL_PY}" "${POOL_ARGS[@]}" \
            2>&1 | tee "${LOG_PART}" &
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python "${POOL_PY}" "${POOL_ARGS[@]}" \
            > "${LOG_PART}" 2>&1 &
    fi
    
    PIDS+=($!)
    echo "[GPU ${GPU_ID}] PID: ${PIDS[$i]}"
done

echo
echo "============================================================"
echo "All ${NUM_GPUS} processes started. Waiting for completion..."
echo "============================================================"

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    GPU_ID=${POOL_GPUS[$i]}
    wait ${PID}
    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo "[GPU ${GPU_ID}] Completed (PID: ${PID})"
    else
        echo "[GPU ${GPU_ID}] Failed with exit code ${EXIT_CODE} (PID: ${PID})"
        exit 1
    fi
done

echo
echo "============================================================"
echo "Merging results from all GPUs..."
echo "============================================================"

FINAL_OUT="${OUT_DIR}/${RUN_NAME}_bp${POOL_BP}_mutmax.feather"
MERGE_LOG="${LOG_DIR}/merge_${RUN_NAME}.log"

python - "${TEMP_DIR}" "${FINAL_OUT}" <<'PY' 2>&1 | tee "${MERGE_LOG}"
import pandas as pd
import glob
import sys

if len(sys.argv) < 3:
    print("Error: Missing arguments", file=sys.stderr)
    sys.exit(1)

temp_dir = sys.argv[1]
final_out = sys.argv[2]

parts = sorted(glob.glob(temp_dir + "/*.feather"))
print("Found {} parts to merge".format(len(parts)))

if not parts:
    print("Error: No feather files found in {}".format(temp_dir), file=sys.stderr)
    sys.exit(1)

dfs = [pd.read_feather(p) for p in parts]
merged = pd.concat(dfs, axis=0, ignore_index=True)

merged.to_feather(final_out)
print("Merged output saved: {}".format(final_out))
print("   Shape: {}".format(merged.shape))
print("   Columns: {}".format(list(merged.columns)))
PY

rm -rf "${TEMP_DIR}"

echo
echo "============================================================"
echo "NTv3 Parallel Pooling DONE"
echo "   Best run  : ${BEST_RUN_DIR}"
echo "   Output    : ${FINAL_OUT}"
echo "   Merge log : ${MERGE_LOG}"
echo "   GPU logs  : ${LOG_DIR}/pool_${RUN_NAME}_gpu*.log"
echo "============================================================"