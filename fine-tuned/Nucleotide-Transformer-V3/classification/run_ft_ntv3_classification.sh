#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# RUN ROOT
# =========================================================
# **Varies depending on the experiment**
export WANDB_ENTITY="${WANDB_ENTITY:-autism}"
export WANDB_PROJECT="${WANDB_PROJECT:-ntv3_ft_bend}"

: "${RUN_ROOT:=/path/to/experiments/ntv3/bend}"
export RUN_ROOT

# Paths
: "${BASE_DIR:=/fine-tuned/Nucleotide-Transformer-V3/classification}"
: "${ANNOT_PATH:=/path/to/data/BEND_annot.feather}"
: "${SEQ_PATH:=/path/to/data/disease_seq_split_NTv3.parquet}"
: "${SPAN_BP:=1024}"
: "${LABEL_COL:=bend_label}"
: "${CHECKPOINT:=InstaDeepAI/NTv3_650M_post}"
: "${SPECIES:=human}"

FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
FT_LOG_ROOT="${RUN_ROOT}/ft/logs"
mkdir -p "${FT_OUTPUT_ROOT}" "${FT_LOG_ROOT}"

PY="${BASE_DIR}/ft_ntv3_classification.py"

# GPU settings
if [[ -z "${FT_GPU_IDS+x}" ]]; then
    FT_GPU_IDS=(0)
fi
NUM_GPUS=${#FT_GPU_IDS[@]}
GPU_IDS=("${FT_GPU_IDS[@]}")

# Fixed knobs
export SEED=42
export WARMUP_STEPS=200

cd "${BASE_DIR}"

# =========================================================
# 1) Create agent script
# =========================================================
AGENT_SCRIPT="${BASE_DIR}/wandb_agent_runner_ntv3_bend.sh"

cat > "${AGENT_SCRIPT}" <<'AGENT_EOF'
#!/usr/bin/env bash
set -euo pipefail

LR=""
WD=""
BS=""
R=""
DROP=""
CENTER_CROP=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --learning_rate=*) LR="${1#*=}"; shift ;;
    --weight_decay=*) WD="${1#*=}"; shift ;;
    --per_device_train_batch_size=*) BS="${1#*=}"; shift ;;
    --lora_r=*) R="${1#*=}"; shift ;;
    --lora_dropout=*) DROP="${1#*=}"; shift ;;
    --keep_target_center_fraction=*) CENTER_CROP="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

R_INT="$(printf "%s" "${R}" | sed 's/[^0-9].*//')"
if [[ -z "${R_INT}" ]]; then
  echo "[ERROR] Invalid lora_r: ${R}"
  exit 1
fi

ALPHA=$((R_INT * 2))
RUN_ID="${WANDB_RUN_ID:-local}"

if [[ -z "${CENTER_CROP}" ]]; then
  CENTER_CROP="0.375"
fi

RUN_NAME="lr${LR}_r${R}_a${ALPHA}_bs${BS}_wd${WD}_dp${DROP}_cc${CENTER_CROP}_seed${SEED}_wu${WARMUP_STEPS}_${RUN_ID}"
OUTPUT_SUBDIR="${RUN_NAME}"
LOG_SUBDIR="${RUN_NAME}"

echo "============================================"
echo "RUN_NAME: ${RUN_NAME}"
echo "RUN_ROOT: ${RUN_ROOT}"
echo "FT_OUTPUT_ROOT: ${FT_OUTPUT_ROOT}"
echo "FT_LOG_ROOT   : ${FT_LOG_ROOT}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

python "${PY}" \
  --annot_path "${ANNOT_PATH}" \
  --seq_path "${SEQ_PATH}" \
  --span_bp "${SPAN_BP}" \
  --label_col "${LABEL_COL}" \
  --checkpoint "${CHECKPOINT}" \
  --species "${SPECIES}" \
  --output_root "${FT_OUTPUT_ROOT}" \
  --output_subdir "${OUTPUT_SUBDIR}" \
  --bf16 \
  --log_root "${FT_LOG_ROOT}" \
  --log_subdir "${LOG_SUBDIR}" \
  --num_train_epochs 100 \
  --per_device_train_batch_size "${BS}" \
  --per_device_eval_batch_size "${BS}" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --keep_target_center_fraction "${CENTER_CROP}" \
  --use_lora \
  --lora_r "${R}" \
  --lora_alpha "${ALPHA}" \
  --lora_dropout "${DROP}" \
  --lora_target_modules "query_head.linear, value_head.linear" \
  --keep_layernorm_trainable 0 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --early_stopping \
  --early_stopping_patience 5 \
  --early_stopping_threshold 0.0 \
  --logging_steps 100 \
  --overwrite_output_dir \
  --remove_unused_columns False \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}"
AGENT_EOF

chmod +x "${AGENT_SCRIPT}"

# Export for agent script
export RUN_ROOT
export AGENT_SCRIPT FT_OUTPUT_ROOT FT_LOG_ROOT ANNOT_PATH SEQ_PATH SPAN_BP LABEL_COL CHECKPOINT SPECIES PY
export NUM_GPUS SEED WARMUP_STEPS

# =========================================================
# 2) Create Sweep
# =========================================================
echo "[PHASE 1] Creating Sweep"

SWEEP_OUT="$(python - <<'PY'
import os, wandb

ENTITY = os.environ["WANDB_ENTITY"]
PROJECT = os.environ["WANDB_PROJECT"]
AGENT_SCRIPT = os.environ["AGENT_SCRIPT"]

cfg = {
    "method": "grid",
    "metric": {"name": "test/AUROC", "goal": "maximize"},
    "parameters": {
        "learning_rate":               {"values": [5e-5]},
        "weight_decay":                {"values": [0.01]},
        "per_device_train_batch_size": {"values": [256]},
        "lora_r":                      {"values": [64]},
        "lora_dropout":                {"values": [0.05]},
        "keep_target_center_fraction": {"values": [0.375]},
    },
    "command": ["${env}", "bash", AGENT_SCRIPT, "${args}"]
}

sid = wandb.sweep(cfg, entity=ENTITY, project=PROJECT)
print(f"{ENTITY}/{PROJECT}/{sid}")
PY
)"

SWEEP_PATH="$(echo "${SWEEP_OUT}" | grep -Eo '[^ ]+/[^ ]+/[A-Za-z0-9]+' | tail -n 1)"
echo "[OK] SWEEP_PATH=${SWEEP_PATH}"

# =========================================================
# 3) Run Agents
# =========================================================
echo "[PHASE 2] Running Agents"

PIDS=()
for gpu_id in "${GPU_IDS[@]}"; do
(
  export CUDA_VISIBLE_DEVICES=${gpu_id}
  wandb agent "${SWEEP_PATH}"
) &
PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

echo "SWEEP DONE"

# =========================================================
# 4) Pick BEST by test_AUROC (ONLY ONE best_run.txt in ft/output)
# =========================================================
echo "[PHASE 3] Selecting BEST run"

python - <<'PY'
import os, glob, pandas as pd

root = os.environ["FT_OUTPUT_ROOT"]
span = os.environ["SPAN_BP"]

pattern = os.path.join(root, "*", "test_metrics.csv")
paths = glob.glob(pattern)

best_p = None
best_auc = -1.0

for p in paths:
    try:
        df = pd.read_csv(p)
        auc = float(df.loc[0, "test_AUROC"])
        if auc > best_auc:
            best_auc = auc
            best_p = p
    except:
        pass

if best_p is None:
    raise SystemExit("No valid test_metrics.csv found")

run_dir = os.path.dirname(best_p)

out_txt = os.path.join(root, "best_run.txt")
with open(out_txt, "w") as f:
    f.write(run_dir + "\n")

print("BEST_AUROC =", best_auc)
print("BEST_RUN_DIR =", run_dir)
print("Saved:", out_txt)
PY

rm -f "${AGENT_SCRIPT}"

echo "ALL DONE"
echo "RUN_ROOT     : ${RUN_ROOT}"
echo "Output root  : ${FT_OUTPUT_ROOT}"
echo "Logs root    : ${FT_LOG_ROOT}"
echo "Best run info: ${FT_OUTPUT_ROOT}/best_run.txt"