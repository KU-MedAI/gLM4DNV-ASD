#!/usr/bin/env bash
set -euo pipefail

export WANDB_ENTITY="${WANDB_ENTITY:-autism}"
export WANDB_PROJECT="${WANDB_PROJECT:-hyenadna_ft_bend}"

: "${RUN_ROOT:=/path/to/experiments/hyenadna/bend}"
export RUN_ROOT

BASE_DIR="/fine-tuned/HyenaDNA/classification"
ANNOT_PATH="/path/to/data/BEND_annot.parquet"

SPAN_BP=200
LABEL_COL="bend_label"
CHECKPOINT="LongSafari/hyenadna-large-1m-seqlen-hf"

FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
FT_LOG_ROOT="${RUN_ROOT}/ft/logs"
mkdir -p "${FT_OUTPUT_ROOT}" "${FT_LOG_ROOT}"

PY="${BASE_DIR}/ft_hyenadna_classification.py"

NUM_GPUS=4
GPU_IDS=(1)

SEED=42
export WARMUP_STEPS="${WARMUP_STEPS:-200}"

cd "${BASE_DIR}"

AGENT_SCRIPT="${BASE_DIR}/wandb_agent_runner.sh"

cat > "${AGENT_SCRIPT}" <<'AGENT_EOF'
#!/usr/bin/env bash
set -euo pipefail

LR=""
WD=""
BS=""
R=""
DROP=""
ALPHA_MULT=""
WU=""
KEEP_LN=""
GC=""
BF16=""
FP16=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --learning_rate=*) LR="${1#*=}"; shift ;;
    --weight_decay=*) WD="${1#*=}"; shift ;;
    --per_device_train_batch_size=*) BS="${1#*=}"; shift ;;
    --lora_r=*) R="${1#*=}"; shift ;;
    --lora_dropout=*) DROP="${1#*=}"; shift ;;
    --alpha_mult=*) ALPHA_MULT="${1#*=}"; shift ;;
    --warmup_steps=*) WU="${1#*=}"; shift ;;
    --keep_ln=*) KEEP_LN="${1#*=}"; shift ;;
    --gradient_checkpointing=*) GC="${1#*=}"; shift ;;
    --bf16=*) BF16="${1#*=}"; shift ;;
    --fp16=*) FP16="${1#*=}"; shift ;;
    *) shift ;;
  esac
done

if [[ -z "${LR}" || -z "${WD}" || -z "${BS}" || -z "${R}" || -z "${DROP}" || -z "${ALPHA_MULT}" ]]; then
  echo "Missing required args."
  exit 1
fi

if [[ -z "${WU}" ]]; then WU="${WARMUP_STEPS:-50}"; fi
if [[ -z "${KEEP_LN}" ]]; then KEEP_LN="0"; fi
if [[ -z "${GC}" ]]; then GC="0"; fi
if [[ -z "${BF16}" ]]; then BF16="0"; fi
if [[ -z "${FP16}" ]]; then FP16="0"; fi

if [[ "${BF16}" == "1" && "${FP16}" == "1" ]]; then
  echo "bf16 and fp16 cannot both be 1"
  exit 1
fi

R_INT="$(printf "%s" "${R}" | sed 's/[^0-9].*//')"
if [[ -z "${R_INT}" ]]; then
  echo "Invalid lora_r: ${R}"
  exit 1
fi

ALPHA="$(python - <<PY
r=int("${R_INT}")
m=float("${ALPHA_MULT}")
a=int(round(r*m))
print(max(1,a))
PY
)"

RUN_ID="${WANDB_RUN_ID:-local}"

RUN_NAME="lr${LR}_r${R_INT}_am${ALPHA_MULT}_a${ALPHA}_se${NUM_GPUS}_bs${BS}_wd${WD}_dp${DROP}_seed${SEED}_wu${WU}_bf${BF16}_fp${FP16}_gc${GC}_${RUN_ID}"
OUTPUT_SUBDIR="${RUN_NAME}"
LOG_SUBDIR="${RUN_NAME}"

ARGS=(
  --annot_path "${ANNOT_PATH}"
  --span_bp "${SPAN_BP}"
  --label_col "${LABEL_COL}"
  --checkpoint "${CHECKPOINT}"
  --output_root "${FT_OUTPUT_ROOT}"
  --output_subdir "${OUTPUT_SUBDIR}"
  --log_root "${FT_LOG_ROOT}"
  --log_subdir "${LOG_SUBDIR}"
  --num_train_epochs 100
  --per_device_train_batch_size "${BS}"
  --per_device_eval_batch_size "${BS}"
  --learning_rate "${LR}"
  --weight_decay "${WD}"
  --warmup_steps "${WU}"
  --use_lora
  --lora_r "${R_INT}"
  --lora_alpha "${ALPHA}"
  --lora_dropout "${DROP}"
  --lora_target_modules "in_proj,out_proj"
  --pooling "mean"
  --eval_strategy epoch
  --save_strategy epoch
  --save_total_limit 2
  --load_best_model_at_end
  --metric_for_best_model eval_loss
  --early_stopping
  --early_stopping_patience 5
  --early_stopping_threshold 0
  --logging_steps 100
  --overwrite_output_dir
  --remove_unused_columns False
  --report_to wandb
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --keep_layernorm_trainable "${KEEP_LN}"
)

if [[ "${GC}" == "1" ]]; then ARGS+=( --gradient_checkpointing ); fi
if [[ "${BF16}" == "1" ]]; then ARGS+=( --bf16 ); fi
if [[ "${FP16}" == "1" ]]; then ARGS+=( --fp16 ); fi

python "${PY}" "${ARGS[@]}"
AGENT_EOF

chmod +x "${AGENT_SCRIPT}"

export RUN_ROOT
export AGENT_SCRIPT FT_OUTPUT_ROOT FT_LOG_ROOT ANNOT_PATH SPAN_BP LABEL_COL CHECKPOINT PY
export NUM_GPUS SEED WARMUP_STEPS

echo "Creating Sweep"

SWEEP_OUT="$(python - <<'EOF'
import os, wandb
ENTITY = os.environ["WANDB_ENTITY"]
PROJECT = os.environ["WANDB_PROJECT"]
AGENT_SCRIPT = os.environ["AGENT_SCRIPT"]
cfg = {
    "method": "grid",
    "metric": {"name": "test/AUROC", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [5e-5]},
        "weight_decay": {"values": [0.01]},
        "per_device_train_batch_size": {"values": [256]},
        "lora_r": {"values": [4]},
        "alpha_mult": {"values": [2]},
        "lora_dropout": {"values": [0.05]},
        "warmup_steps": {"values": [200]},
        "keep_ln": {"values": [0]},
        "gradient_checkpointing": {"values": [0]},
        "bf16": {"values": [1]},
        "fp16": {"values": [0]},
    },
    "command": ["${env}", "bash", AGENT_SCRIPT, "${args}"],
}
sid = wandb.sweep(cfg, entity=ENTITY, project=PROJECT)
print(f"{ENTITY}/{PROJECT}/{sid}")
EOF
)"

SWEEP_PATH="$(echo "${SWEEP_OUT}" | grep -Eo '[^ ]+/[^ ]+/[A-Za-z0-9]+' | tail -n 1)"
echo "SWEEP_PATH=${SWEEP_PATH}"

echo "Running Agents"

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

echo "Selecting TOP5 runs"

python - <<'PY'
import os, glob, pandas as pd
root = os.environ["FT_OUTPUT_ROOT"]
pattern = os.path.join(root, "*", "test_metrics.csv")
paths = glob.glob(pattern)
rows = []
for p in paths:
    try:
        df = pd.read_csv(p)
        auc = float(df.loc[0, "test_AUROC"])
        run_dir = os.path.dirname(p)
        rows.append((auc, run_dir, p))
    except Exception:
        pass
if not rows:
    raise SystemExit("No valid test_metrics.csv found")
rows.sort(key=lambda x: x[0], reverse=True)
topk = rows[:5]
out_txt = os.path.join(root, "best_runs_top5.txt")
with open(out_txt, "w") as f:
    for auc, run_dir, _ in topk:
        f.write(f"{auc}\t{run_dir}\n")
print("TOP5 selected")
PY

echo "ALL DONE"