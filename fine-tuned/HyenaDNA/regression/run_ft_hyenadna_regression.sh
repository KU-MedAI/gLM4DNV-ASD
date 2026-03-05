#!/usr/bin/env bash
set -euo pipefail

export WANDB_ENTITY="${WANDB_ENTITY:-autism}"
export WANDB_PROJECT="${WANDB_PROJECT:-hyenadna_ft_ncre}"

: "${RUN_ROOT:=/path/to/experiments/hyenadna/ncre}"
export RUN_ROOT

# Paths
BASE_DIR="/fine-tuned/HyenaDNA/regression"
DATA_PATH="/path/to/data/ncre_annot.feather"
SEQ_COL="NCREs_seq"
LABEL_COL="activity_score"
SPLIT_COL="split"
CHECKPOINT="LongSafari/hyenadna-large-1m-seqlen-hf"

# **fix**
FT_OUTPUT_ROOT="${RUN_ROOT}/ft/output"
FT_LOG_ROOT="${RUN_ROOT}/ft/logs"
mkdir -p "${FT_OUTPUT_ROOT}" "${FT_LOG_ROOT}"

PY="${BASE_DIR}/ft_hyenadna_regression.py"

# GPU settings
GPU_IDS=(1)
SEED=42
WARMUP_STEPS=200

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

while [[ $# -gt 0 ]]; do
  case $1 in
    --learning_rate=*) LR="${1#*=}"; shift ;;
    --weight_decay=*) WD="${1#*=}"; shift ;;
    --per_device_train_batch_size=*) BS="${1#*=}"; shift ;;
    --lora_r=*) R="${1#*=}"; shift ;;
    --lora_dropout=*) DROP="${1#*=}"; shift ;;
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
RUN_NAME="lr${LR}_r${R}_bs${BS}_wd${WD}_dp${DROP}_seed${SEED}_${RUN_ID}"

python "${PY}" \
  --data_path "${DATA_PATH}" \
  --seq_col "${SEQ_COL}" \
  --label_col "${LABEL_COL}" \
  --split_col "${SPLIT_COL}" \
  --checkpoint "${CHECKPOINT}" \
  --output_root "${FT_OUTPUT_ROOT}" \
  --output_subdir "${RUN_NAME}" \
  --log_root "${FT_LOG_ROOT}" \
  --log_subdir "${RUN_NAME}" \
  --num_train_epochs 100 \
  --per_device_train_batch_size "${BS}" \
  --per_device_eval_batch_size "${BS}" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --bf16 \
  --use_lora \
  --lora_r "${R}" \
  --lora_alpha "${ALPHA}" \
  --lora_dropout "${DROP}" \
  --lora_target_modules "in_proj,out_proj" \
  --keep_layernorm_trainable 0 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --early_stopping \
  --early_stopping_patience 5 \
  --early_stopping_threshold 0 \
  --logging_steps 100 \
  --overwrite_output_dir \
  --remove_unused_columns False \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --seed "${SEED}"
AGENT_EOF

chmod +x "${AGENT_SCRIPT}"
export AGENT_SCRIPT FT_OUTPUT_ROOT FT_LOG_ROOT DATA_PATH CHECKPOINT PY SEQ_COL LABEL_COL SPLIT_COL SEED WARMUP_STEPS

SWEEP_OUT="$(python - <<'EOF'
import os, wandb
ENTITY, PROJECT, AGENT_SCRIPT = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"], os.environ["AGENT_SCRIPT"]
cfg = {
    "method": "grid",
    "metric": {"name": "test/PCC", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [5e-5]},
        "weight_decay": {"values": [0.01]},
        "per_device_train_batch_size": {"values": [256]},
        "lora_r": {"values": [4]},
        "lora_dropout": {"values": [0.05]},
    },
    "command": ["${env}", "bash", AGENT_SCRIPT, "${args}"]
}
sid = wandb.sweep(cfg, entity=ENTITY, project=PROJECT)
print(f"{ENTITY}/{PROJECT}/{sid}")
EOF
)"

SWEEP_PATH="$(echo "${SWEEP_OUT}" | grep -Eo '[^ ]+/[^ ]+/[A-Za-z0-9]+' | tail -n 1)"
echo "[OK] SWEEP_PATH=${SWEEP_PATH}"

PIDS=()
for gpu_id in "${GPU_IDS[@]}"; do
(
  export CUDA_VISIBLE_DEVICES=${gpu_id}
  wandb agent "${SWEEP_PATH}"
) &
PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "SWEEP DONE"

python - <<'PY'
import os, glob, pandas as pd, math
root = os.environ["FT_OUTPUT_ROOT"]
pattern = os.path.join(root, "*", "test_metrics.csv")
paths = glob.glob(pattern)
best_p, best_pcc = None, -float("inf")

for p in paths:
    try:
        df = pd.read_csv(p)
        if "test_PCC" not in df.columns: continue
        pcc = float(df.loc[0, "test_PCC"])
        if math.isfinite(pcc) and pcc > best_pcc:
            best_pcc, best_p = pcc, p
    except Exception: pass

if best_p:
    run_dir = os.path.dirname(best_p)
    with open(os.path.join(root, "best_run.txt"), "w") as f: f.write(run_dir + "\n")
    print(f"BEST_PCC = {best_pcc}\nBEST_RUN_DIR = {run_dir}")
PY

rm -f "${AGENT_SCRIPT}"
echo "ALL DONE"