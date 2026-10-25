#!/usr/bin/env bash
# =============================================================================
# Single entrypoint that runs *inside* the NeMo AutoModel container.
#
# The same script is used by:
#   * hpc/run_local_docker.sh        (local Docker)
#   * hpc/run_local_singularity.sh   (local Apptainer/Singularity)
#   * hpc/pbs_*.pbs                  (Griffith cluster, singularity exec --nv)
#
# Usage (inside the container):
#   bash run_in_container.sh check          # GPU / torch / automodel sanity
#   bash run_in_container.sh prefetch       # download $MODEL_ID into HF cache
#   bash run_in_container.sh dataset-smoke  # tokenizer + dataloader check
#   bash run_in_container.sh smoke          # 512-row, 1-epoch training test
#   bash run_in_container.sh train          # full PEFT training
#   bash run_in_container.sh infer          # format-check a trained adapter
#
# Configuration is by environment variable (see LOCAL_TESTING.md and
# CLUSTER_RUNBOOK.md): MODEL_ID, DATA_PATH, CHECKPOINT_DIR, RUN_NAME,
# GLOBAL_BATCH_SIZE, LOCAL_BATCH_SIZE, NUM_EPOCHS, MAX_LENGTH, LR,
# GROUP_BY_LENGTH, USE_TRITON, NPROC_PER_NODE, HF_HOME, ...
# =============================================================================
set -euo pipefail

PEFT_DIR="${PEFT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODE="${1:-smoke}"
MODEL_ID="${MODEL_ID:-${PEFT_BASE_MODEL:-Qwen/Qwen3-8B}}"
DATA_PATH="${DATA_PATH:-$PEFT_DIR/dataset/train.jsonl}"
RUN_NAME="${RUN_NAME:-${PBS_JOBID:-local}}"
# PBS job ids can contain characters we do not want in a directory name.
RUN_NAME="$(printf '%s' "$RUN_NAME" | tr -cs 'A-Za-z0-9._-' '_')"

export HF_HOME="${HF_HOME:-$(cd "$PEFT_DIR/.." && pwd)/.cache/huggingface}"
mkdir -p "$HF_HOME" "$PEFT_DIR/runs/recipes" "$PEFT_DIR/logs"
cd "$PEFT_DIR"

log() { printf '\n[run_in_container] %s\n' "$*"; }

print_environment() {
  log "mode=$MODE host=$(hostname) date=$(date -Is 2>/dev/null || date)"
  echo "PEFT_DIR=$PEFT_DIR"
  echo "MODEL_ID=$MODEL_ID"
  echo "DATA_PATH=$DATA_PATH"
  echo "HF_HOME=$HF_HOME"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    echo "WARNING: nvidia-smi not found in this container" >&2
  fi
}

check_python_stack() {
  python3 - <<'PY'
import sys
print("python:", sys.version.split()[0], sys.executable)
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
    print("cuda capability:", torch.cuda.get_device_capability(0))
else:
    raise SystemExit(
        "torch.cuda.is_available() is False. Check Docker --gpus / "
        "Singularity --nv, the PBS ngpus request, and the host driver."
    )
PY
  command -v automodel >/dev/null

  echo "automodel: $(command -v automodel)"
}

require_dataset() {
  if [[ ! -f "$DATA_PATH" ]]; then
    cat >&2 <<EOF
ERROR: dataset not found at $DATA_PATH
On the host (not inside the read-only image), extract it first:
  cd 0.peft/dataset && unzip -o sft_data.zip
The recipe expects 0.peft/dataset/train.jsonl (35,000 rows).
EOF
    exit 1
  fi
  echo "dataset rows: $(wc -l < "$DATA_PATH")"
}

prefetch_model() {
  if [[ -d "$MODEL_ID" ]]; then
    echo "MODEL_ID is an existing local directory; skipping download: $MODEL_ID"
    return 0
  fi
  MODEL_ID="$MODEL_ID" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
model = os.environ["MODEL_ID"]
print(f"Downloading {model} into HF_HOME={os.environ.get('HF_HOME')}")
path = snapshot_download(model)
print("model snapshot:", path)
PY
}

RENDERED_RECIPE=""
render_for_run() {
  local rendered="$PEFT_DIR/runs/recipes/${MODE}-${RUN_NAME}.yaml"
  local args=(
    --base "$PEFT_DIR/recipe.yaml"
    --out "$rendered"
    --container-peft-dir "$PEFT_DIR"
    --model "$MODEL_ID"
    --data "$DATA_PATH"
  )
  [[ "$MODE" == "smoke" ]] && args+=(--smoke)
  [[ -n "${CHECKPOINT_DIR:-}" ]] && args+=(--checkpoint-dir "$CHECKPOINT_DIR")
  [[ -n "${PRESET:-}" ]] && args+=(--preset "$PRESET")
  [[ -n "${GLOBAL_BATCH_SIZE:-}" ]] && args+=(--global-batch-size "$GLOBAL_BATCH_SIZE")
  [[ -n "${LOCAL_BATCH_SIZE:-}" ]] && args+=(--local-batch-size "$LOCAL_BATCH_SIZE")
  [[ -n "${NUM_EPOCHS:-}" ]] && args+=(--num-epochs "$NUM_EPOCHS")
  [[ -n "${MAX_LENGTH:-}" ]] && args+=(--max-length "$MAX_LENGTH")
  [[ -n "${LR:-}" ]] && args+=(--lr "$LR")
  [[ -n "${CKPT_EVERY_STEPS:-}" ]] && args+=(--ckpt-every-steps "$CKPT_EVERY_STEPS")
  [[ -n "${VAL_EVERY_STEPS:-}" ]] && args+=(--val-every-steps "$VAL_EVERY_STEPS")
  [[ -n "${NUM_SAMPLES_LIMIT:-}" ]] && args+=(--num-samples-limit "$NUM_SAMPLES_LIMIT")
  [[ -n "${VAL_NUM_SAMPLES_LIMIT:-}" ]] && args+=(--val-num-samples-limit "$VAL_NUM_SAMPLES_LIMIT")
  [[ -n "${GROUP_BY_LENGTH:-}" ]] && args+=(--group-by-length "$GROUP_BY_LENGTH")
  [[ -n "${USE_TRITON:-}" ]] && args+=(--use-triton "$USE_TRITON")
  [[ -n "${SEED:-}" ]] && args+=(--seed "$SEED")

  python3 "$PEFT_DIR/render_recipe.py" "${args[@]}"
  RENDERED_RECIPE="$rendered"
}

run_training() {
  require_dataset
  check_python_stack
  render_for_run
  log "starting automodel with rendered recipe: $RENDERED_RECIPE"

  if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
    automodel --nproc-per-node="$NPROC_PER_NODE" "$RENDERED_RECIPE"
  else
    automodel "$RENDERED_RECIPE"
  fi

  log "training finished. checkpoint directories:"
  local output_dir="${CHECKPOINT_DIR:-}"
  if [[ -z "$output_dir" ]]; then
    if [[ "$MODE" == "smoke" ]]; then
      output_dir="$PEFT_DIR/checkpoints-smoke"
    else
      output_dir="$PEFT_DIR/checkpoints"
    fi
  fi
  find "$output_dir" -maxdepth 2 -type d 2>/dev/null | sort || true
}

run_infer() {
  check_python_stack
  local args=(
    --base "$MODEL_ID"
    --dataset "$DATA_PATH"
    --num-samples "${INFER_NUM_SAMPLES:-10}"
    --max-new-tokens "${INFER_MAX_NEW_TOKENS:-512}"
  )
  if [[ -n "${ADAPTER_PATH:-}" ]]; then
    args+=(--adapter "$ADAPTER_PATH")
  elif [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    args+=(--checkpoint-dir "$CHECKPOINT_DIR")
  fi
  [[ "${INFER_SHOW_EXPECTED:-0}" == "1" ]] && args+=(--show-expected)
  if [[ -n "${INFER_QUESTION:-}" ]]; then
    args+=(--question "$INFER_QUESTION")
  fi
  python3 "$PEFT_DIR/infer.py" "${args[@]}"
}

print_environment

case "$MODE" in
  check)
    check_python_stack
    ;;
  prefetch)
    prefetch_model
    ;;
  dataset-smoke)
    require_dataset
    python3 "$PEFT_DIR/smoke_test_dataset.py" \
      --base "$MODEL_ID" \
      --path "$DATA_PATH" \
      --max-length "${MAX_LENGTH:-2048}" \
      --num-examples "${NUM_EXAMPLES:-3}"
    ;;
  smoke|train)
    run_training
    ;;
  infer)
    run_infer
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Valid modes: check, prefetch, dataset-smoke, smoke, train, infer" >&2
    exit 2
    ;;
esac
