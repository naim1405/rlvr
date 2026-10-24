#!/usr/bin/env bash
# =============================================================================
# Run the PEFT container entrypoint on YOUR machine with Apptainer/Singularity.
# This exercises the exact .sif + bind-mount layout used by the PBS scripts.
#
#   bash 0.peft/hpc/run_local_singularity.sh check
#   bash 0.peft/hpc/run_local_singularity.sh dataset-smoke
#   bash 0.peft/hpc/run_local_singularity.sh smoke
#   MODEL_ID=Qwen/Qwen3-0.6B bash 0.peft/hpc/run_local_singularity.sh train
#
# Environment: SIF_PATH, MODEL_ID, RUN_NAME, CHECKPOINT_DIR (container path),
# plus every training knob understood by run_in_container.sh.
# =============================================================================
set -euo pipefail

MODE="${1:-smoke}"
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEFT_HOST_DIR="$(cd "$HPC_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PEFT_HOST_DIR/.." && pwd)"
SIF_PATH="${SIF_PATH:-$REPO_ROOT/Containers/nemo-automodel_26.04.00.sif}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"

BIN="${CONTAINER_BIN:-}"
if [[ -z "$BIN" ]]; then
  BIN="$(command -v singularity || command -v apptainer || true)"
fi
if [[ -z "$BIN" ]]; then
  echo "ERROR: singularity/apptainer not found. Install Apptainer, or use" >&2
  echo "       bash 0.peft/hpc/run_local_docker.sh $MODE" >&2
  exit 1
fi

if [[ ! -s "$SIF_PATH" ]]; then
  echo "SIF not found: $SIF_PATH"
  echo "Building it with hpc/build_sif.sh ..."
  SIF_PATH="$SIF_PATH" bash "$HPC_DIR/build_sif.sh"
fi

if [[ ! -f "$PEFT_HOST_DIR/dataset/train.jsonl" ]]; then
  echo "ERROR: $PEFT_HOST_DIR/dataset/train.jsonl is missing." >&2
  echo "Extract it first:  cd 0.peft/dataset && unzip -o sft_data.zip" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/.cache/huggingface"

# Values that must be visible *inside* the container. Both prefixes are set
# for SingularityCE/Apptainer compatibility. Container-side paths only.
export SINGULARITYENV_MODEL_ID="$MODEL_ID"
export APPTAINERENV_MODEL_ID="$MODEL_ID"
export SINGULARITYENV_PEFT_DIR=/workspace/rlvr/0.peft
export APPTAINERENV_PEFT_DIR=/workspace/rlvr/0.peft
export SINGULARITYENV_HF_HOME=/workspace/rlvr/.cache/huggingface
export APPTAINERENV_HF_HOME=/workspace/rlvr/.cache/huggingface
export SINGULARITYENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONUNBUFFERED=1

# Forward optional run configuration without needing --env (better support
# on older Singularity versions).
for var in DATA_PATH CHECKPOINT_DIR RUN_NAME PRESET GLOBAL_BATCH_SIZE \
           LOCAL_BATCH_SIZE NUM_EPOCHS MAX_LENGTH LR CKPT_EVERY_STEPS \
           VAL_EVERY_STEPS NUM_SAMPLES_LIMIT VAL_NUM_SAMPLES_LIMIT \
           GROUP_BY_LENGTH SEED NPROC_PER_NODE ADAPTER_PATH \
           INFER_NUM_SAMPLES INFER_MAX_NEW_TOKENS INFER_SHOW_EXPECTED \
           INFER_QUESTION HF_TOKEN HF_HUB_OFFLINE PYTORCH_CUDA_ALLOC_CONF; do
  if [[ -n "${!var:-}" ]]; then
    export "SINGULARITYENV_${var}=${!var}"
    export "APPTAINERENV_${var}=${!var}"
  fi
done

echo "runtime: $BIN"
echo "sif:     $SIF_PATH"
echo "bind:    $REPO_ROOT -> /workspace/rlvr"
echo "mode:    $MODE    model: $MODEL_ID"

exec "$BIN" exec --nv \
  --bind "$REPO_ROOT:/workspace/rlvr" \
  --pwd /workspace/rlvr/0.peft \
  "$SIF_PATH" \
  bash /workspace/rlvr/0.peft/run_in_container.sh "$MODE"
