#!/usr/bin/env bash
# =============================================================================
# Run the PEFT container entrypoint on YOUR machine with Docker.
# This is the easiest local test on Windows/WSL2 or Linux: it validates the
# same run_in_container.sh, rendered recipe, dataset and checkpoint writing
# that the cluster PBS jobs use, without needing Apptainer locally.
#
#   bash 0.peft/hpc/run_local_docker.sh check
#   bash 0.peft/hpc/run_local_docker.sh dataset-smoke
#   bash 0.peft/hpc/run_local_docker.sh smoke
#   MODEL_ID=Qwen/Qwen3-0.6B bash 0.peft/hpc/run_local_docker.sh train
#
# Environment: DOCKER_IMAGE, MODEL_ID, PULL_IMAGE=1, SHM_SIZE, plus every
# training knob understood by run_in_container.sh.
# =============================================================================
set -euo pipefail

MODE="${1:-smoke}"
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEFT_HOST_DIR="$(cd "$HPC_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PEFT_HOST_DIR/.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/nemo-automodel:26.04.00}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker, or use run_local_singularity.sh." >&2
  exit 1
fi

if [[ ! -f "$PEFT_HOST_DIR/dataset/train.jsonl" ]]; then
  echo "ERROR: $PEFT_HOST_DIR/dataset/train.jsonl is missing." >&2
  echo "Extract it first:  cd 0.peft/dataset && unzip -o sft_data.zip" >&2
  exit 1
fi

if [[ "${PULL_IMAGE:-0}" == "1" ]] || ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Pulling $DOCKER_IMAGE ..."
  docker pull "$DOCKER_IMAGE"
fi

mkdir -p "$REPO_ROOT/.cache/huggingface"

args=(
  run --rm --gpus all
  --shm-size="${SHM_SIZE:-8g}"
  --volume "$REPO_ROOT:/workspace/rlvr"
  --workdir /workspace/rlvr/0.peft
  --env "MODEL_ID=$MODEL_ID"
  --env PEFT_DIR=/workspace/rlvr/0.peft
  --env HF_HOME=/workspace/rlvr/.cache/huggingface
  --env PYTHONUNBUFFERED=1
)

# Forward optional configuration only when it is set on the host.
for var in DATA_PATH CHECKPOINT_DIR RUN_NAME PRESET GLOBAL_BATCH_SIZE \
           LOCAL_BATCH_SIZE NUM_EPOCHS MAX_LENGTH LR CKPT_EVERY_STEPS \
           VAL_EVERY_STEPS NUM_SAMPLES_LIMIT VAL_NUM_SAMPLES_LIMIT \
           GROUP_BY_LENGTH USE_TRITON SEED NPROC_PER_NODE ADAPTER_PATH \
           INFER_NUM_SAMPLES INFER_MAX_NEW_TOKENS INFER_SHOW_EXPECTED \
           INFER_QUESTION HF_TOKEN HF_HUB_OFFLINE \
           PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF; do
  if [[ -n "${!var:-}" ]]; then
    args+=(--env "${var}=${!var}")
  fi
done

# A tty makes interactive debugging nicer and is harmless for batch modes.
if [[ -t 0 && -t 1 ]]; then
  args+=(-it)
fi

args+=("$DOCKER_IMAGE" bash /workspace/rlvr/0.peft/run_in_container.sh "$MODE")

echo "image: $DOCKER_IMAGE"
echo "bind:  $REPO_ROOT -> /workspace/rlvr"
echo "mode:  $MODE    model: $MODEL_ID"
exec docker "${args[@]}"
