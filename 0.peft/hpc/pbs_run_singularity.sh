#!/usr/bin/env bash
# =============================================================================
# Host-side launcher used by the pbs_*.pbs job scripts on the cluster.
# It finds Singularity/Apptainer, validates the .sif and dataset, exports
# the container environment, and execs run_in_container.sh with --nv.
#
# Normally you do not run this directly; qsub one of:
#   0.peft/hpc/pbs_smoke.pbs   0.peft/hpc/pbs_train.pbs   0.peft/hpc/pbs_infer.pbs
#
# Configuration arrives via qsub -v, for example:
#   qsub -v MODEL_ID=Qwen/Qwen3-8B,SIF_PATH=/scratch/$USER/Containers/nemo.sif \
#        0.peft/hpc/pbs_train.pbs
# =============================================================================
set -euo pipefail

MODE="${1:?usage: pbs_run_singularity.sh <check|prefetch|dataset-smoke|smoke|train|infer>}"

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEFT_HOST_DIR="$(cd "$HPC_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$PEFT_HOST_DIR/.." && pwd)}"
SIF_PATH="${SIF_PATH:-$REPO_ROOT/Containers/nemo-automodel_26.04.00.sif}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B}"

# --- find the container runtime ------------------------------------------------
# Older Griffith nodes: Singularity is a module. New nodes: system install.
# The parent PBS script runs with `bash -l`, but initialise the module
# system explicitly as a fallback in case `module` is not inherited here.
load_container_module() {
  if ! type module >/dev/null 2>&1; then
    local init
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
      if [[ -r "$init" ]]; then
        set +u # module init scripts are not always nounset-clean
        # shellcheck disable=SC1090
        source "$init" 2>/dev/null || true
        set -u
        break
      fi
    done
  fi
  if type module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
  fi
}

BIN="${CONTAINER_BIN:-}"
if [[ -z "$BIN" ]]; then
  BIN="$(command -v singularity || command -v apptainer || true)"
fi
if [[ -z "$BIN" ]]; then
  load_container_module
  BIN="$(command -v singularity || command -v apptainer || true)"
fi
if [[ -z "$BIN" ]]; then
  echo "ERROR: singularity/apptainer not found on host $(hostname)." >&2
  echo "Try: module avail singularity apptainer   (then set CONTAINER_BIN if needed)" >&2
  exit 1
fi

# --- validate inputs *before* burning GPU queue time ---------------------------
if [[ ! -s "$SIF_PATH" ]]; then
  echo "ERROR: SIF image not found: $SIF_PATH" >&2
  echo "Build it first (see CLUSTER_RUNBOOK.md step 3):" >&2
  echo "  SIF_PATH=$SIF_PATH bash 0.peft/hpc/build_sif.sh" >&2
  exit 1
fi
if [[ ! -f "$PEFT_HOST_DIR/dataset/train.jsonl" ]]; then
  echo "ERROR: $PEFT_HOST_DIR/dataset/train.jsonl is missing." >&2
  echo "On the login node:  cd 0.peft/dataset && unzip -o sft_data.zip" >&2
  exit 1
fi
if [[ ! -f "$PEFT_HOST_DIR/run_in_container.sh" ]]; then
  echo "ERROR: $PEFT_HOST_DIR/run_in_container.sh is missing - wrong REPO_ROOT?" >&2
  echo "REPO_ROOT=$REPO_ROOT" >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/.cache/huggingface" "$PEFT_HOST_DIR/logs"

# --- environment for inside the container (container-side paths) ---------------
export SINGULARITYENV_MODEL_ID="$MODEL_ID"
export APPTAINERENV_MODEL_ID="$MODEL_ID"
export SINGULARITYENV_PEFT_DIR=/workspace/rlvr/0.peft
export APPTAINERENV_PEFT_DIR=/workspace/rlvr/0.peft
export SINGULARITYENV_HF_HOME=/workspace/rlvr/.cache/huggingface
export APPTAINERENV_HF_HOME=/workspace/rlvr/.cache/huggingface
export SINGULARITYENV_PYTHONUNBUFFERED=1
export APPTAINERENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_RUN_NAME="${RUN_NAME:-${PBS_JOBID:-cluster}}"
export APPTAINERENV_RUN_NAME="${RUN_NAME:-${PBS_JOBID:-cluster}}"

for var in DATA_PATH CHECKPOINT_DIR PRESET GLOBAL_BATCH_SIZE LOCAL_BATCH_SIZE \
           NUM_EPOCHS MAX_LENGTH LR CKPT_EVERY_STEPS VAL_EVERY_STEPS \
           NUM_SAMPLES_LIMIT VAL_NUM_SAMPLES_LIMIT GROUP_BY_LENGTH USE_TRITON SEED \
           NPROC_PER_NODE ADAPTER_PATH INFER_NUM_SAMPLES INFER_MAX_NEW_TOKENS \
           INFER_SHOW_EXPECTED INFER_QUESTION HF_TOKEN HF_HUB_OFFLINE \
           PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF; do
  if [[ -n "${!var:-}" ]]; then
    export "SINGULARITYENV_${var}=${!var}"
    export "APPTAINERENV_${var}=${!var}"
  fi
done

echo "=================================================================="
echo "PBS host:   $(hostname)"
echo "PBS job:    ${PBS_JOBID:-<not under PBS>}  queue: ${PBS_QUEUE:-?}"
echo "runtime:    $BIN ($("$BIN" --version 2>/dev/null || echo unknown))"
echo "sif:        $SIF_PATH"
echo "bind:       $REPO_ROOT -> /workspace/rlvr"
echo "mode:       $MODE"
echo "model:      $MODEL_ID"
echo "visible GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=================================================================="

exec "$BIN" exec --nv \
  --bind "$REPO_ROOT:/workspace/rlvr" \
  --pwd /workspace/rlvr/0.peft \
  "$SIF_PATH" \
  bash /workspace/rlvr/0.peft/run_in_container.sh "$MODE"
