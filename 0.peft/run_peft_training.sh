#!/usr/bin/env bash
# =============================================================================
# Scheduler-agnostic cluster entrypoint for the PEFT stage.
#
# Give this one script to whoever writes the PBS job. It contains NO PBS
# directives and submits nothing: it assumes it is already running on an
# allocated GPU compute node, finds Singularity/Apptainer, starts the NeMo
# AutoModel container with --nv, and runs the PEFT pipeline inside it.
#
# A PBS job body only needs something like:
#
#   export MODEL_ID=Qwen/Qwen3-0.6B
#   export SIF_PATH=/scratch/$USER/rlvr/Containers/nemo-automodel_26.04.00.sif
#   bash /scratch/$USER/rlvr/0.peft/run_peft_training.sh smoke
#
# Modes (run in this order):
#   check           GPU/framework sanity only
#   prefetch        download $MODEL_ID into the shared HF cache
#   dataset-smoke   tokenizer + dataloader check, no training
#   smoke           512-row / 1-epoch training smoke test   <- first real job
#   train           full PEFT training
#   infer           format-check the newest trained adapter
#
# Optional environment (see CLUSTER_HANDOFF.md for the full list):
#   MODEL_ID, SIF_PATH, RUN_NAME, CHECKPOINT_DIR, LR, LOCAL_BATCH_SIZE,
#   GLOBAL_BATCH_SIZE, MAX_LENGTH, NUM_EPOCHS, GROUP_BY_LENGTH,
#   HF_HUB_OFFLINE, ADAPTER_PATH, INFER_SHOW_EXPECTED, LOG_DIR
#
# Output is mirrored to 0.peft/logs/<mode>-<timestamp>-<job>.log and the
# script exits with the container command's exit code.
# =============================================================================
set -uo pipefail

usage() {
  sed -n '3,42p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

MODE="${1:-}"
case "$MODE" in
  check|prefetch|dataset-smoke|smoke|train|infer) ;;
  -h|--help|help) usage; exit 0 ;;
  *)
    echo "ERROR: a mode is required: check|prefetch|dataset-smoke|smoke|train|infer" >&2
    echo "Run with --help for details. Start with: smoke" >&2
    exit 2
    ;;
esac

PEFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_LAUNCHER="$PEFT_DIR/hpc/pbs_run_singularity.sh"

if [[ ! -f "$HOST_LAUNCHER" ]]; then
  echo "ERROR: expected host launcher not found: $HOST_LAUNCHER" >&2
  echo "Run this script from a complete checkout of the rlvr repository." >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-$PEFT_DIR/logs}"
mkdir -p "$LOG_DIR"

stamp="$(date +%Y%m%dT%H%M%S%z 2>/dev/null || date +%s)"
job_tag="${RUN_NAME:-${PBS_JOBID:-manual}}"
job_tag="$(printf '%s' "$job_tag" | tr -cs 'A-Za-z0-9._-' '_')"
log_file="$LOG_DIR/${MODE}-${stamp}-${job_tag}.log"

{
  echo "[run_peft_training] mode=$MODE"
  echo "[run_peft_training] script=$PEFT_DIR/run_peft_training.sh"
  echo "[run_peft_training] log_file=$log_file"
  echo "[run_peft_training] MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}"
  echo "[run_peft_training] SIF_PATH=${SIF_PATH:-<repo default: ../Containers/nemo-automodel_26.04.00.sif>}"
} | tee -a "$log_file"

bash "$HOST_LAUNCHER" "$MODE" 2>&1 | tee -a "$log_file"
status=${PIPESTATUS[0]}

echo "[run_peft_training] finished mode=$MODE exit_code=$status log_file=$log_file" | tee -a "$log_file"
exit "$status"
