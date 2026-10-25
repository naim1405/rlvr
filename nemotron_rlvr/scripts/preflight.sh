#!/usr/bin/env bash
# Pre-flight checks before launching train.py (single node).
#
#   bash scripts/preflight.sh          # report only
#   bash scripts/preflight.sh --kill   # also stop leftover Ray / Gym / uv processes
#
# Why this exists: an interrupted run (Ctrl-C, kill, lost SSH session) leaves
# Ray workers holding GPU memory, orphaned Gym servers (bash -> python app.py),
# and sometimes an orphaned `uv pip install` holding uv's cache lock. The next
# launch then hangs at "3 / 4 servers ready" or OOMs in vLLM with no obvious
# error. This script surfaces those states and, with --kill, clears them.

set -u

KILL=0
for arg in "$@"; do
  case "$arg" in
    --kill) KILL=1 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GYM_ROOT="${NEMO_GYM_ROOT:-}"
if [[ -z "$GYM_ROOT" ]]; then
  for c in /opt/nemo-rl/3rdparty/Gym-workspace/Gym /opt/nemo-rl/3rdparty/Gym /opt/Gym; do
    [[ -d "$c/resources_servers" ]] && GYM_ROOT="$c" && break
  done
fi
# NeMo RL passes NEMO_GYM_VENV_DIR to Gym as uv_venv_dir; Gym's own default is the Gym root.
VENV_ROOT="${NEMO_GYM_VENV_DIR:-${GYM_ROOT:-/nonexistent}}"
VERIFIER_VENV="$VENV_ROOT/resources_servers/nemotron_verifier/.venv"
UV_CACHE="${UV_CACHE_DIR:-$HOME/.cache/uv}"

ok()   { printf '  \033[32m[ok]\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }

PROBLEMS=0

echo "== 1. Leftover processes =="
LEFTOVER_PATTERN='raylet|gcs_server|Gym/resources_servers|Gym/responses_api|uv pip install|uv venv|RayWorkerWrapper|MegatronPolicyWorker|VllmAsyncGenerationWorker|VllmGenerationWorker'
# Exclude ourselves and the grep/pgrep machinery.
mapfile -t LEFTOVERS < <(pgrep -af -- "$LEFTOVER_PATTERN" | grep -v -E "preflight.sh|pgrep" || true)
# python app.py children run with cwd inside the Gym root; find them via /proc.
if [[ -n "$GYM_ROOT" ]]; then
  for pid in $(pgrep -f -- '(^|/)python[0-9.]* app\.py$' || true); do
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
    case "$cwd" in
      "$GYM_ROOT"/*) LEFTOVERS+=("$pid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) (cwd $cwd)") ;;
    esac
  done
fi
if ((${#LEFTOVERS[@]} == 0)); then
  ok "no Ray / Gym / uv processes running"
else
  bad "${#LEFTOVERS[@]} leftover process(es):"
  printf '         %s\n' "${LEFTOVERS[@]}" | cut -c1-160
  if ((KILL)); then
    echo "  -> stopping Ray"
    if command -v ray >/dev/null 2>&1; then ray stop --force >/dev/null 2>&1 || true
    else python3 -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true; fi
    echo "  -> killing Gym servers and uv installs"
    for line in "${LEFTOVERS[@]}"; do kill "${line%% *}" 2>/dev/null || true; done
    sleep 3
    for line in "${LEFTOVERS[@]}"; do kill -9 "${line%% *}" 2>/dev/null || true; done
    ok "sent SIGTERM/SIGKILL; re-run without --kill to confirm"
  else
    PROBLEMS=1
    echo "         re-run with --kill to clean these up"
  fi
fi

echo "== 2. GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_APPS="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -z "$GPU_APPS" ]]; then
    ok "no compute processes on the GPU"
  else
    bad "GPU still in use (vLLM needs the whole card at colocated gpu_memory_utilization):"
    printf '         pid,used_memory: %s\n' $GPU_APPS
    PROBLEMS=1
  fi
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/         memory used,total: /'
else
  warn "nvidia-smi not found"
fi

echo "== 3. Disk (need roughly 40 GB: 5 checkpoints incl. optimizer state) =="
CKPT_DIR="$HERE/checkpoints"; mkdir -p "$CKPT_DIR" 2>/dev/null || true
for d in "$CKPT_DIR" /tmp "$UV_CACHE" "$VENV_ROOT"; do
  [[ -e "$d" ]] || continue
  avail_kb="$(df -Pk "$d" | awk 'NR==2{print $4}')"
  avail_gb="$(awk -v kb="$avail_kb" 'BEGIN{printf "%.1f", kb/1048576}')"
  if [[ "$d" == /tmp ]]; then
    # Ray's session dir (object store spill, logs) lives here; it filled up in an earlier run.
    if awk -v g="$avail_gb" 'BEGIN{exit !(g < 5)}'; then bad "$d: ${avail_gb} GB free (Ray spills here; rm -rf /tmp/ray/session_*)"; PROBLEMS=1
    else ok "$d: ${avail_gb} GB free"; fi
  elif awk -v g="$avail_gb" 'BEGIN{exit !(g < 15)}'; then bad "$d: ${avail_gb} GB free"; PROBLEMS=1
  elif awk -v g="$avail_gb" 'BEGIN{exit !(g < 40)}'; then warn "$d: ${avail_gb} GB free (tight; old step_* dirs are safe to delete)"
  else ok "$d: ${avail_gb} GB free"; fi
done

echo "== 4. Gym verifier venv =="
if [[ -z "$GYM_ROOT" ]]; then
  bad "cannot find the Gym checkout; set NEMO_GYM_ROOT"; PROBLEMS=1
elif [[ ! -x "$VERIFIER_VENV/bin/python" || ! -f "$VERIFIER_VENV/bin/activate" ]]; then
  warn "$VERIFIER_VENV missing -> Gym builds it on first spin-up (several minutes of '(nemotron_verifier)' uv output; this is normal)"
else
  SERVER_DIR="$GYM_ROOT/resources_servers/nemotron_verifier"
  [[ -f "$SERVER_DIR/app.py" ]] || SERVER_DIR="$HERE/resources_servers/nemotron_verifier"
  ERR="$(mktemp)"
  (cd "$SERVER_DIR" && timeout 120 "$VERIFIER_VENV/bin/python" -c "import app, sympy, z3" >/dev/null 2>"$ERR")
  rc=$?
  if ((rc == 0)); then
    ok "$VERIFIER_VENV imports app + sympy + z3"
  elif ((rc == 124)); then
    bad "importing app.py in the verifier venv HUNG for 120 s (a stale Ray/uv lock, or a full disk) - this matches '3 / 4 servers ready' forever"
    PROBLEMS=1
  else
    bad "verifier venv is broken: $(grep -E 'Error|error' "$ERR" | tail -1)"
    echo "         fix: rm -rf $VERIFIER_VENV   # Gym rebuilds it on the next launch (several minutes)"
    PROBLEMS=1
  fi
  rm -f "$ERR"
fi

echo
if ((PROBLEMS)); then
  echo "Fix the [FAIL] items above before launching."
  exit 1
fi
echo "Ready. Launch inside tmux so the run survives a dropped SSH session:"
echo "  tmux new -s rlvr"
echo "  cd $HERE && python3 train.py --config configs/single_gpu_overnight.yaml 2>&1 | tee logs/overnight.log"
echo "Gym prints '3 / 4 servers ready' while the verifier venv builds; wait up to ~10 min the first time."
