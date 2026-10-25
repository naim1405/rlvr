#!/usr/bin/env bash
# =============================================================================
# Convert the NVIDIA NeMo AutoModel Docker image into a Singularity/Apptainer
# .sif file. Run this on a machine that is allowed to do large downloads:
# your cluster login/transfer node, or your own Linux/WSL machine (in which
# case copy the resulting .sif to the cluster afterwards).
#
#   bash 0.peft/hpc/build_sif.sh
#   SIF_PATH=/scratch/$USER/Containers/nemo-automodel.sif bash 0.peft/hpc/build_sif.sh
#   FORCE=1 bash 0.peft/hpc/build_sif.sh        # rebuild if the .sif exists
#
# Environment:
#   IMAGE_URI   default docker://nvcr.io/nvidia/nemo-automodel:26.04.00
#   SIF_PATH    default <repo>/Containers/nemo-automodel_26.04.00.sif
#   CACHE_DIR   default <dir of .sif>/.singularity-cache  (keep off $HOME!)
# =============================================================================
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HPC_DIR/../.." && pwd)"
IMAGE_URI="${IMAGE_URI:-docker://nvcr.io/nvidia/nemo-automodel:26.04.00}"
SIF_PATH="${SIF_PATH:-$REPO_ROOT/Containers/nemo-automodel_26.04.00.sif}"
CACHE_DIR="${CACHE_DIR:-$(dirname "$SIF_PATH")/.singularity-cache}"

# Older Griffith nodes expose Singularity as an environment module; new
# nodes install it system-wide. When this script is run as `bash build_sif.sh`
# the interactive shell's `module` function may not be inherited, so try to
# initialise the module system as a fallback.
load_container_module() {
  if ! type module >/dev/null 2>&1; then
    local init
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
      if [[ -r "$init" ]]; then
        set +u; # module init scripts are not always nounset-clean
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
  echo "ERROR: neither 'singularity' nor 'apptainer' was found, and no module provided one." >&2
  echo "On the cluster:  module avail singularity apptainer" >&2
  echo "On your machine: install Apptainer, or use hpc/run_local_docker.sh instead." >&2
  exit 1
fi

mkdir -p "$(dirname "$SIF_PATH")" "$CACHE_DIR" "${TMPDIR:-/tmp}"

# Both variable names are set so the same script works with SingularityCE
# and with Apptainer (which renamed the variables but keeps compatibility).
export SINGULARITY_CACHEDIR="$CACHE_DIR"
export APPTAINER_CACHEDIR="$CACHE_DIR"

if [[ -s "$SIF_PATH" && "${FORCE:-0}" != "1" ]]; then
  echo "SIF already exists (set FORCE=1 to rebuild): $SIF_PATH"
  ls -lh "$SIF_PATH"
  exit 0
fi

echo "container runtime: $BIN ($("$BIN" --version 2>/dev/null || echo unknown))"
echo "image URI:         $IMAGE_URI"
echo "SIF output:        $SIF_PATH"
echo "pull cache:        $CACHE_DIR"
echo "This is a large download/conversion; make sure the filesystem has room."

"$BIN" pull --force "$SIF_PATH" "$IMAGE_URI"

echo "Built:"
ls -lh "$SIF_PATH"
"$BIN" inspect "$SIF_PATH" >/dev/null && echo "inspect: OK"
