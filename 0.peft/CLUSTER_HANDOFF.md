# Cluster handoff — PEFT training job (for the person writing the PBS script)

This is the one-page version. It says exactly **what to run** and **what
resources to request** for the PEFT (LoRA SFT) stage of the `rlvr` project.
Background and troubleshooting are in
[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md); the PBS files in `hpc/pbs_*.pbs`
are working examples of the requests below, kept for reference.

**Division of labour:** you create the PBS job script(s) and choose the
queue/account details. My code provides a single scheduler-agnostic run
script that the PBS body calls — it has no `#PBS` directives, submits
nothing, needs no root/Docker, and exits non-zero on failure.

---

## 1. The running script

```
<repo>/0.peft/run_peft_training.sh <mode>
```

It locates everything relative to itself, so it can be called by absolute
path from any working directory. It launches the NeMo AutoModel
Singularity image with `--nv`, renders the training config for
`$MODEL_ID`, trains, and mirrors all output to
`<repo>/0.peft/logs/<mode>-<timestamp>-<job>.log`.

Three sequential jobs (one running at a time is fine — they must run in
this order anyway):

| # | Command (PBS body) | Purpose |
|---|---|---|
| 1 | `bash <repo>/0.peft/run_peft_training.sh smoke` | 512-row, 1-epoch end-to-end test: GPU, container, dataset, training, checkpoint write. **Please run this first.** |
| 2 | `bash <repo>/0.peft/run_peft_training.sh train` | Full training: 35k rows, 2 epochs, checkpoint every 500 steps. |
| 3 | `bash <repo>/0.peft/run_peft_training.sh infer` | Loads base model + newest adapter, verifies the `<think>/<answer>` output format. |

PBS body sketch (directives are yours to write; this is only the payload):

```bash
export MODEL_ID=Qwen/Qwen3-0.6B
export SIF_PATH=/path/to/rlvr/Containers/nemo-automodel_26.04.00.sif
bash /path/to/rlvr/0.peft/run_peft_training.sh smoke
```

## 2. Required resources per job

All jobs are **single-node, single-GPU, no MPI**. (The queue note says one
GPU per user at a time; these requests match that.)

| Resource | Job 1 `smoke` | Job 2 `train` (Qwen3-0.6B) | Job 3 `infer` |
|---|---:|---:|---:|
| Nodes | 1 | 1 | 1 |
| GPUs (`ngpus`) | **1 × A100 80GB** | **1 × A100 80GB** | **1 × A100 80GB** |
| CPUs (`ncpus`) | 4 | 8 | 4 |
| Host RAM (`mem`) | 32 GB | 64 GB | 32 GB |
| Walltime requested | 2 h | 24 h (generous; expect less) | 2 h (expect minutes) |
| Queue | `gpuq2` | `gpuq2` | `gpuq2` |
| Open ports / services | none | none | none |

**Disk (shared filesystem, e.g. scratch, writable by the job):** plan for
≈75 GB free where the repo lives — container image ~10–20 GB, pull cache
of similar size during the one-time build, model cache ~2 GB (0.6B),
checkpoints + logs a few GB. Everything persistent is written *inside the
repo tree* (`0.peft/checkpoints*/`, `0.peft/logs/`, `0.peft/runs/`,
`.cache/`), so the repo must sit on storage visible to compute nodes
and be aware of any scratch purge policy.

### If training the bigger model instead of 0.6B

The compute *shape* stays identical (1 GPU); only memory headroom and
walltime change. The script auto-lowers the training micro-batch for
larger models, keeping the effective batch size fixed. Confirm which
model before writing the `train` job — set it via `MODEL_ID`:

| `MODEL_ID` | Host RAM | Suggested walltime* | Model cache | Disk to allow |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` (or 1.7B) | 64 GB | 24 h | ~1–3 GB | ~75 GB |
| `Qwen/Qwen3-4B` | 64 GB | 48 h | ~8 GB | ~90 GB |
| `Qwen/Qwen3-8B` | 64 GB | 48 h | ~16 GB | ~100 GB |
| `Qwen/Qwen3-14B` | 96 GB | up to 100 h | ~28 GB | ~120 GB |
| `Qwen/Qwen3-32B` | — | — | — | Not supported on 1 GPU with this recipe (≈64 GB weights alone); needs a different setup — please don't schedule it as-is. |

*Walltimes for 4B+ are request ceilings, not measurements: the right
number comes from timing the **smoke job run with the same `MODEL_ID`**
and extrapolating to ≈2,140 optimizer steps. Checkpoints every 500 steps
limit the loss if a job hits its walltime.

## 3. One-time setup before the first job (login/transfer node)

```bash
# 1) code, on shared storage:
git clone https://github.com/naim1405/rlvr.git
cd rlvr && git switch feat/singularity-peft-port

# 2) dataset (required; the .jsonl is not in git):
cd 0.peft/dataset && unzip -o sft_data.zip && cd ../..   # -> train.jsonl, 35,000 rows

# 3) container image: Docker -> Singularity (.sif), once:
bash 0.peft/hpc/build_sif.sh
#    default output: Containers/nemo-automodel_26.04.00.sif
#    (source image: nvcr.io/nvidia/nemo-automodel:26.04.00)
```

Environment facts the setup depends on:

- **Singularity or Apptainer available on compute nodes**, with NVIDIA
  support (`singularity exec --nv`). The run script finds either binary,
  or loads a `singularity`/`apptainer` environment module. If the binary
  has an unusual path, set `CONTAINER_BIN=/path/to/it` in the PBS body.
- **Host NVIDIA driver** new enough for the image's CUDA — the smoke job
  prints the driver and fails early and clearly if this is wrong.
- **Internet:** the image pull (step 3) and the first model download need
  it. If compute nodes are offline, that's fine provided the model is
  prefetched once into `<repo>/.cache/huggingface/` from a node with
  internet — `run_peft_training.sh prefetch` does exactly that — and
  later jobs set `HF_HUB_OFFLINE=1`. If registry pulls from the cluster
  are rate-limited, the `.sif` can be built elsewhere and copied in;
  it's a single self-contained file.
- **No secrets needed**: the Qwen3 models are public. (`HF_TOKEN` is only
  required if a gated model is ever substituted.)
- The image filesystem is read-only and the job runs as the submitting
  user — standard Singularity behaviour; nothing to install on nodes.

## 4. Environment variables the PBS body may set

Only the first two are ever likely to be needed:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-0.6B` | Must be **identical** across smoke/train/infer. |
| `SIF_PATH` | `<repo>/Containers/nemo-automodel_26.04.00.sif` | If the image lives elsewhere. |
| `RUN_NAME` | PBS job id | Labels logs and the rendered config. |
| `LR` | recipe default 2e-4 | e.g. `1e-4` for 8B-class models. |
| `CHECKPOINT_DIR` | `.../0.peft/checkpoints` (smoke: `checkpoints-smoke`) | Container path; use a distinct dir per model. |
| `HF_HUB_OFFLINE` | unset | `1` once the model is prefetched. |

Full reference: `CLUSTER_RUNBOOK.md` §9.

## 5. Outputs, and what "success" looks like

| Job | Success looks like | Key outputs |
|---|---|---|
| `smoke` | Exit 0; log shows 1× A100 80GB, `dataset rows: 35000`, ~32 training steps with finite loss | `0.peft/checkpoints-smoke/epoch_*/model/` |
| `train` | Exit 0; ~2,140 steps over 2 epochs; decreasing loss | `0.peft/checkpoints/epoch_*/model/` (the LoRA adapter — this is the deliverable of the stage) |
| `infer` | Per-sample `FORMAT: PASS`, summary `==== SUMMARY: N/N passed format check ====` | console/log only |

Every run also leaves its **exact effective config** at
`0.peft/runs/recipes/<mode>-<job>.yaml` and a full log at
`0.peft/logs/<mode>-*.log` (in addition to the PBS output file). If a job
fails, those two files plus the PBS stdout/stderr are what I need to
diagnose it — please send them back rather than re-running blind.

## 6. Quick answers to likely questions

- **Can the three jobs run in parallel?** No need — they're sequential by
  design, which suits the one-job-at-a-time queue limit.
- **Multi-GPU/multi-node?** Not used. Model-parallel machinery exists in
  the framework but this run is deliberately 1 GPU.
- **What if the `train` job is killed?** Checkpoints every 500 steps are
  on disk under `0.peft/checkpoints/`; tell me the last `epoch_*` present
  and the tail of the log before resubmitting.
- **Why a container at all?** The training stack (NeMo AutoModel) has
  pinned CUDA/framework dependencies; the `.sif` carries them so nothing
  is installed on cluster nodes and my local test used the identical
  environment.
