# 0.peft — LoRA SFT for `<think>/<answer>` format

Stage 1 of the pipeline: train a LoRA adapter (NeMo AutoModel) on 35k rows so
the base Qwen model reliably emits `<think>...</think><answer>...</answer>`.
The resulting adapter (`checkpoints/epoch_*/model/`) is later merged
(`nemotron_rlvr/scripts/merge_peft.py`) and used as the starting checkpoint
for GRPO/RLVR.

## 📖 Start with the two runbooks

- **[LOCAL_TESTING.md](LOCAL_TESTING.md)** — test the whole PEFT step on your
  own machine first (Docker, or Apptainer/Singularity if you have it).
- **[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md)** — run it on the Griffith
  cluster (Singularity + PBS, `gpuq2`, 1× A100 80 GB).

Both go through the same entrypoint, `run_in_container.sh`, and both render
the actual training recipe with `render_recipe.py`, so a local pass means the
cluster job is exercising the same code path.

## Quick reference (details in the runbooks)

```bash
# 1. data (host, once)
cd 0.peft/dataset && unzip -o sft_data.zip        # -> train.jsonl, 35k rows

# 2. local smoke test with Docker (from repo root)
bash 0.peft/hpc/run_local_docker.sh dataset-smoke
bash 0.peft/hpc/run_local_docker.sh smoke

# 3. cluster (from repo root, after building the .sif)
bash 0.peft/hpc/build_sif.sh
qsub 0.peft/hpc/pbs_smoke.pbs
qsub 0.peft/hpc/pbs_train.pbs
qsub -v MODEL_ID=Qwen/Qwen3-0.6B 0.peft/hpc/pbs_infer.pbs
```

**Changing the base model** is `MODEL_ID=Qwen/Qwen3-4B` (etc.) in front of the
same commands — see "Using a bigger Qwen model" in `CLUSTER_RUNBOOK.md` for
what else changes with size (micro-batch, and the limits at 32B).

## How the pieces fit

| File | Role |
|---|---|
| `recipe.yaml` | Template training config (defaults = Qwen3-0.6B). Don't hand-edit per run. |
| `render_recipe.py` | Renders a per-run recipe (model, container paths, batch sizes, smoke preset) into `runs/recipes/` and validates it. |
| `run_in_container.sh` | The one entrypoint *inside* the container: `check`, `prefetch`, `dataset-smoke`, `smoke`, `train`, `infer`. |
| `mydataset.py` | Chat-format SFT dataset; loss on assistant tokens only. |
| `smoke_test_dataset.py` | Dataloader/tokenizer check (no training). |
| `infer.py` | Checkpoint tester: generation + format PASS/FAIL. Base model comes from `--base`/`$MODEL_ID`/`adapter_config.json`. |
| `test_render_recipe.py` | CPU-only unit tests for the renderer (run anywhere with PyYAML). |
| `hpc/build_sif.sh` | Docker image → `.sif` conversion (login/transfer node or local Linux). |
| `hpc/run_local_docker.sh` | Local Docker launcher. |
| `hpc/run_local_singularity.sh` | Local Apptainer/Singularity launcher (exact cluster layout). |
| `hpc/pbs_run_singularity.sh` | Host-side launcher shared by the PBS scripts. |
| `hpc/pbs_smoke.pbs` / `pbs_train.pbs` / `pbs_infer.pbs` | Griffith `gpuq2` batch jobs. |
| `dataset/sft_data.zip` | `train.jsonl`, 35k rows (the extracted `.jsonl` is git-ignored). |
| `Tuning.md` | [Parameter tuning guide](Tuning.md): how VRAM scales, per-knob advice. |

## Legacy manual Docker flow (kept for reference)

The launchers above automate this; it still works by hand. Note the repo is
mounted at `/workspace/rlvr` (previously `/tmp/rlvr`):

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.04.00
docker run --gpus all -it --rm --shm-size=8g \
  -v "$(pwd):/workspace/rlvr" nvcr.io/nvidia/nemo-automodel:26.04.00
# inside:
cd /workspace/rlvr/0.peft && automodel recipe.yaml
python infer.py --num-samples 10 --show-expected
python infer.py --question "What is 12*13?"
```
