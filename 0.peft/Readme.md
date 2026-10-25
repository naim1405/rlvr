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
- **[CLUSTER_HANDOFF.md](CLUSTER_HANDOFF.md)** — if someone else (e.g. your
  Professor) writes the PBS script, hand them this: the one run script to
  call (`run_peft_training.sh`) and the exact resources to request.

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

# 3. cluster: someone else writes the PBS script -> its body only needs:
#      bash <repo>/0.peft/run_peft_training.sh smoke    # then: train, then: infer
#    (see CLUSTER_HANDOFF.md for the resources to request)
#    ...or submit the ready-made examples yourself:
bash 0.peft/hpc/build_sif.sh
qsub 0.peft/hpc/pbs_smoke.pbs
qsub 0.peft/hpc/pbs_train.pbs
qsub -v MODEL_ID=Qwen/Qwen3-0.6B 0.peft/hpc/pbs_infer.pbs

# 4. optional, AFTER training: upload one chosen adapter dir to Hugging Face.
#    This is a separate manual step -- training never uploads anything, and
#    HF_TOKEN alone does not turn uploads on.
export HF_TOKEN=<write-token>          # keep out of git/chat; see CLUSTER_RUNBOOK.md §7
python3 0.peft/upload_to_huggingface.py \
  --repo-url https://huggingface.co/<you>/<private-adapter-repo> \
  --upload-folder 0.peft/checkpoints/<epoch_step>/model \
  --require-adapter --dry-run           # remove --dry-run to actually upload
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
| `run_peft_training.sh` | Scheduler-agnostic cluster entrypoint to hand to whoever writes the PBS script; runs the same modes and logs to `logs/`. |
| `mydataset.py` | Chat-format SFT dataset; loss on assistant tokens only. |
| `smoke_test_dataset.py` | Dataloader/tokenizer check (no training). |
| `infer.py` | Checkpoint tester: generation + format PASS/FAIL. Base model comes from `--base`/`$MODEL_ID`/`adapter_config.json`. |
| `upload_to_huggingface.py` | Optional **separate** post-training uploader: takes a HF write token (`HF_TOKEN`/`--token-file`), a repo URL or id, and one folder (normally `checkpoints/epoch_*/model/`) and uploads it with `huggingface_hub`. Never called by training. |
| `test_render_recipe.py` | CPU-only unit tests for the renderer (run anywhere with PyYAML). |
| `test_upload_to_huggingface.py` | CPU-only unit tests for the uploader (fake HF client; no network or real token). |
| `hpc/build_sif.sh` | Docker image → `.sif` conversion (login/transfer node or local Linux). |
| `hpc/run_local_docker.sh` | Local Docker launcher. |
| `hpc/run_local_singularity.sh` | Local Apptainer/Singularity launcher (exact cluster layout). |
| `hpc/pbs_run_singularity.sh` | Host-side launcher shared by the PBS scripts. |
| `hpc/pbs_smoke.pbs` / `pbs_train.pbs` / `pbs_infer.pbs` | Griffith `gpuq2` batch jobs — submit them yourself, or use as reference examples for whoever writes the PBS script. |
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
