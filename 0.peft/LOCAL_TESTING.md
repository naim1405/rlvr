# Testing the PEFT step on your own machine

**Goal:** prove, on hardware you control, that the exact code the cluster will
run — same container image, same `run_in_container.sh`, same rendered recipe,
same dataset — trains a LoRA adapter and that the adapter emits
`<think>...</think><answer>...</answer>`. Only the *launcher* differs between
your machine and the cluster (Docker/Apptainer directly here, PBS + Singularity
there), so a green run here de-risks almost everything except PBS itself.

Use the small model (**Qwen3-0.6B**, the repo default) for local testing. Its
job is to validate the pipeline, not to be the final model.

---

## 0. Choose your local container runtime

| Your machine | Use | Notes |
|---|---|---|
| Linux + NVIDIA GPU + Docker | **Docker** (`hpc/run_local_docker.sh`) | Easiest. Also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so `--gpus all` works. |
| Linux (or WSL2 distro) + Apptainer/Singularity | **Singularity** (`hpc/run_local_singularity.sh`) | Closest to the cluster: same `.sif`, same `--nv`, same bind mounts. |
| Windows + Docker Desktop | **Docker, via WSL2** | Clone the repo *inside the WSL filesystem* (e.g. `~/rlvr`), **not** under `/mnt/c/...` — file I/O through the Windows mount is slow and line-ending/permission quirks appear. Run all commands from a WSL shell. |
| macOS / no NVIDIA GPU | Unit tests only | Test level 0 below works; training needs an NVIDIA GPU (that's what the cluster is for). |

You need roughly **30 GB free disk** (container image + model cache +
checkpoints) and, for the smoke/full runs at their default settings, a GPU
with **≥16 GB VRAM** (the 0.6B recipe peaks around 10–13 GB; see
`Tuning.md`). A 4 GB laptop GPU can still run a scaled-down pipeline-only
smoke test — see §5.1.

All commands below are run **from the repo root** unless stated otherwise.

## 1. Get the code and the dataset

```bash
git clone https://github.com/naim1405/rlvr.git
cd rlvr
git switch feat/singularity-peft-port

cd 0.peft/dataset && unzip -o sft_data.zip && cd ../..

wc -l 0.peft/dataset/train.jsonl     # expect: 35000
```

The zip is the versioned artifact; `train.jsonl` itself is git-ignored, so
this extraction step is required on every fresh clone (local and cluster).

## 2. Test level 0 — recipe renderer (no GPU, no container, 1 minute)

`render_recipe.py` turns the template `recipe.yaml` into the concrete file a
run will use (model id, absolute container paths, batch sizes). It is the one
new piece of logic in this port, so it has unit tests:

```bash
python3 0.peft/test_render_recipe.py
# .......  Ran 7 tests ... OK        (needs only PyYAML: pip install pyyaml)

# The optional HF uploader is tested the same way (fake HF client: no
# network, no real token, huggingface_hub not even required):
python3 0.peft/test_upload_to_huggingface.py
# ......................  Ran 22 tests ... OK

# Eyeball a rendered recipe yourself:
python3 0.peft/render_recipe.py --model Qwen/Qwen3-8B \
    --out /tmp/recipe-8b.yaml && sed -n '1,60p' /tmp/recipe-8b.yaml
python3 0.peft/render_recipe.py --model Qwen/Qwen3-0.6B --smoke \
    --out /tmp/recipe-smoke.yaml
```

Check that `pretrained_model_name_or_path`, the `/workspace/rlvr/0.peft` paths
and `local_batch_size` (1 for 8B, 4 for 0.6B) look right. Also syntax-check
the shell scripts you are about to trust:

```bash
for f in 0.peft/run_in_container.sh 0.peft/hpc/*.sh 0.peft/hpc/*.pbs; do bash -n "$f"; done && echo "shell syntax OK"
```

## 3. Test level 1 — container, GPU and stack sanity (first run downloads the image)

Docker path:

```bash
bash 0.peft/hpc/run_local_docker.sh check
```

Singularity path (builds `Containers/nemo-automodel_26.04.00.sif` first if
missing — this is the same converter script you'll use for the cluster):

```bash
bash 0.peft/hpc/build_sif.sh
bash 0.peft/hpc/run_local_singularity.sh check
```

**Pass criteria:** the log shows your GPU (`nvidia-smi -L`),
`torch.cuda.is_available(): True` with the device name, and the `automodel`
executable path. If CUDA is unavailable *inside* the container but `nvidia-smi`
works on the host, the problem is the container GPU plumbing (Docker:
Container Toolkit / `--gpus all`; Singularity: `--nv` and a new-enough host
driver), not your code.

## 4. Test level 2 — dataset smoke (tokenizer + masking, no training)

```bash
bash 0.peft/hpc/run_local_docker.sh dataset-smoke        # or run_local_singularity.sh
```

This downloads the Qwen3-0.6B tokenizer into `.cache/huggingface/` (inside the
repo, so it's shared by every later run), loads `train.jsonl` through
`mydataset.py`, and prints 3 examples with token statistics.

**Pass criteria:** `Loaded 34300 samples for split='train'` (35,000 minus the
2% validation split), prompt/completion previews look like your data, and
`supervised_tokens` is **> 0 and < input_ids length for ordinary rows** — the
prompt is masked with `-100`, the assistant tokens are not. One exception is
expected and is *not* a failure: a row whose prompt alone exceeds
`max_length` is truncated before any assistant token survives, so it shows
`input_ids == max_length` and **`supervised_tokens: 0`** (e.g. a long P4
planning prompt at length 2048). Such rows contribute no loss; they are a
known cost of the chosen `max_length` (see the truncation table in
`Tuning.md`), and lowering `max_length` makes them more frequent. This test
is what catches "the chat template changed under us" when you switch models
later — what matters is that *some* examples show proper masking.

## 5. Test level 3 — training smoke (the real pipeline, 512 rows)

```bash
bash 0.peft/hpc/run_local_docker.sh smoke                # or run_local_singularity.sh
```

What this does, in order: renders `0.peft/runs/recipes/smoke-local.yaml`
(512 train rows, 1 epoch, `max_length` 1024, global batch 16, checkpoint +
validation every 10 steps), runs `automodel` on it, and writes adapters under
`0.peft/checkpoints-smoke/`. Expect minutes, not hours.

**Pass criteria:**

- Training starts, loss prints and is finite (it need not converge — 512 rows
  is about plumbing, not quality).
- `find 0.peft/checkpoints-smoke -maxdepth 2 -type d` shows at least one
  `epoch_*/model/` directory.
- `0.peft/runs/recipes/smoke-local.yaml` matches what you intended (keep this
  habit: the rendered file *is* the config that ran).

### 5.1 The 4 GB laptop variant (e.g. RTX 3050 Laptop)

The default smoke preset (micro-batch 4 × 1024 tokens) is sized for a
~20 GB card and OOMs immediately on 4 GB — roughly 1.2 GB of weights plus
activations/logits for 4,096 token positions do not fit. Scale the
*micro-batch token count* down ~10× instead. This verifies the identical
pipeline (same renderer, dataset, container) — just don't read anything
into the loss, and don't attempt the full run on this GPU:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
USE_TRITON=false \
LOCAL_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_LENGTH=384 \
NUM_SAMPLES_LIMIT=128 VAL_NUM_SAMPLES_LIMIT=32 \
GROUP_BY_LENGTH=false \
bash 0.peft/hpc/run_local_singularity.sh smoke
```

Why each knob: `LOCAL_BATCH_SIZE=1` + `MAX_LENGTH=384` cut per-step token
positions 4×1024 → 1×≤384 (peak activation/logit memory scales with
this); `GLOBAL_BATCH_SIZE=4` keeps the run to 32 optimizer steps over the
128-row subset (accumulation does not affect peak memory);
`USE_TRITON=false` drops the Triton LoRA kernel workspace;
`GROUP_BY_LENGTH=false` avoids length-sorted peaks;
`expandable_segments` reduces fragmentation. Close other GPU apps first —
a laptop GPU shares its 4 GB with your desktop. If this fits, retry
`MAX_LENGTH=512`; if it still OOMs, go to 256 before touching anything
else. Expect more `supervised_tokens: 0` rows at these lengths (prompt
truncation) — the run is a plumbing test only. The same settings work via
`run_local_docker.sh`.

## 6. Test level 4 — does the smoke adapter produce the format?

```bash
CHECKPOINT_DIR=/workspace/rlvr/0.peft/checkpoints-smoke INFER_SHOW_EXPECTED=1 \
  bash 0.peft/hpc/run_local_docker.sh infer
```

(`CHECKPOINT_DIR` here is a path **inside** the container — the repo is always
mounted at `/workspace/rlvr`.) Or directly, in any environment with the
dependencies installed:

```bash
cd 0.peft
python3 infer.py --checkpoint-dir checkpoints-smoke --num-samples 10 --show-expected
python3 infer.py --checkpoint-dir checkpoints-smoke --question "What is 12*13?"
```

**Pass criteria:** samples end with a well-formed
`<think>...</think><answer>...</answer>` and the summary shows PASSes. A smoke
adapter trained on 512 rows may still fail some — if *all* fail or the output
is empty/garbage, something structural is wrong (wrong base model for the
adapter, chat-template mismatch, truncated generation — the FAIL detail in the
output says which).

## 7. Optional — full local training

If your card has ~20 GB, the complete 0.6B run (35k rows, 2 epochs, ≈2,140
steps) fits — it was originally tuned on a 20 GB card:

```bash
bash 0.peft/hpc/run_local_docker.sh train
# RAM-fragmentation safety on small cards (see Tuning.md):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash 0.peft/hpc/run_local_docker.sh train
```

Output: `0.peft/checkpoints/epoch_*/model/`. Validate with level 4 pointed at
`checkpoints` (that's `infer.py`'s default search path, so no
`CHECKPOINT_DIR` is needed). This full local run is a good baseline, but it is
**not required** before using the cluster — levels 0–4 are the pre-flight.

If you want a Hugging Face copy of a local adapter, rehearse the (separate,
manual) upload with `--dry-run` first — it validates the repo URL/id and the
adapter directory without a token or network; the real upload workflow is in
[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) §7:

```bash
python3 0.peft/upload_to_huggingface.py \
  --repo-url https://huggingface.co/<you>/<private-adapter-repo> \
  --upload-folder 0.peft/checkpoints/<epoch_step>/model \
  --require-adapter --dry-run
```

## 8. Trying a different model size locally

The model is one environment variable everywhere:

```bash
MODEL_ID=Qwen/Qwen3-1.7B bash 0.peft/hpc/run_local_docker.sh smoke
```

`render_recipe.py` infers the size from the model id and lowers
`local_batch_size` (micro-batch) for larger models, keeping the effective
global batch at 32 through gradient accumulation. Locally, only try sizes your
VRAM can hold (0.6B/1.7B on 16–24 GB; 4B in bf16 is ~8 GB of weights *before*
activations). The size-by-size table, LR notes, and the 32B caveat live in
[CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md) — they apply here too, with your
card's VRAM instead of 80 GB.

Remember the golden rule when you *do* change models: **everything downstream
must name the same base model** — `infer.py --base` (or `$MODEL_ID`), the merge
script's `--base`, and eventually the RLVR configs. `infer.py` now tries to
read the base back from `adapter_config.json`, but explicit is better.

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cuda available: False` in `check` | GPU not exposed to the container | Docker: install NVIDIA Container Toolkit, restart Docker. Singularity: you're already getting `--nv` from the launcher; check host driver ≥ container's CUDA requirement. |
| Container can't write checkpoints / HF cache | Writing outside the bind mount (image is read-only) | Keep everything under the repo (mounted at `/workspace/rlvr`). The launchers already do; don't override `CHECKPOINT_DIR`/`HF_HOME` to non-mounted paths. |
| `Dataset path does not exist` / `ERROR: dataset not found` | `train.jsonl` not extracted, or wrong mount | `cd 0.peft/dataset && unzip -o sft_data.zip`; always launch via the `hpc/` scripts so the mount is `/workspace/rlvr`. |
| `_target_` import error for `mydataset.py` | Recipe rendered for a different container path | Re-run through `run_in_container.sh` (it renders with the correct `--container-peft-dir`); inspect `0.peft/runs/recipes/*.yaml`. |
| Download stalls / 401 from Hugging Face | Network, or a gated custom model | Qwen3 models are public; for gated models set `HF_TOKEN=...` in the environment (the launchers forward it — for downloads only; uploading is the separate `upload_to_huggingface.py` step). After one online run, `HF_HUB_OFFLINE=1` proves the cache is complete. |
| CUDA OOM at step 0 | Micro-batch × sequence too big | `LOCAL_BATCH_SIZE=2` (or 1), then `MAX_LENGTH=1024`. Keep `GLOBAL_BATCH_SIZE` divisible by the local size. |
| CUDA OOM mid-epoch, fine at the start | A long `group_by_length` batch | `GROUP_BY_LENGTH=false` (smooths peaks, slightly slower) — see `Tuning.md`. |
| Everything passes but infer is all FAIL | Base/adapter mismatch or truncation | Confirm the `Base:` line printed by `infer.py` equals the training model; raise `INFER_MAX_NEW_TOKENS`. |
| Very slow data loading on Windows | Repo under `/mnt/c` | Move the clone into the WSL home directory. |

## 10. What "ready for the cluster" looks like

- [ ] Level 0 unit tests green; you have read one rendered recipe end-to-end.
- [ ] `check` shows the GPU inside the container.
- [ ] `dataset-smoke` shows 34,300 rows and `0 < supervised_tokens < len`.
- [ ] `smoke` wrote `checkpoints-smoke/epoch_*/model/`.
- [ ] `infer` on the smoke adapter produces well-formed `<think>/<answer>` output.
- [ ] (Singularity track) you have a built `.sif` — you can copy this file to
      the cluster and skip building it there.

Then head to [CLUSTER_RUNBOOK.md](CLUSTER_RUNBOOK.md).
