# Running the PEFT step on the Griffith cluster (Singularity + PBS)

**Audience:** you, on your Professor's university cluster: PBS scheduler, queue
`gpuq2`, nodes with **5× NVIDIA A100 80GB PCIe**, no Docker, Singularity
supported ([Griffith wiki: Singularity](https://griffith.atlassian.net/wiki/spaces/GHCD/pages/4030761/Singularity)).

**What runs where:** you edit and submit from a *login node*; PBS allocates a
*compute node* (e.g. `gn061`) and runs your job script there, unattended. The
job script runs the container with Singularity; the container runs
`run_in_container.sh`. You never train on the login node.

**Already done the local pre-flight?** This runbook assumes
[LOCAL_TESTING.md](LOCAL_TESTING.md) levels 0–3 passed at least once. The
cluster flow deliberately re-uses the same scripts, so most of what can break
has already broken (and been fixed) on your machine.

> **If your Professor/admin writes the PBS script for you** (the current
> arrangement), you don't submit anything yourself: hand them
> **[CLUSTER_HANDOFF.md](CLUSTER_HANDOFF.md)** — it gives them the single
> run script to call (`0.peft/run_peft_training.sh <smoke|train|infer>`)
> and the exact resources to request. This runbook then serves as the
> technical reference behind it, and `hpc/pbs_*.pbs` are ready-made
> examples of those resource requests. The `qsub` commands below are the
> self-service equivalent if you later get submission access.

---

## 1. How the cluster constraints shape this run

From the admins' answers, translated into the choices baked into the
`hpc/pbs_*.pbs` scripts:

| Constraint | Consequence here |
|---|---|
| "Max GPU memory at once: 80 GB" + `max_run_res.ngpus=[u:PBS_GENERIC=1]` | You effectively get **one A100 80GB at a time**, even though nodes have five. All jobs request `ngpus=1` and train single-GPU. For PEFT that's fine — see the model-size table in §8. |
| "Parallel inference + training: not possible" | Not needed for this stage. PEFT is one ordinary training job. (The later GRPO stage will need its colocated single-GPU design — a separate port.) |
| "500 GB RAM max — but your job will never run" | PBS starts a job only when a node can satisfy the *whole* request. Scripts ask for **32 GB** (smoke/infer) / **64 GB** (train), 4–8 CPUs — enough, and schedulable. |
| Walltime max 100 h | Smoke asks 2 h, training asks 24 h. Checkpoints every 500 steps (10 in smoke) bound your loss if a job is killed. |
| No Docker; convert images to Singularity | One-time conversion produces a single `.sif` file (§4). Inside jobs it's `singularity exec --nv ... .sif`. |

Queue limits for reference: `resources_max` = 500 GB RAM, 95 CPUs, 5 GPUs,
100 h walltime; `max_run=[u:PBS_GENERIC=1]` also means **one of your jobs runs
at a time** in this queue — queue the next one and PBS will serialise them.

## 2. One-time orientation (login node, ~10 minutes)

Replace `$USER` paths with what you actually discover here — several later
steps depend on these answers:

```bash
# Which container runtime, if any, is on PATH?
which singularity apptainer; singularity --version 2>/dev/null || apptainer --version 2>/dev/null
module avail 2>&1 | grep -i -E 'singularity|apptainer'    # older nodes use modules

# Where may you put tens of gigabytes? (image + HF cache + checkpoints)
pwd; df -h ~ . ; quota -s 2>/dev/null || true
ls -ld ~/scratch /scratch/$USER /project/* 2>/dev/null    # whatever exists

# Can this node reach the two registries you need?
curl -I https://nvcr.io/v2/ 2>&1 | head -3               # NVIDIA image registry
curl -I https://huggingface.co 2>&1 | head -3            # model download

# What does the queue look like?
qstat -Q gpuq2 2>/dev/null || qstat -Q
```

Decisions to make now (write them down, e.g. in a `cluster.env` shell file):

- **`REPO_ROOT`** — where the clone lives. Must be on a filesystem visible to
  compute nodes, with room for the `.sif` (~10–20 GB), the model cache
  (1–30 GB depending on model) and checkpoints. Scratch/project space is
  usually right; home usually isn't. Check the **purge policy** of scratch —
  if files vanish after N days, copy finished adapters somewhere durable.
- **`SIF_PATH`** — default `$REPO_ROOT/Containers/nemo-automodel_26.04.00.sif`
  (works if the repo is in scratch). Otherwise pick e.g.
  `/scratch/$USER/Containers/nemo-automodel_26.04.00.sif` and pass it to every
  command below.
- **Do computes nodes have internet?** If not, pre-download the model into the
  repo's `.cache/huggingface/` from the login node (§5) and add
  `HF_HUB_OFFLINE=1` to job submissions as a proof.

If your group needs a PBS account/project code, open the `hpc/pbs_*.pbs`
files and activate the commented `##PBS -A ...` / `group_list` lines (remove
one leading `#` and fill in the values) — the queue config shows a default
Qlist, but only the admins/your Professor can tell you your allocation code.

## 3. Get code and data in place

```bash
# in your chosen scratch/project directory:
git clone https://github.com/naim1405/rlvr.git
cd rlvr
git switch feat/singularity-peft-port

cd 0.peft/dataset && unzip -o sft_data.zip && cd ../..
wc -l 0.peft/dataset/train.jsonl          # expect 35000
python3 0.peft/test_render_recipe.py      # renderer sanity (needs PyYAML)
```

If GitHub is unreachable from the cluster, clone locally and
`rsync -az rlvr/ <user>@<cluster>:<scratch>/rlvr/` instead (exclude `.cache/`
and `Containers/` if you already built them locally and transfer them
separately — or deliberately include the `.sif`, see §4).

**Every `qsub` in this runbook is submitted from the repo root** — the job
scripts use `$PBS_O_WORKDIR` as the repo location (override with
`-v REPO_ROOT=/abs/path` if you must submit from elsewhere).

## 4. Build (or transfer) the Singularity image — once

`singularity pull` converts the Docker image directly; no Docker daemon is
involved:

```bash
# from repo root; adjust SIF_PATH if you chose a different home for it:
SIF_PATH=$PWD/Containers/nemo-automodel_26.04.00.sif bash 0.peft/hpc/build_sif.sh
ls -lh Containers/
```

The build script keeps the pull cache next to the `.sif`
(`SINGULARITY_CACHEDIR`/`APPTAINER_CACHEDIR`) instead of filling your home
quota. Three practical warnings from the Griffith wiki:

- Direct pulls of **Docker Hub** images from Griffith hit rate limits; the
  wiki's recommended workaround is "download externally and transfer the
  image in". Your image lives on NVIDIA's registry (`nvcr.io`), which may
  behave better — but if the pull fails or is unusably slow, build the `.sif`
  on your own machine (LOCAL_TESTING.md §3) and copy it up:
  `scp Containers/nemo-automodel_26.04.00.sif <user>@<cluster>:<path>/Containers/`
- A pull/conversion is a big CPU/disk job. If login-node policies forbid that,
  use a data-transfer node or ask the admins where image builds should happen;
  don't just run it and hope.
- If `nvcr.io` asks for credentials during the pull, that's NGC auth:
  `export SINGULARITY_DOCKER_USERNAME='$oauthtoken'` and
  `SINGULARITY_DOCKER_PASSWORD=<your NGC API key>` before building.

Verify the image without a GPU:

```bash
singularity inspect Containers/nemo-automodel_26.04.00.sif | head -20
```

## 5. Pre-download the model (skip only if compute nodes have internet)

The first training run downloads `MODEL_ID` from Hugging Face into
`$REPO_ROOT/.cache/huggingface/` — pass `prefetch` once so that cost (and any
network surprise) doesn't come out of a GPU job. From the login node you can
run the container without `--nv` for this:

```bash
singularity exec --bind "$PWD:/workspace/rlvr" --pwd /workspace/rlvr/0.peft \
  Containers/nemo-automodel_26.04.00.sif \
  bash -c 'HF_HOME=/workspace/rlvr/.cache/huggingface MODEL_ID=Qwen/Qwen3-0.6B \
           bash /workspace/rlvr/0.peft/run_in_container.sh prefetch'
```

Swap in your real `MODEL_ID`. If even the login node can't reach Hugging
Face, download on your machine into a cache and `rsync` the whole `.cache/`
directory up — it's just files.

After a successful prefetch, add `HF_HUB_OFFLINE=1` to the `qsub -v` list in
later steps. If a job then runs, your cache is provably complete; if it fails
fast with a "not found in cache" error, you find out in the smoke job, cheaply.

## 6. Smoke job first — always

```bash
qsub 0.peft/hpc/pbs_smoke.pbs
qstat -u $USER                       # Q = queued, R = running
# when finished, output is in peft-smoke.o<jobid> in the repo root:
less peft-smoke.o*
```

What you're looking for, in order:

1. Header block: compute-node hostname, runtime version, your `.sif` path,
   `CUDA_VISIBLE_DEVICES` set by PBS.
2. `nvidia-smi` inside the container lists one **A100 80GB** — this also
   reveals the **host driver version**; the container's CUDA needs a
   new-enough driver. A driver/library mismatch fails here, clearly.
3. `torch.cuda.is_available(): True`, `dataset rows: 35000`.
4. A rendered recipe at `0.peft/runs/recipes/smoke-<jobid>.yaml` — read it;
   it's the exact config that ran.
5. ~32 training steps (512 rows ÷ global batch 16), finite loss, and a
   checkpoint under `0.peft/checkpoints-smoke/epoch_*/model/`.

Then format-check that smoke adapter (queued behind nothing, 2 h limit but
should take minutes):

```bash
qsub -v CHECKPOINT_DIR=/workspace/rlvr/0.peft/checkpoints-smoke,INFER_SHOW_EXPECTED=1 \
     0.peft/hpc/pbs_infer.pbs
```

Note `CHECKPOINT_DIR` is a **container** path — the repo is always bound at
`/workspace/rlvr`. `MODEL_ID` defaults to `Qwen/Qwen3-0.6B` here; when you
train a different model, the infer job must be told the same id (see §8).

If interactive jobs are allowed, this debug loop is faster interactively —
ask, then e.g.
`qsub -I -q gpuq2 -l select=1:ncpus=4:ngpus=1:mem=32gb,walltime=01:00:00`
and inside it run the `singularity exec --nv ... run_in_container.sh check`
line from `hpc/pbs_run_singularity.sh` by hand.

## 7. Full training job

```bash
qsub 0.peft/hpc/pbs_train.pbs
qstat -u $USER
tail -f peft-train.o*                    # once it starts
```

Defaults: Qwen3-0.6B, all 34,300 train rows, 2 epochs, global batch 32,
`max_length` 2048, checkpoints every 500 steps into `0.peft/checkpoints/`.
When it completes:

1. Confirm the final adapter exists: `ls 0.peft/checkpoints/epoch_*/model/`
   (expect `adapter_config.json` + safetensors weights).
2. Format-check it:
   `qsub -v MODEL_ID=Qwen/Qwen3-0.6B,INFER_SHOW_EXPECTED=1 0.peft/hpc/pbs_infer.pbs`
   and read the PASS/FAIL summary in `peft-infer.o*`.
3. **Copy the adapter (and the job log + rendered recipe) out of scratch** if
   scratch is purged: `rsync -az 0.peft/checkpoints/epoch_1/ <you>@<your-machine>:...`
   The adapter is the input to the merge step
   (`nemotron_rlvr/scripts/merge_peft.py --base Qwen/Qwen3-0.6B --adapter ...`),
   wherever you run that next.

Useful monitoring while it runs: `qstat -f <jobid>` (resources used vs
requested — check host RAM headroom), and your `.o` log for loss/step lines.
You can't `nvidia-smi` a running job from the login node unless the site gives
you node access; the training log is your window.

## 8. Using a bigger Qwen model — what actually changes

**Your intuition is mostly right:** within the same dense text-only Qwen3
family, the *code* doesn't change — same chat template usage in
`mydataset.py`, same `*_proj` LoRA targets (q/k/v/o/gate/up/down projections
exist under those names), same masked-loss logic, same `<think>/<answer>`
contract that the RLVR verifiers later depend on. `render_recipe.py` was
built exactly so the model is one variable:

```bash
qsub -v MODEL_ID=Qwen/Qwen3-4B 0.peft/hpc/pbs_smoke.pbs     # smoke FIRST
qsub -v MODEL_ID=Qwen/Qwen3-4B 0.peft/hpc/pbs_train.pbs
```

But "just change the model" hides four things that **do** change, and the
renderer + this workflow handle (or expose) each of them:

### 8.1 Memory: the micro-batch must shrink (handled automatically)

Activation memory scales with `local_batch_size × max_length × model width`,
and the frozen bf16 weights grow linearly. The renderer infers the size from
the model id and lowers `local_batch_size`; the effective batch
(`global_batch_size: 32`) is preserved by gradient accumulation, so the
optimisation trajectory stays comparable — steps just take longer.

| Model (dense Qwen3) | bf16 weights | Renderer `local_batch_size` (1×80 GB, len 2048) | Accumulation to global 32 | Starting LR to try* | Verdict on 1× A100 80GB |
|---|---:|---:|---:|---:|---|
| 0.6B / 1.7B | ~1–3 GB | 4 | 8 | 2e-4 (template) | Easy; the pipeline-test size |
| 4B | ~8 GB | 2 | 16 | 1e-4 – 2e-4 | Comfortable |
| 8B | ~16 GB | 1 | 32 | 1e-4 | Comfortable; slower per step |
| 14B | ~28 GB | 1 | 32 | 7e-5 – 1e-4 | Workable starting point; smoke test at full length is essential, keep `GROUP_BY_LENGTH=false` in your pocket for mid-epoch OOM on long batches |
| 32B | ~64 GB | 1 (renderer warns) | 32 | — | **Not a drop-in swap.** Weights alone leave almost nothing for activations with this recipe (no activation checkpointing/sharding/quantisation configured, and you only have 1 GPU). Needs separate engineering (multi-GPU FSDP/TP, activation checkpointing or QLoRA-style quantised base, shorter sequences) — out of scope for this port. |

*LR is *not* auto-changed by the renderer — pass `LR=...` yourself
(e.g. `-v MODEL_ID=Qwen/Qwen3-8B,LR=1e-4,RUN_NAME=8b-lr1e4`). Larger adapters
on larger bases usually want a somewhat lower LR than the 0.6B's 2e-4; treat
the table values as starting points and judge by the loss curve + infer
PASS rate, exactly as in `Tuning.md`. Any value in the batch table can be
overridden the same way, e.g. `LOCAL_BATCH_SIZE=2,MAX_LENGTH=1536`.

Also budget **host** resources: a 14B load is heavier — if the job dies during
model load with host OOM, raise `mem=64gb` in `pbs_train.pbs` to `96gb`
(still far from "never runs" territory) rather than touching GPU settings.

### 8.2 The base model name must follow the adapter everywhere

An adapter is meaningless without its exact base. All of these must name the
**same** id you trained with:

- infer jobs: `-v MODEL_ID=Qwen/Qwen3-4B` (`infer.py` also falls back to the
  base recorded in `adapter_config.json`, but don't rely on the fallback);
- the merge step: `merge_peft.py --base Qwen/Qwen3-4B`;
- later, every RLVR config's `policy.model_name` points at that merged model.

Mismatched base/adapter usually fails loudly (shape mismatch) — but a
same-size *different variant* swap can fail silently with garbage output,
which is worse. `RUN_NAME` (e.g. `RUN_NAME=qwen3-4b`) makes rendered recipes
and logs self-describing.

### 8.3 Family boundaries: what is *not* a drop-in swap

The "just change `MODEL_ID`" rule holds for **dense, text-only Qwen3**
checkpoints with the same tokenizer/chat template. It does not automatically
hold for Qwen3-VL (vision tower), Qwen3 MoE variants (`30B-A3B`, `235B` —
expert module names and memory profile differ), Qwen2.5 (different template
behaviour; `enable_thinking` in `infer.py` is a Qwen3-ism), or "base"
(non-instruct) checkpoints whose chat template may differ. Any of those needs
a fresh `dataset-smoke` + template audit before you trust it — the masked-token
statistics and the format PASS rate are your detectors.

### 8.4 Time, not just memory

Global batch is preserved, but each optimizer step processes the same 32
sequences through a much larger network: an 8B epoch is many times slower than
0.6B. The 24 h default walltime is sized with headroom for the small model —
after the smoke job, time 10 steps of your target model, extrapolate to
≈2,140 steps, and set `walltime` in `pbs_train.pbs` accordingly (max 100 h).
Checkpointing every 500 steps means a walltime kill costs at most 500 steps
*if* your next submission resumes from the checkpoint directory — verify the
checkpoint you need is complete before deleting the job's output.

## 9. Configuration reference (all via `qsub -v A=1,B=2`)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `Qwen/Qwen3-0.6B` | HF id or local path of the base model. |
| `SIF_PATH` | `$REPO_ROOT/Containers/nemo-automodel_26.04.00.sif` | Host path of the image. |
| `REPO_ROOT` | submit dir | Host path of this repo (bound to `/workspace/rlvr`). |
| `RUN_NAME` | PBS job id | Names the rendered recipe; make it meaningful. |
| `CHECKPOINT_DIR` | auto (`checkpoints`, smoke: `checkpoints-smoke`) | **Container** path. Use distinct dirs per model, e.g. `/workspace/rlvr/0.peft/checkpoints-4b`. |
| `GLOBAL_BATCH_SIZE`, `LOCAL_BATCH_SIZE` | 32 / preset | Must divide evenly (renderer enforces). |
| `NUM_EPOCHS`, `MAX_LENGTH`, `LR`, `SEED` | 2 / 2048 / 2e-4 / 1111 | Training knobs (see `Tuning.md`). |
| `GROUP_BY_LENGTH` | `true` | Set `false` if OOM strikes mid-epoch. |
| `HF_HUB_OFFLINE` | unset | `1` after prefetch, to prove the cache is complete. |
| `HF_TOKEN` | unset | Only for gated models. Never commit it. |
| `ADAPTER_PATH` | newest in `CHECKPOINT_DIR` | Infer jobs: **container** path to one adapter. |
| `INFER_NUM_SAMPLES`, `INFER_SHOW_EXPECTED`, `INFER_QUESTION` | 10 / 0 / – | Infer jobs only. |

Example — a complete 8B first attempt:

```bash
qsub -v MODEL_ID=Qwen/Qwen3-8B,LR=1e-4,RUN_NAME=qwen3-8b-smoke,HF_HUB_OFFLINE=1 \
     0.peft/hpc/pbs_smoke.pbs
qsub -v MODEL_ID=Qwen/Qwen3-8B,LR=1e-4,RUN_NAME=qwen3-8b,HF_HUB_OFFLINE=1,CHECKPOINT_DIR=/workspace/rlvr/0.peft/checkpoints-8b \
     0.peft/hpc/pbs_train.pbs
qsub -v MODEL_ID=Qwen/Qwen3-8B,CHECKPOINT_DIR=/workspace/rlvr/0.peft/checkpoints-8b,INFER_SHOW_EXPECTED=1 \
     0.peft/hpc/pbs_infer.pbs
```

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Job sits in `Q` forever | Request too large, wrong queue props, or your 1 running job slot is busy | Compare `qstat -f` with the queue `resources_max`; don't raise `mem` speculatively. `qstat -u $USER` shows your running job. |
| Job rejected immediately | Missing account/Qlist | Fill the `##PBS -A`/`group_list` lines; ask your Professor for the allocation code. |
| `singularity: command not found` in job log | Runtime is a module on that node | The launcher tries `module load singularity` already; run `module avail` in an interactive job and set `CONTAINER_BIN=<path>` via `-v` if the name differs. |
| `nvidia-smi` fails / driver error inside container | Host driver too old for the image's CUDA | Report the driver version from the smoke log to the admins; options are a host with a newer driver or an older `nemo-automodel` image tag (re-render `.sif` with `IMAGE_URI=docker://nvcr.io/nvidia/nemo-automodel:<older>`). |
| Hangs at model download; killed by walltime | Compute nodes have no internet | §5 prefetch + `HF_HUB_OFFLINE=1`. The recipe sets a 30-min NCCL timeout partly to survive slow first downloads, but offline nodes will never succeed. |
| CUDA OOM at step 0 | Micro-batch too big for this model/length | `-v LOCAL_BATCH_SIZE=1` (renderer presets should already be safe), then `MAX_LENGTH=1536`. |
| CUDA OOM mid-epoch | Long `group_by_length` batch | `-v GROUP_BY_LENGTH=false`. |
| Host (RAM) OOM / job killed with no Python traceback | `mem` too small for model load | Raise `mem` in `pbs_train.pbs` (64→96 GB); check `qstat -f` `resources_used.mem`. |
| Checkpoints missing after job | Wrote outside the bind mount, or scratch purged | Keep `CHECKPOINT_DIR` under `/workspace/rlvr/...`; copy adapters out promptly. |
| Infer all FAIL after "successful" training | Base/adapter mismatch, or undertrained | Check the `Base:`/`Adapter:` lines in the infer log against §8.2; if they match, train longer / adjust LR. |
| `.sif` build fails with registry rate limits | Registry throttling from campus IPs | Build the `.sif` at home and `scp` it (§4) — the file is self-contained. |

## 11. End-to-end checklist

- [ ] Orientation done: runtime command, scratch path + quota/purge known, internet question answered for login **and** compute nodes.
- [ ] Repo cloned on `feat/singularity-peft-port` in scratch; `train.jsonl` extracted (35,000 lines).
- [ ] `.sif` built/transferred; `singularity inspect` passes; you know its size and path.
- [ ] Model prefetched (or online first run accepted); smoke passes with `HF_HUB_OFFLINE=1` if applicable.
- [ ] `pbs_smoke` green: A100 visible in-container, loss finite, `checkpoints-smoke/epoch_*/model/` exists, rendered recipe read.
- [ ] `pbs_infer` on the smoke adapter shows well-formed `<think>/<answer>`.
- [ ] `pbs_train` submitted with correct `MODEL_ID` (+ size-appropriate `LR`/walltime from §8).
- [ ] Final adapter format-checked with the **same** `MODEL_ID`.
- [ ] Adapter + `peft-train.o*` log + rendered recipe copied out of scratch; base-model id recorded with them for the merge step.

---

*Scope note:* this branch ports **stage 1 (PEFT)** only. The GRPO/RLVR stage
(`nemotron_rlvr/`) brings Ray + vLLM + NeMo Gym into a second container and a
different PBS shape (still one GPU, colocated) — port it the same way
(one in-container entrypoint, rendered configs, smoke job first) when you get
there, and expect the multi-GPU `cluster_8gpu.yaml` path to be irrelevant
under this queue's 1-GPU-per-user limit.
