# Custom RLVR Pipeline (7-Problem Family Suite P0–P6)

Reinforcement Learning with Verifiable Rewards for Qwen across seven families:

- **P0**: Factual Retrieval (`p0_retrieval`)
- **P1**: Arithmetic (`p1_arithmetic`)
- **P2**: Symbolic Transformation (`p2_symbolic`)
- **P3**: Deductive Logic (`p3_logic`)
- **P4**: Procedural / Planning (`p4_planning`)
- **P5**: Pattern / Inductive Reasoning (`p5_pattern`)
- **P6**: Constraint Satisfaction (`p6_constraint`)

Rewards come from a **NeMo Gym resources server** (`nemotron_verifier`) that
routes each example to the matching Python class in `verifiers/`.

---

## Workspace Structure

```text
rlvr/
├── 0.peft/                          # PEFT SFT (XML <think>/<answer> format)
└── nemotron_rlvr/                   # GRPO + Gym
    ├── dataset/                     # Corpus metadata (per-family JSONL is local-only)
    ├── configs/
    │   ├── default.yaml
    │   ├── single_gpu.yaml          # 1×20GB colocated sync-GRPO
    │   ├── cluster_8gpu.yaml
    │   ├── models/policy_model.yaml # Gym vLLM training yaml (fallback)
    │   └── resources_servers/       # DEPRECATED class-entrypoint stubs
    ├── resources_servers/nemotron_verifier/
    │   ├── app.py                   # Gym FastAPI wrapper
    │   ├── score.py                 # P0–P6 routing (Gym-free)
    │   ├── pyproject.toml           # ONLY this file — no requirements.txt
    │   └── configs/nemotron_verifier.yaml
    ├── verifiers/                   # p0 … p6 .verify(response, extra_env_info)
    ├── scripts/
    │   ├── reformat_dataset.py      # corpus → Gym JSONL
    │   ├── install_gym_server.py    # copy server into $GYM_ROOT
    │   └── merge_peft.py
    ├── data/                        # train-split.jsonl / val-split.jsonl
    ├── train_utils.py
    └── train.py
```

---

## Execution Workflow

### Step 1: Train PEFT format model (in `0.peft`)

```bash
automodel recipe.yaml
```

### Step 2: Merge the PEFT adapter

```bash
cd nemotron_rlvr
python3 scripts/merge_peft.py \
    --base Qwen/Qwen3-0.6B \
    --adapter /path/to/0.peft/checkpoints/epoch_2/model \
    --output /workspace/rlvr/0.peft/checkpoints-smoke/qwen_merged_sft
```

### Step 3: Reformat the RLVR corpus to Gym JSONL

Datasets are **not** in git. Point `--corpus` at the local P0–P6 tree
(`p0/train.jsonl` … `p6/test_id.jsonl`):

```bash
python3 scripts/reformat_dataset.py --corpus /path/to/dataset --out-dir data
```

Each line looks like:

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "..."}]},
  "expected_answer": "...",
  "task_name": "p0_retrieval",
  "family": "P0",
  "verifier_meta": {},
  "agent_ref": {"type": "responses_api_agents", "name": "nemotron_verifier_simple_agent"}
}
```

### Step 4: Install the Gym server (also done automatically by `train.py`)

Gym's `setup_env_command` opens

```
$GYM_ROOT/resources_servers/nemotron_verifier/pyproject.toml
```

A copy that only lives in this repo is not enough. On the Super3 container:

```bash
export NEMO_GYM_ROOT=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
python3 scripts/install_gym_server.py
```

Do **not** add `requirements.txt` next to `pyproject.toml` — Gym rejects having both.

### Step 5: Train

Single GPU (RTX A4500 20GB) — run **`configs/single_gpu.yaml`**. Do not use a Super leftover yaml pasted from chat (`async_grpo` + colocated 1 GPU, `async_engine: false`, p0–p6 class entrypoints).

```bash
python3 train.py --config configs/single_gpu.yaml
```

8×A100:

```bash
python3 train.py --config configs/cluster_8gpu.yaml
```

`train.py` copies the server into Gym, rewrites `config_paths` to absolute
paths, then spins Gym up. If `setup()` already returns a Gym actor (Super3 /
current NeMo RL), that actor is reused; on v0.5.0 a `NemoGym.remote` is created
after the 10-tuple `setup()`.

---

## Hardware notes

| Parameter | Single GPU (20GB) | 8 GPU cluster |
| :--- | :--- | :--- |
| `cluster.gpus_per_node` | `1` | `8` |
| `async_grpo.enabled` | **`false`** (colocated 1 GPU) | yaml default |
| `vllm_cfg.async_engine` | `true` (Gym HTTP) | `true` |
| `vllm_cfg.expose_http_server` | `true` | `true` |
| `gpu_memory_utilization` | `0.45` | `0.7` |
| `num_prompts_per_step` × gens | `2 × 4` (gbs 8) | `32 × 8` |
| `max_total_sequence_length` | `2048` (`max_new_tokens` 1536) | `32768` |
| `max_val_samples` / `val_batch_size` | `8` / `4` | not capped |

`train.py` does **not** overwrite `val_batch_size` with `len(val)` when the yaml
already sets `max_val_samples`.

---

## Local verifier tests (no Gym)

```bash
python3 resources_servers/nemotron_verifier/tests/test_score.py
```
