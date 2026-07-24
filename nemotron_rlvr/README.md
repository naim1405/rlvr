# Custom RLVR Pipeline (7-Problem Family Suite P0–P6)

This repository contains a full **Reinforcement Learning with Verifiable Rewards (RLVR)** setup for training LLMs across 7 problem domains:

- **P0**: Factual Retrieval (`p0_retrieval`)
- **P1**: Arithmetic (`p1_arithmetic`)
- **P2**: Symbolic Transformation (`p2_symbolic`)
- **P3**: Deductive Logic (`p3_logic`)
- **P4**: Procedural / Planning (`p4_planning`)
- **P5**: Pattern / Inductive Reasoning (`p5_pattern`)
- **P6**: Constraint Satisfaction (`p6_constraint`)

---

## Project Layout

```
.
├── configs/
│   ├── default.yaml                 # Master RLVR & NeMo Gym config
│   └── resources_servers/           # 7 NeMo Gym environment configs (p0.yaml .. p6.yaml)
├── verifiers/                       # Python verifiers for P0-P6
│   ├── p0_retrieval.py
│   ├── p1_arithmetic.py
│   ├── p2_symbolic.py
│   ├── p3_logic.py
│   ├── p4_planning.py
│   ├── p5_pattern.py
│   └── p6_constraint.py
├── scripts/
│   ├── reformat_dataset.py          # Formats phase2 dataset to JSONL splits
│   └── merge_peft.py                # Merges LoRA PEFT adapter into base Qwen model
├── data/                            # Generated train/val JSONL splits
├── train_utils.py                   # Dataset loading helper functions
├── train.py                         # Main RLVR Async-GRPO training script
└── README.md
```

---

## Execution Workflow

### Step 1: Reformat Dataset
```bash
python scripts/reformat_dataset.py
```

### Step 2: Merge PEFT Adapter (if using LoRA)
```bash
python scripts/merge_peft.py \
    --base /path/to/qwen_base \
    --adapter /path/to/peft_adapter \
    --output checkpoints/qwen_merged_sft
```

---

## Deployment Configurations

### Option A: Single GPU Testing Setup (e.g., RTX A4500 / 20GB VRAM)

Use the dedicated single GPU configuration file (`configs/single_gpu.yaml`):

```bash
python train.py --config configs/single_gpu.yaml
```

---

### Option B: 8 x A100 GPU Cluster Deployment Setup (80GB VRAM GPUs)

Use the dedicated 8 GPU cluster configuration file (`configs/cluster_8gpu.yaml`):

```bash
python train.py --config configs/cluster_8gpu.yaml
```

---

## Hardware Configuration Matrix

| Parameter | Single GPU Test (20GB VRAM) | 8 GPU Cluster (80GB VRAM Each) |
| :--- | :--- | :--- |
| **`cluster.gpus_per_node`** | `1` | `8` |
| **`tensor_model_parallel_size`** | `1` | `2` (or `4`) |
| **`context_parallel_size`** | `1` | `2` |
| **`colocated.enabled`** | `true` | `true` |
| **`gpu_memory_utilization`** | `0.4` | `0.7` |
| **`max_total_sequence_length`** | `4,096` tokens | `32,768` tokens |
| **`num_prompts_per_step`** | `4` | `32` |
| **`num_generations_per_prompt`**| `4` | `8` |
