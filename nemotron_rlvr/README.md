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

## Workspace Structure

```text
rlvr/
├── 0.peft/                          # PEFT SFT module (Format learning)
│   ├── recipe.yaml                  # NeMo AutoModel PEFT SFT recipe
│   ├── mydataset.py                 # QwenPEFTDataset loader
│   ├── smoke_test_dataset.py        # Dataset verification script
│   └── dataset/                     # PEFT SFT dataset (train.jsonl)
└── nemotron_rlvr/                   # Custom RLVR pipeline module
    ├── dataset/                     # Multi-family RLVR corpus (p0/ .. p6/)
    ├── configs/
    │   ├── default.yaml             # Master RLVR & NeMo Gym config
    │   ├── single_gpu.yaml          # Single GPU test config
    │   ├── cluster_8gpu.yaml        # 8 x A100 GPU cluster deployment config
    │   └── resources_servers/       # 7 NeMo Gym environment configs (p0.yaml .. p6.yaml)
    ├── verifiers/                   # Python verifiers for P0-P6
    │   ├── p0_retrieval.py
    │   ├── p1_arithmetic.py
    │   ├── p2_symbolic.py
    │   ├── p3_logic.py
    │   ├── p4_planning.py
    │   ├── p5_pattern.py
    │   └── p6_constraint.py
    ├── scripts/
    │   ├── reformat_dataset.py      # Formats dataset/ to JSONL splits (train-split.jsonl)
    │   └── merge_peft.py            # Merges 0.peft adapter into base Qwen model
    ├── data/                        # Generated train/val JSONL splits
    ├── train_utils.py               # Dataset loading helper functions
    ├── train.py                     # Main RLVR Async-GRPO training script
    └── README.md
```

---

## Execution Workflow

### Step 1: Train PEFT Format Model (in `0.peft`)
```bash
# Inside container from /tmp/0.peft:
automodel recipe.yaml
```

### Step 2: Merge PEFT Adapter into Base Model
```bash
cd /home/ezio/Projects/rlhf/rlvr/nemotron_rlvr
python3 scripts/merge_peft.py \
    --base Qwen/Qwen3-0.6B \
    --adapter /path/to/0.peft/checkpoints/epoch_2/model \
    --output /path/to/qwen_merged_sft
```

### Step 3: Reformat RLVR Dataset
```bash
python3 scripts/reformat_dataset.py
```

### Step 4: Run RLVR Training

#### Single GPU Test Setup (e.g. RTX A4500 20GB VRAM)
```bash
python3 train.py --config configs/single_gpu.yaml
```

#### 8 x A100 GPU Cluster Deployment Setup (80GB VRAM GPUs)
```bash
python3 train.py --config configs/cluster_8gpu.yaml
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
