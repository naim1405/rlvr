
# Recipe tuning guide (NeMo AutoModel LoRA)

General reference for every knob in `recipe.yaml`: what it does, what it costs
in VRAM, and how to tune it for any GPU. The `Default` column mirrors the
values in `recipe.yaml`.

## How VRAM scales (the only two laws you need)

1. **Tokens per microbatch drives memory.** Activation memory scales roughly
   linearly with `local_batch_size × max_length`. Double either → ~2x the
   variable memory. Model weights are fixed (~1.2 GB for Qwen3-0.6B in bf16).
2. **`global_batch_size` is free.** Effective batch per optimizer step is
   `local × accum_steps × GPUs`, so raising the global batch only adds cheap
   accumulation steps, not memory. Rule: it must stay divisible by
   `local × GPUs`.

**Method for any card:** run once at a known config, record peak VRAM with
`watch -n 2 nvidia-smi`, treat ~90% of total VRAM as your safe ceiling, then
spend headroom in this order:

1. `max_length` ↑ — buys quality (less truncation, see dataset facts below)
2. `global_batch_size` ↑ — free via accumulation (stabler gradients)
3. `local_batch_size` ↑ — throughput only; raise last
4. Quality knobs (`num_epochs`, `peft.dim`) — no/little memory cost

## Dataset facts (measured on `train.jsonl`, Qwen3 tokenizer + chat template)

Fixed properties of the data — same regardless of GPU:

| Stat | Tokens |
|---|---|
| Rows | 35,000 |
| p50 / p90 / p95 / max full-sequence length | 267 / 1343 / 3404 / 14880 |

| `max_length` | Rows truncated |
|---|---|
| 512 | 34.8% |
| 1024 | 19.3% |
| 2048 | 7.5% |
| 3072 | ~6% (not worth +50% memory over 2048) |

## Parameter reference

### `step_scheduler` — batching & schedule

| Parameter | Default | What it does | VRAM | Throughput / Quality | How to tune |
|---|---|---|---|---|---|
| `global_batch_size` | 32 | Effective batch per optimizer step | None directly (via accumulation) | Higher = stabler grads, fewer steps/epoch | Raise freely; keep divisible by `local × GPUs` |
| `local_batch_size` | 4 | Microbatch per GPU per forward pass | **Linear — the biggest lever** | Higher = better throughput | Raise **last**, after sequence length. Halve it first if OOM at step 0 |
| `num_epochs` | 2 | Full passes over data | None | Time scales linearly; quality ↑ | 1 for quick verification loops, 2+ for real runs |
| `ckpt_every_steps` | 500 | Adapter checkpoint frequency | Negligible | Lower = safer resume, slight slowdown | 500 is fine; lower on flaky/preemptible boxes |
| `val_every_steps` | 500 | Validation frequency (no-grad, cheap) | Tiny | Lower = earlier convergence signal | Lower temporarily when watching a new config |

### `dataset` / `validation_dataset` — sequence length is the quality lever

| Parameter | Default | What it does | VRAM | Throughput / Quality | How to tune |
|---|---|---|---|---|---|
| `max_length` | 2048 | Truncates/pads every sample to this many tokens | **Linear per token** | Longer = less truncation = quality ↑, slower | Spend headroom here **first**. Use the truncation table above to pick: 2048 keeps 92.5% of rows intact |
| `validation_fraction` | 0.02 | Train/val split (~700 val rows) | None | — | Keep unless you need a bigger/smaller val set |
| `include_system_prompt` | false | Prepends system prompt to inputs | Negligible | `false` = format becomes unconditional response style | Keep false (matches RLVR inference, which sends no system prompt) |
| `num_samples_limit` | — / 1024 | Caps rows loaded (train / val) | None | Lower = faster loops, unrepresentative loss | Use on train for fast iteration only; remove for real runs. Val cap ~1024 is cheap to keep |

### `dataloader` — throughput & peak smoothing

| Parameter | Default | What it does | VRAM | Throughput / Quality | How to tune |
|---|---|---|---|---|---|
| `shuffle` | true | Shuffles each epoch | None | Quality (generalization) | Keep true |
| `group_by_length` | true | Batches similar-length rows → less padding waste | **Raises peak**: longest batches hit full `local × max_length` | Faster (less padding) | Keep true normally. If OOM strikes **mid-epoch** (fine at step 0, dies later on a long batch), set **false** to smooth peaks at the cost of padding |
| `collate_fn` | default | Pads batch to longest sample | — | — | Don't touch |

### `peft` — adapter capacity (quality, small memory cost)

| Parameter | Default | What it does | VRAM | Throughput / Quality | How to tune |
|---|---|---|---|---|---|
| `target_modules` | `*_proj` | Which linears get LoRA (q/k/v/o/gate/up/down) | Small | More coverage = quality ↑ | Keep |
| `dim` (rank) | 32 | Adapter bottleneck size | Small (~rank × hidden × layers + Adam states, tens of MB) | Higher = more expressive, slight slowdown | 32 is a solid default; try 64 if underfitting and VRAM allows |
| `alpha` | 64 | LoRA scaling (effective strength = alpha/rank = 2) | None | Higher = stronger adapter updates | Keep at 2× dim by convention |
| `dropout` | 0.05 | Adapter dropout | None | Regularization | Keep |
| `use_triton` | true | Fused Triton LoRA kernels | Extra kernel workspace (~hundreds of MB) | Faster | Keep true; if OOM is marginal (within ~1 GB of fitting), try false |

### `optimizer` — no memory impact, quality only

| Parameter | Default | What it does | How to tune |
|---|---|---|---|
| `lr` | 2.0e-4 | Step size | Standard for LoRA. Sweep 1e-4–3e-4 only if loss misbehaves |
| `betas` / `eps` | 0.9, 0.999 / 1e-8 | Adam moment estimates | Keep |
| `weight_decay` | 0 | Decay applied to adapters | Keep 0 (decaying LoRA weights usually hurts) |

### `model` / `distributed` / `checkpoint` / misc — rarely touched

| Parameter | Default | What it does | How to tune |
|---|---|---|---|
| `torch_dtype` | bf16 | Compute/weight precision | Optimal on Ampere and newer. fp32 ~2x weights+activations — never for LoRA |
| `strategy` / `tp_size` / `cp_size` / `dp_size` | fsdp2 / 1 / 1 / none | Parallelism strategy | Single GPU: nothing to shard, leave as-is. Multi-GPU: see scaling section |
| `save_consolidated` | false | Gather full model when checkpointing | Keep false (adapters only). `true` causes a full-model gather memory spike |
| `packed_sequence_size` | 0 | Packs multiple samples per sequence (0 = off) | Reduces padding waste but changes batching semantics; enable only if the recipe docs confirm label-masking support for packed input |
| `trust_remote_code` | true | Allow model-specific code from HF | Required for Qwen; keep |

## Multi-GPU scaling

Scale out, not up: keep per-GPU `local_batch_size` and `max_length` fixed,
grow the global batch with GPU count:

```
global_batch_size = local × accum_steps × num_GPUs
```

Example: 8 GPUs at `local=4`, `accum=8` → `global = 256`. On large cards
(80 GB) you can additionally raise the per-GPU microbatch for throughput —
re-run the measure-and-spend method above to find the new ceiling.

## Worked example: fitting a 20 GB card

1. Baseline run at `1 × 1024` peaks at 7 GB. Safe ceiling ≈ 17.5 GB (90%).
2. Rough split: ~2.5–3 GB fixed (weights + CUDA ctx + buffers),
   ~4–4.5 GB per 1024 tokens of microbatch → budget ≈ 3–3.5K tokens.
3. Spend it: `max_length` → 2048 (truncation 19.3% → 7.5%),
   `global_batch_size` → 16 (free, accum 16). Expected peak ~10–13 GB. ✔
4. Optional: if peak stays < ~14 GB, try `local_batch_size` → 2 for
   throughput; revert on OOM — costs only a few minutes.

## Monitoring cheat sheet

```bash
watch -n 2 nvidia-smi                                   # peak VRAM per GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True automodel recipe.yaml  # fragmentation guard
# OOM in .backward() at step 0  -> cut local_batch_size or max_length
# OOM mid-epoch (was fine, then dies) -> a long group_by_length batch: set group_by_length: false or cut max_length
# 2+ GB "reserved but unallocated" -> fragmentation: keep expandable_segments on
```
