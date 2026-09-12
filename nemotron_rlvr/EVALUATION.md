# Verify and evaluate an RLVR checkpoint

Run these commands **inside the same NeMo RL/Nemotron container used for training**. Keep the starting merged-SFT model as the baseline. The validation split is the primary held-out measurement; the train split is only a memorization diagnostic.

## 1. Inspect the checkpoint

```bash
python3 scripts/inspect_checkpoint.py checkpoints/rlvr_results_single_gpu/step_1500 \
  --json-out evaluation/step_1500_inspection.json
```

## 2. Convert Torch DCP/Megatron weights to Hugging Face

The wrapper invokes NeMo RL's official `examples/converters/convert_megatron_to_hf.py`:

```bash
export NEMO_RL_ROOT=/opt/nemo-rl
python3 scripts/convert_checkpoint.py \
  checkpoints/rlvr_results_single_gpu/step_1500 \
  huggingface/step_1500
```

If the checkout is elsewhere, pass `--converter /path/to/convert_megatron_to_hf.py`. If `config.yaml` contains a stale model path, add `--hf-model-name /path/to/the/original/merged_sft`. Use `--no-strict` only after investigating a strict-conversion error.

Official equivalent command:

```bash
python /opt/nemo-rl/examples/converters/convert_megatron_to_hf.py \
  --config checkpoints/rlvr_results_single_gpu/step_1500/config.yaml \
  --megatron-ckpt-path checkpoints/rlvr_results_single_gpu/step_1500/policy/weights/iter_0000000 \
  --hf-ckpt-path huggingface/step_1500
```

## 3. Prove parameters changed

```bash
python3 scripts/compare_weights.py \
  /workspace/rlvr/0.peft/checkpoints-smoke/qwen_merged_sft \
  huggingface/step_1500 \
  --json-out evaluation/weight_delta.json
```

This proves optimization changed weights, but not that quality improved.

## 4. Evaluate baseline and trained model on all validation rows

Evaluate one model at a time to fit one GPU. Results are flushed after every generation and `--resume` continues interrupted runs.

```bash
python3 scripts/evaluate_model.py \
  --model /workspace/rlvr/0.peft/checkpoints-smoke/qwen_merged_sft \
  --data data/val-split.jsonl --output evaluation/val_baseline.jsonl \
  --temperature 0 --max-new-tokens 1536 --resume

python3 scripts/evaluate_model.py \
  --model huggingface/step_1500 \
  --data data/val-split.jsonl --output evaluation/val_step1500.jsonl \
  --temperature 0 --max-new-tokens 1536 --resume

python3 scripts/compare_evaluations.py \
  --baseline evaluation/val_baseline.jsonl \
  --trained evaluation/val_step1500.jsonl \
  --markdown evaluation/val_report.md \
  --json-out evaluation/val_report.json
```

Do not use the training-time `max_val_samples: 8` cap for final evaluation. Evaluate the complete validation file.

## 5. Sampled evaluation (recommended)

Use identical seeds/settings for both models:

```bash
# Repeat for baseline and step_1500, changing --model and --output.
python3 scripts/evaluate_model.py --model MODEL --data data/val-split.jsonl \
  --output OUTPUT.jsonl --samples 4 --temperature 1.0 --top-p 1.0 \
  --max-new-tokens 1536 --seed 42 --resume
```

Compare these files with `compare_evaluations.py`. `samples=4` reports mean sampled reward; per-prompt pass@4 can also be computed from the retained JSONL.

## 6. Optional train-set diagnostic

Repeat step 4 with `data/train-split.jsonl`. Interpret outcomes as follows:

- train and validation improve: useful learning/generalization;
- train improves, validation does not: likely memorization or overfitting;
- weights changed but neither improves: optimization happened but did not help this metric;
- trained model regresses: inspect earlier checkpoints and reward trends.

For strongest evidence, evaluate several saved checkpoints and an untouched test split. The latest checkpoint need not be the best.
