# RLVR pipeline — fixes applied

This tree is the reviewed `rlvr-main` with the review's critical and high items
fixed, plus the tooling the transfer-matrix experiment needs. Every change is
pure-Python or config and is covered by tests that run without a GPU. Nothing in
the NeMo-RL / Gym training path was restructured.

Run the whole regression set:

```bash
cd nemotron_rlvr
python3 resources_servers/nemotron_verifier/tests/test_score.py          # existing, 10 tests
python3 resources_servers/nemotron_verifier/tests/test_verifier_fixes.py  # new, 19 tests
python3 scripts/test_reformat_and_install.py
(cd ../0.peft && python3 test_render_recipe.py && python3 test_upload_to_huggingface.py)
```

## Reward soundness (verifiers)

- **P1 arithmetic** (`verifiers/p1_arithmetic.py`) — rewritten to pure-Python
  exact `Fraction` compare. The answer must equal the gold **and** be a reduced
  literal (integer or `p/q`). Equivalence alone was unsound: the prompt's own
  expression is equivalent to its value, so an unevaluated restatement scored
  1.0. No SymPy on the answer path, so P1 is also fast and safe.
- **P2 symbolic** (`verifiers/p2_symbolic.py`) — after the equivalence check the
  answer must be in the form the task demands (factored / expanded / a ≥2-term
  partial-fraction / no-more-complex-than-gold simplification) and must not be a
  verbatim echo of the input expression. Mode is taken from `verifier_meta.mode`
  or inferred from the task verb. The parser now runs with SymPy's public
  namespace and no Python builtins.
- **P3 logic** (`verifiers/p3_logic.py`) — `_yesno` rejects ambiguous answers
  (both a yes- and a no-token present) instead of taking the first token. K&K is
  scored by **constraint satisfaction** (the submitted assignment must make each
  knight's statement true and each knave's false), so any valid solution is
  accepted; unchanged for unique puzzles.
- **P5 pattern** (`verifiers/p5_pattern.py`) — bracket-insensitive: a single
  value matches whether gold or answer is `10` or `[10]`. Removes the silent
  zero where the SFT target `[10]` disagreed with a scalar gold.
- **P0 retrieval** (`verifiers/p0_retrieval.py`) — accepts either one valid span
  (membership) or the full set (set-equality) by default; pin with
  `verifier_meta.span_match` = `any` | `all`. Resolves the "name one" prompt vs.
  full-list gold inconsistency.

## Reward-server robustness

- **Isolation + limits** (`resources_servers/nemotron_verifier/score.py`) — P2
  and P3 now score in a child process with CPU (10s) and address-space (2 GiB)
  rlimits, so pathological answer text cannot stall the scorer (the thread
  timeout did not bound off-main-thread SymPy work) or exhaust memory. P1 stays
  in-process (pure-Python, fast).

## Training config

- **`configs/cluster_5xA100.yaml`** — 4 Megatron trainer GPUs (TP2/CP2) + 1
  dedicated vLLM rollout GPU, bf16 weights with FP32 optimizer and gradient
  reduction, group size 8 with leave-one-out baseline, and a **nonzero KL
  penalty** (`reference_policy_kl_penalty: 0.01`, to be swept). The shipped
  configs trained with KL = 0.

## Transfer-matrix tooling (new, in `scripts/`)

- **`split_by_family.py`** — splits the train JSONL into the seven conditions
  (P1..P6 single-family + a balanced mix) at matched size.
- **`make_transfer_configs.py`** — renders one GRPO config per condition × seed
  from a base config and writes a launch plan.
- **`build_transfer_matrix.py`** — assembles the 6×6 T[i,j] from per-condition
  eval JSONL, paired by instance_id, with 95% bootstrap CIs, a P0 stability
  column and an R-Mix row.
- **`validate_corpus.py`** — gold round-trip (every gold must score 1.0 under
  its verifier), global content-hash dedup within and across splits, and yes/no
  label balance. Run this before training.
- **`extract_subspaces.py`** — Tier-2 geometry: per-family activation subspaces
  from the base model, the principal-angle matrix S at several depths, and its
  correlation with the transfer matrix (the go/no-go signal). Linear algebra is
  pure NumPy and unit-testable; model loading is behind lazy torch imports.

## SFT think-block

- **`0.peft/diversify_think.py`** — the shipped SFT corpus uses one identical
  `<think>` string on all 35k rows, which SFTs a fixed placeholder prior. This
  script replaces it with varied content-free scaffolds (`--mode vary`) or drops
  the think scaffold entirely (`--mode strip`), leaving `<answer>` untouched. It
  injects no reasoning content. Regenerate the SFT data through it before Stage
  0, or regenerate the corpus upstream so the think block is not constant.

## Not changed here (needs the generators, which are not in this repo)

- Capping P1/P2 tier-4 answer magnitude (thousand-digit golds) and P6 unique-
  solution / infeasible-instance filtering live in the problem generators, not
  in this training repo. `validate_corpus.py` will surface the symptoms
  (oversized answers via gold round-trip failures; duplicates) so the fixes can
  be verified once the generators are updated.

## Suggested order to run

```bash
cd nemotron_rlvr
python3 scripts/reformat_dataset.py --corpus /path/to/P0-P6/corpus --out-dir data
python3 scripts/validate_corpus.py --file train=data/train-split.jsonl --file val=data/val-split.jsonl
python3 scripts/split_by_family.py --in data/train-split.jsonl --out-dir data/conditions --size 10000
python3 scripts/make_transfer_configs.py --base configs/cluster_5xA100.yaml \
    --conditions-dir data/conditions --out-dir configs/transfer --seeds 1 2 3
# ... run configs/transfer/*.yaml (train.py), convert checkpoints, evaluate each on the eval set ...
python3 scripts/build_transfer_matrix.py --baseline eval/sft_baseline.jsonl \
    --condition P1=eval/trained_p1.jsonl ... --condition MIX=eval/trained_mix.jsonl \
    --out-json eval/transfer_matrix.json --out-markdown eval/transfer_matrix.md
python3 scripts/extract_subspaces.py --model <merged_sft> --data data/val-split.jsonl \
    --transfer eval/transfer_matrix.json --out eval/subspaces.json
```
