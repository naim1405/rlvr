#!/usr/bin/env python3
"""Tier-2 activation-subspace analysis: does base-model geometry predict transfer?

Runs the base (post-SFT) model over the SRP eval set, takes the last-input-token
hidden state at several depths, and for each reasoning family computes the
principal subspace its activations span. The principal angles between families'
subspaces give a 6x6 geometry matrix S (0 = aligned, 1 = orthogonal). If a
transfer matrix is supplied, S is correlated with the off-diagonal transfer
entries — the go/no-go signal for the paper's central claim.

The linear algebra (subspace_basis, principal_angles, subspace_matrix,
correlate_offdiag) is pure NumPy and unit-testable without a GPU. Model loading
and hooking live in main() behind lazy torch/transformers imports.

Usage (from nemotron_rlvr/):
  python3 scripts/extract_subspaces.py --model /path/to/merged_sft \
      --data data/val-split.jsonl --layers 0.25 0.37 0.5 0.75 --k 20 \
      --transfer eval/transfer_matrix.json --out eval/subspaces.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_FAMILIES = ["P1", "P2", "P3", "P4", "P5", "P6"]


def subspace_basis(X: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal directions of rows of X (each row one activation vector).

    Returns a (k, d) matrix of orthonormal basis vectors for the subspace that
    explains the most variance in X (mean-centered).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # rows are samples; right singular vectors (Vt) are directions in feature space
    _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, vt.shape[0])
    return vt[:k]


def principal_angles(Qi: np.ndarray, Qj: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two row-orthonormal bases."""
    # singular values of Qi Qj^T are the cosines of the principal angles
    s = np.linalg.svd(Qi @ Qj.T, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def subspace_matrix(acts: dict[str, np.ndarray], k: int) -> dict:
    """Build the 6x6 mean-principal-angle matrix (normalized to [0,1])."""
    bases = {f: subspace_basis(acts[f], k) for f in EVAL_FAMILIES if f in acts}
    fams = [f for f in EVAL_FAMILIES if f in bases]
    S = {fi: {} for fi in fams}
    for fi in fams:
        for fj in fams:
            if fi == fj:
                S[fi][fj] = 0.0
                continue
            ang = principal_angles(bases[fi], bases[fj])
            S[fi][fj] = float(np.degrees(ang).mean() / 90.0)  # 0 aligned .. 1 orthogonal
    return S


def correlate_offdiag(S: dict, T: dict) -> dict:
    """Pearson r between (1 - S) similarity and transfer T on off-diagonal pairs."""
    xs, ys = [], []
    for fi in EVAL_FAMILIES:
        for fj in EVAL_FAMILIES:
            if fi == fj or fi not in S or fj not in S.get(fi, {}):
                continue
            t = T.get(fi, {}).get(fj)
            if t is None:
                continue
            tv = t["delta"] if isinstance(t, dict) else t
            xs.append(1.0 - S[fi][fj])   # higher = more aligned
            ys.append(float(tv))
    if len(xs) < 3:
        return {"n": len(xs), "pearson_r": float("nan")}
    x = np.array(xs); y = np.array(ys)
    xm, ym = x.mean(), y.mean()
    denom = math.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    r = float(((x - xm) * (y - ym)).sum() / denom) if denom else float("nan")
    return {"n": len(xs), "pearson_r": r}


def _layer_indices(n_layers: int, fracs: list[float]) -> list[int]:
    return sorted({max(0, min(n_layers - 1, int(round(fr * n_layers)))) for fr in fracs})


def _family(row: dict) -> str:
    f = str(row.get("family") or row.get("task_name") or "").upper()[:2]
    return f if f in EVAL_FAMILIES else f


def _prompt(row: dict) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    for m in reversed((row.get("responses_create_params") or {}).get("input") or []):
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--layers", type=float, nargs="+", default=[0.25, 0.37, 0.5, 0.75])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-per-family", type=int, default=150)
    ap.add_argument("--transfer", type=Path, help="transfer_matrix.json to correlate against")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    dtype = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        output_hidden_states=True,
    ).eval()
    n_layers = model.config.num_hidden_layers
    layers = _layer_indices(n_layers, a.layers)
    print(f"{n_layers} layers; extracting at indices {layers}")

    rows = [json.loads(l) for l in a.data.open(encoding="utf-8") if l.strip()]
    per_fam_count: dict[str, int] = defaultdict(int)
    # layer -> family -> list[vec]
    acts: dict[int, dict[str, list]] = {L: defaultdict(list) for L in layers}

    for r in rows:
        fam = _family(r)
        if fam not in EVAL_FAMILIES or per_fam_count[fam] >= a.max_per_family:
            continue
        per_fam_count[fam] += 1
        text = _prompt(r)
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**enc)
        hs = out.hidden_states  # tuple: embeddings + per-layer
        for L in layers:
            vec = hs[L + 1][0, -1, :].float().cpu().numpy()  # last input token
            acts[L][fam].append(vec)

    result = {"model": a.model, "n_layers": n_layers, "layers": layers, "k": a.k,
              "per_family": dict(per_fam_count), "S_by_layer": {}, "correlation_by_layer": {}}
    T = None
    if a.transfer and a.transfer.exists():
        T = json.loads(a.transfer.read_text()).get("matrix", {})

    for L in layers:
        mats = {f: np.array(v) for f, v in acts[L].items() if len(v) >= a.k}
        S = subspace_matrix(mats, a.k)
        result["S_by_layer"][str(L)] = S
        if T is not None:
            result["correlation_by_layer"][str(L)] = correlate_offdiag(S, T)
            r = result["correlation_by_layer"][str(L)]
            print(f"layer {L} (~{L/n_layers:.0%} depth): geometry-transfer r = {r['pearson_r']:.3f} (n={r['n']})")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
