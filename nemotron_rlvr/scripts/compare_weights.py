#!/usr/bin/env python3
"""Prove two HF checkpoints differ without loading both models into GPU/RAM."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open

def shards(root):
 ps=sorted(root.glob('*.safetensors'))
 if not ps: raise SystemExit(f'No .safetensors under {root}')
 return ps
def index(root):
 out={}
 for p in shards(root):
  with safe_open(p,framework='pt',device='cpu') as f:
   for k in f.keys(): out[k]=p
 return out
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('baseline',type=Path); ap.add_argument('trained',type=Path); ap.add_argument('--json-out',type=Path); a=ap.parse_args()
 bi,ti=index(a.baseline),index(a.trained); common=sorted(bi.keys()&ti.keys()); changed=total=0; sum_abs=sum_sq=base_sq=max_abs=0.0; cache={}
 for n in common:
  key=('b',bi[n]);
  if key not in cache: cache[key]=safe_open(bi[n],framework='pt',device='cpu'); cache[key].__enter__()
  key2=('t',ti[n]);
  if key2 not in cache: cache[key2]=safe_open(ti[n],framework='pt',device='cpu'); cache[key2].__enter__()
  x=cache[key].get_tensor(n).float(); y=cache[key2].get_tensor(n).float()
  if x.shape!=y.shape: continue
  d=y-x; num=d.numel(); total+=num; ma=float(d.abs().max()); max_abs=max(max_abs,ma); changed+=int(torch.count_nonzero(d)); sum_abs+=float(d.abs().sum()); sum_sq+=float((d*d).sum()); base_sq+=float((x*x).sum())
 for f in cache.values(): f.__exit__(None,None,None)
 report={'common_tensors':len(common),'baseline_only':len(bi.keys()-ti.keys()),'trained_only':len(ti.keys()-bi.keys()),'parameters':total,'changed_parameters':changed,'changed_fraction':changed/total if total else 0,'mean_absolute_delta':sum_abs/total if total else 0,'max_absolute_delta':max_abs,'relative_l2_delta':(sum_sq/base_sq)**.5 if base_sq else None}
 print(json.dumps(report,indent=2));
 if a.json_out:a.json_out.write_text(json.dumps(report,indent=2)+'\n')
 if changed==0: raise SystemExit('FAIL: no parameter differences found')
 print('PASS: model parameters changed during training.')
if __name__=='__main__':main()
