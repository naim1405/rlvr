#!/usr/bin/env python3
"""Validate a NeMo-RL checkpoint and summarize evidence that training progressed."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import yaml

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20), b''): h.update(chunk)
    return h.hexdigest()

def find_iter(step: Path) -> Path:
    root=step/'policy'/'weights'
    latest=root/'latest_checkpointed_iteration.txt'
    if latest.is_file():
        raw=latest.read_text().strip()
        candidates=[root/f'iter_{int(raw):07d}', root/f'iter_{int(raw):06d}', root/f'iter_{raw}'] if raw.isdigit() else []
        for p in candidates:
            if p.is_dir(): return p
    dirs=sorted(root.glob('iter_*'))
    if len(dirs)!=1: raise SystemExit(f'Expected one iter_* directory under {root}, found {len(dirs)}')
    return dirs[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path, help='step_N checkpoint directory')
    ap.add_argument('--hash-shards', action='store_true', help='SHA256 large .distcp files (slow)')
    ap.add_argument('--json-out', type=Path)
    a=ap.parse_args(); step=a.checkpoint.resolve()
    required=[step/'config.yaml',step/'training_info.json',step/'policy'/'weights']
    missing=[str(p) for p in required if not p.exists()]
    if missing: raise SystemExit('Incomplete checkpoint; missing: '+', '.join(missing))
    cfg=yaml.safe_load((step/'config.yaml').read_text()) or {}
    info=json.loads((step/'training_info.json').read_text())
    idir=find_iter(step); shards=sorted(idir.glob('*.distcp'))
    files=[]
    for p in sorted(idir.iterdir()):
        if p.is_file():
            row={'name':p.name,'bytes':p.stat().st_size}
            if a.hash_shards or p.suffix!='.distcp': row['sha256']=sha256(p)
            files.append(row)
    name_step=int(step.name.split('_')[-1]) if step.name.startswith('step_') else None
    report={'checkpoint':str(step),'directory_step':name_step,'training_info':info,
      'model_name':cfg.get('policy',{}).get('model_name'),
      'tokenizer_name':cfg.get('policy',{}).get('tokenizer',{}).get('name'),
      'iteration_dir':str(idir),'distcp_shards':len(shards),
      'distcp_bytes':sum(p.stat().st_size for p in shards),'files':files,
      'conversion_ready':bool(shards and (idir/'metadata.json').is_file())}
    print(json.dumps(report,indent=2,default=str))
    if not shards: raise SystemExit('FAIL: no .distcp weight shards')
    if not (idir/'metadata.json').is_file(): raise SystemExit('FAIL: metadata.json missing')
    if a.json_out: a.json_out.parent.mkdir(parents=True,exist_ok=True); a.json_out.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print('\nPASS: checkpoint structure is complete. This proves a save occurred, not that quality improved.')
if __name__=='__main__': main()
