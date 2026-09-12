#!/usr/bin/env python3
"""Invoke NeMo RL's official Torch-DCP -> Hugging Face converter.
Run this inside the same NeMo RL/Nemotron container used for training.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

def find_iter(step: Path)->Path:
    ds=sorted((step/'policy'/'weights').glob('iter_*'))
    if len(ds)!=1: raise SystemExit(f'Expected exactly one iter_* directory, found {len(ds)}')
    return ds[0]
def find_converter(explicit: str|None)->Path:
    candidates=[]
    if explicit: candidates.append(Path(explicit))
    if os.getenv('NEMO_RL_ROOT'): candidates.append(Path(os.environ['NEMO_RL_ROOT'])/'examples/converters/convert_megatron_to_hf.py')
    candidates += [Path('/opt/nemo-rl/examples/converters/convert_megatron_to_hf.py'),Path('/workspace/nemo-rl/examples/converters/convert_megatron_to_hf.py')]
    for p in candidates:
        if p.is_file(): return p
    raise SystemExit('Cannot find converter. Set NEMO_RL_ROOT or pass --converter. Expected examples/converters/convert_megatron_to_hf.py in the NeMo RL checkout.')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('checkpoint',type=Path); ap.add_argument('output',type=Path)
    ap.add_argument('--converter'); ap.add_argument('--hf-model-name'); ap.add_argument('--no-strict',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(); step=a.checkpoint.resolve(); out=a.output.resolve(); conv=find_converter(a.converter)
    idir=find_iter(step)
    cmd=[sys.executable,str(conv),'--config',str(step/'config.yaml'),'--megatron-ckpt-path',str(idir),'--hf-ckpt-path',str(out)]
    if a.hf_model_name: cmd += ['--hf-model-name',a.hf_model_name]
    if a.no_strict: cmd += ['--no-strict']
    print('Running:', ' '.join(map(str,cmd)))
    if a.dry_run:return
    out.mkdir(parents=True,exist_ok=True); subprocess.run(cmd,check=True)
    expected=[out/'config.json',out/'tokenizer_config.json']
    if not expected[0].is_file(): raise SystemExit(f'Converter returned successfully but {expected[0]} is absent')
    print(f'Converted Hugging Face checkpoint: {out}')
if __name__=='__main__':main()
