#!/usr/bin/env python3
"""Compare paired baseline/RL evaluation JSONL and write Markdown + JSON reports."""
from __future__ import annotations
import argparse,json,math,random
from collections import defaultdict
from pathlib import Path

def load(p):
 d={}
 for l in p.open():
  x=json.loads(l); d[(x['row_index'],x['sample_index'])]=x
 return d
def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--baseline',type=Path,required=True); ap.add_argument('--trained',type=Path,required=True); ap.add_argument('--markdown',type=Path,required=True); ap.add_argument('--json-out',type=Path); ap.add_argument('--bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
 b,t=load(a.baseline),load(a.trained); keys=sorted(b.keys()&t.keys())
 if not keys: raise SystemExit('No paired (row_index,sample_index) records')
 fam=defaultdict(list); wins=losses=0; diffs=[]
 for k in keys:
  br,tr=float(b[k]['reward']),float(t[k]['reward']); f=str(t[k].get('family') or t[k].get('task_name') or 'unknown'); fam[f].append((br,tr)); diffs.append(tr-br); wins+=tr>br; losses+=tr<br
 rng=random.Random(a.seed); boots=[mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(a.bootstrap)]; boots.sort(); ci=[boots[int(.025*len(boots))],boots[min(len(boots)-1,int(.975*len(boots)))]]
 summary={'pairs':len(keys),'baseline_mean_reward':mean([b[k]['reward'] for k in keys]),'trained_mean_reward':mean([t[k]['reward'] for k in keys]),'delta':mean(diffs),'bootstrap_95pct_ci':ci,'baseline_to_trained_wins':wins,'regressions':losses,'ties':len(keys)-wins-losses,'families':{f:{'n':len(v),'baseline':mean([x for x,_ in v]),'trained':mean([y for _,y in v]),'delta':mean([y-x for x,y in v])} for f,v in sorted(fam.items())},'format_valid_baseline':mean([bool(b[k].get('format_valid')) for k in keys]),'format_valid_trained':mean([bool(t[k].get('format_valid')) for k in keys])}
 lines=['# Baseline vs RLVR checkpoint','',f"Paired generations: **{len(keys)}**",'', '| Family | N | Baseline reward | Trained reward | Delta |','|---|---:|---:|---:|---:|']
 for f,v in summary['families'].items(): lines.append(f"| {f} | {v['n']} | {v['baseline']:.4f} | {v['trained']:.4f} | {v['delta']:+.4f} |")
 lines += ['',f"**Overall:** {summary['baseline_mean_reward']:.4f} → {summary['trained_mean_reward']:.4f} ({summary['delta']:+.4f}; bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])",'',f"Paired wins / regressions / ties: **{wins} / {losses} / {summary['ties']}**",f"Valid answer format: **{summary['format_valid_baseline']:.2%} → {summary['format_valid_trained']:.2%}**",'', '> A changed checkpoint proves optimization occurred; held-out reward improvement is the evidence that it helped. Inspect verifier errors separately from wrong answers.']
 a.markdown.parent.mkdir(parents=True,exist_ok=True); a.markdown.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
 if a.json_out: a.json_out.write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
