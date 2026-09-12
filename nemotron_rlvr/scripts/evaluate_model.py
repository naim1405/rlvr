#!/usr/bin/env python3
"""Generate and verifier-score a Hugging Face model on Gym JSONL.
One model is evaluated per invocation so a baseline and RL model fit on one GPU.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'resources_servers/nemotron_verifier'))
from score import extra_env_info_from_mapping, score_response
from verifiers._util import extract_answer

def prompt_of(x):
    r=x.get('responses_create_params',{}); inp=r.get('input',[])
    for m in reversed(inp):
        if m.get('role')=='user':
            c=m.get('content','')
            if isinstance(c,str): return c
            if isinstance(c,list): return ''.join(str(i.get('text','')) if isinstance(i,dict) else str(i) for i in c)
    return str(x.get('prompt',''))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--data',required=True,type=Path); ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--limit',type=int); ap.add_argument('--samples',type=int,default=1); ap.add_argument('--temperature',type=float,default=0.0); ap.add_argument('--top-p',type=float,default=1.0); ap.add_argument('--max-new-tokens',type=int,default=1536); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--dtype',choices=['auto','bfloat16','float16','float32'],default='auto'); ap.add_argument('--trust-remote-code',action='store_true'); ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    rows=[json.loads(l) for l in a.data.open() if l.strip()]; rows=rows[:a.limit] if a.limit else rows
    dtype={'auto':'auto','bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[a.dtype]
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=a.trust_remote_code); model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype,device_map='auto',trust_remote_code=a.trust_remote_code).eval()
    done=set(); mode='a' if a.resume and a.output.exists() else 'w'
    if mode=='a':
        for l in a.output.open():
            try: z=json.loads(l); done.add((z['row_index'],z['sample_index']))
            except Exception: pass
    a.output.parent.mkdir(parents=True,exist_ok=True); good=total=0
    with a.output.open(mode,encoding='utf-8') as out:
      for i,x in enumerate(rows):
       prompt=prompt_of(x); messages=[{'role':'user','content':prompt}]
       text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) if getattr(tok,'chat_template',None) else prompt
       inputs=tok(text,return_tensors='pt').to(model.device)
       for s in range(a.samples):
        if (i,s) in done: continue
        seed=a.seed+i*1009+s; random.seed(seed); torch.manual_seed(seed); 
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        genkw={'max_new_tokens':a.max_new_tokens,'do_sample':a.temperature>0,'pad_token_id':tok.eos_token_id}
        if a.temperature>0: genkw.update(temperature=a.temperature,top_p=a.top_p)
        t=time.time()
        with torch.inference_mode(): ids=model.generate(**inputs,**genkw)
        response=tok.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        extra=extra_env_info_from_mapping(x); reward=float(score_response(response,extra)); ans=extract_answer(response)
        rec={'row_index':i,'sample_index':s,'seed':seed,'instance_id':x.get('instance_id',''),'family':x.get('family',''),'task_name':x.get('task_name',''),'prompt':prompt,'expected_answer':x.get('expected_answer'),'response':response,'extracted_answer':ans,'format_valid':ans is not None,'reward':reward,'generated_tokens':int(ids.shape[1]-inputs['input_ids'].shape[1]),'seconds':round(time.time()-t,3),'model':a.model}
        out.write(json.dumps(rec,ensure_ascii=False,default=str)+'\n'); out.flush(); total+=1; good+=reward
        print(f'[{i+1}/{len(rows)} sample {s+1}/{a.samples}] {rec["family"]} reward={reward:g} running={good/total:.3f}')
    print(f'Wrote {a.output}. New generations={total}, mean_reward={good/total if total else float("nan"):.4f}')
if __name__=='__main__':main()
