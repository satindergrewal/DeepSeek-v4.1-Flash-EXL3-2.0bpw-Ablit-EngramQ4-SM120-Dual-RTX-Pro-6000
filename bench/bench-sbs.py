#!/usr/bin/env python3
"""Side-by-side on the REAL workload shape. Non-streaming (no client parse cap).
  PREFILL: unique uncached prompt per stream, max_tokens=1 -> wall time is prefill.
  DECODE : warm the prefix cache, then short(16)/long(400) pair -> subtraction cancels prefill."""
import json, os, random, sys, threading, time, urllib.request
K=os.environ["API_KEY"]
M=os.environ.get("SERVED_MODEL","deepseek-v4.1-flash"); CTX=int(os.environ.get("CTX","46000"))
CONC=int(os.environ.get("CONC","1")); URL=f"http://localhost:{os.environ.get('PORT','8000')}/v1/chat/completions"
W="alpha beta gamma delta epsilon zeta eta theta iota kappa system network memory kernel buffer thread packet cluster tensor matrix".split()
def prompt(seed):
    r=random.Random(seed)
    return ("Reference document:\n"+" ".join(r.choice(W) for _ in range(int(CTX*0.95)))+
            "\n\nIgnore the document above. Write a detailed technical explanation of how speculative decoding works in LLM inference.")
def call(p,nmax):
    body=json.dumps({"model":M,"max_tokens":nmax,"temperature":0,"messages":[{"role":"user","content":p}]}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json","Authorization":f"Bearer {K}"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=1800) as r: d=json.loads(r.read())
    return time.time()-t0, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]
bar=threading.Barrier(CONC); res=[None]*CONC
def worker(i):
    seed=i*7919+int(time.time())
    p=prompt(seed)
    try:
        bar.wait(); t_pf,_,ptok=call(p,1)          # uncached prefill
        call(p,16)                                  # warm
        bar.wait(); t_s,n_s,_=call(p,16); t_l,n_l,_=call(p,400)
        dec=(n_l-n_s)/(t_l-t_s) if t_l>t_s else 0.0
        res[i]=dict(prefill_s=t_pf, ptok=ptok, prefill_tps=ptok/t_pf, decode_tps=dec)
    except Exception as e:
        res[i]=dict(err=str(e)[:150])
ts=[threading.Thread(target=worker,args=(i,)) for i in range(CONC)]
[t.start() for t in ts]; [t.join() for t in ts]
ok=[r for r in res if r and "err" not in r]; bad=[r for r in res if r and "err" in r]
if bad: print(f"  ERRORS: {bad}")
if ok:
    print(f"  {M} ctx={CTX} conc={CONC}: prompt_tok={ok[0]['ptok']} | "
          f"prefill {sum(o['prefill_tps'] for o in ok):8.0f} tok/s agg ({sum(o['prefill_tps'] for o in ok)/len(ok):7.0f}/stream, TTFT {max(o['prefill_s'] for o in ok):5.1f}s) | "
          f"decode {sum(o['decode_tps'] for o in ok):6.1f} tok/s agg ({sum(o['decode_tps'] for o in ok)/len(ok):5.1f}/stream)")
