"""Deterministic mathematical/mechanism invariants for Aetherion."""
from dataclasses import dataclass, asdict
import json

@dataclass
class AuditResult:
    name:str
    passed:bool
    expected:object
    observed:object
    note:str=""

def compute_ratio_vs_dense(d_model:int,n_features:int,k:int)->float:
    """Transparent local work proxy: sparse/dense = k/n_features."""
    if not (0 <= k <= n_features): raise ValueError("k must satisfy 0 <= k <= n_features")
    if n_features <= 0: raise ValueError("n_features must be positive")
    return k/n_features

def audit_k_bounds(k,n_features):
    return AuditResult("k_bounds",0<=k<=n_features,{"min":0,"max":n_features},k)

def audit_compute_ratio(d_model,n_features,k):
    expected=k/n_features; observed=compute_ratio_vs_dense(d_model,n_features,k)
    return AuditResult("compute_ratio_vs_dense",abs(expected-observed)<1e-12,expected,observed,
                       "This is a transparent work proxy, not measured GPU speedup.")

def audit_topk_mask(mask,k,n_features):
    import torch
    counts=mask.bool().sum(-1)
    expected=max(0,min(k,n_features))
    lo=int(counts.min()) if counts.numel() else 0
    hi=int(counts.max()) if counts.numel() else 0
    return AuditResult("topk_active_count",lo==expected and hi==expected,
                       expected,{"min":lo,"max":hi},
                       "Nudge/override paths can intentionally violate exact-k.")

def audit_active_fraction(mask,n_features):
    active=mask.bool().float().sum(-1).mean().item()
    observed=active/n_features
    return AuditResult("active_fraction",0<=observed<=1, "active/n_features", observed)

def run_basic_audit(d_model,n_features,k,batch_tokens=32):
    import torch
    mask=torch.zeros(batch_tokens,n_features,dtype=torch.bool)
    if k: mask[:,:k]=True
    return [audit_k_bounds(k,n_features),audit_compute_ratio(d_model,n_features,k),
            audit_topk_mask(mask,k,n_features),audit_active_fraction(mask,n_features)]

def results_to_json(results): return [asdict(x) for x in results]

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--d-model",type=int,default=64)
    p.add_argument("--n-features",type=int,default=256); p.add_argument("--k",type=int,default=64)
    a=p.parse_args(); r=run_basic_audit(a.d_model,a.n_features,a.k)
    print(json.dumps({"all_passed":all(x.passed for x in r),"results":results_to_json(r)},indent=2))
    raise SystemExit(0 if all(x.passed for x in r) else 1)
