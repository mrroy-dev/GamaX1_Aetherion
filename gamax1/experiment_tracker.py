"""Persistent experiment tracking for GamaX1/Aetherion.

Each run gets its own directory. Metrics are append-only JSONL, while plots are
regenerated from that run's metrics. This deliberately records measurements
rather than inventing a speedup: the Aetherion compute ratio is kept separate
from measured wall-clock/GPU throughput.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path

class ExperimentTracker:
    def __init__(self, base_dir="experiments", run_name=None, config=None, resume=False):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        if run_name:
            self.run_dir = self.base / run_name
        else:
            stamp = time.strftime("run_%Y%m%d_%H%M%S")
            self.run_dir = self.base / stamp
            i=2
            while self.run_dir.exists():
                self.run_dir = self.base / f"{stamp}_{i}"; i+=1
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        self.metrics_path=self.run_dir/"metrics.jsonl"
        self.timing_path=self.run_dir/"checkpoint_timing.jsonl"
        if config is not None and not resume:
            (self.run_dir/"config.json").write_text(json.dumps(config,indent=2,default=str))
        self.start_wall=time.time()
        self.last_checkpoint_step=None
        self.last_checkpoint_wall=None
        self.last_checkpoint_metrics=None

    def log(self, **metrics):
        record={"timestamp":time.time(),"elapsed_sec":time.time()-self.start_wall,**metrics}
        with self.metrics_path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,sort_keys=True,default=float)+"\n")
        return record

    def checkpoint_timing(self, step:int, checkpoint_kind="training", extra=None):
        now=time.time()
        interval_steps=None if self.last_checkpoint_step is None else step-self.last_checkpoint_step
        interval_sec=None if self.last_checkpoint_wall is None else now-self.last_checkpoint_wall
        steps_per_sec=(interval_steps/interval_sec) if interval_sec and interval_steps is not None and interval_sec>0 else None
        rec={"timestamp":now,"checkpoint_kind":checkpoint_kind,"step":step,
             "interval_steps":interval_steps,"interval_sec":interval_sec,
             "steps_per_sec":steps_per_sec}
        if extra: rec.update(extra)
        with self.timing_path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(rec,sort_keys=True,default=float)+"\n")
        self.last_checkpoint_step=step; self.last_checkpoint_wall=now
        self.last_checkpoint_metrics=rec
        return rec

    def write_summary(self, **summary):
        (self.run_dir/"summary.json").write_text(json.dumps(summary,indent=2,default=str))

    def plot(self):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            (self.run_dir/"plot_warning.txt").write_text(f"matplotlib unavailable: {exc}\n")
            return
        rows=[]
        if self.metrics_path.exists():
            for line in self.metrics_path.read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
        if not rows: return
        def series(key):
            xs=[]; ys=[]
            for r in rows:
                if key in r and r[key] is not None:
                    xs.append(r.get("step",len(xs))); ys.append(r[key])
            return xs,ys
        for key,title,ylabel in [("train_loss","Training loss","loss"),("val_loss","Validation loss","loss"),
                                 ("train_ppl","Training perplexity","PPL"),("val_ppl","Validation perplexity","PPL"),
                                 ("lr","Learning rate","LR"),("active_units_per_token","Active units/token","units")]:
            x,y=series(key)
            if not y: continue
            fig=plt.figure(figsize=(7,4)); ax=fig.add_subplot(111); ax.plot(x,y); ax.set_title(title); ax.set_xlabel("step"); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(self.run_dir/"plots"/(key+".png")); plt.close(fig)
        # Checkpoint interval timing as its own graph.
        if self.timing_path.exists():
            ts=[]
            for line in self.timing_path.read_text().splitlines():
                try:
                    r=json.loads(line)
                    if r.get("interval_sec") is not None: ts.append(r)
                except json.JSONDecodeError: pass
            if ts:
                fig=plt.figure(figsize=(7,4)); ax=fig.add_subplot(111); ax.plot([r["step"] for r in ts],[r["interval_sec"] for r in ts],marker="o"); ax.set_title("Checkpoint interval time"); ax.set_xlabel("checkpoint step"); ax.set_ylabel("seconds since previous checkpoint"); fig.tight_layout(); fig.savefig(self.run_dir/"plots"/"checkpoint_interval_seconds.png"); plt.close(fig)

def compare_runs(experiments_dir="experiments", output="comparison.json"):
    base=Path(experiments_dir); result=[]
    if not base.exists(): return result
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        summary={"run":d.name}
        sp=d/"summary.json"
        if sp.exists():
            try: summary.update(json.loads(sp.read_text()))
            except Exception: pass
        result.append(summary)
    (base/output).write_text(json.dumps(result,indent=2,default=str))
    return result
