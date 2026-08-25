from __future__ import annotations
import csv, json
from pathlib import Path
import jax.numpy as jnp
def _native(x):
    if isinstance(x, dict): return {k:_native(v) for k,v in x.items()}
    if hasattr(x,"tolist"): return x.tolist()
    return x
class NISLogger:
    def __init__(self, directory): self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
    def log(self, metrics):
        row=_native(metrics)
        with (self.directory/'metrics.jsonl').open('a') as f: f.write(json.dumps(row,default=str)+'\n')
        csvpath=self.directory/'summary.csv'; exists=csvpath.exists()
        with csvpath.open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row.keys());
            if not exists: writer.writeheader()
            writer.writerow(row)
    def save_config(self, config): (self.directory/'config.json').write_text(json.dumps(_native(config),indent=2,default=str))
