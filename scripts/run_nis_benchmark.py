"""Configuration front-end for project-specific NIS benchmarks.

The Hamiltonian/model construction remains intentionally project-owned; this
script validates and records all portable NIS knobs before a launcher imports it.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from diagnostics.nis_logging import NISLogger
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/nis/default.yaml')
 for n,t in [('system',str),('Lx',int),('Ly',int),('j2',float),('num_samples',int),('n_proposals',int),('proposal_model',str),('proposal_checkpoint',str),('target_checkpoint',str),('resample_interval',int),('ESS_threshold',float),('temperature',float),('tempering_schedule',str),('max_weight_ratio',float),('num_iterations',int),('lr',float),('optimizer',str),('diag_shift',float),('seed',int),('diagnostics_dir',str),('resample_method',str),('proposal_train_steps',int),('proposal_train_lr',float),('proposal_loss',str),('proposal_batch_size',int)]: p.add_argument('--'+n,type=t,default=None)
 for n in ['clip_weights','use_exact_probs_for_small_N','debug_mode','compare_to_mcmc','compare_to_existing_nir','allow_mcmc_fallback']: p.add_argument('--'+n,action='store_true',default=None)
 a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()) or {}; cfg.update({k:v for k,v in vars(a).items() if k!='config' and v is not None})
 NISLogger(cfg['diagnostics_dir']).save_config(cfg); print(json.dumps(cfg,indent=2))
if __name__=='__main__': main()
