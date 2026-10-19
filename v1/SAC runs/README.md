# Scalable Actor Critic for the SIS epidemic network (Qu, Wierman, Li, arXiv:1912.02906)

Primary application: the SIS epidemic of Sec. 2.2 (eq. 3-4).  `line` and `wireless` (Sec. 5) are kept as supplementary environments.

| file | role |
|---|---|
| `sis_environment.py` | epidemic networked MDP (eq. 3-4, locality eq. 1, discounted objective eq. 2) |
| `sac.py` | Algorithm 1: truncated-Q TD critic over kappa-hop neighbourhoods + policy-gradient actor |
| `run_sac.py` | **main CLI** (training, metrics, 10 snapshots per policy rollout, time-complexity study) |
| `policies.py`, `run_baselines.py`, `run_demo.py` | baseline policies, baseline evaluation, single-policy demo |
| `paper_environments.py` | supplementary Sec. 5.1 line and Sec. 5.2 wireless environments |
| `test_environment.py`, `test_sac.py` | 32 tests (`python test_sac.py` or `pytest`) |

```
python run_sac.py                                   # Algorithm 1 as written (policy zeta_i(a_i|s_i)), ring n=20
python run_sac.py --policy-radius 1 --warm-start -M 3000 --rollout-steps 200
python run_sac.py --graph-type grid -n 36 --runs 3 --kappas 0 1 2
python run_sac.py --complexity --complexity-sizes 10 20 40 80 160 --complexity-kappas 0 1 2 3
python run_sac.py --help
```
Plots: `training_sis.png` (global discounted reward + benchmarks + stationarity proxy), `complexity_vs_n.png`,
`complexity_vs_kappa.png`, `snapshots/sis/<policy>/step_*.png` (10 per rollout; every rollout_steps/10 steps,
every step if rollout_steps <= 10).  `example_run_sis/` contains one complete run.

Not done / caveats: epidemic parameters, T, h, t0, eta are not given in the paper (CLI-tunable, my defaults);
`--policy-radius 1` is the paper's Remark 1 generalisation and `--warm-start` its suggested extension (both off by
default); the optimality gap needs a known optimum (only `line`); the paper's convergence guarantee is local.
