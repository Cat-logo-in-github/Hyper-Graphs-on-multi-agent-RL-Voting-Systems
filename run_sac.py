#!/usr/bin/env python3
"""
run_sac.py
==========

Train and evaluate the Scalable Actor Critic of Qu, Wierman, Li (arXiv:1912.02906) from the command line.

PRIMARY APPLICATION: the SIS epidemic network of Sec. 2.2 (default --env sis).  line / wireless are the paper's
other two systems, kept as supplementary environments.

  --env sis       epidemic, eq. (3)-(4).  Benchmarks: never / always / the 4 deterministic policies of the agent's
                  own state / a neighbour-aware local heuristic.
  --policy-radius 0   zeta_i(a_i | s_i)       (paper's base model; little headroom in the epidemic)
  --policy-radius 1   zeta_i(a_i | s_{N_i})   (paper's Remark 1; kappa then matters)
  --warm-start        paper Sec. 3.2 extension (Qhat^0 = previous Qhat^T); default off = Algorithm 1 exactly
  --env line      Sec. 5.1 synthetic line (optimal J known = 1/(n(1-gamma)) -> optimality gap, Fig. 1-2)
  --env wireless  Sec. 5.2 multi-access wireless grid (benchmark: localized ALOHA, Fig. 4)

Examples
    python run_sac.py                                           # epidemic, ring n=20, kappa 0 1 2, Algorithm 1 as written
    python run_sac.py --policy-radius 1 --warm-start -M 3000 --kappas 0 1 2 --rollout-steps 200
    python run_sac.py --graph-type grid -n 36 --infection-rate 0.6 --recovery-rate 0.2 --runs 3
    python run_sac.py --env line --num-nodes 8 --kappas 0 1 2 3 --iterations 3000
    python run_sac.py --env wireless --grid-size 6 --kappas 0 1 --iterations 3000
    python run_sac.py --env sis --graph-type ring -n 20 --kappas 0 1 2 --rollout-steps 200
    python run_sac.py --env sis --complexity --complexity-sizes 10 20 40 80 160 --complexity-kappas 0 1 2 3
    python run_sac.py --help

Metrics (as in the paper)
    * global discounted reward  J(theta) = E sum_t gamma^t r(s_t, a_t)  during training (Fig. 1/4/5),
      estimated from the training trajectory of each iteration (smoothed in the plot), and for the final
      policy from --eval-episodes fresh rollouts;
    * optimality gap J* - J vs kappa (Fig. 2, environments with a known optimum);
    * benchmark policies as dotted lines;
    * (extra) running estimate of sum_m eta_m |g(m)|^2 / sum_m eta_m, the quantity bounded by Theorem 4.

Plots
    training_<env>.png, optimality_gap_<env>.png, complexity_*.png, and for every policy (each trained kappa,
    each benchmark) 10 snapshots of an environment rollout of --rollout-steps steps, one every
    rollout_steps/10 steps (every step if rollout_steps <= 10), in snapshots/<env>/<policy>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paper_environments import LineEnv, WirelessEnv  # noqa: E402
from sac import (ScalableActorCritic, estimate_value, rollout_snapshots, sac_act_fn,  # noqa: E402
                 snapshot_steps)
from sis_environment import GRAPH_TYPES, SISGraphEnv  # noqa: E402

DEFAULT_NODES = {"sis": 20, "line": 8}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scalable Actor Critic (Qu, Wierman, Li 2019/2021).",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    e = p.add_argument_group("environment")
    e.add_argument("--env", choices=["line", "wireless", "sis"], default="sis",
                   help="sis = the paper's epidemic application (primary); line/wireless = supplementary Sec. 5 systems")
    e.add_argument("-n", "--num-nodes", type=int, default=None, help="agents (line: 8, sis: 20)")
    e.add_argument("--grid-size", type=int, default=6, help="wireless: users form a grid-size x grid-size grid")
    e.add_argument("--deadline", type=int, default=2, help="wireless: packet deadline d_i")
    e.add_argument("--graph-type", choices=GRAPH_TYPES, default="ring", help="sis topology")
    e.add_argument("--random-edge-prob", type=float, default=0.2)
    e.add_argument("--infection-rate", type=float, default=0.15, help="sis: beta(0)")
    e.add_argument("--recovery-rate", type=float, default=0.25, help="sis: delta")
    e.add_argument("--intervention-effect", type=float, default=0.8, help="sis: beta reduction at strongest action")
    e.add_argument("--intervention-cost", type=float, default=0.2, help="sis: cost of strongest action")
    e.add_argument("--num-actions", type=int, default=2, help="sis: |A_i|")
    e.add_argument("--initial-infection-prob", type=float, default=0.2, help="sis: pi_0")
    e.add_argument("--gamma", type=float, default=0.7, help="discount factor (paper: 0.7)")
    a = p.add_argument_group("SAC (Algorithm 1)")
    a.add_argument("--kappas", type=int, nargs="+", default=None, help="truncation radii (line: 0 1 2 3, else 0 1 2)")
    a.add_argument("-M", "--iterations", type=int, default=2000, help="outer iterations M")
    a.add_argument("-T", "--horizon", type=int, default=50, help="trajectory length T per outer iteration")
    a.add_argument("--critic-h", type=float, default=5.0, help="critic step alpha_t = h/(t+t0)")
    a.add_argument("--critic-t0", type=float, default=10.0, help="critic step alpha_t = h/(t+t0)")
    a.add_argument("--actor-eta", type=float, default=2.0, help="actor step eta_m = eta/sqrt(m+1)")
    a.add_argument("--warm-start", action="store_true",
                   help="initialise each critic from the previous iteration (extension named in paper Sec. 3.2; default: off = Algorithm 1)")
    a.add_argument("--policy-radius", type=int, default=1,
                   help="0: zeta_i(a_i|s_i) (paper base model); 1: zeta_i(a_i|s_{N_i}) (paper Remark 1)")
    a.add_argument("--runs", type=int, default=1, help="independent training runs per kappa")
    a.add_argument("--seed", type=int, default=0)
    v = p.add_argument_group("evaluation / plots")
    v.add_argument("--eval-episodes", type=int, default=100, help="rollouts for final J and benchmark J")
    v.add_argument("--rollout-steps", type=int, default=100,
                   help="length of the rollouts that are plotted (10 snapshots: every steps/10 steps)")
    v.add_argument("--num-plots", type=int, default=10, help="snapshots per rollout")
    v.add_argument("--aloha-send-prob", type=float, default=0.5, help="wireless benchmark send probability")
    v.add_argument("--no-baselines", action="store_true")
    v.add_argument("--no-snapshots", action="store_true")
    v.add_argument("--output-dir", default="sac_results")
    v.add_argument("--log-every", type=int, default=0, help="progress print interval (default M/10)")
    c = p.add_argument_group("time-complexity study")
    c.add_argument("--complexity", action="store_true", help="run the time-complexity study (and plots)")
    c.add_argument("--complexity-sizes", type=int, nargs="+", default=[10, 20, 40, 80, 160],
                   help="node counts (wireless: rounded to a square grid)")
    c.add_argument("--complexity-kappas", type=int, nargs="+", default=[0, 1, 2, 3])
    c.add_argument("--complexity-iters", type=int, default=15, help="timed outer iterations per point")
    c.add_argument("--complexity-fixed-kappa", type=int, default=1, help="kappa used in the time-vs-n study")
    c.add_argument("--complexity-fixed-n", type=int, default=None, help="n used in the time-vs-kappa study")
    c.add_argument("--only-complexity", action="store_true", help="skip training/evaluation")
    return p


def make_env(args: argparse.Namespace, num_nodes: Optional[int] = None, seed: Optional[int] = None) -> Any:
    ms = max(args.horizon, args.rollout_steps) + 1
    seed = args.seed if seed is None else seed
    if args.env == "line":
        return LineEnv(num_nodes or args.num_nodes or DEFAULT_NODES["line"], args.gamma, ms, seed)
    if args.env == "wireless":
        side = args.grid_size if num_nodes is None else max(2, int(round(np.sqrt(num_nodes))))
        return WirelessEnv(side, args.deadline, args.gamma, ms, seed)
    return SISGraphEnv(num_nodes=num_nodes or args.num_nodes or DEFAULT_NODES["sis"], graph_type=args.graph_type,
                       observation_radius=0,  # policies are zeta_i(a_i | s_i), as in the paper's base model
                       max_steps=ms, gamma=args.gamma, infection_rate=args.infection_rate,
                       recovery_rate=args.recovery_rate, intervention_effect=args.intervention_effect,
                       intervention_cost=args.intervention_cost, num_actions=args.num_actions,
                       initial_infection_prob=args.initial_infection_prob,
                       random_edge_prob=args.random_edge_prob, seed=seed)


def benchmark_policies(env: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """name -> act_fn(env, state, rng)."""
    uniform = lambda e, s, rng: rng.integers(0, e.num_actions, size=e.num_nodes)  # noqa: E731
    out: Dict[str, Any] = {"uniform_random": uniform}
    if args.env == "line":
        out["optimal_all_ones"] = lambda e, s, rng: np.ones(e.num_nodes, dtype=np.int64)
    elif args.env == "wireless":
        out["aloha"] = lambda e, s, rng: e.aloha_actions(rng, args.aloha_send_prob)
    else:
        top = env.num_actions - 1
        adj = env.get_adjacency()
        out["never"] = lambda e, s, rng: np.zeros(e.num_nodes, dtype=np.int64)
        out["always"] = lambda e, s, rng: np.full(e.num_nodes, top, dtype=np.int64)
        for a_s in sorted({0, top}):            # deterministic policies of the agent's OWN state only
            for a_i in sorted({0, top}):
                out[f"own_state[S->{a_s},I->{a_i}]"] = (lambda e, s, rng, a_s=a_s, a_i=a_i:
                                                        np.where(s == 0, a_s, a_i).astype(np.int64))
        out["local_heuristic(sees neighbours)"] = lambda e, s, rng: np.where((s + adj @ s) > 0, top, 0).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training(args, curves: Dict[int, Dict[str, np.ndarray]], bench: Dict[str, float], optimum, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    win = max(1, args.iterations // 50)
    for kappa, hist in curves.items():
        J = np.atleast_2d(hist["J"])                       # (runs, M)
        m, s = smooth(J.mean(axis=0), win), smooth(J.std(axis=0), win)
        x = np.arange(len(m)) + win
        axes[0].plot(x, m, label=f"SAC κ={kappa}")
        if J.shape[0] > 1:
            axes[0].fill_between(x, m - s, m + s, alpha=0.15)
        G, E = np.atleast_2d(hist["grad_sq"]), np.atleast_2d(hist["eta"])
        axes[1].plot(np.cumsum(E * G, axis=1).mean(axis=0) / np.cumsum(E, axis=1).mean(axis=0), label=f"κ={kappa}")
    for name, val in bench.items():
        axes[0].axhline(val, ls=":", lw=1.2, color="red" if name in ("aloha", "optimal_all_ones") else "gray")
        axes[0].text(axes[0].get_xlim()[1], val, f" {name}", fontsize=7, va="bottom", ha="right")
    if optimum is not None:
        axes[0].axhline(optimum, ls="--", color="k", lw=1, label="optimal")
    axes[0].set_xlabel("training iteration m")
    axes[0].set_ylabel("global discounted reward")
    axes[0].set_title(f"Training curves ({args.env}, moving avg. {win})")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("training iteration m")
    axes[1].set_ylabel("Σ η_m |ĝ(m)|² / Σ η_m")
    axes[1].set_yscale("log")
    axes[1].set_title("Stationarity proxy (cf. Theorem 4; ĝ is a noisy estimate)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gap(summary: Dict[int, Dict[str, float]], path: str) -> None:
    ks = sorted(summary)
    gap = [max(summary[k]["gap"], 1e-6) for k in ks]
    err = [summary[k]["gap_sem"] for k in ks]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.errorbar(ks, gap, yerr=err, marker="o", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel("κ")
    ax.set_ylabel("optimality gap J* − J(θ)")
    ax.set_title("Optimality gap vs κ")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0]) if len(x) > 1 and y.min() > 0 else None


def run_complexity(args, outdir: str) -> Dict[str, Any]:
    """Time per outer iteration vs n (fixed kappa) and vs kappa (fixed n), split into simulate/critic/actor."""
    def measure(n, kappa):
        env = make_env(args, num_nodes=n)
        sac = ScalableActorCritic(env, kappa, args.horizon, args.critic_h, args.critic_t0, args.actor_eta, args.seed, args.warm_start, args.policy_radius)
        sac.iterate()  # warm-up
        recs = [sac.iterate() for _ in range(args.complexity_iters)]
        mean = lambda k: float(np.mean([r[k] for r in recs]))  # noqa: E731
        return {"n": env.num_nodes, "kappa": kappa, "f_kappa": sac.f_kappa,
                "t_sim": mean("t_sim"), "t_critic": mean("t_critic"), "t_actor": mean("t_actor"),
                "t_total": mean("t_sim") + mean("t_critic") + mean("t_actor"),
                "q_entries_max": mean("q_entries_max"),
                "q_bound": float((sac.S * sac.A) ** sac.f_kappa)}

    k0, kk = args.complexity_fixed_kappa, args.complexity_kappas
    sizes = sorted(set(args.complexity_sizes))
    n_fixed = args.complexity_fixed_n or sizes[len(sizes) // 2]
    vs_n = [measure(n, k0) for n in sizes]
    vs_k = [measure(n_fixed, k) for k in kk]
    ns = [r["n"] for r in vs_n]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for key, lab in (("t_total", "total"), ("t_sim", "simulation"), ("t_critic", "critic (TD)"), ("t_actor", "actor")):
        y = [r[key] for r in vs_n]
        ax.loglog(ns, y, marker="o", label=f"{lab} (slope {_fit(ns, y):.2f})" if _fit(ns, y) is not None else lab)
    ax.set_xlabel("number of agents n")
    ax.set_ylabel("seconds per outer iteration")
    ax.set_title(f"Time per SAC iteration vs n  ({args.env}, κ={k0}, T={args.horizon})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "complexity_vs_n.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fk = [r["f_kappa"] for r in vs_k]
    axes[0].semilogy(kk, [r["t_total"] for r in vs_k], marker="o", label="total")
    axes[0].semilogy(kk, [r["t_critic"] for r in vs_k], marker="s", label="critic (TD)")
    axes[0].semilogy(kk, [r["t_actor"] for r in vs_k], marker="^", label="actor")
    axes[0].set_xlabel("κ")
    axes[0].set_ylabel("seconds per outer iteration")
    axes[0].set_title(f"Time vs κ  (n={vs_k[0]['n']})")
    axes[1].semilogy(kk, [r["q_entries_max"] for r in vs_k], marker="o", label="visited Q̂ entries (max over agents)")
    axes[1].semilogy(kk, [r["q_bound"] for r in vs_k], marker="x", ls="--", label="(|S||A|)^f(κ)  (worst-case table size)")
    axes[1].set_xlabel("κ")
    axes[1].set_title("Truncated Q-table size vs κ")
    for a_ in axes:
        a_.grid(alpha=0.3, which="both")
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "complexity_vs_kappa.png"), dpi=120)
    plt.close(fig)

    res = {"vs_n": vs_n, "vs_kappa": vs_k, "fixed_kappa": k0, "fixed_n": vs_k[0]["n"],
           "slopes_vs_n": {k: _fit(ns, [r[k] for r in vs_n]) for k in ("t_total", "t_sim", "t_critic", "t_actor")},
           "f_kappa": fk}
    print("\nTime complexity (seconds per outer iteration)")
    print(f"  vs n at kappa={k0}:  " + ", ".join(f"n={r['n']}: {r['t_total'] * 1e3:.1f} ms" for r in vs_n))
    print("  fitted exponent in n:", {k: (None if v is None else round(v, 2)) for k, v in res["slopes_vs_n"].items()})
    print(f"  vs kappa at n={res['fixed_n']}:  " + ", ".join(
        f"κ={r['kappa']} (f={r['f_kappa']}): {r['t_total'] * 1e3:.1f} ms" for r in vs_k))
    return res


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.kappas is None:
        args.kappas = [0, 1, 2, 3] if args.env == "line" else [0, 1, 2]
    if args.iterations < 1 or args.runs < 1 or args.horizon < 1 or args.rollout_steps < 1:
        parser.error("iterations, runs, horizon and rollout-steps must be >= 1")
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)
    results: Dict[str, Any] = {"config": vars(args)}

    if not args.only_complexity:
        env = make_env(args)
        optimum = getattr(env, "optimal_value", None)
        print(f"env={args.env} n={env.num_nodes} |S|={getattr(env, 'num_states', 2)} |A|={env.num_actions} "
              f"gamma={args.gamma} | M={args.iterations} T={args.horizon} kappas={args.kappas} runs={args.runs}")

        # ---- benchmarks ---------------------------------------------------
        benches = {} if args.no_baselines else benchmark_policies(env, args)
        bench_val = {}
        for name, fn in benches.items():
            bench_val[name] = estimate_value(env, fn, args.eval_episodes, args.horizon, seed=args.seed + 999)
        # ---- training -----------------------------------------------------
        curves, summary, thetas = {}, {}, {}
        log_every = args.log_every or max(1, args.iterations // 10)
        for kappa in args.kappas:
            hists, finals, times = [], [], []
            for run in range(args.runs):
                renv = make_env(args, seed=args.seed + 1000 * run)
                sac = ScalableActorCritic(renv, kappa, args.horizon, args.critic_h, args.critic_t0,
                                          args.actor_eta, seed=args.seed + 1000 * run + kappa,
                                          warm_start=args.warm_start, policy_radius=args.policy_radius)
                t0 = time.perf_counter()
                hist = sac.train(args.iterations, callback=(lambda m, r, k=kappa, rn=run: print(
                    f"  κ={k} run {rn}  iter {m:6d}  J(traj)={r['J']:.4f}") if m % log_every == 0 else None))
                times.append(time.perf_counter() - t0)
                hists.append(hist)
                finals.append(estimate_value(renv, sac.act_fn(), args.eval_episodes, args.horizon,
                                             seed=args.seed + 777 + run)["mean"])
                if run == 0:
                    thetas[kappa] = sac.act_fn()
            curves[kappa] = {k: np.stack([h[k] for h in hists]) for k in hists[0]}
            fin = np.asarray(finals)
            summary[kappa] = {"J_mean": float(fin.mean()),
                              "J_sem": float(fin.std(ddof=1) / np.sqrt(len(fin))) if len(fin) > 1 else 0.0,
                              "train_seconds": float(np.mean(times)),
                              "q_entries_mean": float(curves[kappa]["q_entries"].mean()),
                              "f_kappa": int(sac.f_kappa)}
            if optimum is not None:
                summary[kappa]["gap"] = float(optimum - fin.mean())
                summary[kappa]["gap_sem"] = summary[kappa]["J_sem"]

        # ---- report -------------------------------------------------------
        print(f"\n{'policy':<22} | {'J (final, mean±sem)':<22}" + (" | gap to optimum" if optimum else ""))
        for name, v in bench_val.items():
            print(f"{'benchmark:' + name:<22} | {v['mean']:.4f} ± {v['sem']:.4f}")
        for kappa, s in summary.items():
            print(f"{'SAC kappa=' + str(kappa):<22} | {s['J_mean']:.4f} ± {s['J_sem']:.4f}" +
                  (f"        | {s['gap']:.4f}" if optimum else "") + f"   [{s['train_seconds']:.1f}s, f(κ)={s['f_kappa']}]")
        if optimum is not None:
            print(f"optimal J* = {optimum:.4f}")

        plot_training(args, curves, {k: v["mean"] for k, v in bench_val.items()}, optimum,
                      os.path.join(outdir, f"training_{args.env}.png"))
        if optimum is not None and len(summary) > 1:
            plot_gap(summary, os.path.join(outdir, f"optimality_gap_{args.env}.png"))

        # ---- 10 snapshots per policy rollout ------------------------------
        if not args.no_snapshots:
            policies = {f"SAC_kappa{k}": fn for k, fn in thetas.items()}
            policies.update({f"benchmark_{k}": v for k, v in benches.items()})
            venv = make_env(args)
            width = len(str(args.rollout_steps))
            for label, fn in policies.items():
                d = os.path.join(outdir, "snapshots", args.env, label)
                os.makedirs(d, exist_ok=True)

                def save(t, label=label, d=d):
                    fig, ax = plt.subplots(figsize=(5.5, 5.5))
                    venv.render(ax=ax)
                    ax.set_title(f"{label} | {ax.get_title()}", fontsize=9)
                    fig.savefig(os.path.join(d, f"step_{t:0{width}d}.png"), dpi=110, bbox_inches="tight")
                    plt.close(fig)

                taken = rollout_snapshots(venv, fn, args.rollout_steps, args.seed + 4242, save, args.num_plots)
            print(f"snapshots: {len(taken)} per policy at steps {taken} -> {outdir}/snapshots/{args.env}/")

        results.update({"benchmarks": bench_val, "optimal": optimum,
                        "summary": {str(k): v for k, v in summary.items()},
                        "curves": {str(k): {m: h.tolist() for m, h in c.items() if m in ("J", "grad_sq", "eta")}
                                   for k, c in curves.items()}})

    if args.complexity or args.only_complexity:
        results["complexity"] = run_complexity(args, outdir)

    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nresults.json and plots written to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
