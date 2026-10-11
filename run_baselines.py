#!/usr/bin/env python3
"""
run_baselines.py
================

Evaluate baseline policies on the multi-agent SIS environment
(Qu, Wierman, Li, arXiv:1912.02906, Sec. 2.2) entirely from the command line.

Examples
--------
    python run_baselines.py                                    # defaults: ring, n=20, 10 episodes
    python run_baselines.py -n 50 --graph-type grid --episodes 30 --policies never local
    python run_baselines.py --graph-type random --random-edge-prob 0.1 --observation-radius 2
    python run_baselines.py --num-actions 3 --heuristic-action 1 --gamma 0.9
    python run_baselines.py --scaling-sizes 20 40 80 160 320    # empirical time-complexity study
    python run_baselines.py --help

What is reported
----------------
* Discounted return  sum_t gamma^t r(s_t, a_t)  -- a Monte-Carlo estimate of the
  paper's objective J(theta) (eq. 2).  The undiscounted episode sum is reported too.
* Average / peak infected fraction, mean intervention rate.
* Timing: wall-clock per `policy.act`, per `env.step`, per episode (perf_counter).
* Optional (`--scaling-sizes`): per-step time vs. number of nodes and the fitted
  log-log slope (empirical exponent of n).

Plots (unless --no-plots)
-------------------------
For every policy, `--num-plots` (default 10) graph snapshots are written during
the evaluated episode `--viz-episode`, one every max_steps/num_plots steps
(max_steps=100 -> steps 10, 20, ..., 100), plus a montage of them, plus one
comparison plot of the infected fraction over time for all policies.

Fairness: episode k uses seed (--seed + k) for the environment, so all policies
see the same initial states and the same random-number stream (common random
numbers).  For --graph-type random each episode draws its own G(n, p) graph
(again identical across policies).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: we only write files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from policies import POLICY_NAMES, Policy, make_policy  # noqa: E402
from sis_environment import GRAPH_TYPES, SISGraphEnv  # noqa: E402


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    """Every environment knob, shared by run_baselines.py and run_demo.py."""
    g = parser.add_argument_group("environment")
    g.add_argument("-n", "--num-nodes", type=int, default=20, help="number of agents (default 20)")
    g.add_argument("--graph-type", choices=GRAPH_TYPES, default="ring", help="topology (default ring)")
    g.add_argument("--random-edge-prob", type=float, default=0.2, help="edge prob. for graph-type random")
    g.add_argument("--observation-radius", type=int, default=1,
                   help="agents observe states within this many hops (default 1; 0 = own state only)")
    g.add_argument("--max-steps", type=int, default=100, help="episode length (default 100)")
    g.add_argument("--gamma", type=float, default=0.7, help="discount factor (paper: 0.7)")
    g.add_argument("--infection-rate", type=float, default=0.5, help="beta(0)")
    g.add_argument("--recovery-rate", type=float, default=0.3, help="delta")
    g.add_argument("--intervention-effect", type=float, default=0.8,
                   help="relative reduction of beta at the strongest action")
    g.add_argument("--intervention-cost", type=float, default=0.2, help="cost of the strongest action")
    g.add_argument("--num-actions", type=int, default=2, help="size A of the local action set")
    g.add_argument("--initial-infection-prob", type=float, default=0.2, help="P(node starts infected)")
    g.add_argument("--terminate-when-extinct", action="store_true",
                   help="end the episode once no node is infected")
    g.add_argument("--seed", type=int, default=0, help="base seed; episode k uses seed + k")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("plots / output")
    g.add_argument("--num-plots", type=int, default=10,
                   help="graph snapshots per policy, evenly spaced over the episode (default 10)")
    g.add_argument("--viz-episode", type=int, default=0, help="which evaluated episode is plotted (default 0)")
    g.add_argument("--no-plots", action="store_true", help="skip all plots")
    g.add_argument("--output-dir", default="results", help="where plots / results.json go")
    g.add_argument("--quiet", action="store_true", help="less console output")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate baseline policies on the SIS networked environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_env_arguments(p)
    g = p.add_argument_group("evaluation")
    g.add_argument("--policies", nargs="+", choices=list(POLICY_NAMES) + ["all"], default=["all"],
                   help="policies to evaluate")
    g.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
    g.add_argument("--policy-seed", type=int, default=42, help="seed of the random policy")
    g.add_argument("--heuristic-action", type=int, default=None,
                   help="action level used by the local heuristic (default: strongest)")
    g.add_argument("--scaling-sizes", type=int, nargs="+", default=None, metavar="N",
                   help="run the empirical time-complexity study over these node counts")
    g.add_argument("--scaling-episodes", type=int, default=3, help="episodes per size in the scaling study")
    add_run_arguments(p)
    return p


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, keys: Sequence[str]) -> None:
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.num_plots < 0:
        parser.error("--num-plots must be >= 0")
    if "local" in keys and args.observation_radius < 1:
        parser.error("the 'local' policy needs --observation-radius >= 1")
    if not 0 <= args.viz_episode < args.episodes:
        parser.error("--viz-episode must be in [0, episodes-1]")


def resolve_policy_keys(requested: Sequence[str]) -> List[str]:
    if "all" in requested:
        return list(POLICY_NAMES)
    return list(dict.fromkeys(requested))  # unique, keep order


def make_env(args: argparse.Namespace, seed: int, num_nodes: Optional[int] = None) -> SISGraphEnv:
    return SISGraphEnv(
        num_nodes=num_nodes if num_nodes is not None else args.num_nodes,
        graph_type=args.graph_type,
        observation_radius=args.observation_radius,
        max_steps=args.max_steps,
        gamma=args.gamma,
        infection_rate=args.infection_rate,
        recovery_rate=args.recovery_rate,
        intervention_effect=args.intervention_effect,
        intervention_cost=args.intervention_cost,
        num_actions=args.num_actions,
        initial_infection_prob=args.initial_infection_prob,
        terminate_when_extinct=args.terminate_when_extinct,
        random_edge_prob=args.random_edge_prob,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def snapshot_steps(max_steps: int, num_plots: int) -> List[int]:
    """Steps at which snapshots are taken: max_steps=100, num_plots=10 -> 10, 20, ..., 100."""
    if num_plots <= 0:
        return []
    return sorted({max(1, int(round(max_steps * k / num_plots))) for k in range(1, num_plots + 1)})


def _title(env: SISGraphEnv, label: str) -> str:
    return f"{label}  |  t = {env.t}, infected {int(env.get_state().sum())}/{env.num_nodes}"


def save_snapshot(env: SISGraphEnv, label: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax=ax)
    ax.set_title(_title(env, label), fontsize=10)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Episode / policy evaluation
# --------------------------------------------------------------------------- #
def run_episode(
    env: SISGraphEnv,
    policy: Policy,
    seed: int,
    *,
    snap_steps: Sequence[int] = (),
    out_dir: Optional[str] = None,
    print_every: int = 0,
) -> Dict[str, Any]:
    """Run one episode; time `policy.act` and `env.step` separately (plotting is NOT timed)."""
    obs, info = env.reset(seed=seed)
    policy.reset()

    fractions = [info["fraction_infected"]]  # s_0 .. s_T
    intervention_rates: List[float] = []
    act_ns: List[int] = []
    step_ns: List[int] = []
    saved: List[str] = []
    snap_set = set(snap_steps)
    width = len(str(env.max_steps))

    montage = None
    if out_dir is not None and snap_set:
        os.makedirs(out_dir, exist_ok=True)
        cols = math.ceil(len(snap_steps) / 2)
        fig_m, axes = plt.subplots(2 if len(snap_steps) > 1 else 1, cols, figsize=(3.2 * cols, 6.4), squeeze=False)
        montage = (fig_m, list(axes.ravel()))

    done = False
    while not done:
        t0 = time.perf_counter_ns()
        actions = policy.act(env, obs)
        t1 = time.perf_counter_ns()
        obs, _, term, trunc, info = env.step(actions)
        t2 = time.perf_counter_ns()
        act_ns.append(t1 - t0)
        step_ns.append(t2 - t1)
        fractions.append(info["fraction_infected"])
        intervention_rates.append(info["intervention_rate"])
        done = term or trunc

        if print_every and (env.t - 1) % print_every == 0:
            print(f"  step {env.t - 1:4d} | infected fraction {info['fraction_infected']:.2f} "
                  f"| mean reward {info['global_reward']:.3f}")

        if out_dir is not None and env.t in snap_set:
            path = os.path.join(out_dir, f"step_{env.t:0{width}d}.png")
            save_snapshot(env, policy.name, path)
            saved.append(path)
            if montage is not None:
                ax = montage[1][len(saved) - 1]
                env.render(ax=ax)
                ax.set_title(f"t = {env.t}", fontsize=9)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

    if montage is not None:
        fig_m, axes_list = montage
        for ax in axes_list[len(saved):]:
            ax.set_axis_off()
        fig_m.suptitle(f"{policy.name}: infection spread over time (blue = S, red = I)")
        fig_m.tight_layout()
        fig_m.savefig(os.path.join(out_dir, "montage.png"), dpi=120)
        plt.close(fig_m)

    m = env.get_metrics()
    return {
        "discounted_return": m["discounted_return"],
        "total_reward": m["cumulative_reward"],
        "mean_infection": float(np.mean(fractions[:-1])),  # states s_0..s_{T-1} (those that earned reward)
        "peak_infection": float(np.max(fractions)),
        "mean_intervention_rate": float(np.mean(intervention_rates)),
        "steps": env.t,
        "trajectory": fractions,
        "act_ns": act_ns,
        "step_ns": step_ns,
        "snapshots": saved,
    }


def _stat(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=float)
    sem = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "sem": sem}


def _timing(act_ns: Sequence[int], step_ns: Sequence[int], episodes: int, n: int) -> Dict[str, float]:
    act_us = np.asarray(act_ns, dtype=float) / 1e3
    step_us = np.asarray(step_ns, dtype=float) / 1e3
    total_s = float(act_us.sum() + step_us.sum()) / 1e6
    return {
        "act_us_per_step_mean": float(act_us.mean()),
        "act_us_per_step_std": float(act_us.std()),
        "act_us_per_step_p95": float(np.percentile(act_us, 95)),
        "env_step_us_per_step_mean": float(step_us.mean()),
        "env_step_us_per_step_std": float(step_us.std()),
        "env_step_us_per_step_p95": float(np.percentile(step_us, 95)),
        "us_per_agent_step": float((act_us + step_us).mean() / n),
        "seconds_per_episode": total_s / episodes,
        "steps_per_second": float(len(act_us) / total_s) if total_s > 0 else float("inf"),
    }


def evaluate_policy(
    args: argparse.Namespace,
    key: str,
    *,
    episodes: Optional[int] = None,
    plot_dir: Optional[str] = None,
    print_every: int = 0,
) -> Dict[str, Any]:
    """Evaluate one policy over several episodes; returns aggregate statistics."""
    episodes = args.episodes if episodes is None else episodes
    policy = make_policy(key, seed=args.policy_seed, heuristic_action=args.heuristic_action)
    snaps = snapshot_steps(args.max_steps, args.num_plots) if plot_dir else []

    runs: List[Dict[str, Any]] = []
    for ep in range(episodes):
        env = make_env(args, seed=args.seed + ep)
        viz = plot_dir is not None and ep == args.viz_episode
        runs.append(run_episode(env, policy, seed=args.seed + ep,
                                snap_steps=snaps if viz else (), out_dir=plot_dir if viz else None,
                                print_every=print_every if ep == 0 else 0))

    n = args.num_nodes
    horizon = max(len(r["trajectory"]) for r in runs)
    padded = np.array([r["trajectory"] + [r["trajectory"][-1]] * (horizon - len(r["trajectory"])) for r in runs])
    return {
        "policy_key": key,
        "policy_name": policy.name,
        "episodes": episodes,
        "discounted_return": _stat([r["discounted_return"] for r in runs]),
        "total_reward": _stat([r["total_reward"] for r in runs]),
        "mean_infection": _stat([r["mean_infection"] for r in runs]),
        "peak_infection": _stat([r["peak_infection"] for r in runs]),
        "mean_intervention_rate": _stat([r["mean_intervention_rate"] for r in runs]),
        "timing": _timing([x for r in runs for x in r["act_ns"]], [x for r in runs for x in r["step_ns"]],
                          episodes, n),
        "snapshots": [p for r in runs for p in r["snapshots"]],
        "_trajectories": padded,  # stripped before JSON
    }


# --------------------------------------------------------------------------- #
# Empirical time-complexity study
# --------------------------------------------------------------------------- #
def run_scaling(args: argparse.Namespace, keys: Sequence[str], sizes: Sequence[int]) -> Dict[str, Any]:
    """Per-step wall time as a function of n; slope = fitted exponent in time ~ n^slope."""
    out: Dict[str, Any] = {"sizes": list(sizes), "policies": {}, "env_init_seconds": []}
    for n in sizes:  # environment construction (graph, adjacency, all-pairs distances)
        t0 = time.perf_counter()
        make_env(args, seed=args.seed, num_nodes=n)
        out["env_init_seconds"].append(time.perf_counter() - t0)

    for key in keys:
        act_us, step_us = [], []
        for n in sizes:
            sargs = argparse.Namespace(**vars(args))
            sargs.num_nodes = n
            res = evaluate_policy(sargs, key, episodes=args.scaling_episodes)
            act_us.append(res["timing"]["act_us_per_step_mean"])
            step_us.append(res["timing"]["env_step_us_per_step_mean"])
        out["policies"][key] = {
            "act_us_per_step": act_us,
            "env_step_us_per_step": step_us,
            "act_exponent": _loglog_slope(sizes, act_us),
            "env_step_exponent": _loglog_slope(sizes, step_us),
        }
    out["env_init_exponent"] = _loglog_slope(sizes, out["env_init_seconds"])
    return out


def _loglog_slope(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 2 or min(y) <= 0:
        return None
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


def plot_scaling(scaling: Dict[str, Any], path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sizes = scaling["sizes"]
    for key, d in scaling["policies"].items():
        axes[0].loglog(sizes, d["act_us_per_step"], marker="o", label=f"{key} (slope {_fmt(d['act_exponent'])})")
        axes[1].loglog(sizes, d["env_step_us_per_step"], marker="o", label=f"{key} (slope {_fmt(d['env_step_exponent'])})")
    axes[0].set_title("policy.act: time per step")
    axes[1].set_title("env.step: time per step")
    for ax in axes:
        ax.set_xlabel("number of nodes n")
        ax.set_ylabel("microseconds")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(results: List[Dict[str, Any]], gamma: float) -> None:
    pm = lambda s, d=3: f"{s['mean']:.{d}f}±{s['sem']:.{d}f}"  # noqa: E731
    head = (f"{'Policy':<17} | {'J=Σγ^t r':<14} | {'Σ r (undisc.)':<15} | {'Avg infected':<14} | "
            f"{'Peak infected':<14} | {'Interv. rate':<13} | {'act µs/step':<12} | {'env µs/step':<12}")
    print(f"\nMean ± s.e.m. over episodes (gamma = {gamma})")
    print(head)
    print("-" * len(head))
    for r in results:
        t = r["timing"]
        print(f"{r['policy_name']:<17} | {pm(r['discounted_return']):<14} | {pm(r['total_reward'], 2):<15} | "
              f"{pm(r['mean_infection']):<14} | {pm(r['peak_infection']):<14} | "
              f"{pm(r['mean_intervention_rate']):<13} | {t['act_us_per_step_mean']:<12.1f} | "
              f"{t['env_step_us_per_step_mean']:<12.1f}")


def print_scaling(scaling: Dict[str, Any]) -> None:
    sizes = scaling["sizes"]
    print("\nEmpirical time complexity (time per step vs. number of nodes n)")
    print(f"{'n':>8} | " + " | ".join(f"{k + ' act µs':>16} {k + ' env µs':>16}" for k in scaling["policies"]))
    for i, n in enumerate(sizes):
        row = " | ".join(f"{d['act_us_per_step'][i]:>16.1f} {d['env_step_us_per_step'][i]:>16.1f}"
                         for d in scaling["policies"].values())
        print(f"{n:>8} | {row}")
    print("fitted exponents (time ~ n^slope):")
    for k, d in scaling["policies"].items():
        print(f"  {k:<8} act: {_fmt(d['act_exponent'])}   env.step: {_fmt(d['env_step_exponent'])}")
    print(f"  env construction (graph + adjacency + all-pairs distances): {_fmt(scaling['env_init_exponent'])}")


def plot_curves(results: List[Dict[str, Any]], path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in results:
        traj = r["_trajectories"]
        mean, std = traj.mean(axis=0), traj.std(axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=r["policy_name"])
        ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.15)
    ax.set_xlabel("timestep")
    ax.set_ylabel("infected fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_evaluation(args: argparse.Namespace, keys: Sequence[str], print_every: int = 0) -> List[Dict[str, Any]]:
    results = []
    for key in keys:
        plot_dir = None if args.no_plots else os.path.join(args.output_dir, key)
        if not args.quiet:
            print(f"Evaluating '{key}' ...")
        results.append(evaluate_policy(args, key, plot_dir=plot_dir, print_every=print_every))
    return results


def save_results(args: argparse.Namespace, results: List[Dict[str, Any]], scaling: Optional[Dict[str, Any]]) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    payload = {"config": vars(args), "results": clean}
    if scaling is not None:
        payload["scaling"] = scaling
    path = os.path.join(args.output_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    keys = resolve_policy_keys(args.policies)
    validate_args(parser, args, keys)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.quiet:
        print(f"Config: n={args.num_nodes}, graph={args.graph_type}, radius={args.observation_radius}, "
              f"steps={args.max_steps}, gamma={args.gamma}, episodes={args.episodes}, seed={args.seed}")

    results = run_evaluation(args, keys)
    print_table(results, args.gamma)

    if not args.no_plots:
        plot_curves(results, os.path.join(args.output_dir, "infection_curves.png"),
                    f"Infected fraction (mean ± std over {args.episodes} episodes) – {args.graph_type}, n={args.num_nodes}")

    scaling = None
    if args.scaling_sizes:
        scaling = run_scaling(args, keys, sorted(set(args.scaling_sizes)))
        print_scaling(scaling)
        if not args.no_plots:
            plot_scaling(scaling, os.path.join(args.output_dir, "scaling.png"))

    path = save_results(args, results, scaling)
    if not args.quiet:
        print(f"\nResults written to {path}" + ("" if args.no_plots else f"; plots in {args.output_dir}/<policy>/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
