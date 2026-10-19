#!/usr/bin/env python3
"""
run_demo.py
===========

Run ONE policy on the SIS environment, print progress, and write
  * `--num-plots` (default 10) graph snapshots, one every max_steps/num_plots steps,
  * a montage of those snapshots,
  * `demo_plot.png`: infected fraction over time,
  * timing information (policy.act / env.step per step) in the console and results.json.

Everything is configurable from the command line (same environment flags as
run_baselines.py).  Examples:

    python run_demo.py
    python run_demo.py --policy never -n 40 --graph-type grid --max-steps 200
    python run_demo.py --policy random --graph-type random --random-edge-prob 0.15 --seed 7
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from policies import POLICY_NAMES  # noqa: E402
from run_baselines import (  # noqa: E402
    add_env_arguments,
    add_run_arguments,
    evaluate_policy,
    save_results,
    validate_args,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-policy epidemic demo with timing and periodic snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_env_arguments(p)
    p.add_argument("--policy", choices=POLICY_NAMES, default="local", help="policy to run")
    p.add_argument("--policy-seed", type=int, default=42, help="seed of the random policy")
    p.add_argument("--heuristic-action", type=int, default=None,
                   help="action level used by the local heuristic (default: strongest)")
    p.add_argument("--print-every", type=int, default=10, help="print progress every k steps (0 = never)")
    add_run_arguments(p)
    # demo defaults (match the original demo): seed 42, 50 steps, own output folder
    p.set_defaults(seed=42, max_steps=50, output_dir="demo_output")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.episodes = 1          # a demo is a single episode
    args.viz_episode = 0
    validate_args(parser, args, [args.policy])
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Running epidemic demo: policy={args.policy}, n={args.num_nodes}, graph={args.graph_type}, "
          f"radius={args.observation_radius}, steps={args.max_steps}, seed={args.seed}")
    plot_dir = None if args.no_plots else os.path.join(args.output_dir, args.policy)
    res = evaluate_policy(args, args.policy, episodes=1, plot_dir=plot_dir, print_every=args.print_every)

    traj = res["_trajectories"][0]
    if not args.no_plots:
        plt.figure(figsize=(8, 4))
        plt.plot(traj, marker="o", markersize=3)
        plt.title(f"Epidemic progression – {res['policy_name']} ({args.graph_type}, n={args.num_nodes})")
        plt.xlabel("Timestep")
        plt.ylabel("Infected Fraction")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "demo_plot.png"), dpi=120)
        plt.close()

    t = res["timing"]
    print(f"\nDiscounted return J = {res['discounted_return']['mean']:.4f} (gamma={args.gamma}); "
          f"final infected fraction = {traj[-1]:.2f}")
    print(f"Timing: policy.act {t['act_us_per_step_mean']:.1f} µs/step, env.step "
          f"{t['env_step_us_per_step_mean']:.1f} µs/step, {t['seconds_per_episode'] * 1e3:.2f} ms/episode "
          f"({t['us_per_agent_step']:.2f} µs per agent-step)")
    path = save_results(args, [res], None)
    print(f"Done. Results: {path}" + ("" if args.no_plots else f"; plots in {args.output_dir}/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
