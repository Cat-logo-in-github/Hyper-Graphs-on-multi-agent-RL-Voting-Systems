"""
Tests for the SIS environment, the policies and the command-line tools.

Run with either
    pytest -q test_environment.py
or (no pytest needed)
    python test_environment.py

The first group checks the environment against the model in Sec. 2.2 of
Qu, Wierman, Li (arXiv:1912.02906): transition law (3), reward (4), locality of
the dynamics (1), own-action-only dependence (Remark 1), the discounted
objective (2) and the size f(kappa) of kappa-hop neighbourhoods.
"""

import json
import math
import os
import tempfile

import numpy as np

import run_baselines
import run_demo
from policies import AlwaysIntervene, LocalHeuristic, NeverIntervene, RandomPolicy, make_policy
from sis_environment import INFECTED, SUSCEPTIBLE, SISGraphEnv


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- #
# Original tests (kept)
# --------------------------------------------------------------------------- #
def test_initialization():
    env = SISGraphEnv(num_nodes=10, graph_type="ring")
    assert env.num_nodes == 10


def test_reset():
    env = SISGraphEnv(num_nodes=10)
    obs, info = env.reset(seed=42)
    assert len(obs) == 10
    assert "fraction_infected" in info


def test_reproducibility():
    env1 = SISGraphEnv(num_nodes=10, seed=42)
    env2 = SISGraphEnv(num_nodes=10, seed=42)
    env1.reset(seed=42)
    env2.reset(seed=42)
    actions = np.zeros(10, dtype=int)
    _, r1, _, _, _ = env1.step(actions)
    _, r2, _, _, _ = env2.step(actions)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(env1.get_state(), env2.get_state())


def test_local_observations():
    env = SISGraphEnv(num_nodes=5, graph_type="line", observation_radius=0)
    obs, _ = env.reset(seed=1)
    assert len(obs[0]) == 1  # radius 0 -> only self
    env1 = SISGraphEnv(num_nodes=5, graph_type="line", observation_radius=1)
    obs1, _ = env1.reset(seed=1)
    assert len(obs1[0]) == 2  # end node of a line: self + 1 neighbour


# --------------------------------------------------------------------------- #
# Fidelity to the paper's epidemic model
# --------------------------------------------------------------------------- #
def test_transition_probabilities_match_eq3():
    # line 0-1-2, beta(0)=0.5, beta(1)=0.5*(1-0.8)=0.1, delta=0.3
    env = SISGraphEnv(num_nodes=3, graph_type="line", seed=0)
    env.reset(seed=0, initial_state=[1, 0, 1])
    p = env.next_infection_probabilities([0, 0, 0])
    np.testing.assert_allclose(p, [0.7, 1 - 0.5 ** 2, 0.7])   # I stays I w.p. 1-delta; S: 1-(1-beta)^m
    p = env.next_infection_probabilities([0, 1, 0])
    np.testing.assert_allclose(p[1], 1 - 0.9 ** 2)            # own intervention lowers beta


def test_only_own_action_matters_remark1():
    env = SISGraphEnv(num_nodes=3, graph_type="line", seed=0)
    env.reset(seed=0, initial_state=[1, 0, 1])
    p_a = env.next_infection_probabilities([1, 0, 1])
    p_b = env.next_infection_probabilities([0, 0, 0])
    assert p_a[1] == p_b[1]  # neighbours' actions do not enter node 1's transition


def test_dynamics_are_local_eq1():
    env = SISGraphEnv(num_nodes=6, graph_type="line", seed=0)
    env.reset(seed=0)
    a = np.zeros(6, dtype=int)
    base = np.array([0, 0, 0, 1, 0, 0])
    far = base.copy()
    far[5] = 1  # differs only at node 5
    p0 = env.next_infection_probabilities(a, base)
    p1 = env.next_infection_probabilities(a, far)
    np.testing.assert_allclose(p0[:4], p1[:4])  # nodes 0..3 are not in N_5
    assert p0[4] != p1[4]                      # node 4 is a neighbour of 5


def test_reward_uses_state_before_transition_eq4():
    env = SISGraphEnv(num_nodes=3, graph_type="line", seed=0)  # c(1) = 0.2
    env.reset(seed=0, initial_state=[1, 0, 0])
    _, r, _, _, _ = env.step([1, 0, 0])
    np.testing.assert_allclose(r, [0.0 - 0.2, 1.0, 1.0])


def test_discounted_return_matches_definition_eq2():
    env = SISGraphEnv(num_nodes=8, graph_type="ring", gamma=0.7, seed=3, max_steps=20)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    manual = 0.0
    for t in range(20):
        _, _, _, _, info = env.step(rng.integers(0, 2, size=8))
        manual += 0.7 ** t * info["global_reward"]
    assert math.isclose(env.get_metrics()["discounted_return"], manual, rel_tol=1e-12)


def test_monte_carlo_matches_analytic_probability():
    env = SISGraphEnv(num_nodes=3, graph_type="line", seed=11)
    hits, trials = 0, 4000
    for _ in range(trials):
        env.reset(initial_state=[1, 0, 1])
        env.step([0, 0, 0])
        hits += int(env.get_state()[1] == INFECTED)
    assert abs(hits / trials - 0.75) < 0.03


def test_neighbourhood_size_ring():
    env = SISGraphEnv(num_nodes=12, graph_type="ring")
    for kappa in range(4):
        assert all(len(env.get_neighborhood(i, kappa)) == 2 * kappa + 1 for i in range(12))


def test_invalid_actions_rejected():
    env = SISGraphEnv(num_nodes=4, graph_type="ring")
    env.reset(seed=0)
    assert _raises(ValueError, env.step, [0, 0, 0, 5])
    assert _raises(ValueError, env.step, [0, 0, 0])


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def test_basic_policies():
    env = SISGraphEnv(num_nodes=6, graph_type="ring", observation_radius=1, num_actions=3)
    obs, _ = env.reset(seed=0)
    assert not NeverIntervene().act(env, obs).any()
    assert (AlwaysIntervene().act(env, obs) == 2).all()
    r1, r2 = RandomPolicy(seed=1), RandomPolicy(seed=1)
    r1.reset(); r2.reset()
    a1, a2 = r1.act(env, obs), r2.act(env, obs)
    np.testing.assert_array_equal(a1, a2)
    assert a1.min() >= 0 and a1.max() < 3


def test_local_heuristic_is_local():
    env = SISGraphEnv(num_nodes=7, graph_type="line", observation_radius=1)
    pol = LocalHeuristic()
    obs, _ = env.reset(seed=0, initial_state=[0] * 7)
    a_clean = pol.act(env, obs)
    obs, _ = env.reset(initial_state=[0, 0, 0, 0, 0, 0, 1])
    a_far = pol.act(env, obs)
    assert a_clean[0] == a_far[0] == 0     # agent 0 cannot see node 6
    assert a_far[5] == 1 and a_far[6] == 1  # neighbour / self infected -> intervene
    assert a_far[4] == 0


def test_local_heuristic_needs_radius_1():
    env = SISGraphEnv(num_nodes=5, observation_radius=0)
    obs, _ = env.reset(seed=0)
    assert _raises(ValueError, LocalHeuristic().act, env, obs)


def test_make_policy_registry():
    for key in ("never", "always", "local", "random"):
        assert make_policy(key, seed=0) is not None
    assert _raises(ValueError, make_policy, "nope")


# --------------------------------------------------------------------------- #
# Command-line tools
# --------------------------------------------------------------------------- #
def test_cli_baselines_makes_ten_plots_per_policy_and_timing():
    with tempfile.TemporaryDirectory() as d:
        rc = run_baselines.main(["-n", "8", "--max-steps", "20", "--episodes", "2", "--quiet",
                                 "--policies", "never", "local", "--output-dir", d])
        assert rc == 0
        for key in ("never", "local"):
            steps = [f for f in os.listdir(os.path.join(d, key)) if f.startswith("step_")]
            assert len(steps) == 10
        assert os.path.exists(os.path.join(d, "infection_curves.png"))
        res = json.load(open(os.path.join(d, "results.json")))
        t = res["results"][0]["timing"]
        assert t["act_us_per_step_mean"] > 0 and t["env_step_us_per_step_mean"] > 0


def test_cli_scaling_study():
    with tempfile.TemporaryDirectory() as d:
        run_baselines.main(["--max-steps", "5", "--episodes", "1", "--quiet", "--no-plots",
                            "--policies", "local", "--scaling-sizes", "8", "16", "32",
                            "--scaling-episodes", "1", "--output-dir", d])
        res = json.load(open(os.path.join(d, "results.json")))
        assert res["scaling"]["sizes"] == [8, 16, 32]
        assert res["scaling"]["policies"]["local"]["act_exponent"] is not None


def test_cli_demo():
    with tempfile.TemporaryDirectory() as d:
        run_demo.main(["--policy", "never", "-n", "10", "--max-steps", "30", "--output-dir", d,
                       "--graph-type", "grid"])
        assert os.path.exists(os.path.join(d, "demo_plot.png"))
        assert len([f for f in os.listdir(os.path.join(d, "never")) if f.startswith("step_")]) == 10


if __name__ == "__main__":  # run without pytest
    failed = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
