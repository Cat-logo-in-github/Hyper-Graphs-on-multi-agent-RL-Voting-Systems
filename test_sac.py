"""Tests for SAC and the paper's environments.  Run: pytest -q test_sac.py   or   python test_sac.py"""
import json
import os
import tempfile

import numpy as np

import run_sac
from paper_environments import LineEnv, WirelessEnv
from sac import ContextEncoder, ScalableActorCritic, estimate_value, sac_act_fn, snapshot_steps
from sis_environment import SISGraphEnv


def test_snapshot_steps_rule():
    assert snapshot_steps(100) == list(range(10, 101, 10))
    assert snapshot_steps(200) == list(range(20, 201, 20))
    assert snapshot_steps(7) == [1, 2, 3, 4, 5, 6, 7]       # <= 10 steps: one plot per step
    assert len(snapshot_steps(13)) == 10


def test_line_env_matches_section_5_1():
    env = LineEnv(8, 0.7)
    s = np.ones(8, dtype=np.int64)
    np.testing.assert_allclose(env.next_one_probabilities(s, np.ones(8, dtype=np.int64)), np.ones(8))
    s0 = s.copy(); s0[3] = 0
    p = env.next_one_probabilities(s0, np.ones(8, dtype=np.int64))
    assert p[2] == 0.8 and p[1] == 1.0 and p[0] == 1.0       # a_i=1 but s_{i+1}=0 -> 0.8
    assert env.next_one_probabilities(s, np.zeros(8, dtype=np.int64))[1:].sum() == 0
    assert abs(env.optimal_value - 1 / (8 * 0.3)) < 1e-12


def test_optimal_policy_attains_optimum():
    env = LineEnv(8, 0.7, max_steps=60)
    v = estimate_value(env, lambda e, s, rng: np.ones(8, dtype=np.int64), 5, 60)
    assert abs(v["mean"] - env.optimal_value) < 1e-3


def test_wireless_table1_conflict_and_success():
    env = WirelessEnv(2, 2, max_steps=10, p=np.zeros(4), q=np.ones(9))
    env.reset(seed=0, initial_state=[2, 2, 0, 0])             # users 0,1 hold a packet (bits 1,0)
    _, r, _, _, _ = env.step([2, 1, 0, 0])                    # user0 -> AP1, user1 -> AP1 (corner 0): conflict
    assert r.sum() == 0
    env.reset(initial_state=[2, 2, 0, 0])
    _, r, _, _, info = env.step([2, 0, 0, 0])                 # only user 0 sends: success, reward 1
    assert r[0] == 1 and r.sum() == 1 and info["state"][0] == 0


def test_wireless_empty_queue_cannot_send():
    env = WirelessEnv(2, 2, max_steps=10, p=np.zeros(4), q=np.ones(9))
    env.reset(seed=0, initial_state=[0, 0, 0, 0])
    _, r, _, _, _ = env.step([1, 1, 1, 1])
    assert r.sum() == 0


def test_kappa0_only_agent0_moves_on_line():
    # reward only at node 0 => Qhat_j = 0 exactly for j != 0; with kappa=0 only agent 0 gets gradient (paper Fig. 1)
    env = LineEnv(8, 0.7, max_steps=30)
    sac = ScalableActorCritic(env, 0, horizon=20, seed=0)
    sac.train(20)
    assert np.all(sac.theta[1:] == 0)


def test_kappa1_learns_agent1():
    env = LineEnv(8, 0.7, max_steps=40)
    sac = ScalableActorCritic(env, 1, horizon=30, actor_eta=5.0, seed=0)
    sac.train(600)
    assert np.all(sac.theta[2:] == 0)                          # outside N_0^1 -> no signal
    assert sac.policy_probs()[1, 1, 1] > 0.8                   # agent 1 learns to send a=1


def test_sac_reproducible_and_runs_on_sis():
    out = []
    for _ in range(2):
        env = SISGraphEnv(num_nodes=10, graph_type="ring", observation_radius=0, max_steps=21, seed=1)
        sac = ScalableActorCritic(env, 1, horizon=20, seed=5)
        sac.train(5)
        out.append(sac.theta.copy())
    np.testing.assert_array_equal(out[0], out[1])
    assert out[0].shape == (10, 2, 2)


def test_table_size_bounded_by_neighbourhood():
    env = SISGraphEnv(num_nodes=30, graph_type="ring", observation_radius=0, max_steps=21, seed=1)
    sac = ScalableActorCritic(env, 1, horizon=20, seed=0)
    rec = sac.iterate()
    assert sac.f_kappa == 3 and rec["q_entries_max"] <= (2 * 2) ** 3


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        run_sac.main(["--env", "line", "--kappas", "0", "1", "-M", "20", "-T", "10", "--rollout-steps", "100",
                      "--eval-episodes", "5", "--output-dir", d, "--complexity", "--complexity-sizes", "6", "12",
                      "--complexity-kappas", "0", "1", "--complexity-iters", "2"])
        snaps = os.path.join(d, "snapshots", "line", "SAC_kappa1")
        assert len(os.listdir(snaps)) == 10
        for f in ("training_line.png", "optimality_gap_line.png", "complexity_vs_n.png",
                  "complexity_vs_kappa.png", "results.json"):
            assert os.path.exists(os.path.join(d, f)), f
        assert "complexity" in json.load(open(os.path.join(d, "results.json")))


def test_context_encoder_remark1_is_local():
    env = SISGraphEnv(num_nodes=8, graph_type="ring", observation_radius=0, max_steps=10, seed=0)
    enc = ContextEncoder(env, 1, 2)
    assert enc.size == 8
    a = np.zeros(8, dtype=np.int64)
    b = a.copy(); b[4] = 1                                   # node 4 is outside N_0^1 = {7, 0, 1}
    assert enc(a)[0] == enc(b)[0] and enc(a)[3] != enc(b)[3]
    assert np.array_equal(ContextEncoder(env, 0, 2)(b), b)  # radius 0 = own state


def test_warm_start_keeps_critic_tables():
    env = SISGraphEnv(num_nodes=10, graph_type="ring", observation_radius=0, max_steps=21, seed=1)
    warm = ScalableActorCritic(env, 1, horizon=20, seed=0, warm_start=True)
    cold = ScalableActorCritic(env, 1, horizon=20, seed=0, warm_start=False)
    warm.iterate(); n1 = warm.iterate()["q_entries"]; cold.iterate(); c1 = cold.iterate()["q_entries"]
    assert n1 >= c1


def test_sac_policy_radius1_on_sis():
    env = SISGraphEnv(num_nodes=10, graph_type="ring", observation_radius=0, max_steps=21, seed=1)
    sac = ScalableActorCritic(env, 1, horizon=20, seed=5, policy_radius=1)
    sac.train(3)
    assert sac.theta.shape == (10, 8, 2)
    v = estimate_value(env, sac.act_fn(), 3, 10)
    assert np.isfinite(v["mean"])


def test_code_equals_literal_algorithm1():
    import verify_paper
    assert all(r["pass"] for r in verify_paper.check_algorithm_identity())


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn(); print("PASS ", name)
        except Exception as exc:  # noqa: BLE001
            bad += 1; print("FAIL ", name, type(exc).__name__, exc)
    print(f"{len(tests) - bad}/{len(tests)} passed")
    raise SystemExit(1 if bad else 0)
