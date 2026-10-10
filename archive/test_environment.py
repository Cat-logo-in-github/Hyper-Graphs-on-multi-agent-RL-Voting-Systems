import pytest
import numpy as np
from sis_environment import SISGraphEnv, SUSCEPTIBLE, INFECTED

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
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    actions = np.zeros(10, dtype=int)
    _, r1, _, _, _ = env1.step(actions)
    _, r2, _, _, _ = env2.step(actions)
    
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(env1.get_state(), env2.get_state())

def test_local_observations():
    env = SISGraphEnv(num_nodes=5, graph_type="line", observation_radius=0)
    obs, _ = env.reset(seed=1)
    # Radius 0 observation length should be 1 (only self)
    assert len(obs[0]) == 1
    
    env1 = SISGraphEnv(num_nodes=5, graph_type="line", observation_radius=1)
    obs1, _ = env1.reset(seed=1)
    # Node 0 in a line graph has 1 neighbor -> total radius-1 neighborhood size = 2
    assert len(obs1[0]) == 2