"""
policies.py
===========

Simple decentralised policies for `SISGraphEnv`.

A policy is any object with

    reset() -> None
    act(env, observations) -> np.ndarray of shape (n,)  (integer actions)

where `observations` is the dict returned by `env.get_observations()`.  Every
agent's action is computed ONLY from its own observation (plus static
graph-structure metadata `env.observation_distances`), so these are genuinely
localized policies in the sense of the paper (zeta_i(a_i | s_i), or
zeta_i(a_i | s_{N_i}) under the paper's Remark 1).  A learned SAC policy can be
dropped in by implementing the same two methods.

Policies are looked up by short CLI keys through `make_policy(key, ...)`:

    never | always | local | random
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from sis_environment import INFECTED, SISGraphEnv


class Policy:
    """Base class for policies."""

    name: str = "policy"

    def reset(self) -> None:
        """Called at the start of every episode (override for stateful policies)."""

    def act(self, env: SISGraphEnv, observations: Dict[int, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class NeverIntervene(Policy):
    """Baseline 1: a_i = 0 for every agent."""

    name = "Never intervene"

    def act(self, env: SISGraphEnv, observations: Dict[int, np.ndarray]) -> np.ndarray:
        return np.zeros(env.num_nodes, dtype=np.int64)


class AlwaysIntervene(Policy):
    """Baseline 2: every agent always takes the strongest intervention (A - 1)."""

    name = "Always intervene"

    def act(self, env: SISGraphEnv, observations: Dict[int, np.ndarray]) -> np.ndarray:
        return np.full(env.num_nodes, env.num_actions - 1, dtype=np.int64)


class LocalHeuristic(Policy):
    """Baseline 3: intervene iff the agent is infected OR has an infected neighbour.

    Needs `env.observation_radius >= 1` so that agents can see their neighbours.
    (With a larger radius only the nodes at hop distance <= 1 are used.)

    Parameters
    ----------
    action : intervention level used when triggered (default: strongest, A - 1).
    """

    name = "Local heuristic"

    def __init__(self, action: Optional[int] = None) -> None:
        self.action = action

    def act(self, env: SISGraphEnv, observations: Dict[int, np.ndarray]) -> np.ndarray:
        if env.observation_radius < 1:
            raise ValueError(
                "LocalHeuristic needs observation_radius >= 1 (agents must see their neighbours)."
            )
        level = env.num_actions - 1 if self.action is None else int(self.action)
        if not 0 <= level < env.num_actions:
            raise ValueError(f"heuristic action must lie in [0, {env.num_actions - 1}], got {level}.")
        actions = np.zeros(env.num_nodes, dtype=np.int64)
        for i in range(env.num_nodes):
            within_one_hop = env.observation_distances[i] <= 1
            if np.any(observations[i][within_one_hop] == INFECTED):
                actions[i] = level
        return actions


class RandomPolicy(Policy):
    """Each agent picks a uniformly random action (own random generator).

    If a seed is given, `reset()` re-seeds the generator from (seed, episode
    index), so episode k always sees the same random stream regardless of what
    happened in earlier episodes.
    """

    name = "Random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._episode = -1
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._episode += 1
        if self._seed is not None:
            self._rng = np.random.default_rng([self._seed, self._episode])

    def act(self, env: SISGraphEnv, observations: Dict[int, np.ndarray]) -> np.ndarray:
        return self._rng.integers(0, env.num_actions, size=env.num_nodes)


# --------------------------------------------------------------------------- #
# Registry used by the command-line tools
# --------------------------------------------------------------------------- #
POLICY_NAMES: Tuple[str, ...] = ("never", "always", "local", "random")


def make_policy(key: str, seed: Optional[int] = None, heuristic_action: Optional[int] = None) -> Policy:
    """Build a policy from its CLI key."""
    factories: Dict[str, Callable[[], Policy]] = {
        "never": NeverIntervene,
        "always": AlwaysIntervene,
        "local": lambda: LocalHeuristic(action=heuristic_action),
        "random": lambda: RandomPolicy(seed=seed),
    }
    if key not in factories:
        raise ValueError(f"Unknown policy '{key}'. Choose one of {POLICY_NAMES}.")
    return factories[key]()
