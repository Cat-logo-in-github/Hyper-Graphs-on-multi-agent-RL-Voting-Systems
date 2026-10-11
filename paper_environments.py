"""
paper_environments.py
=====================

The two numerical environments of Qu, Wierman, Li (arXiv:1912.02906), written
against the same interface as `SISGraphEnv` (Sec. 2.2 epidemic example), so the
Scalable Actor Critic in `sac.py` runs unchanged on all three.

Common interface (what SAC needs)
---------------------------------
    num_nodes, num_states (|S_i|), num_actions (|A_i|), gamma, max_steps
    get_neighborhood(i, radius) -> node ids within `radius` hops, INCLUDING i
    reset(seed=None, initial_state=None) -> (obs, info)      info["state"] = s(0)
    step(actions) -> (obs, rewards (n,), terminated, truncated, info)
        rewards are r_i(s_i(t), a_i(t)) on the state BEFORE the transition and
        info["state"] is s(t+1).  Local states are integers in [0, |S_i|).

Sec. 5.1  `LineEnv`      synthetic line, binary states/actions, known optimum
Sec. 5.2  `WirelessEnv`  multi-access wireless grid (Table 1), ALOHA benchmark
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np


class _NetworkedBase:
    """Graph bookkeeping shared by the environments below."""

    num_nodes: int
    num_states: int
    num_actions: int
    gamma: float
    max_steps: int

    def _setup_graph(self, g: nx.Graph, pos: Dict[int, Tuple[float, float]]) -> None:
        n = g.number_of_nodes()
        self._graph, self._pos, self.num_nodes = g, pos, n
        dist = np.full((n, n), np.inf)
        for src, lengths in nx.all_pairs_shortest_path_length(g):
            for dst, d in lengths.items():
                dist[src, dst] = d
        self._dist = dist
        self._t = 0
        self._rng = np.random.default_rng()
        self._state = np.zeros(n, dtype=np.int64)

    def get_neighborhood(self, agent: int, radius: int) -> np.ndarray:
        d = self._dist[agent]
        idx = np.flatnonzero(d <= radius)
        return idx[np.lexsort((idx, d[idx]))]

    def get_graph(self) -> nx.Graph:
        return self._graph.copy()

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def t(self) -> int:
        return self._t

    def _validate_actions(self, actions: Any) -> np.ndarray:
        a = np.asarray(actions)
        if a.shape != (self.num_nodes,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"actions must be an int array of shape ({self.num_nodes},).")
        if a.min() < 0 or a.max() >= self.num_actions:
            raise ValueError(f"actions must lie in [0, {self.num_actions - 1}].")
        return a.astype(np.int64)

    def _reseed(self, seed: Optional[int]) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Sec. 5.1 -- synthetic line example
# --------------------------------------------------------------------------- #
class LineEnv(_NetworkedBase):
    """Line of n nodes, S_i = A_i = {0, 1}   (paper Sec. 5.1; node 1 there = node 0 here).

    * r_0 = 1(s_0 = 1); every other reward is 0.
    * s_0(t+1) = 1 iff s_1(t) = 1 (deterministic, a_0 is irrelevant).
    * 1 <= i <= n-2:  P(s_i(t+1)=1) = 1 if (s_{i+1}=1, a_i=1);  0.8 if (s_{i+1}=0, a_i=1);  0 otherwise.
    * s_{n-1}(t+1) = 1 iff a_{n-1} = 1.
    * pi_0: all nodes start at 1.
    Optimal policy: a_i = 1 always; optimal J = (1/n) * 1/(1-gamma).
    """

    num_states = 2
    num_actions = 2

    def __init__(self, num_nodes: int = 8, gamma: float = 0.7, max_steps: int = 100,
                 seed: Optional[int] = None) -> None:
        if num_nodes < 3:
            raise ValueError("LineEnv needs at least 3 nodes.")
        self.gamma, self.max_steps = float(gamma), int(max_steps)
        self._setup_graph(nx.path_graph(num_nodes), {i: (float(i), 0.0) for i in range(num_nodes)})
        self._reseed(seed)

    @property
    def optimal_value(self) -> float:
        return 1.0 / (self.num_nodes * (1.0 - self.gamma))

    def reset(self, seed: Optional[int] = None, initial_state: Optional[Any] = None):
        self._reseed(seed)
        self._state = np.ones(self.num_nodes, dtype=np.int64) if initial_state is None \
            else np.asarray(initial_state, dtype=np.int64)
        self._t = 0
        return self._state.copy(), {"state": self._state.copy(), "t": 0}

    def next_one_probabilities(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        p = np.zeros(self.num_nodes)
        p[0] = float(state[1] == 1)
        p[1:-1] = actions[1:-1] * np.where(state[2:] == 1, 1.0, 0.8)
        p[-1] = float(actions[-1] == 1)
        return p

    def step(self, actions: Any):
        a = self._validate_actions(actions)
        s = self._state
        rewards = np.zeros(self.num_nodes)
        rewards[0] = float(s[0] == 1)
        self._state = (self._rng.random(self.num_nodes) < self.next_one_probabilities(s, a)).astype(np.int64)
        self._t += 1
        info = {"state": self._state.copy(), "t": self._t, "global_reward": float(rewards.mean())}
        return self._state.copy(), rewards, False, self._t >= self.max_steps, info

    def render(self, ax) -> None:
        ax.clear()
        colors = ["#4C9BE8" if v == 1 else "#BBBBBB" for v in self._state]
        nx.draw_networkx_edges(self._graph, self._pos, ax=ax, edge_color="0.6")
        nx.draw_networkx_nodes(self._graph, self._pos, node_color=colors, edgecolors="black", ax=ax, node_size=350)
        nx.draw_networkx_labels(self._graph, self._pos, ax=ax, font_size=8)
        ax.set_title(f"t = {self._t}   (blue: s=1, grey: s=0)")
        ax.set_axis_off()


# --------------------------------------------------------------------------- #
# Sec. 5.2 -- multi-access wireless communication
# --------------------------------------------------------------------------- #
class WirelessEnv(_NetworkedBase):
    """Grid of L x L users, one per square block; access points on the block corners (Fig. 3).

    Local state: d-bit queue (bit l = a packet with remaining deadline l+1; bit 0 = leftmost =
    earliest), encoded as an integer in [0, 2^d).  Local action: 0 = null, k = 1..4 = send the
    earliest packet to the k-th corner access point of the user's block.
    Transition = Table 1 of the paper: a send succeeds (flip leftmost 1 w.p. q_k) iff the queue is
    non-empty and no other user with a non-empty queue sends to the same access point; then left shift
    and append Bernoulli(p_i).  Reward r_i = 1 exactly in the last row of Table 1 (non-empty queue,
    send, no conflict), as in the paper's definition of r_i.
    p_i, q_k ~ U[0,1] (fixed at construction from `seed`); pi_0: queues uniform over {0,1}^d.
    Interaction graph = conflict graph (users sharing an access point).
    """

    num_actions = 5

    def __init__(self, grid_size: int = 6, deadline: int = 2, gamma: float = 0.7, max_steps: int = 100,
                 seed: Optional[int] = None, p: Optional[np.ndarray] = None, q: Optional[np.ndarray] = None) -> None:
        if grid_size < 2 or deadline < 1:
            raise ValueError("grid_size >= 2 and deadline >= 1 required.")
        L, n = grid_size, grid_size * grid_size
        self.grid_size, self.deadline = L, deadline
        self.num_states = 2 ** deadline
        self.gamma, self.max_steps = float(gamma), int(max_steps)
        params = np.random.default_rng(seed)
        self.p = params.random(n) if p is None else np.asarray(p, float)
        self.q = params.random((L + 1) ** 2) if q is None else np.asarray(q, float)
        r, c = np.divmod(np.arange(n), L)
        self.ap_of = np.stack([r * (L + 1) + c, r * (L + 1) + c + 1,
                               (r + 1) * (L + 1) + c, (r + 1) * (L + 1) + c + 1], axis=1)
        self.users_per_ap = np.bincount(self.ap_of.ravel(), minlength=(L + 1) ** 2)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for ap in range(self.q.size):
            users = np.flatnonzero((self.ap_of == ap).any(axis=1))
            g.add_edges_from((int(a), int(b)) for i, a in enumerate(users) for b in users[i + 1:])
        self._setup_graph(g, {i: (float(i % L), -float(i // L)) for i in range(n)})
        self._weights = 2 ** np.arange(deadline - 1, -1, -1)
        self._bits = np.zeros((n, deadline), dtype=np.int8)
        self._reseed(None if seed is None else seed + 1)

    def _sync_state(self) -> None:
        self._state = (self._bits.astype(np.int64) @ self._weights)

    def reset(self, seed: Optional[int] = None, initial_state: Optional[Any] = None):
        self._reseed(seed)
        if initial_state is None:
            self._bits = (self._rng.random((self.num_nodes, self.deadline)) < 0.5).astype(np.int8)
        else:
            st = np.asarray(initial_state, dtype=np.int64)
            self._bits = ((st[:, None] >> np.arange(self.deadline - 1, -1, -1)) & 1).astype(np.int8)
        self._sync_state()
        self._t = 0
        return self._state.copy(), {"state": self._state.copy(), "t": 0}

    def step(self, actions: Any):
        a = self._validate_actions(actions)
        n = self.num_nodes
        B = self._bits
        send = (a > 0) & B.any(axis=1)
        ap = self.ap_of[np.arange(n), np.maximum(a - 1, 0)]
        counts = np.bincount(ap[send], minlength=self.q.size)
        attempt = send & (counts[ap] == 1)                  # last row of Table 1
        rewards = attempt.astype(float)
        success = attempt & (self._rng.random(n) < self.q[ap])
        nb = B.copy()
        rows = np.flatnonzero(success)
        nb[rows, B[rows].argmax(axis=1)] = 0                # flip the leftmost 1
        arrivals = (self._rng.random(n) < self.p).astype(np.int8)
        self._bits = np.concatenate([nb[:, 1:], arrivals[:, None]], axis=1)  # left shift + append
        self._sync_state()
        self._t += 1
        info = {"state": self._state.copy(), "t": self._t, "global_reward": float(rewards.mean()),
                "num_senders": int(send.sum()), "num_conflicts": int((send & ~attempt).sum())}
        return self._state.copy(), rewards, False, self._t >= self.max_steps, info

    def aloha_actions(self, rng: np.random.Generator, send_prob: float = 0.5) -> np.ndarray:
        """Localized ALOHA benchmark: send w.p. `send_prob`, to an access point chosen with
        probability proportional to q_k / (#users sharing it).  (Uses q_k, which SAC never sees.)"""
        w = self.q[self.ap_of] / self.users_per_ap[self.ap_of]
        cum = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        k = np.minimum((rng.random((self.num_nodes, 1)) > cum).sum(axis=1), 3)
        return np.where(rng.random(self.num_nodes) < send_prob, k + 1, 0).astype(np.int64)

    def render(self, ax) -> None:
        ax.clear()
        L = self.grid_size
        packets = self._bits.sum(axis=1).reshape(L, L)
        ax.imshow(packets, cmap="Blues", vmin=0, vmax=self.deadline)
        for i in range(self.num_nodes):
            ax.text(i % L, i // L, str(int(self._bits[i].sum())), ha="center", va="center", fontsize=8)
        ax.set_title(f"t = {self._t}   queued packets / user")
        ax.set_axis_off()
