"""
sac.py
======

Scalable Actor Critic (SAC), Algorithm 1 of Qu, Wierman, Li (arXiv:1912.02906).

For every outer iteration m = 0, 1, ...
  1. sample s(0) ~ pi_0; every agent draws a_i(t) ~ zeta_i^{theta_i}(. | s_i(t)) (localized softmax
     policy, one logit table theta_i[s_i, a_i] per agent) along a trajectory of length T;
  2. CRITIC (lines 3-9): every agent i runs TD(0) on a *truncated* Q-function
        Qhat_i(s_{N_i^kappa}, a_{N_i^kappa})
     initialised to 0, step size alpha_{t-1} = h / (t - 1 + t0):
        Qhat_i(z_Nk(t-1)) <- (1-alpha) Qhat_i(z_Nk(t-1)) + alpha (r_i(t-1) + gamma Qhat_i(z_Nk(t)))
  3. ACTOR (lines 10-12):
        g_i(m) = sum_{t=0}^T gamma^t (1/n) sum_{j in N_i^kappa} Qhat_j(z_{N_j^kappa}(t)) grad log zeta_i(a_i(t)|s_i(t))
        theta_i <- theta_i + eta_m g_i,     eta_m = eta / sqrt(m + 1).

Q-tables are sparse dictionaries keyed by the (state, action) pattern of the kappa-hop neighbourhood, so
memory/time depend on the neighbourhood size f(kappa), not on n (the paper's scalability claim).
kappa = 0 is the independent-learner special case.  Constants h, t0, eta are command-line tunable: the
theorem's conditions on them involve the (unknown) exploration constants sigma and tau.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def snapshot_steps(steps: int, num_plots: int = 10) -> List[int]:
    """Steps at which an environment rollout is plotted.

    steps=100 -> 10, 20, ..., 100;  steps=200 -> 20, 40, ..., 200;  steps <= 10 -> every step.
    """
    if steps <= num_plots:
        return list(range(1, steps + 1))
    return sorted({int(round(steps * k / num_plots)) for k in range(1, num_plots + 1)})


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ContextEncoder:
    """Maps the global state to each agent's policy context.

    radius 0 -> context = own state s_i                (paper's base model, zeta_i(a_i | s_i))
    radius r -> context = states of N_i^r, base-|S|    (paper's Remark 1,   zeta_i(a_i | s_{N_i}))
    Implemented as one integer matrix:  context = state @ W.T.
    """

    def __init__(self, env: Any, radius: int, num_states: int) -> None:
        n = env.num_nodes
        nodes = [np.asarray(sorted(env.get_neighborhood(i, radius))) for i in range(n)]
        self.radius, self.identity = int(radius), radius == 0
        self.size = int(num_states ** max(len(x) for x in nodes))
        if self.size > 2 ** 16:
            raise ValueError(f"policy radius {radius} gives {self.size} contexts per agent (> 65536); use a smaller radius.")
        self.W = np.zeros((n, n), dtype=np.int64)
        for i, nd in enumerate(nodes):
            self.W[i, nd] = num_states ** np.arange(len(nd))

    def __call__(self, state: np.ndarray) -> np.ndarray:
        return state if self.identity else state @ self.W.T


def sample_actions(theta: np.ndarray, state: np.ndarray, rng: np.random.Generator,
                   enc: Optional[ContextEncoder] = None) -> np.ndarray:
    """a_i ~ softmax(theta_i[context_i, :]) independently for every agent (localized policy)."""
    n = theta.shape[0]
    ctx = state if enc is None else enc(state)
    probs = softmax_rows(theta[np.arange(n), ctx])
    cum = np.cumsum(probs, axis=1)
    return np.minimum((rng.random((n, 1)) > cum).sum(axis=1), theta.shape[2] - 1).astype(np.int64)


# --------------------------------------------------------------------------- #
# the algorithm
# --------------------------------------------------------------------------- #
class ScalableActorCritic:
    def __init__(self, env: Any, kappa: int, horizon: int = 50, critic_h: float = 5.0,
                 critic_t0: float = 10.0, actor_eta: float = 2.0, seed: int = 0,
                 warm_start: bool = False, policy_radius: int = 0) -> None:
        if kappa < 0:
            raise ValueError("kappa must be >= 0.")
        if horizon < 1:
            raise ValueError("horizon T must be >= 1.")
        if env.max_steps < horizon:
            raise ValueError(f"env.max_steps ({env.max_steps}) must be >= horizon T ({horizon}).")
        self.env, self.kappa, self.T = env, int(kappa), int(horizon)
        self.h, self.t0, self.eta = float(critic_h), float(critic_t0), float(actor_eta)
        self.gamma = float(env.gamma)
        self.n, self.S, self.A = env.num_nodes, getattr(env, "num_states", 2), env.num_actions
        self.rng = np.random.default_rng(seed)
        self.enc = ContextEncoder(env, policy_radius, self.S)
        self.theta = np.zeros((self.n, self.enc.size, self.A))   # uniform initial policy
        self.nodes = [np.asarray(sorted(env.get_neighborhood(i, kappa))) for i in range(self.n)]
        self.f_kappa = max(len(x) for x in self.nodes)           # f(kappa) = sup_i |N_i^kappa|
        self.nmat = np.zeros((self.n, self.n))                   # nmat[i, j] = 1 iff j in N_i^kappa
        for i, nodes in enumerate(self.nodes):
            self.nmat[i, nodes] = 1.0
        base = self.S * self.A
        self._radix = [bool(len(x) * np.log2(base) < 62) for x in self.nodes]
        self._weights = [(base ** np.arange(len(x), dtype=np.int64)) if ok else None
                         for x, ok in zip(self.nodes, self._radix)]
        self.warm_start = bool(warm_start)  # paper Sec. 3.2 'Discussion': Qhat^0 := previous Qhat^T
        self.m = 0
        self.q_tables: List[Dict[Any, float]] = [dict() for _ in range(self.n)]

    # ---- policy -------------------------------------------------------- #
    def policy_probs(self) -> np.ndarray:
        return softmax_rows(self.theta)

    def act(self, state: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return sample_actions(self.theta, state, rng or self.rng, self.enc)

    def act_fn(self) -> "ActFn":
        """Frozen copy of the current policy as act_fn(env, state, rng)."""
        theta, enc = self.theta.copy(), self.enc
        return lambda env, state, rng: sample_actions(theta, state, rng, enc)

    # ---- keys of the truncated Q-tables ---------------------------------- #
    def _keys(self, i: int, z: np.ndarray) -> List[Any]:
        zi = z[:, self.nodes[i]]
        if self._radix[i]:
            return (zi @ self._weights[i]).tolist()
        return [row.tobytes() for row in zi.astype(np.int16)]

    # ---- one outer iteration (Algorithm 1, lines 2-12) -------------------- #
    def iterate(self) -> Dict[str, float]:
        env, n, T, gamma = self.env, self.n, self.T, self.gamma
        t_start = time.perf_counter()

        # trajectory under the current (fixed) policy: states s(0..T), actions a(0..T), rewards r(0..T-1)
        _, info = env.reset(seed=int(self.rng.integers(2 ** 31 - 1)))
        S = np.zeros((T + 1, n), dtype=np.int64)
        Act = np.zeros((T + 1, n), dtype=np.int64)
        R = np.zeros((T, n))
        S[0] = info["state"]
        for t in range(T):
            Act[t] = sample_actions(self.theta, S[t], self.rng, self.enc)
            _, r, _, _, info = env.step(Act[t])
            R[t], S[t + 1] = r, info["state"]
        Act[T] = sample_actions(self.theta, S[T], self.rng, self.enc)
        t_sim = time.perf_counter()

        # critic: TD(0) on the truncated Q-function, Qhat^0 = 0
        Z = S * self.A + Act
        if not (self.warm_start and self.m > 0):
            self.q_tables = [dict() for _ in range(n)]
        q_vals = np.zeros((n, T + 1))
        for i in range(n):
            keys, Q, ri = self._keys(i, Z), self.q_tables[i], R[:, i].tolist()
            for t in range(1, T + 1):
                alpha = self.h / (t - 1 + self.t0)
                k0 = keys[t - 1]
                Q[k0] = (1.0 - alpha) * Q.get(k0, 0.0) + alpha * (ri[t - 1] + gamma * Q.get(keys[t], 0.0))
            q_vals[i] = [Q.get(k, 0.0) for k in keys]
        t_critic = time.perf_counter()

        # actor: g_i = sum_t gamma^t (1/n) sum_{j in N_i^kappa} Qhat_j(t) * grad log zeta_i(a_i(t)|s_i(t))
        CTX = self.enc(S)                                                    # policy contexts (T+1, n)
        probs = softmax_rows(self.theta[np.arange(n)[None, :], CTX])      # (T+1, n, A)
        w = (self.nmat @ q_vals) / n                                        # (n, T+1)
        coeff = (gamma ** np.arange(T + 1))[None, :] * w
        delta = np.eye(self.A)[Act] - probs
        grad = np.zeros_like(self.theta)
        np.add.at(grad, (np.broadcast_to(np.arange(n), S.shape), CTX), coeff.T[:, :, None] * delta)
        eta_m = self.eta / np.sqrt(self.m + 1)
        self.theta += eta_m * grad
        self.m += 1
        t_actor = time.perf_counter()

        return {
            "J": float(((gamma ** np.arange(T)) * R.mean(axis=1)).sum()),  # sampled discounted return
            "grad_sq": float((grad ** 2).sum()),
            "eta": float(eta_m),
            "t_sim": t_sim - t_start, "t_critic": t_critic - t_sim, "t_actor": t_actor - t_critic,
            "q_entries": float(sum(len(q) for q in self.q_tables)),
            "q_entries_max": float(max(len(q) for q in self.q_tables)),
        }

    def train(self, iterations: int, callback: Optional[Callable[[int, Dict[str, float]], None]] = None
              ) -> Dict[str, np.ndarray]:
        hist: Dict[str, List[float]] = {}
        for _ in range(iterations):
            rec = self.iterate()
            for k, v in rec.items():
                hist.setdefault(k, []).append(v)
            if callback:
                callback(self.m, rec)
        return {k: np.asarray(v) for k, v in hist.items()}


# --------------------------------------------------------------------------- #
# evaluation of arbitrary policies:  act_fn(env, state, rng) -> actions
# --------------------------------------------------------------------------- #
ActFn = Callable[[Any, np.ndarray, np.random.Generator], np.ndarray]


def sac_act_fn(theta: np.ndarray, enc: Optional[ContextEncoder] = None) -> ActFn:
    return lambda env, state, rng: sample_actions(theta, state, rng, enc)


def estimate_value(env: Any, act_fn: ActFn, episodes: int, steps: int, seed: int = 0) -> Dict[str, float]:
    """Monte-Carlo estimate of J = E sum_t gamma^t r(s_t, a_t) (global reward = mean of r_i), truncated at `steps`."""
    rng = np.random.default_rng(seed)
    disc = env.gamma ** np.arange(steps)
    vals = []
    for _ in range(episodes):
        _, info = env.reset(seed=int(rng.integers(2 ** 31 - 1)))
        s, total = info["state"], 0.0
        for t in range(steps):
            _, r, _, _, info = env.step(act_fn(env, s, rng))
            total += disc[t] * float(r.mean())
            s = info["state"]
        vals.append(total)
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0}


def rollout_snapshots(env: Any, act_fn: ActFn, steps: int, seed: int, save: Callable[[int], None],
                      num_plots: int = 10) -> List[int]:
    """Run one rollout of `steps` steps; call save(t) after the steps listed by snapshot_steps()."""
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed)
    s, wanted, taken = info["state"], set(snapshot_steps(steps, num_plots)), []
    for t in range(1, steps + 1):
        _, _, _, _, info = env.step(act_fn(env, s, rng))
        s = info["state"]
        if t in wanted:
            save(t)
            taken.append(t)
    return taken
