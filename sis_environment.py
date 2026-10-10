"""
sis_environment.py
==================

Multi-agent SIS (Susceptible-Infected-Susceptible) epidemic environment on a
graph, implementing the epidemic example of

    G. Qu, A. Wierman, N. Li,
    "Scalable Reinforcement Learning for Multi-Agent Networked Systems"
    (arXiv:1912.02906), Section 2.2, "Epidemic Network".

Model (equations (1), (3), (4) of the paper)
--------------------------------------------
* Graph G = (N, E): undirected. N_i is the neighbourhood of i INCLUDING i.
* Local state  s_i in {susceptible = 0, infected = 1}.
* Local action a_i in a finite set {0, ..., A-1}  (0 = no intervention).
* Synchronous transition, independent across nodes given (s(t), a(t)):

      P(s_i(t+1) = susceptible | s_{N_i}(t), a_i(t)) =
            (1 - beta_i(a_i(t))) ** m_i(t)     if s_i(t) = susceptible
            delta_i                            if s_i(t) = infected

  with m_i(t) = #{ j in N_i \\ {i} : s_j(t) = infected }.
  beta_i(a) is the transmission rate under action a, delta_i the recovery rate.
  Only the node's OWN action enters its transition (paper's Remark 1).
* Local reward  r_i(s_i, a_i) = 1(s_i = susceptible) - c_i(a_i),
  global reward r(s, a) = (1/n) * sum_i r_i(s_i, a_i).
* Objective: maximise the discounted global reward with discount gamma.

TIMING CONVENTION (important, follows Algorithm 1 of the paper):
    r_i(t) = r_i(s_i(t), a_i(t)) is evaluated on the state BEFORE the
    transition.  `step(a_t)` therefore returns the rewards of (s_t, a_t) and
    the observations of s_{t+1}.

Things the paper does NOT specify for the epidemic example and that are
therefore configurable defaults here (see README): beta, delta, c, the action
set, the discount for this example, the initial distribution, the graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUSCEPTIBLE: int = 0
INFECTED: int = 1
PAD_VALUE: int = -1  # padding value in padded observation arrays

GRAPH_TYPES: Tuple[str, ...] = ("ring", "line", "complete", "grid", "random")
DEFAULT_NUM_NODES: int = 20

_SUSCEPTIBLE_COLOR = "#4C9BE8"
_INFECTED_COLOR = "#E8574C"
_NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "ps", "svg", "pgf", "cairo", "template")
_MAX_LABELLED_NODES = 30  # draw node ids only on small graphs


def interactive_backend() -> bool:
    """True if the active Matplotlib backend can show windows."""
    import matplotlib

    return matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def grid_shape(num_nodes: int) -> Tuple[int, int]:
    """(rows, cols) of the grid used for `num_nodes` nodes.

    rows = floor(sqrt(n)), cols = ceil(n / rows).  If rows * cols > n the last
    (rows * cols - n) cells of the last row are simply left out, so any node
    count works and the graph stays connected.
    """
    rows = math.isqrt(num_nodes)
    cols = math.ceil(num_nodes / rows)
    return rows, cols


def build_graph(
    graph_type: str,
    num_nodes: int,
    random_edge_prob: float = 0.2,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Create one of the supported topologies with nodes labelled 0..n-1."""
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"Unknown graph_type '{graph_type}'. Choose one of {GRAPH_TYPES}.")
    if not isinstance(num_nodes, (int, np.integer)) or num_nodes < 1:
        raise ValueError(f"num_nodes must be a positive integer, got {num_nodes!r}.")
    num_nodes = int(num_nodes)

    if graph_type == "ring":
        if num_nodes < 3:
            raise ValueError("A ring needs at least 3 nodes.")
        g = nx.cycle_graph(num_nodes)
    elif graph_type == "line":
        g = nx.path_graph(num_nodes)
    elif graph_type == "complete":
        g = nx.complete_graph(num_nodes)
    elif graph_type == "grid":
        rows, cols = grid_shape(num_nodes)
        g = nx.grid_2d_graph(rows, cols)
        g = nx.relabel_nodes(g, {(r, c): r * cols + c for r in range(rows) for c in range(cols)})
        g.remove_nodes_from([v for v in list(g.nodes) if v >= num_nodes])
    else:  # "random": Erdos-Renyi G(n, p)
        if not 0.0 <= random_edge_prob <= 1.0:
            raise ValueError("random_edge_prob must be in [0, 1].")
        g = nx.erdos_renyi_graph(num_nodes, random_edge_prob, seed=seed)

    # Canonical form: nodes 0..n-1 inserted in order.
    canonical = nx.Graph()
    canonical.add_nodes_from(range(num_nodes))
    canonical.add_edges_from(g.edges())
    return canonical


def default_positions(graph: nx.Graph, graph_type: str) -> Dict[int, Tuple[float, float]]:
    """Node positions used for drawing."""
    n = graph.number_of_nodes()
    if graph_type == "grid":
        _, cols = grid_shape(n)
        return {i: (float(i % cols), -float(i // cols)) for i in range(n)}
    if graph_type == "line":
        return {i: (float(i), 0.0) for i in range(n)}
    if graph_type in ("ring", "complete"):
        return nx.circular_layout(graph)
    return nx.spring_layout(graph, seed=0)


# --------------------------------------------------------------------------- #
# Small validation helpers
# --------------------------------------------------------------------------- #
def _check_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return value


def _as_table(values: Any, num_nodes: int, name: str) -> np.ndarray:
    """Return a (num_nodes, num_actions) float table from a 1-D or 2-D input."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = np.tile(arr, (num_nodes, 1))
    elif arr.ndim != 2 or arr.shape[0] != num_nodes:
        raise ValueError(
            f"{name} must have shape (num_actions,) or (num_nodes={num_nodes}, num_actions); "
            f"got {np.asarray(values).shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _as_node_vector(values: Any, num_nodes: int, name: str) -> np.ndarray:
    """Return a (num_nodes,) float vector from a scalar or a length-n array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(num_nodes, float(arr))
    if arr.shape != (num_nodes,):
        raise ValueError(f"{name} must be a scalar or have shape ({num_nodes},); got {arr.shape}.")
    return arr


@dataclass(frozen=True)
class LocalTransition:
    """One transition seen from agent i's kappa-hop neighbourhood.

    All arrays are ordered like `nodes` (nodes[0] is the agent itself, then by
    hop distance, then by node index).  This is exactly the data needed by the
    truncated-Q TD update of Algorithm 1:
        (s_{N_i^k}(t), a_{N_i^k}(t), r_i(t), s_{N_i^k}(t+1)).
    """

    agent: int
    nodes: np.ndarray
    state: np.ndarray       # s_{N}(t)
    actions: np.ndarray     # a_{N}(t)
    reward: float           # r_i(t) = r_i(s_i(t), a_i(t))
    next_state: np.ndarray  # s_{N}(t+1)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class SISGraphEnv:
    """Synchronous multi-agent SIS environment on an undirected graph.

    Parameters
    ----------
    num_nodes : number of agents (ignored if `graph` is given; must then match).
    graph_type : "ring", "line", "complete", "grid" or "random".
    graph : optional custom undirected `networkx.Graph` (overrides graph_type).
    infection_rate : beta(0), transmission rate with no intervention.   [default]
    intervention_effect : relative reduction of beta at the strongest action:
        beta(a) = infection_rate * (1 - intervention_effect * a / (A-1)).  [default]
    intervention_cost : cost of the strongest action:
        c(a) = intervention_cost * a / (A-1).                             [default]
        (For A = 2 this gives beta(0)=infection_rate, beta(1)=infection_rate *
        (1-effect), c(0)=0, c(1)=intervention_cost.)
    recovery_rate : delta_i, scalar or per-node array.                    [default]
    transmission_rates, action_costs : optional explicit tables beta_i(a), c_i(a)
        of shape (A,) or (n, A); override the three parameters above.
    num_actions : A (default 2, or inferred from the explicit tables).
    observation_radius : agent i observes the states of its radius-hop
        neighbourhood (0 = own state only, as in the paper's base model).
    max_steps : episode length (the paper's problem is infinite-horizon; the
        episode is simply truncated after `max_steps` steps).
    gamma : discount factor (paper uses 0.7 in its experiments).
    initial_infection_prob : each node starts infected independently with this
        probability (pi_0 is not specified for the epidemic example).
    terminate_when_extinct : if True, `terminated=True` as soon as no node is
        infected (the all-susceptible state is absorbing).
    random_edge_prob : edge probability for graph_type="random".
    seed : seed for the graph (random type) and for the dynamics.
    """

    def __init__(
        self,
        num_nodes: Optional[int] = None,
        graph_type: str = "ring",
        graph: Optional[nx.Graph] = None,
        *,
        infection_rate: float = 0.5,
        recovery_rate: Union[float, Sequence[float]] = 0.3,
        intervention_effect: float = 0.8,
        intervention_cost: float = 0.2,
        transmission_rates: Optional[Any] = None,
        action_costs: Optional[Any] = None,
        num_actions: Optional[int] = None,
        observation_radius: int = 0,
        max_steps: int = 100,
        gamma: float = 0.7,
        initial_infection_prob: float = 0.2,
        terminate_when_extinct: bool = False,
        random_edge_prob: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        # ---- graph --------------------------------------------------------
        if graph is not None:
            if not isinstance(graph, nx.Graph):
                raise TypeError("graph must be a networkx.Graph.")
            if graph.is_directed():
                raise ValueError("Directed graphs are not supported; pass an undirected nx.Graph.")
            if graph.number_of_nodes() == 0:
                raise ValueError("graph must contain at least one node.")
            if num_nodes is not None and num_nodes != graph.number_of_nodes():
                raise ValueError(
                    f"num_nodes={num_nodes} does not match the given graph "
                    f"({graph.number_of_nodes()} nodes)."
                )
            g = nx.Graph(graph)  # simple-graph copy
            g.remove_edges_from(list(nx.selfloop_edges(g)))
            g = nx.convert_node_labels_to_integers(g, ordering="default")
            self.graph_type = "custom"
        else:
            if num_nodes is None:
                num_nodes = DEFAULT_NUM_NODES
            g = build_graph(graph_type, num_nodes, random_edge_prob, seed)
            self.graph_type = graph_type
        self._graph: nx.Graph = g
        self.num_nodes: int = g.number_of_nodes()
        n = self.num_nodes
        self._agents = np.arange(n)
        self._adj = nx.to_numpy_array(g, nodelist=range(n), dtype=float)
        self._pos = default_positions(g, self.graph_type)

        # hop distances (inf if unreachable)
        dist = np.full((n, n), np.inf)
        for src, lengths in nx.all_pairs_shortest_path_length(g):
            for dst, d in lengths.items():
                dist[src, dst] = d
        self._dist = dist

        # ---- epidemic parameters -----------------------------------------
        infection_rate = _check_unit_interval(infection_rate, "infection_rate")
        intervention_effect = _check_unit_interval(intervention_effect, "intervention_effect")
        if intervention_cost < 0:
            raise ValueError("intervention_cost must be >= 0.")
        self.recovery_rates = _as_node_vector(recovery_rate, n, "recovery_rate")
        if np.any(self.recovery_rates < 0) or np.any(self.recovery_rates > 1):
            raise ValueError("recovery_rate must lie in [0, 1].")

        given = [np.asarray(t, dtype=float) for t in (transmission_rates, action_costs) if t is not None]
        for arr in given:
            if arr.ndim not in (1, 2):
                raise ValueError(
                    "transmission_rates / action_costs must have shape (A,) or (num_nodes, A)."
                )
        sizes = {arr.shape[-1] for arr in given}
        if len(sizes) > 1:
            raise ValueError("transmission_rates and action_costs imply different numbers of actions.")
        if sizes:
            a_count = sizes.pop()
            if num_actions is not None and num_actions != a_count:
                raise ValueError(f"num_actions={num_actions} conflicts with the tables ({a_count} actions).")
        else:
            a_count = 2 if num_actions is None else int(num_actions)
        if a_count < 2:
            raise ValueError("At least 2 actions are required (0 = no intervention, ...).")
        self.num_actions: int = a_count

        levels = np.arange(a_count) / (a_count - 1)
        if transmission_rates is None:
            transmission_rates = infection_rate * (1.0 - intervention_effect * levels)
        if action_costs is None:
            action_costs = intervention_cost * levels
        self.transmission_rates = _as_table(transmission_rates, n, "transmission_rates")
        self.action_costs = _as_table(action_costs, n, "action_costs")
        if np.any(self.transmission_rates < 0) or np.any(self.transmission_rates > 1):
            raise ValueError("transmission_rates must lie in [0, 1].")
        if np.any(self.action_costs < 0):
            raise ValueError("action_costs must be >= 0.")

        # ---- episode / observation settings -------------------------------
        if not isinstance(max_steps, (int, np.integer)) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1).")
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.initial_infection_prob = _check_unit_interval(initial_infection_prob, "initial_infection_prob")
        self.terminate_when_extinct = bool(terminate_when_extinct)

        self.observation_radius = self._check_radius(observation_radius)
        self.observation_nodes: List[np.ndarray] = [
            self.get_neighborhood(i, self.observation_radius) for i in range(n)
        ]
        self.observation_distances: List[np.ndarray] = [
            self._dist[i, nodes].astype(int) for i, nodes in enumerate(self.observation_nodes)
        ]
        self.max_observation_size: int = max(len(nodes) for nodes in self.observation_nodes)

        # ---- random numbers and dynamic state ------------------------------
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(n, dtype=np.int8)
        self._t = 0
        self._has_state = False
        self._episode_over = False
        self._cumulative_reward = 0.0
        self._last_step: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_radius(radius: int) -> int:
        if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
            raise ValueError(f"radius must be a non-negative integer, got {radius!r}.")
        return int(radius)

    def _require_state(self) -> None:
        if not self._has_state:
            raise RuntimeError("Call reset() before using the environment.")

    def _validate_state(self, state: Any) -> np.ndarray:
        arr = np.asarray(state)
        if arr.shape != (self.num_nodes,):
            raise ValueError(f"state must have shape ({self.num_nodes},); got {arr.shape}.")
        if not np.all(np.isin(arr, (SUSCEPTIBLE, INFECTED))):
            raise ValueError("state entries must be 0 (susceptible) or 1 (infected).")
        return arr.astype(np.int8)

    def _validate_actions(self, actions: Any) -> np.ndarray:
        n = self.num_nodes
        if isinstance(actions, Mapping):
            if set(actions.keys()) != set(range(n)):
                raise ValueError(f"actions dict must have exactly the keys 0..{n - 1}.")
            arr = np.array([actions[i] for i in range(n)])
        else:
            arr = np.asarray(actions)
        if arr.shape != (n,):
            raise ValueError(f"actions must have shape ({n},) (one action per agent); got shape {arr.shape}.")
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int64)
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"actions must be integers in [0, {self.num_actions - 1}]; got dtype {arr.dtype}.")
        if arr.min() < 0 or arr.max() >= self.num_actions:
            raise ValueError(
                f"actions must lie in [0, {self.num_actions - 1}]; got values {sorted(set(arr.tolist()))}."
            )
        return arr.astype(np.int64)

    # ------------------------------------------------------------------ #
    # Graph / neighbourhood queries
    # ------------------------------------------------------------------ #
    def get_graph(self) -> nx.Graph:
        """A copy of the interaction graph (nodes are 0..n-1)."""
        return self._graph.copy()

    def get_adjacency(self) -> np.ndarray:
        """Copy of the (n, n) 0/1 adjacency matrix (no self loops)."""
        return self._adj.copy()

    def get_neighborhood(self, agent: int, radius: int) -> np.ndarray:
        """Nodes within `radius` hops of `agent` (INCLUDING the agent, N_i^radius).

        Ordered by (hop distance, node index), so element 0 is the agent itself.
        """
        radius = self._check_radius(radius)
        if not 0 <= agent < self.num_nodes:
            raise ValueError(f"agent must be in [0, {self.num_nodes - 1}], got {agent}.")
        d = self._dist[agent]
        idx = np.flatnonzero(d <= radius)
        order = np.lexsort((idx, d[idx]))
        return idx[order]

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(
        self, seed: Optional[int] = None, initial_state: Optional[Any] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """Start a new episode.

        `seed` re-seeds the dynamics generator (the graph is NOT regenerated).
        `initial_state` optionally fixes s(0); otherwise each node is infected
        independently with probability `initial_infection_prob`.

        Returns (observations, info).
        """
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        if initial_state is None:
            state = (self._rng.random(self.num_nodes) < self.initial_infection_prob).astype(np.int8)
        else:
            state = self._validate_state(initial_state)
        self._state = state
        self._t = 0
        self._has_state = True
        self._episode_over = False
        self._cumulative_reward = 0.0
        self._last_step = None
        info = self.get_metrics()
        info["state"] = self.get_state()
        return self.get_observations(), info

    def next_infection_probabilities(self, actions: Any, state: Optional[Any] = None) -> np.ndarray:
        """P(s_i(t+1) = infected | s(t), a(t)) for every node i (equation (3)).

        susceptible i:  1 - (1 - beta_i(a_i)) ** (#infected neighbours)
        infected i:     1 - delta_i
        """
        a = self._validate_actions(actions)
        if state is None:
            self._require_state()
            s = self._state
        else:
            s = self._validate_state(state)
        infected_neighbors = self._adj @ s.astype(float)  # m_i(t); self excluded (no self loops)
        beta = self.transmission_rates[self._agents, a]
        p_if_susceptible = 1.0 - np.power(1.0 - beta, infected_neighbors)
        p_if_infected = 1.0 - self.recovery_rates
        return np.where(s == INFECTED, p_if_infected, p_if_susceptible)

    def step(
        self, actions: Any
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray, bool, bool, Dict[str, Any]]:
        """Advance all agents simultaneously from s(t) to s(t+1).

        Parameters
        ----------
        actions : int array / list of shape (n,) or dict {agent: action}.

        Returns
        -------
        observations : dict {agent: states of its observation neighbourhood at t+1}
        rewards : (n,) array, r_i(s_i(t), a_i(t)) (local rewards, state BEFORE transition)
        terminated, truncated : bool
        info : dict (see README) with prev_state, state, actions, rewards,
               global_reward, num_infected, fraction_infected, ...
        """
        self._require_state()
        if self._episode_over:
            raise RuntimeError("Episode is over; call reset() before step().")
        a = self._validate_actions(actions)
        s = self._state

        # Local rewards use (s_t, a_t): r_i = 1(s_i = susceptible) - c_i(a_i)
        costs = self.action_costs[self._agents, a]
        rewards = (s == SUSCEPTIBLE).astype(float) - costs

        # Synchronous transition: every node uses the SAME s(t).
        p_infected = self.next_infection_probabilities(a, s)
        u = self._rng.random(self.num_nodes)
        s_next = (u < p_infected).astype(np.int8)

        self._state = s_next
        self._t += 1
        self._cumulative_reward += float(rewards.mean())
        self._last_step = (s.copy(), a.copy(), rewards.copy(), s_next.copy())

        terminated = bool(self.terminate_when_extinct and s_next.sum() == 0)
        truncated = bool(self._t >= self.max_steps)
        self._episode_over = terminated or truncated

        info = self.get_metrics()
        info.update(
            {
                "prev_state": s.copy(),
                "state": s_next.copy(),
                "actions": a.copy(),
                "rewards": rewards.copy(),
                "global_reward": float(rewards.mean()),
                "total_intervention_cost": float(costs.sum()),
                "mean_intervention_cost": float(costs.mean()),
                "num_interventions": int(np.count_nonzero(a)),
                "intervention_rate": float(np.count_nonzero(a) / self.num_nodes),
                "infection_probabilities": p_infected.copy(),
            }
        )
        return self.get_observations(), rewards, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # State, observations, metrics
    # ------------------------------------------------------------------ #
    @property
    def t(self) -> int:
        """Number of steps taken in the current episode."""
        return self._t

    def get_state(self) -> np.ndarray:
        """Global state s(t) as an int8 array of shape (n,) (0 = S, 1 = I)."""
        self._require_state()
        return self._state.copy()

    def get_observations(self) -> Dict[int, np.ndarray]:
        """Local observations {agent: states of its `observation_radius`-hop neighbourhood}.

        The array of agent i lists the states of `observation_nodes[i]`, ordered
        by (hop distance, node index); entry 0 is always the agent's own state.
        """
        self._require_state()
        return {i: self._state[nodes] for i, nodes in enumerate(self.observation_nodes)}

    def get_padded_observations(self) -> np.ndarray:
        """Same data as `get_observations` as an (n, max_observation_size) int8 array
        padded on the right with PAD_VALUE (-1)."""
        self._require_state()
        out = np.full((self.num_nodes, self.max_observation_size), PAD_VALUE, dtype=np.int8)
        for i, nodes in enumerate(self.observation_nodes):
            out[i, : len(nodes)] = self._state[nodes]
        return out

    def get_neighborhood_state(self, agent: int, kappa: int) -> np.ndarray:
        """Current s_{N_i^kappa} (ordered like `get_neighborhood(agent, kappa)`)."""
        self._require_state()
        return self._state[self.get_neighborhood(agent, kappa)]

    def get_local_transition(self, agent: int, kappa: int) -> LocalTransition:
        """Last transition (s_t, a_t, r_t, s_{t+1}) restricted to N_agent^kappa."""
        if self._last_step is None:
            raise RuntimeError("No transition yet; call step() first.")
        prev, a, r, nxt = self._last_step
        nodes = self.get_neighborhood(agent, kappa)
        return LocalTransition(
            agent=agent,
            nodes=nodes,
            state=prev[nodes].copy(),
            actions=a[nodes].copy(),
            reward=float(r[agent]),
            next_state=nxt[nodes].copy(),
        )

    def get_local_transitions(self, kappa: int) -> List[LocalTransition]:
        """`get_local_transition(i, kappa)` for every agent i."""
        return [self.get_local_transition(i, kappa) for i in range(self.num_nodes)]

    def get_metrics(self) -> Dict[str, Any]:
        """Statistics of the CURRENT state and of the reward collected so far."""
        self._require_state()
        k = int(self._state.sum())
        n = self.num_nodes
        return {
            "t": self._t,
            "num_infected": k,
            "num_susceptible": n - k,
            "fraction_infected": k / n,
            "cumulative_reward": self._cumulative_reward,  # sum_t r(s_t, a_t), undiscounted
            "average_reward": self._cumulative_reward / self._t if self._t > 0 else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, ax: Optional[Any] = None, pause: float = 0.0) -> Any:
        """Draw the graph; blue = susceptible, red = infected.

        If `ax` is given, draw into it; otherwise reuse (or create) the
        environment's own figure.  `pause` (seconds) is only applied to the
        environment's own figure and only on interactive backends.
        """
        self._require_state()
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        own_figure = ax is None
        if own_figure:
            if self._fig is None or not plt.fignum_exists(self._fig.number):
                self._fig, self._ax = plt.subplots(figsize=(6, 6))
            ax = self._ax
        ax.clear()

        n = self.num_nodes
        colors = [_INFECTED_COLOR if s == INFECTED else _SUSCEPTIBLE_COLOR for s in self._state]
        node_size = float(np.clip(6000.0 / n, 40.0, 600.0))
        nx.draw_networkx_edges(self._graph, self._pos, ax=ax, edge_color="0.7", width=1.0)
        nx.draw_networkx_nodes(
            self._graph, self._pos, nodelist=list(range(n)), node_color=colors,
            edgecolors="black", node_size=node_size, ax=ax,
        )
        if n <= _MAX_LABELLED_NODES:
            nx.draw_networkx_labels(self._graph, self._pos, font_size=8, ax=ax)
        k = int(self._state.sum())
        ax.set_title(f"t = {self._t}   infected {k}/{n}")
        ax.set_axis_off()
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=_SUSCEPTIBLE_COLOR,
                   markeredgecolor="black", label="Susceptible"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=_INFECTED_COLOR,
                   markeredgecolor="black", label="Infected"),
        ]
        ax.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)

        if own_figure:
            self._fig.canvas.draw_idle()
            if pause > 0 and interactive_backend():
                plt.pause(pause)
        return ax

    def close(self) -> None:
        """Close the rendering window, if any."""
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def __repr__(self) -> str:
        return (
            f"SISGraphEnv(num_nodes={self.num_nodes}, graph_type='{self.graph_type}', "
            f"num_actions={self.num_actions}, observation_radius={self.observation_radius}, "
            f"max_steps={self.max_steps}, gamma={self.gamma})"
        )
