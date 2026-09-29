# Higher-Order Interactions in Multi-Agent Reinforcement Learning for Strategic Voting

 A course project exploring how the **structure of interactions between voters** affects behavior learned by multi-agent reinforcement learning (MARL) agents.

 We compare **pairwise graph interactions** with **higher-order hypergraph interactions** in a simple iterative voting environment, while keeping the voting mechanism and learning algorithm fixed.

 ## Research Question

 > **Do group interactions change the voting strategies learned by RL agents compared with pairwise interactions?**

 More specifically, we study whether changing the interaction topology affects:

 - Social welfare
- Consensus
- Voting stability
- Individual welfare and inequality

 ## Core Idea

 Traditional graphs represent pairwise relationships:

```
A ─ B ─ C ─ D
```

 Hypergraphs allow multiple agents to participate in the same interaction:

```
{A, B, C}
{B, C, D}
```

 Our goal is to investigate whether these **higher-order interactions** lead to different emergent voting behavior.

 ## Experimental Setup

 The initial environment will use:

 - 6 agents
- 3 candidates
- Synthetic preference profiles
- Iterative plurality voting
- Discrete voting actions
- Graph and hypergraph interaction structures
- PPO/MAPPO-based MARL
- Multiple random seeds for evaluation

 The main comparison will be:

```
No interaction
      │
      ▼
Pairwise graph
      │
      ▼
3-agent hypergraph
      │
      ▼
4-agent hypergraph
```

 The learning algorithm, voting mechanism, preference distribution, and training budget will otherwise remain fixed.

 ## Research Motivation

 This project sits at the intersection of:

 - **Multi-Agent Reinforcement Learning**
- **Graph Theory**
- **Hypergraph Theory**
- **Computational Social Choice**
- **Game Theory / Strategic Behavior**

 Previous work has studied reinforcement learning for strategic voting and hypergraphs for multi-agent coordination. This project explores the intersection by treating interaction topology as an experimental variable in a voting environment.

 ## Planned Experiments

 ### Experiment 1 — Baseline

 Compare learned agents against random/non-learning voting behavior.

 ### Experiment 2 — Graph vs Hypergraph

 Compare pairwise interactions with higher-order group interactions.

 ### Experiment 3 — Hyperedge Size

 Study how changing the size of interacting groups affects learned behavior.

 ### Evaluation

 We will measure:

 - Average social welfare
- Individual utility
- Consensus rate
- Vote-switching rate
- Welfare inequality
- Learning convergence

 ## Project Structure

```
.
├── environment/
│   ├── voting_env.py
│   ├── graph.py
│   └── hypergraph.py
│
├── agents/
│   ├── random_agent.py
│   └── marl_agent.py
│
├── experiments/
│   ├── baseline.py
│   ├── graph_vs_hypergraph.py
│   └── hyperedge_size.py
│
├── analysis/
│   ├── metrics.py
│   └── plots.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│
├── requirements.txt
└── README.md
```

 ## Scope

 This is a **course project**, so the objective is not to develop a new MARL algorithm or reproduce state-of-the-art hypergraph neural networks.

 Instead, we aim to build a small, controlled environment and answer a focused empirical question about the relationship between **interaction topology and learned strategic behavior**.

 ## References

 The project is informed by research in:

 - Iterative and strategic voting
- Computational social choice
- Multi-agent reinforcement learning
- Graph-based MARL
- Hypergraph-based MARL
- Opinion dynamics on hypergraphs

 Relevant papers and resources will be collected in `references/` as the project develops.

 ## Status

 🚧 **Work in progress**

 Currently defining the environment, interaction models, and experimental protocol.
