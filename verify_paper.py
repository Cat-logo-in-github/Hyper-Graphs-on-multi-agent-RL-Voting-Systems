#!/usr/bin/env python3
"""
verify_paper.py -- does the code implement the paper's algorithm, and do the paper's theory claims hold in the SIS epidemic?

CHECK 1 (same algorithm)   A literal, loop-by-loop transcription of Algorithm 1 (lines 3-12, tuple-keyed
                           dictionaries, no vectorisation) is run on the same trajectory as `sac.py`; the resulting
                           theta must be identical (kappa = 0, 1, 2; policy radius 0 and 1).
CHECK 2 (Lemma 2a / Def. 1) Exact Q_i^theta of the SIS epidemic on a ring (n=8: 256 states x 256 joint actions) is
                           computed by solving the Bellman equation; the largest change of Q_i when only states/actions
                           OUTSIDE the kappa-hop neighbourhood change is compared with  (r_range/(1-gamma)) gamma^(kappa+1).
CHECK 3 (Lemma 3a / Thm 5) The SAC critic (the code's own TD loop, one long trajectory, alpha_t = h/(t+t0)) is compared with
                           the exact Q_i: the steady-state error should shrink with kappa (until sampling noise dominates).

Usage:  python verify_paper.py [--output-dir verify_results] [--critic-steps 40000] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sac import ContextEncoder, ScalableActorCritic  # noqa: E402
from sis_environment import SISGraphEnv  # noqa: E402


# --------------------------------------------------------------------------- #
# CHECK 1: literal transcription of Algorithm 1
# --------------------------------------------------------------------------- #
def reference_algorithm1(env, kappa, theta0, S, Act, R, h, t0, eta, m, enc):
    """Algorithm 1, lines 3-12, written exactly as the pseudocode (S: s(0..T), Act: a(0..T), R: r(0..T-1))."""
    n, T, gamma, A = env.num_nodes, S.shape[0] - 1, env.gamma, env.num_actions
    N = [sorted(env.get_neighborhood(i, kappa)) for i in range(n)]               # N_i^kappa
    ctx = lambda t, i: int(enc(S[t])[i])                                           # policy context of agent i
    zN = lambda i, t: tuple((int(S[t][j]), int(Act[t][j])) for j in N[i])          # (s_{N_i^k}(t), a_{N_i^k}(t))
    Qhat = [dict() for _ in range(n)]                                              # line 3: all-zero
    for t in range(1, T + 1):                                                      # lines 4-9
        alpha = h / (t - 1 + t0)                                                   # line 6
        for i in range(n):
            k_prev, k_now = zN(i, t - 1), zN(i, t)
            Qhat[i][k_prev] = (1 - alpha) * Qhat[i].get(k_prev, 0.0) + \
                alpha * (R[t - 1][i] + gamma * Qhat[i].get(k_now, 0.0))            # line 7 (line 8: others unchanged)
    theta = theta0.copy()
    g = np.zeros_like(theta)
    for i in range(n):                                                             # line 11
        for t in range(T + 1):
            c = ctx(t, i)
            p = np.exp(theta0[i, c] - theta0[i, c].max()); p /= p.sum()
            grad_log = -p.copy(); grad_log[int(Act[t][i])] += 1.0                  # grad_theta_i log softmax
            q = sum(Qhat[j].get(zN(j, t), 0.0) for j in N[i]) / n                  # (1/n) sum_{j in N_i^k} Qhat_j
            g[i, c] += gamma ** t * q * grad_log
    return theta + eta / np.sqrt(m + 1) * g                                        # line 12


def check_algorithm_identity(rng_seed=0):
    results = []
    for radius in (0, 1):
        for kappa in (0, 1, 2):
            env = SISGraphEnv(num_nodes=9, graph_type="ring", observation_radius=0, max_steps=31, seed=3)
            sac = ScalableActorCritic(env, kappa, horizon=30, seed=rng_seed, policy_radius=radius)
            sac.theta = np.random.default_rng(1).normal(0, 0.5, sac.theta.shape)   # non-trivial policy
            theta0 = sac.theta.copy()
            sac.iterate()                                                           # code under test
            S, Act, R = sac.last_trajectory
            ref = reference_algorithm1(env, kappa, theta0, S, Act, R, sac.h, sac.t0, sac.eta, 0, sac.enc)
            err = float(np.abs(ref - sac.theta).max())
            results.append({"kappa": kappa, "policy_radius": radius, "max_abs_diff": err, "pass": err < 1e-10})
    return results


# --------------------------------------------------------------------------- #
# exact Q-functions of the SIS epidemic on a small graph
# --------------------------------------------------------------------------- #
def exact_q(env, theta):
    """Exact Q_i^theta(s, a) for every agent (policy zeta_i(a|s_i) = softmax(theta_i[s_i]))."""
    n, A = env.num_nodes, env.num_actions
    NS, NA = 2 ** n, A ** n
    bits = lambda K, base: np.array([[(x // base ** i) % base for i in range(n)] for x in range(K)])
    sb, ab = bits(NS, 2), bits(NA, A)                                              # state / joint-action digits
    # P1[s, a, i] = P(node i infected at t+1 | s, a), from the environment's own transition law (eq. 3)
    P1 = np.zeros((NS, NA, n))
    for si in range(NS):
        for ai in range(NA):
            P1[si, ai] = env.next_infection_probabilities(ab[ai], sb[si])
    # reward table r(s_i, a_i) from the environment
    rt = np.zeros((2, A))
    for s_val in (0, 1):
        for a_val in range(A):
            env.reset(seed=0, initial_state=np.full(n, s_val))
            rt[s_val, a_val] = env.step(np.full(n, a_val))[1][0]
    zeta = np.exp(theta - theta.max(axis=-1, keepdims=True)); zeta /= zeta.sum(axis=-1, keepdims=True)  # (n, 2, A)
    # state transition under the policy, P_pi[s, s'] (product over nodes of the action-averaged marginals)
    p_mix = np.zeros((NS, n))
    for x in range(A):
        allx = int(sum(x * A ** i for i in range(n)))
        p_mix += zeta[np.arange(n)[None, :], sb, x] * P1[:, allx, :]
    P_pi = np.ones((NS, NS))
    for i in range(n):
        P_pi *= np.where(sb[None, :, i] == 1, p_mix[:, i:i + 1], 1 - p_mix[:, i:i + 1])
    # full transition P[s, a, s'] with deterministic joint action
    Pfull = np.ones((NS, NA, NS))
    for i in range(n):
        Pfull *= np.where(sb[None, None, :, i] == 1, P1[:, :, i:i + 1], 1 - P1[:, :, i:i + 1])
    Q = []
    for i in range(n):
        rbar = (zeta[i, sb[:, i], :] * rt[sb[:, i], :]).sum(axis=1)
        V = np.linalg.solve(np.eye(NS) - env.gamma * P_pi, rbar)
        Q.append(rt[sb[:, i][:, None], ab[:, i][None, :]] + env.gamma * Pfull @ V)  # (NS, NA)
    return np.stack(Q), sb, ab, rt


def check_decay(env, Q, sb, ab, rt, kappas):
    n, gamma = env.num_nodes, env.gamma
    r_range = float(rt.max() - rt.min())
    out = []
    for kappa in kappas:
        worst = 0.0
        for i in range(n):
            nb = np.asarray(sorted(env.get_neighborhood(i, kappa)))
            ks = (sb[:, nb] @ (2 ** np.arange(len(nb))))[:, None] * (env.num_actions ** len(nb)) + \
                 (ab[:, nb] @ (env.num_actions ** np.arange(len(nb))))[None, :]
            key, q = ks.ravel(), Q[i].ravel()
            order = np.argsort(key, kind="stable")
            key, q = key[order], q[order]
            starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
            spread = np.maximum.reduceat(q, starts) - np.minimum.reduceat(q, starts)
            worst = max(worst, float(spread.max()))
        bound = r_range / (1 - gamma) * gamma ** (kappa + 1)
        out.append({"kappa": kappa, "max_dependence_outside_Nk": worst, "lemma2a_bound": bound, "pass": worst <= bound + 1e-12})
    return out


def check_critic_error(env, theta, Q, sb, ab, kappas, steps, h=5.0, t0=10.0):
    n, A = env.num_nodes, env.num_actions
    out = []
    for kappa in kappas:
        sac = ScalableActorCritic(env, kappa, horizon=steps, critic_h=h, critic_t0=t0, seed=11)
        sac.theta = theta.copy()
        sac.iterate()                                                               # one critic run (policy fixed)
        S, Act, _ = sac.last_trajectory
        s_idx = (S * (2 ** np.arange(n))).sum(axis=1)
        a_idx = (Act * (A ** np.arange(n))).sum(axis=1)
        Z = S * sac.A + Act
        errs = []
        for i in range(n):
            keys, tab = sac._keys(i, Z), sac.q_tables[i]
            est = np.array([tab.get(k, 0.0) for k in keys])
            errs.append(np.abs(est - Q[i][s_idx, a_idx]))
        errs = np.stack(errs)
        out.append({"kappa": kappa, "mean_abs_error": float(errs.mean()), "p95_abs_error": float(np.percentile(errs, 95)),
                    "max_abs_error": float(errs.max()), "f_kappa": sac.f_kappa,
                    "table_entries_max": int(max(len(t) for t in sac.q_tables))})
    return out


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="verify_results")
    p.add_argument("--critic-steps", type=int, default=40000, help="T of the single critic run in check 3")
    p.add_argument("--quick", action="store_true", help="skip the (slow) exact-Q checks")
    args = p.parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    res = {}

    print("CHECK 1  code == literal transcription of Algorithm 1 (lines 3-12)")
    res["algorithm_identity"] = check_algorithm_identity()
    for r in res["algorithm_identity"]:
        print(f"   kappa={r['kappa']} policy_radius={r['policy_radius']}: max|theta_code - theta_pseudocode| = "
              f"{r['max_abs_diff']:.2e}  {'PASS' if r['pass'] else 'FAIL'}")
    if args.quick:
        json.dump(res, open(os.path.join(args.output_dir, "verify_results.json"), "w"), indent=1)
        return 0

    env = SISGraphEnv(num_nodes=8, graph_type="ring", observation_radius=0, max_steps=args.critic_steps + 1, seed=0)
    theta = np.random.default_rng(5).normal(0, 1.0, (8, 2, 2))                       # a generic non-uniform policy
    print("\n   computing exact Q-functions (8-node ring, 256 states x 256 joint actions) ...")
    Q, sb, ab, rt = exact_q(env, theta)
    # sanity: exact Q must satisfy the discounted-return definition (J = mean_i E V_i); report Bellman residual
    kap = [0, 1, 2, 3, 4]
    print("\nCHECK 2  Lemma 2(a): exponential decay of Q_i in the distance to the perturbed nodes")
    res["decay"] = check_decay(env, Q, sb, ab, rt, kap)
    for r in res["decay"]:
        print(f"   kappa={r['kappa']}: sup dependence outside N^kappa = {r['max_dependence_outside_Nk']:.5f}"
              f"  <= bound {r['lemma2a_bound']:.5f}  {'PASS' if r['pass'] else 'FAIL'}")
    d = [r["max_dependence_outside_Nk"] for r in res["decay"] if r["max_dependence_outside_Nk"] > 1e-12]
    if len(d) > 1:
        res["measured_decay_rate_rho"] = float(np.exp(np.polyfit(np.arange(len(d)), np.log(d), 1)[0]))
        print(f"   fitted decay rate rho ~ {res['measured_decay_rate_rho']:.3f}  (paper: rho <= gamma = {env.gamma})")

    print("\nCHECK 3  Theorem 5 / Lemma 3: critic error of Qhat^T vs exact Q (single run, T=%d)" % args.critic_steps)
    print("         Theorem 5 needs h >= (1/sigma) max(2, 1/(1-sqrt(gamma))), sigma = visit frequency of a table entry;")
    print("         we therefore compare a small step-size constant (h=5, t0=10) with a large one (h=200, t0=400).")
    res["critic_error"] = {}
    for label, (h, t0) in {"h=5,t0=10": (5.0, 10.0), "h=200,t0=400": (200.0, 400.0)}.items():
        res["critic_error"][label] = check_critic_error(env, theta, Q, sb, ab, kap, args.critic_steps, h, t0)
        for r in res["critic_error"][label]:
            print(f"   [{label}] kappa={r['kappa']} (f={r['f_kappa']}): mean|err| = {r['mean_abs_error']:.4f}, p95 = "
                  f"{r['p95_abs_error']:.4f}, max = {r['max_abs_error']:.4f}, entries <= {r['table_entries_max']}")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.semilogy(kap, [max(r["max_dependence_outside_Nk"], 1e-9) for r in res["decay"]], "o-", label="measured (exact Q)")
    ax.semilogy(kap, [r["lemma2a_bound"] for r in res["decay"]], "k--", label=r"Lemma 2(a): $\frac{\bar r}{1-\gamma}\gamma^{\kappa+1}$")
    ax.set_xlabel("κ"); ax.set_ylabel("sup |Q_i(s,a) − Q_i(s',a')|  (differences outside N_i^κ)")
    ax.set_title("Exponential decay property (Definition 1), SIS ring n=8"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "verify_decay.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label, rows in res["critic_error"].items():
        ax.semilogy(kap, [r["mean_abs_error"] for r in rows], "o-", label=f"mean |Q̂ − Q|, {label}")
    ax.set_xlabel("κ"); ax.set_ylabel("critic error vs exact Q")
    ax.set_title(f"Critic error of truncated Q̂ (T={args.critic_steps}), SIS ring n=8"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(args.output_dir, "verify_critic_error.png"), dpi=120); plt.close(fig)

    json.dump(res, open(os.path.join(args.output_dir, "verify_results.json"), "w"), indent=1, default=float)
    ok = all(r["pass"] for r in res["algorithm_identity"]) and all(r["pass"] for r in res["decay"])
    print("\nOVERALL (checks 1-2 are pass/fail; check 3 is reported):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
