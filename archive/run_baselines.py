import numpy as np
from sis_environment import SISGraphEnv
from policies import NeverIntervene, AlwaysIntervene, LocalHeuristic, RandomPolicy

def evaluate_policy(policy, graph_type="ring", num_nodes=20, episodes=10, max_steps=100):
    rewards, infections, peaks = [], [], []
    for ep in range(episodes):
        env = SISGraphEnv(num_nodes=num_nodes, graph_type=graph_type, observation_radius=1, max_steps=max_steps, seed=ep)
        obs, _ = env.reset(seed=ep)
        policy.reset()
        
        ep_reward = 0
        ep_infections = []
        
        done = False
        while not done:
            actions = policy.act(env, obs)
            obs, r, term, trunc, info = env.step(actions)
            ep_reward += info["global_reward"]
            ep_infections.append(info["fraction_infected"])
            done = term or trunc
            
        rewards.append(ep_reward)
        infections.append(np.mean(ep_infections))
        peaks.append(np.max(ep_infections))
        
    return np.mean(rewards), np.mean(infections), np.mean(peaks)

def main():
    policies = [NeverIntervene(), AlwaysIntervene(), LocalHeuristic(), RandomPolicy(seed=42)]
    print(f"{'Policy':<20} | {'Avg Reward':<12} | {'Avg Infection':<15} | {'Peak Infection':<15}")
    print("-" * 70)
    for pol in policies:
        avg_r, avg_inf, peak_inf = evaluate_policy(pol)
        print(f"{pol.name:<20} | {avg_r:<12.2f} | {avg_inf:<15.2f} | {peak_inf:<15.2f}")

if __name__ == "__main__":
    main()