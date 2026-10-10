import matplotlib.pyplot as plt
from sis_environment import SISGraphEnv
from policies import LocalHeuristic

def main():
    env = SISGraphEnv(num_nodes=20, graph_type="ring", observation_radius=1, seed=42)
    obs, info = env.reset(seed=42)
    policy = LocalHeuristic()
    
    history_infected = []
    
    print("Running epidemic simulation demo...")
    for step in range(50):
        actions = policy.act(env, obs)
        obs, rewards, term, trunc, info = env.step(actions)
        history_infected.append(info["fraction_infected"])
        if step % 10 == 0:
            print(f"Step {step:02d} | Infected Fraction: {info['fraction_infected']:.2f} | Mean Reward: {info['global_reward']:.2f}")
        if term or trunc:
            break
            
    plt.figure(figsize=(8, 4))
    plt.plot(history_infected, marker='o')
    plt.title("Epidemic Progression (Fraction Infected Over Time)")
    plt.xlabel("Timestep")
    plt.ylabel("Infected Fraction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("demo_plot.png")
    print("Demo completed. Plot saved as demo_plot.png.")

if __name__ == "__main__":
    main()