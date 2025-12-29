"""
Multi-Armed Bandit Simulation with Epsilon-Greedy Strategy.

This module simulates the epsilon-greedy algorithm on a 10-armed bandit problem
with different action value distributions (Gaussian, Uniform, Random).
"""

import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    """
    A k-armed bandit environment with configurable action value distributions.
    """
    def __init__(self, k: int = 10, distribution: str = "gaussian"):
        """
        Initialize the bandit with k arms.

        Args:
            k: Number of arms (actions).
            distribution: How to initialize true action values.
                - "gaussian": q*(a) ~ N(0, 1)
                - "uniform": q*(a) evenly spaced in [-2, 2]
                - "random": q*(a) ~ Uniform(-2, 2)
        """
        self.k = k
        self.distribution = distribution
        self.q_true = self._init_action_values()
        self.optimal_action = np.argmax(self.q_true)

    def _init_action_values(self) -> np.ndarray:
        """Initialize true action values based on distribution type."""
        if self.distribution == "gaussian":
            return np.random.randn(self.k)
        elif self.distribution == "uniform":
            return np.linspace(-2, 2, self.k)
        elif self.distribution == "random":
            return np.random.uniform(-2, 2, self.k)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def step(self, action: int) -> float:
        """
        Take an action and receive a reward.

        Args:
            action: The arm to pull (0 to k-1).

        Returns:
            Reward sampled from N(q*(action), 1).
        """
        return np.random.randn() + self.q_true[action]

    def optimal_value(self) -> float:
        """Return the maximum true action value."""
        return np.max(self.q_true)


class EpsilonGreedyAgent:
    """
    An agent that uses epsilon-greedy action selection.
    """
    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Initialize the agent.

        Args:
            k: Number of actions.
            epsilon: Probability of exploring (choosing random action).
        """
        self.k = k
        self.epsilon = epsilon
        self.q_estimates = np.zeros(k)  # Estimated action values
        self.action_counts = np.zeros(k)  # Number of times each action taken

    def choose_action(self) -> int:
        """
        Select an action using epsilon-greedy policy.

        Returns:
            The selected action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            # Break ties randomly
            max_q = np.max(self.q_estimates)
            max_actions = np.where(self.q_estimates == max_q)[0]
            return np.random.choice(max_actions)

    def update(self, action: int, reward: float):
        """
        Update action value estimate using incremental mean.

        Args:
            action: The action that was taken.
            reward: The reward received.
        """
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_estimates[action] += (reward - self.q_estimates[action]) / n


def run_simulation(
    n_steps: int = 1000,
    n_runs: int = 2000,
    k: int = 10,
    epsilon: float = 0.1,
    distribution: str = "gaussian"
) -> np.ndarray:
    """
    Run multiple bandit simulations and return average rewards.

    Args:
        n_steps: Number of steps per run.
        n_runs: Number of independent runs to average over.
        k: Number of arms.
        epsilon: Exploration rate.
        distribution: Action value distribution type.

    Returns:
        Array of average rewards at each step (shape: n_steps).
    """
    rewards = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        bandit = MultiArmedBandit(k=k, distribution=distribution)
        agent = EpsilonGreedyAgent(k=k, epsilon=epsilon)

        for step in range(n_steps):
            action = agent.choose_action()
            reward = bandit.step(action)
            agent.update(action, reward)
            rewards[run, step] = reward

    return np.mean(rewards, axis=0)


def plot_results(distribution: str, n_steps: int = 1000, n_runs: int = 2000):
    """
    Plot average rewards for different epsilon values.

    Args:
        distribution: The action value distribution to use.
        n_steps: Number of steps per run.
        n_runs: Number of runs to average over.
    """
    epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

    plt.figure(figsize=(12, 6))

    # Calculate and plot optimal value (averaged over many bandits)
    optimal_values = []
    for _ in range(n_runs):
        bandit = MultiArmedBandit(k=10, distribution=distribution)
        optimal_values.append(bandit.optimal_value())
    avg_optimal = np.mean(optimal_values)

    plt.axhline(y=avg_optimal, color='black', linestyle='--',
                label=f'Optimal ({avg_optimal:.2f})', linewidth=2)

    # Run simulations for each epsilon
    for epsilon, color in zip(epsilons, colors):
        print(f"  Running epsilon={epsilon}...")
        avg_rewards = run_simulation(
            n_steps=n_steps,
            n_runs=n_runs,
            epsilon=epsilon,
            distribution=distribution
        )
        plt.plot(avg_rewards, color=color, label=f'ε={epsilon}', alpha=0.8)

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'Epsilon-Greedy on 10-Armed Bandit\n({distribution.capitalize()} Distribution, {n_runs} runs)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def main():
    """Run simulations for all distributions and save plots."""
    distributions = ['gaussian', 'uniform', 'random']
    n_steps = 1000
    n_runs = 2000

    print(f"Running {n_runs} simulations with {n_steps} steps each...\n")

    for dist in distributions:
        print(f"Distribution: {dist}")
        fig = plot_results(dist, n_steps=n_steps, n_runs=n_runs)
        filename = f'epsilon_greedy_{dist}.png'
        fig.savefig(filename, dpi=150)
        print(f"  Saved: {filename}\n")
        plt.close(fig)

    # Also create a combined figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']

    for idx, dist in enumerate(distributions):
        ax = axes[idx]

        # Calculate optimal value
        optimal_values = []
        for _ in range(n_runs):
            bandit = MultiArmedBandit(k=10, distribution=dist)
            optimal_values.append(bandit.optimal_value())
        avg_optimal = np.mean(optimal_values)

        ax.axhline(y=avg_optimal, color='black', linestyle='--',
                   label=f'Optimal ({avg_optimal:.2f})', linewidth=2)

        for epsilon, color in zip(epsilons, colors):
            avg_rewards = run_simulation(
                n_steps=n_steps,
                n_runs=n_runs,
                epsilon=epsilon,
                distribution=dist
            )
            ax.plot(avg_rewards, color=color, label=f'ε={epsilon}', alpha=0.8)

        ax.set_xlabel('Steps')
        ax.set_ylabel('Average Reward')
        ax.set_title(f'{dist.capitalize()} Distribution')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Epsilon-Greedy Comparison ({n_runs} runs)', fontsize=14)
    plt.tight_layout()
    fig.savefig('epsilon_greedy_comparison.png', dpi=150)
    print("Saved: epsilon_greedy_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
