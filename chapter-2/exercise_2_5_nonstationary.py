"""
Exercise 2.5: Nonstationary Bandit Problem

Demonstrates the difficulties of sample-average methods for nonstationary problems.

Setup:
- 10-armed bandit where all q*(a) start at 0
- Each step, all q*(a) take independent random walks: q*(a) += N(0, 0.01)
- Compare sample-average vs constant step-size (α=0.1) methods
- Both use ε=0.1 for exploration
- 10,000 steps per run

Reference: Sutton & Barto, Reinforcement Learning 2nd Edition, p.33
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class NonstationaryBandit:
    """10-armed bandit with random walk action values."""

    def __init__(self, n_arms=10, walk_std=0.01):
        self.n_arms = n_arms
        self.walk_std = walk_std
        self.reset()

    def reset(self):
        """Reset all action values to zero."""
        self.q_star = np.zeros(self.n_arms)

    def step(self):
        """Take a random walk step for all action values."""
        self.q_star += np.random.normal(0, self.walk_std, self.n_arms)

    def get_reward(self, action):
        """Get reward for an action (sampled from N(q*(a), 1))."""
        return np.random.normal(self.q_star[action], 1)

    def get_optimal_action(self):
        """Return the current optimal action."""
        return np.argmax(self.q_star)


class Agent:
    """Epsilon-greedy agent with configurable step-size."""

    def __init__(self, n_arms=10, epsilon=0.1, alpha=None):
        """
        Args:
            n_arms: Number of arms
            epsilon: Exploration probability
            alpha: Step-size parameter. If None, use sample average (1/n)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha  # None means sample average
        self.reset()

    def reset(self):
        """Reset Q estimates and action counts."""
        self.Q = np.zeros(self.n_arms)
        self.N = np.zeros(self.n_arms)

    def select_action(self):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            # Break ties randomly
            max_q = np.max(self.Q)
            return np.random.choice(np.where(self.Q == max_q)[0])

    def update(self, action, reward):
        """Update Q estimate for the given action."""
        self.N[action] += 1

        if self.alpha is None:
            # Sample average: step_size = 1/n
            step_size = 1.0 / self.N[action]
        else:
            # Constant step-size
            step_size = self.alpha

        self.Q[action] += step_size * (reward - self.Q[action])


def run_experiment(n_runs=2000, n_steps=10000, epsilon=0.1):
    """
    Run the nonstationary bandit experiment.

    Returns:
        Dictionary containing rewards and optimal action percentages
        for both sample-average and constant step-size methods.
    """
    # Track results for both methods
    rewards_sample_avg = np.zeros((n_runs, n_steps))
    rewards_constant = np.zeros((n_runs, n_steps))
    optimal_sample_avg = np.zeros((n_runs, n_steps))
    optimal_constant = np.zeros((n_runs, n_steps))

    for run in tqdm(range(n_runs), desc="Running experiments"):
        # Create bandit and agents
        bandit = NonstationaryBandit()
        agent_sample_avg = Agent(epsilon=epsilon, alpha=None)
        agent_constant = Agent(epsilon=epsilon, alpha=0.1)

        for step in range(n_steps):
            optimal_action = bandit.get_optimal_action()

            # Sample average agent
            action_sa = agent_sample_avg.select_action()
            reward_sa = bandit.get_reward(action_sa)
            agent_sample_avg.update(action_sa, reward_sa)
            rewards_sample_avg[run, step] = reward_sa
            optimal_sample_avg[run, step] = (action_sa == optimal_action)

            # Constant step-size agent
            action_c = agent_constant.select_action()
            reward_c = bandit.get_reward(action_c)
            agent_constant.update(action_c, reward_c)
            rewards_constant[run, step] = reward_c
            optimal_constant[run, step] = (action_c == optimal_action)

            # Random walk for next step
            bandit.step()

    return {
        'sample_avg': {
            'rewards': rewards_sample_avg.mean(axis=0),
            'optimal': optimal_sample_avg.mean(axis=0) * 100
        },
        'constant': {
            'rewards': rewards_constant.mean(axis=0),
            'optimal': optimal_constant.mean(axis=0) * 100
        }
    }


def plot_results(results, save_path='exercise_2_5_nonstationary.png'):
    """Create Figure 2.2 style plots comparing the two methods."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    steps = np.arange(1, len(results['sample_avg']['rewards']) + 1)

    # Plot average reward
    ax1.plot(steps, results['sample_avg']['rewards'],
             label='Sample Average', color='green', alpha=0.8)
    ax1.plot(steps, results['constant']['rewards'],
             label='Constant α=0.1', color='red', alpha=0.8)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Nonstationary 10-Armed Bandit: Average Reward')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot % optimal action
    ax2.plot(steps, results['sample_avg']['optimal'],
             label='Sample Average', color='green', alpha=0.8)
    ax2.plot(steps, results['constant']['optimal'],
             label='Constant α=0.1', color='red', alpha=0.8)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('% Optimal Action')
    ax2.set_title('Nonstationary 10-Armed Bandit: % Optimal Action')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Last 1000 steps performance (steady state)
    sa_reward = results['sample_avg']['rewards'][-1000:].mean()
    c_reward = results['constant']['rewards'][-1000:].mean()
    sa_optimal = results['sample_avg']['optimal'][-1000:].mean()
    c_optimal = results['constant']['optimal'][-1000:].mean()

    print(f"\nPerformance (last 1000 steps average):")
    print(f"{'Method':<20} {'Avg Reward':>12} {'% Optimal':>12}")
    print("-" * 44)
    print(f"{'Sample Average':<20} {sa_reward:>12.3f} {sa_optimal:>11.1f}%")
    print(f"{'Constant α=0.1':<20} {c_reward:>12.3f} {c_optimal:>11.1f}%")

    print(f"\nImprovement with constant step-size:")
    print(f"  Reward: +{((c_reward - sa_reward) / abs(sa_reward)) * 100:.1f}%")
    print(f"  Optimal action: +{c_optimal - sa_optimal:.1f} percentage points")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("Sample averages weight all past rewards equally, making them")
    print("slow to adapt when action values change. Constant step-size")
    print("gives more weight to recent rewards, tracking changes better.")
    print("=" * 60)


if __name__ == '__main__':
    print("Exercise 2.5: Nonstationary Bandit Problem")
    print("-" * 45)
    print("Comparing sample-average vs constant step-size methods")
    print("on a 10-armed bandit with random walk action values.\n")

    # Run experiment
    results = run_experiment(n_runs=2000, n_steps=10000, epsilon=0.1)

    # Generate plot
    plot_results(results)

    # Print summary
    print_summary(results)
