"""
Exercise 2.8: UCB (Upper Confidence Bound) Action Selection Analysis

Simulates UCB action selection with 10 arms:
    A_t = argmax_a [Q_t(a) + c * sqrt(ln(t) / N_t(a))]

Where:
    - Q_t(a): Estimated value of action a (initialized from N(0,1))
    - N_t(a): Number of times action a has been selected
    - c: Exploration parameter (default: 2)
    - t: Current time step

Reference: Sutton & Barto, Reinforcement Learning 2nd Edition
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class UCBBandit:
    """UCB action selection for a 10-armed bandit."""

    def __init__(self, n_actions=10, c=2):
        self.n_actions = n_actions
        self.c = c
        self.reset()

    def reset(self):
        """Initialize true values q*(a) from N(0,1) and reset estimates."""
        self.q_star = np.random.normal(0, 1, self.n_actions)  # True action values
        self.Qt = np.zeros(self.n_actions)  # Estimated values (start at 0)
        self.Nt = np.zeros(self.n_actions)  # Action counts
        self.t = 0  # Time step

    def get_reward(self, action):
        """Get reward for action (sampled from N(q*(a), 1))."""
        return np.random.normal(self.q_star[action], 1)

    def get_optimal_action(self):
        """Return the optimal action."""
        return np.argmax(self.q_star)

    def ucb_values(self):
        """Calculate UCB value for each action."""
        ucb = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.Nt[a] == 0:
                ucb[a] = float('inf')  # Explore unvisited actions first
            else:
                exploration_bonus = self.c * np.sqrt(np.log(self.t) / self.Nt[a])
                ucb[a] = self.Qt[a] + exploration_bonus
        return ucb

    def select_action(self):
        """Select action using UCB."""
        self.t += 1
        ucb = self.ucb_values()
        # Break ties randomly
        max_ucb = np.max(ucb)
        action = np.random.choice(np.where(ucb == max_ucb)[0])
        return action

    def update(self, action, reward):
        """Update Q estimate using incremental sample average."""
        self.Nt[action] += 1
        # Incremental update: Q = Q + (1/n)(R - Q)
        self.Qt[action] += (reward - self.Qt[action]) / self.Nt[action]

    def get_exploration_bonus(self):
        """Get exploration bonus for each action."""
        bonus = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.Nt[a] > 0 and self.t > 0:
                bonus[a] = self.c * np.sqrt(np.log(self.t) / self.Nt[a])
            else:
                bonus[a] = float('inf')
        return bonus


def run_simulation(n_steps=1000, n_runs=100, c=2, seed=42):
    """Run UCB simulation with multiple independent runs and average results."""
    np.random.seed(seed)

    # Track results across all runs
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps))

    for run in tqdm(range(n_runs), desc="Running simulations"):
        bandit = UCBBandit(n_actions=10, c=c)
        optimal_action = bandit.get_optimal_action()

        for step in range(n_steps):
            action = bandit.select_action()
            reward = bandit.get_reward(action)
            bandit.update(action, reward)

            all_rewards[run, step] = reward
            all_optimal[run, step] = 1 if action == optimal_action else 0

    # Average across runs
    avg_reward_per_step = all_rewards.mean(axis=0)
    avg_optimal_per_step = all_optimal.mean(axis=0) * 100  # Convert to percentage

    # Cumulative average reward
    cumulative_avg_reward = np.cumsum(avg_reward_per_step) / np.arange(1, n_steps + 1)

    history = {
        'n_runs': n_runs,
        'n_steps': n_steps,
        'avg_reward_per_step': avg_reward_per_step,
        'cumulative_avg_reward': cumulative_avg_reward,
        'avg_optimal_per_step': avg_optimal_per_step,
        'cumulative_optimal_pct': np.cumsum(all_optimal.mean(axis=0)) / np.arange(1, n_steps + 1) * 100
    }

    return history


def plot_results(history, save_path='exercise_2_8_ucb_analysis.png'):
    """Plot UCB simulation results averaged over multiple runs."""
    n_steps = history['n_steps']
    n_runs = history['n_runs']
    focus_steps = min(100, n_steps)
    spark_steps = 30  # Highlight first 30 steps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average Reward per step (first 100 steps)
    ax1 = axes[0]
    # Plot full range in lighter color
    ax1.plot(range(1, focus_steps + 1), history['avg_reward_per_step'][:focus_steps],
             color='blue', linewidth=1.5, alpha=0.4)
    # Highlight first 30 steps with markers
    ax1.plot(range(1, spark_steps + 1), history['avg_reward_per_step'][:spark_steps],
             color='red', linewidth=2.5, label=f'First {spark_steps} steps')
    ax1.scatter(range(1, spark_steps + 1), history['avg_reward_per_step'][:spark_steps],
                color='red', s=30, zorder=5)
    # Mark the spike/dip points
    rewards_30 = history['avg_reward_per_step'][:spark_steps]
    max_idx = np.argmax(rewards_30)
    min_idx = np.argmin(rewards_30)
    ax1.scatter([max_idx + 1], [rewards_30[max_idx]], color='gold', s=150,
                marker='*', zorder=6, edgecolors='black', linewidths=1,
                label=f'Max: step {max_idx+1} ({rewards_30[max_idx]:.2f})')
    ax1.scatter([min_idx + 1], [rewards_30[min_idx]], color='purple', s=150,
                marker='*', zorder=6, edgecolors='black', linewidths=1,
                label=f'Min: step {min_idx+1} ({rewards_30[min_idx]:.2f})')
    ax1.axvline(x=spark_steps, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title(f'Average Reward per Step (First 100 Steps)\nAveraged over {n_runs} runs', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, focus_steps)

    # Plot 2: % Optimal Action per step (first 100 steps)
    ax2 = axes[1]
    # Plot full range in lighter color
    ax2.plot(range(1, focus_steps + 1), history['avg_optimal_per_step'][:focus_steps],
             color='green', linewidth=1.5, alpha=0.4)
    # Highlight first 30 steps with markers
    ax2.plot(range(1, spark_steps + 1), history['avg_optimal_per_step'][:spark_steps],
             color='red', linewidth=2.5, label=f'First {spark_steps} steps')
    ax2.scatter(range(1, spark_steps + 1), history['avg_optimal_per_step'][:spark_steps],
                color='red', s=30, zorder=5)
    # Mark key points
    optimal_30 = history['avg_optimal_per_step'][:spark_steps]
    max_idx = np.argmax(optimal_30)
    min_idx = np.argmin(optimal_30)
    ax2.scatter([max_idx + 1], [optimal_30[max_idx]], color='gold', s=150,
                marker='*', zorder=6, edgecolors='black', linewidths=1,
                label=f'Max: step {max_idx+1} ({optimal_30[max_idx]:.1f}%)')
    ax2.scatter([min_idx + 1], [optimal_30[min_idx]], color='purple', s=150,
                marker='*', zorder=6, edgecolors='black', linewidths=1,
                label=f'Min: step {min_idx+1} ({optimal_30[min_idx]:.1f}%)')
    ax2.axvline(x=spark_steps, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('% Optimal Action', fontsize=12)
    ax2.set_title(f'% Optimal Action per Step (First 100 Steps)\nAveraged over {n_runs} runs', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, focus_steps)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_summary(history):
    """Print simulation summary."""
    n_steps = history['n_steps']
    n_runs = history['n_runs']

    print("\n" + "=" * 60)
    print("UCB SIMULATION SUMMARY")
    print("=" * 60)

    print(f"\nExperiment Configuration:")
    print(f"  Number of runs: {n_runs}")
    print(f"  Steps per run: {n_steps}")
    print(f"  Exploration parameter c: 2")

    # Performance at different steps
    print(f"\nPerformance Metrics (averaged over {n_runs} runs):")
    print(f"{'Step':<10} {'Avg Reward':>15} {'% Optimal':>15}")
    print("-" * 42)
    for step in [10, 25, 50, 100]:
        if step <= n_steps:
            print(f"  {step:<8} {history['avg_reward_per_step'][step-1]:>15.3f} "
                  f"{history['avg_optimal_per_step'][step-1]:>14.1f}%")

    # Final performance
    print(f"\nFinal Performance (step {n_steps}):")
    print(f"  Average reward: {history['avg_reward_per_step'][-1]:.3f}")
    print(f"  % Optimal action: {history['avg_optimal_per_step'][-1]:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    print("Exercise 2.8: UCB Action Selection Analysis")
    print("-" * 45)
    print("Simulating: A_t = argmax[Q(a) + c√(ln(t)/N(a))]")
    print("10 actions, q*(a) ~ N(0,1), c = 2")
    print("1000 independent runs, 1000 steps each\n")

    # Run simulation with multiple runs
    history = run_simulation(n_steps=1000, n_runs=1000, c=2, seed=42)

    # Plot results
    plot_results(history)

    # Print summary
    print_summary(history)
