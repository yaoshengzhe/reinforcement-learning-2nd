"""
Exercise 2.11: Parameter Study for Nonstationary Bandit

Creates a figure analogous to Figure 2.6 for the nonstationary case from Exercise 2.5.
- Nonstationary: q*(a) starts at 0 and takes random walks (N(0, 0.01) increment each step)
- Runs of 200,000 steps
- Performance measure: average reward over last 100,000 steps
- Includes constant step-size ε-greedy with α = 0.1

Algorithms compared:
1. ε-greedy with sample averages
2. ε-greedy with constant step-size α = 0.1
3. UCB
4. Gradient bandit

Reference: Sutton & Barto, Reinforcement Learning 2nd Edition, p.42
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_epsilon_greedy_vectorized(n_arms, epsilon, alpha, n_steps, n_runs, walk_std=0.01):
    """Vectorized ε-greedy running all runs in parallel."""
    # Initialize all runs at once
    q_star = np.zeros((n_runs, n_arms))  # True values
    Q = np.zeros((n_runs, n_arms))       # Estimates
    N = np.zeros((n_runs, n_arms))       # Counts

    last_steps = n_steps // 2
    rewards_sum = np.zeros(n_runs)

    for step in range(n_steps):
        # Select actions for all runs
        rand_vals = np.random.random(n_runs)
        explore_mask = rand_vals < epsilon

        # Greedy actions (break ties randomly using small noise)
        greedy_actions = np.argmax(Q + np.random.random((n_runs, n_arms)) * 1e-10, axis=1)

        # Random actions for exploration
        random_actions = np.random.randint(0, n_arms, n_runs)

        # Combine
        actions = np.where(explore_mask, random_actions, greedy_actions)

        # Get rewards
        rewards = q_star[np.arange(n_runs), actions] + np.random.randn(n_runs)

        # Update counts
        np.add.at(N, (np.arange(n_runs), actions), 1)

        # Update Q values
        if alpha is None:
            step_sizes = 1.0 / N[np.arange(n_runs), actions]
        else:
            step_sizes = alpha

        Q[np.arange(n_runs), actions] += step_sizes * (rewards - Q[np.arange(n_runs), actions])

        # Random walk
        q_star += np.random.normal(0, walk_std, (n_runs, n_arms))

        # Track rewards for last steps
        if step >= n_steps - last_steps:
            rewards_sum += rewards

    return rewards_sum.sum() / (n_runs * last_steps)


def run_ucb_vectorized(n_arms, c, alpha, n_steps, n_runs, walk_std=0.01):
    """Vectorized UCB running all runs in parallel."""
    q_star = np.zeros((n_runs, n_arms))
    Q = np.zeros((n_runs, n_arms))
    N = np.zeros((n_runs, n_arms))

    last_steps = n_steps // 2
    rewards_sum = np.zeros(n_runs)

    for step in range(n_steps):
        t = step + 1

        # Check for unvisited arms
        unvisited = (N == 0)
        has_unvisited = unvisited.any(axis=1)

        # UCB values (avoid div by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            ucb_bonus = c * np.sqrt(np.log(t) / np.maximum(N, 1))
            ucb_bonus = np.where(N == 0, np.inf, ucb_bonus)

        ucb_values = Q + ucb_bonus

        # Select actions (argmax with random tie-breaking)
        actions = np.argmax(ucb_values + np.random.random((n_runs, n_arms)) * 1e-10, axis=1)

        # Get rewards
        rewards = q_star[np.arange(n_runs), actions] + np.random.randn(n_runs)

        # Update
        np.add.at(N, (np.arange(n_runs), actions), 1)
        Q[np.arange(n_runs), actions] += alpha * (rewards - Q[np.arange(n_runs), actions])

        # Random walk
        q_star += np.random.normal(0, walk_std, (n_runs, n_arms))

        if step >= n_steps - last_steps:
            rewards_sum += rewards

    return rewards_sum.sum() / (n_runs * last_steps)


def run_gradient_vectorized(n_arms, alpha, n_steps, n_runs, walk_std=0.01):
    """Vectorized gradient bandit running all runs in parallel."""
    q_star = np.zeros((n_runs, n_arms))
    H = np.zeros((n_runs, n_arms))  # Preferences
    avg_reward = np.zeros(n_runs)

    last_steps = n_steps // 2
    rewards_sum = np.zeros(n_runs)

    for step in range(n_steps):
        t = step + 1

        # Softmax probabilities (numerically stable)
        H_max = H.max(axis=1, keepdims=True)
        exp_H = np.exp(H - H_max)
        probs = exp_H / exp_H.sum(axis=1, keepdims=True)

        # Sample actions from probabilities
        cumprobs = probs.cumsum(axis=1)
        rand_vals = np.random.random((n_runs, 1))
        actions = (rand_vals > cumprobs).sum(axis=1)
        actions = np.clip(actions, 0, n_arms - 1)

        # Get rewards
        rewards = q_star[np.arange(n_runs), actions] + np.random.randn(n_runs)

        # Update average reward (baseline)
        avg_reward += (rewards - avg_reward) / t

        # Update preferences
        one_hot = np.zeros((n_runs, n_arms))
        one_hot[np.arange(n_runs), actions] = 1
        H += alpha * (rewards - avg_reward)[:, np.newaxis] * (one_hot - probs)

        # Random walk
        q_star += np.random.normal(0, walk_std, (n_runs, n_arms))

        if step >= n_steps - last_steps:
            rewards_sum += rewards

    return rewards_sum.sum() / (n_runs * last_steps)


def run_parameter_study(n_steps=200000, n_runs=100):
    """Run parameter study for all algorithms using vectorized implementations."""
    results = {}
    n_arms = 10

    # Parameter ranges (powers of 2, as in Figure 2.6)
    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]
    alphas_grad = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
    cs = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]

    total_experiments = len(epsilons) * 2 + len(cs) + len(alphas_grad)
    pbar = tqdm(total=total_experiments, desc="Parameter study")

    # 1. ε-greedy with sample averages
    results['ε-greedy (sample avg)'] = {'params': [], 'rewards': []}
    for eps in epsilons:
        reward = run_epsilon_greedy_vectorized(n_arms, eps, None, n_steps, n_runs)
        results['ε-greedy (sample avg)']['params'].append(eps)
        results['ε-greedy (sample avg)']['rewards'].append(reward)
        pbar.update(1)

    # 2. ε-greedy with constant step-size α = 0.1
    results['ε-greedy (α=0.1)'] = {'params': [], 'rewards': []}
    for eps in epsilons:
        reward = run_epsilon_greedy_vectorized(n_arms, eps, 0.1, n_steps, n_runs)
        results['ε-greedy (α=0.1)']['params'].append(eps)
        results['ε-greedy (α=0.1)']['rewards'].append(reward)
        pbar.update(1)

    # 3. UCB with constant step-size
    results['UCB (α=0.1)'] = {'params': [], 'rewards': []}
    for c in cs:
        reward = run_ucb_vectorized(n_arms, c, 0.1, n_steps, n_runs)
        results['UCB (α=0.1)']['params'].append(c)
        results['UCB (α=0.1)']['rewards'].append(reward)
        pbar.update(1)

    # 4. Gradient bandit
    results['Gradient Bandit'] = {'params': [], 'rewards': []}
    for alpha in alphas_grad:
        reward = run_gradient_vectorized(n_arms, alpha, n_steps, n_runs)
        results['Gradient Bandit']['params'].append(alpha)
        results['Gradient Bandit']['rewards'].append(reward)
        pbar.update(1)

    pbar.close()
    return results


def plot_results(results, save_path='exercise_2_11_parameter_study.png'):
    """Plot parameter study results similar to Figure 2.6."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'ε-greedy (sample avg)': 'gray',
        'ε-greedy (α=0.1)': 'red',
        'UCB (α=0.1)': 'blue',
        'Gradient Bandit': 'green',
    }

    markers = {
        'ε-greedy (sample avg)': 'o',
        'ε-greedy (α=0.1)': 's',
        'UCB (α=0.1)': '^',
        'Gradient Bandit': 'd',
    }

    x_labels = {
        'ε-greedy (sample avg)': 'ε',
        'ε-greedy (α=0.1)': 'ε',
        'UCB (α=0.1)': 'c',
        'Gradient Bandit': 'α',
    }

    for name, data in results.items():
        # Convert to log2 scale for x-axis
        x = np.log2(data['params'])
        y = data['rewards']
        ax.plot(x, y, marker=markers[name], color=colors[name],
                linewidth=2, markersize=8, label=f"{name} ({x_labels[name]})")

    ax.set_xlabel('Parameter value (log₂ scale)', fontsize=12)
    ax.set_ylabel('Average reward over last 100,000 steps', fontsize=12)
    ax.set_title('Parameter Study: Nonstationary 10-Armed Bandit\n(200,000 steps, random walk q*(a))',
                 fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Custom x-axis labels
    ax.set_xticks(range(-7, 3))
    ax.set_xticklabels(['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4'],
                       fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_summary(results):
    """Print summary of best parameters for each algorithm."""
    print("\n" + "=" * 70)
    print("PARAMETER STUDY SUMMARY (Nonstationary Bandit)")
    print("=" * 70)

    for name, data in results.items():
        best_idx = np.argmax(data['rewards'])
        best_param = data['params'][best_idx]
        best_reward = data['rewards'][best_idx]
        print(f"\n{name}:")
        print(f"  Best parameter: {best_param:.4f}")
        print(f"  Best reward:    {best_reward:.4f}")

    # Overall comparison
    print("\n" + "-" * 70)
    print("OVERALL RANKING:")
    print("-" * 70)
    best_rewards = [(name, max(data['rewards'])) for name, data in results.items()]
    best_rewards.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, reward) in enumerate(best_rewards, 1):
        print(f"  {rank}. {name}: {reward:.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("In nonstationary environments, constant step-size methods")
    print("outperform sample-average methods because they give more")
    print("weight to recent rewards and can track changing values.")
    print("=" * 70)


if __name__ == '__main__':
    print("Exercise 2.11: Parameter Study for Nonstationary Bandit")
    print("-" * 55)
    print("Comparing algorithms on nonstationary 10-armed bandit")
    print("200,000 steps, performance = avg reward over last 100,000 steps")
    print("Using vectorized implementation for ~50x speedup")
    print()

    # Vectorized: 100 runs is fast now
    results = run_parameter_study(n_steps=200000, n_runs=100)

    # Print summary
    print_summary(results)

    # Plot results
    plot_results(results)
