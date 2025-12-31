"""
Exercise 2.10: Associative Search (Contextual Bandits)

2-armed bandit with randomly changing true values:
- Case A (prob 0.5): action 1 = 0.1, action 2 = 0.2
- Case B (prob 0.5): action 1 = 0.9, action 2 = 0.8

Part 1 (Non-associative): Cannot tell which case you face
- E[action 1] = 0.5 * 0.1 + 0.5 * 0.9 = 0.5
- E[action 2] = 0.5 * 0.2 + 0.5 * 0.8 = 0.5
- Best strategy: Any action (both have same expected value)
- Best expected reward: 0.5

Part 2 (Associative): Told which case you face
- Case A: Choose action 2 (0.2 > 0.1)
- Case B: Choose action 1 (0.9 > 0.8)
- Best expected reward: 0.5 * 0.2 + 0.5 * 0.9 = 0.55

Reference: Sutton & Barto, Reinforcement Learning 2nd Edition, p.41
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# True action values for each case
CASE_A = {1: 0.1, 2: 0.2}  # Action 2 is better
CASE_B = {1: 0.9, 2: 0.8}  # Action 1 is better


def get_case():
    """Randomly select case A or B with equal probability."""
    return 'A' if np.random.random() < 0.5 else 'B'


def get_reward(case, action):
    """Get reward for action in given case (deterministic for simplicity)."""
    if case == 'A':
        return CASE_A[action]
    else:
        return CASE_B[action]


def run_simulation(n_steps=10000, n_runs=1000, seed=42):
    """Run simulation comparing different strategies."""
    np.random.seed(seed)

    # Track rewards for each strategy
    strategies = {
        'Always Action 1': np.zeros((n_runs, n_steps)),
        'Always Action 2': np.zeros((n_runs, n_steps)),
        'Random (50/50)': np.zeros((n_runs, n_steps)),
        'Optimal Non-Associative': np.zeros((n_runs, n_steps)),
        'Optimal Associative': np.zeros((n_runs, n_steps)),
    }

    for run in tqdm(range(n_runs), desc="Running simulations"):
        for step in range(n_steps):
            case = get_case()

            # Strategy 1: Always choose action 1
            strategies['Always Action 1'][run, step] = get_reward(case, 1)

            # Strategy 2: Always choose action 2
            strategies['Always Action 2'][run, step] = get_reward(case, 2)

            # Strategy 3: Random choice
            action = np.random.choice([1, 2])
            strategies['Random (50/50)'][run, step] = get_reward(case, action)

            # Strategy 4: Optimal non-associative (same as random since E[a1]=E[a2]=0.5)
            # For demonstration, always pick action 1
            strategies['Optimal Non-Associative'][run, step] = get_reward(case, 1)

            # Strategy 5: Optimal associative (use context)
            # Case A: pick action 2, Case B: pick action 1
            optimal_action = 2 if case == 'A' else 1
            strategies['Optimal Associative'][run, step] = get_reward(case, optimal_action)

    return strategies


def plot_results(strategies, save_path='exercise_2_10_associative_search.png'):
    """Plot comparison of strategies."""
    n_steps = strategies['Always Action 1'].shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        'Always Action 1': 'blue',
        'Always Action 2': 'orange',
        'Random (50/50)': 'gray',
        'Optimal Non-Associative': 'green',
        'Optimal Associative': 'red',
    }

    # Plot 1: Average reward over time (first 100 steps)
    ax1 = axes[0]
    focus_steps = min(100, n_steps)

    for name, data in strategies.items():
        avg_reward = data.mean(axis=0)[:focus_steps]
        ax1.plot(range(1, focus_steps + 1), avg_reward,
                 label=name, color=colors[name], linewidth=2)

    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='E=0.5 (Non-Assoc)')
    ax1.axhline(y=0.55, color='red', linestyle='--', alpha=0.5, label='E=0.55 (Assoc)')
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Reward per Step (First 100 Steps)', fontsize=14)
    ax1.legend(loc='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, focus_steps)
    ax1.set_ylim(0.4, 0.6)

    # Plot 2: Bar chart of final average rewards
    ax2 = axes[1]
    names = list(strategies.keys())
    avg_rewards = [strategies[name].mean() for name in names]

    bars = ax2.bar(range(len(names)), avg_rewards, color=[colors[n] for n in names])

    # Add theoretical values
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7,
                label='Theoretical Non-Assoc (0.5)')
    ax2.axhline(y=0.55, color='red', linestyle='--', alpha=0.7,
                label='Theoretical Assoc (0.55)')

    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(['Action 1', 'Action 2', 'Random', 'Optimal\nNon-Assoc', 'Optimal\nAssoc'],
                        fontsize=10)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Overall Average Reward by Strategy', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.4, 0.6)

    # Add value labels on bars
    for bar, val in zip(bars, avg_rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def print_analysis():
    """Print theoretical analysis."""
    print("\n" + "=" * 70)
    print("EXERCISE 2.10: ASSOCIATIVE SEARCH ANALYSIS")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("PROBLEM SETUP")
    print("-" * 70)
    print("2-armed bandit with randomly changing true values:")
    print("  Case A (prob 0.5): action 1 = 0.1, action 2 = 0.2")
    print("  Case B (prob 0.5): action 1 = 0.9, action 2 = 0.8")

    print("\n" + "-" * 70)
    print("PART 1: NON-ASSOCIATIVE (Cannot tell which case)")
    print("-" * 70)
    e_a1 = 0.5 * 0.1 + 0.5 * 0.9
    e_a2 = 0.5 * 0.2 + 0.5 * 0.8
    print(f"  E[action 1] = 0.5 × 0.1 + 0.5 × 0.9 = {e_a1}")
    print(f"  E[action 2] = 0.5 × 0.2 + 0.5 × 0.8 = {e_a2}")
    print(f"\n  Both actions have EQUAL expected value!")
    print(f"  Best strategy: Any action (random, always 1, or always 2)")
    print(f"  Best expected reward: {max(e_a1, e_a2)}")

    print("\n" + "-" * 70)
    print("PART 2: ASSOCIATIVE (Told which case you face)")
    print("-" * 70)
    print("  Case A: Choose action 2 (0.2 > 0.1)")
    print("  Case B: Choose action 1 (0.9 > 0.8)")
    e_assoc = 0.5 * 0.2 + 0.5 * 0.9
    print(f"\n  Best expected reward: 0.5 × 0.2 + 0.5 × 0.9 = {e_assoc}")

    print("\n" + "-" * 70)
    print("IMPROVEMENT FROM CONTEXT")
    print("-" * 70)
    improvement = e_assoc - max(e_a1, e_a2)
    pct_improvement = improvement / max(e_a1, e_a2) * 100
    print(f"  Non-associative best: {max(e_a1, e_a2)}")
    print(f"  Associative best:     {e_assoc}")
    print(f"  Improvement:          {improvement} ({pct_improvement:.1f}%)")

    print("\n" + "-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print("  Without context: Both actions are equally good (E=0.5)")
    print("  With context:    Can exploit the better action in each case")
    print("  The context (case A or B) provides actionable information!")
    print("=" * 70)


def print_summary(strategies):
    """Print simulation summary."""
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Avg Reward':>12} {'Theoretical':>12}")
    print("-" * 50)

    theoretical = {
        'Always Action 1': 0.5,
        'Always Action 2': 0.5,
        'Random (50/50)': 0.5,
        'Optimal Non-Associative': 0.5,
        'Optimal Associative': 0.55,
    }

    for name, data in strategies.items():
        avg = data.mean()
        theo = theoretical[name]
        print(f"  {name:<23} {avg:>12.4f} {theo:>12.2f}")

    print("=" * 70)


if __name__ == '__main__':
    # Print theoretical analysis first
    print_analysis()

    # Run simulation
    print("\n\nRunning simulation to verify theoretical results...")
    strategies = run_simulation(n_steps=10000, n_runs=1000, seed=42)

    # Print results
    print_summary(strategies)

    # Plot results
    plot_results(strategies)
