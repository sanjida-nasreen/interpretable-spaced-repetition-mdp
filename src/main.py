"""
main.py
=======
Spaced Repetition SSP-MDP Project — Main Pipeline

Runs the complete project:
  1. Build MDP transition matrices
  2. Solve with value iteration → optimal policy
  3. Check monotonicity of optimal policy
  4. Define interpretable threshold policies
  5. Simulate all policies
  6. Compare results and compute price of interpretability
  7. Generate plots
"""

import sys
from pathlib import Path

# Make sibling modules importable no matter how this file is invoked
# (python src/main.py, python -m src.main, or from an IDE run button).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Anchor outputs to the repo root, not the current working directory.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

from mdp import (
    N_STATES, N_ACTIONS, HALFLIFE_BINS, DIFFICULTY_LEVELS,
    INTERVALS, N_HALFLIFE_BINS, N_DIFFICULTY, GAMMA,
    build_transition_matrices, state_index, TARGET_HALFLIFE,
)
from value_iteration import (
    value_iteration, print_policy_table,
    check_monotonicity, print_monotonicity_results,
)
from threshold_policies import (
    get_all_policies, evaluate_policy, halflife_threshold_policy,
    optimize_threshold_recall,
)
from simulator import simulate_policy, print_comparison_table


def plot_convergence(history, save_path=None):
    """Plot value iteration convergence."""
    save_path = save_path or RESULTS / 'convergence.png'
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Max Bellman Error')
    ax.set_title('Value Iteration Convergence')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_policy_heatmap(policy, title, save_path):
    """Plot a policy as a heatmap of intervals."""
    grid = np.zeros((N_HALFLIFE_BINS, N_DIFFICULTY))
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            grid[h_bin, d_idx] = INTERVALS[policy[s]]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd_r', origin='lower')
    ax.set_yticks(range(N_HALFLIFE_BINS))
    ax.set_yticklabels([f'{h:.0f}' if h >= 1 else f'{h:.1f}'
                       for h in HALFLIFE_BINS], fontsize=8)
    ax.set_xticks(range(N_DIFFICULTY))
    ax.set_xticklabels(DIFFICULTY_LEVELS)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Halflife (days)')
    ax.set_title(title)

    # Add text annotations
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY):
            val = int(grid[h_bin, d_idx])
            ax.text(d_idx, h_bin, str(val), ha='center', va='center', fontsize=7)

    plt.colorbar(im, ax=ax, label='Interval (days)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_value_function(V, save_path=None):
    """Plot value function as a heatmap."""
    save_path = save_path or RESULTS / 'value_function.png'
    grid = np.zeros((N_HALFLIFE_BINS, N_DIFFICULTY))
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            grid[h_bin, d_idx] = V[s]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(grid, aspect='auto', cmap='viridis_r', origin='lower')
    ax.set_yticks(range(N_HALFLIFE_BINS))
    ax.set_yticklabels([f'{h:.0f}' if h >= 1 else f'{h:.1f}'
                       for h in HALFLIFE_BINS], fontsize=8)
    ax.set_xticks(range(N_DIFFICULTY))
    ax.set_xticklabels(DIFFICULTY_LEVELS)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('Halflife (days)')
    ax.set_title('Value Function V(s) — Expected Cost to Mastery')
    plt.colorbar(im, ax=ax, label='Expected Cost')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_bar(results_dict, save_path=None):
    """Bar chart comparing policies on key metrics."""
    save_path = save_path or RESULTS / 'comparison.png'
    names = list(results_dict.keys())
    costs = [results_dict[n]['avg_cost'] for n in names]
    reviews = [results_dict[n]['avg_reviews'] for n in names]
    recall_pcts = [results_dict[n]['avg_recall_prob'] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Average cost
    colors = ['#2ecc71' if n == 'Optimal (VI)' else '#3498db' for n in names]
    axes[0].barh(names, costs, color=colors)
    axes[0].set_xlabel('Average Total Cost')
    axes[0].set_title('Cost to Mastery')
    axes[0].invert_yaxis()

    # Number of reviews
    axes[1].barh(names, reviews, color=colors)
    axes[1].set_xlabel('Average Number of Reviews')
    axes[1].set_title('Review Burden')
    axes[1].invert_yaxis()

    # Recall probability
    axes[2].barh(names, recall_pcts, color=colors)
    axes[2].set_xlabel('Average Recall %')
    axes[2].set_title('Recall Probability')
    axes[2].invert_yaxis()
    axes[2].set_xlim(50, 100)

    fig.suptitle('Policy Comparison: Optimal vs Interpretable', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_price_of_interpretability(results_dict,
                                   save_path=None):
    """Bar chart showing cost overhead vs optimal for each policy."""
    save_path = save_path or RESULTS / 'price_of_interpretability.png'
    if 'Optimal (VI)' not in results_dict:
        return

    opt_cost = results_dict['Optimal (VI)']['avg_cost']
    names = [n for n in results_dict if n != 'Optimal (VI)']
    overheads = [(results_dict[n]['avg_cost'] - opt_cost) / opt_cost * 100
                 for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if o > 20 else '#f39c12' if o > 5 else '#2ecc71'
              for o in overheads]
    bars = ax.barh(names, overheads, color=colors)
    ax.set_xlabel('Cost Overhead vs Optimal (%)')
    ax.set_title('Price of Interpretability')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, overheads):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_difficulty_breakdown(results_dict,
                              save_path=None):
    """Compare cost by difficulty level for key policies."""
    save_path = save_path or RESULTS / 'difficulty_breakdown.png'
    policies_to_plot = ['Optimal (VI)', 'Threshold (p=0.9)',
                        'Difficulty-aware', 'Tiered rules']
    policies_to_plot = [p for p in policies_to_plot if p in results_dict]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DIFFICULTY_LEVELS))
    width = 0.8 / len(policies_to_plot)

    for i, name in enumerate(policies_to_plot):
        costs = [results_dict[name]['by_difficulty'].get(d, {}).get('avg_cost', 0)
                 for d in DIFFICULTY_LEVELS]
        ax.bar(x + i * width, costs, width, label=name)

    ax.set_xticks(x + width * len(policies_to_plot) / 2)
    ax.set_xticklabels([f'd={d}' for d in DIFFICULTY_LEVELS])
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Average Cost to Mastery')
    ax.set_title('Cost by Difficulty Level')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def main():
    RESULTS.mkdir(exist_ok=True)

    # ── Step 1: Build MDP ──
    print("\n" + "=" * 60)
    print("STEP 1: Building MDP transition matrices")
    print("=" * 60)
    P, C = build_transition_matrices()
    print(f"  States: {N_STATES}, Actions: {N_ACTIONS}")
    print(f"  P shape: {P.shape}, C shape: {C.shape}")

    # ── Step 2: Solve with Value Iteration ──
    print("\n" + "=" * 60)
    print("STEP 2: Running Value Iteration")
    print("=" * 60)
    V, optimal_policy, history = value_iteration(P, C)
    print_policy_table(optimal_policy, V)

    # ── Step 3: Check Monotonicity ──
    print("\n" + "=" * 60)
    print("STEP 3: Checking Monotone Structure")
    print("=" * 60)
    mono = check_monotonicity(optimal_policy)
    print_monotonicity_results(mono)

    # ── Step 4: Define Threshold Policies ──
    print("\n" + "=" * 60)
    print("STEP 4: Defining Interpretable Policies")
    print("=" * 60)
    policies = get_all_policies()
    policies['Optimal (VI)'] = optimal_policy

    # Evaluate each policy analytically (using policy evaluation)
    print("\n--- Analytical Policy Values (avg cost from starting states) ---")
    for name, pol in policies.items():
        V_pol = evaluate_policy(pol, P, C, GAMMA)
        # Average value across starting states (h_bin=0 for each difficulty)
        start_values = [V_pol[state_index(0, d_idx)]
                       for d_idx in range(N_DIFFICULTY)]
        print(f"  {name:<25}: avg start cost = {np.mean(start_values):.1f}")

    # ── Step 5: Simulate All Policies ──
    print("\n" + "=" * 60)
    print("STEP 5: Running Simulations (500 episodes per difficulty)")
    print("=" * 60)
    sim_results = {}
    for name, pol in policies.items():
        print(f"  Simulating: {name}...")
        sim_results[name] = simulate_policy(pol, n_episodes=500)

    # ── Step 6: Compare Results ──
    print("\n" + "=" * 60)
    print("STEP 6: Results & Price of Interpretability")
    print("=" * 60)
    print_comparison_table(sim_results)

    # ── Step 7: Generate Plots ──
    print("\n" + "=" * 60)
    print("STEP 7: Generating Plots")
    print("=" * 60)
    plot_convergence(history)
    plot_policy_heatmap(optimal_policy, 'Optimal Policy (Value Iteration)',
                       RESULTS / 'optimal_policy_heatmap.png')
    plot_policy_heatmap(policies['Threshold (p=0.9)'],
                       'Threshold Policy (p=0.9)',
                       RESULTS / 'threshold_policy_heatmap.png')
    plot_policy_heatmap(policies['Tiered rules'],
                       'Tiered Rules Policy',
                       RESULTS / 'tiered_policy_heatmap.png')
    plot_value_function(V)
    plot_comparison_bar(sim_results)
    plot_price_of_interpretability(sim_results)
    plot_difficulty_breakdown(sim_results)

    # ── Save numerical results ──
    np.save(RESULTS / 'value_function.npy', V)
    np.save(RESULTS / 'optimal_policy.npy', optimal_policy)

    print("\n" + "=" * 60)
    print("DONE — all results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
