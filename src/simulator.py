"""
simulator.py
============
Simulates a learner reviewing a single item under different policies.

For each policy, we simulate many episodes of a learner starting with
a new word and reviewing it until mastery (halflife >= 360 days) or
until a maximum number of days is reached.

This gives us empirical estimates of:
  - Total cost to mastery
  - Number of reviews needed
  - Time to mastery (days)
  - Average recall probability across reviews
"""

import numpy as np
from memory_model import (
    recall_probability,
    next_halflife_recall,
    next_halflife_forget,
    starting_halflife,
)
from mdp import (
    HALFLIFE_BINS, DIFFICULTY_LEVELS, INTERVALS,
    halflife_to_bin, state_index, TARGET_HALFLIFE,
    RECALL_COST, FORGET_COST, N_HALFLIFE_BINS, N_DIFFICULTY,
)


def simulate_episode(policy, difficulty=5, max_days=2000, seed=None):
    """
    Simulate one learning episode for a single item.

    Args:
        policy: array mapping state index → action index
        difficulty: initial item difficulty
        max_days: max simulation days
        seed: random seed (optional)

    Returns:
        dict with episode metrics
    """
    if seed is not None:
        np.random.seed(seed)

    d = difficulty
    h = starting_halflife(d)
    d_idx = np.argmin(np.abs(np.array(DIFFICULTY_LEVELS) - d))

    total_cost = 0.0
    total_reviews = 0
    total_recalls = 0
    total_forgets = 0
    recall_probs = []
    day = 0

    while day < max_days and h < TARGET_HALFLIFE:
        # Current state
        h_bin = halflife_to_bin(h)
        s = state_index(h_bin, d_idx)

        # Get action from policy
        a_idx = policy[s]
        interval = INTERVALS[a_idx]

        # Advance time
        day += interval

        # Compute recall probability at review time
        p = recall_probability(h, interval)
        recall_probs.append(p)

        # Simulate review outcome
        total_reviews += 1
        if np.random.random() < p:
            # Successful recall
            total_cost += RECALL_COST
            total_recalls += 1
            h = next_halflife_recall(DIFFICULTY_LEVELS[d_idx], h, p)
        else:
            # Forgot
            total_cost += FORGET_COST
            total_forgets += 1
            h = next_halflife_forget(DIFFICULTY_LEVELS[d_idx], h, p)
            d_idx = min(d_idx + 1, N_DIFFICULTY - 1)

    mastered = h >= TARGET_HALFLIFE

    return {
        'total_cost': total_cost,
        'total_reviews': total_reviews,
        'total_recalls': total_recalls,
        'total_forgets': total_forgets,
        'days_to_mastery': day if mastered else None,
        'mastered': mastered,
        'avg_recall_prob': np.mean(recall_probs) if recall_probs else 0,
        'final_halflife': h,
    }


def simulate_policy(policy, n_episodes=500, difficulties=None,
                    max_days=2000, seed=42):
    """
    Run many episodes under a given policy and aggregate results.

    Args:
        policy: array mapping state index → action index
        n_episodes: number of episodes per difficulty level
        difficulties: list of difficulty levels to test
        max_days: max days per episode
        seed: base random seed

    Returns:
        dict with aggregated metrics
    """
    if difficulties is None:
        difficulties = [1, 3, 5, 7, 10]

    all_results = []
    for d in difficulties:
        for ep in range(n_episodes):
            result = simulate_episode(
                policy, difficulty=d, max_days=max_days,
                seed=seed + ep + d * n_episodes
            )
            result['difficulty'] = d
            all_results.append(result)

    # Aggregate
    mastered = [r for r in all_results if r['mastered']]
    not_mastered = [r for r in all_results if not r['mastered']]

    metrics = {
        'n_episodes': len(all_results),
        'mastery_rate': len(mastered) / len(all_results),
        'avg_cost': np.mean([r['total_cost'] for r in all_results]),
        'avg_cost_mastered': np.mean([r['total_cost'] for r in mastered]) if mastered else None,
        'avg_reviews': np.mean([r['total_reviews'] for r in all_results]),
        'avg_recalls': np.mean([r['total_recalls'] for r in all_results]),
        'avg_forgets': np.mean([r['total_forgets'] for r in all_results]),
        'avg_recall_prob': np.mean([r['avg_recall_prob'] for r in all_results]),
        'avg_days_mastered': np.mean([r['days_to_mastery'] for r in mastered]) if mastered else None,
        'median_days_mastered': np.median([r['days_to_mastery'] for r in mastered]) if mastered else None,
    }

    # Per-difficulty breakdown
    metrics['by_difficulty'] = {}
    for d in difficulties:
        d_results = [r for r in all_results if r['difficulty'] == d]
        d_mastered = [r for r in d_results if r['mastered']]
        metrics['by_difficulty'][d] = {
            'mastery_rate': len(d_mastered) / len(d_results),
            'avg_cost': np.mean([r['total_cost'] for r in d_results]),
            'avg_reviews': np.mean([r['total_reviews'] for r in d_results]),
            'avg_recall_prob': np.mean([r['avg_recall_prob'] for r in d_results]),
        }

    return metrics


def print_comparison_table(results_dict):
    """Print a formatted comparison table of all policies."""
    print("\n" + "=" * 95)
    print(f"{'Policy':<25} {'Avg Cost':>10} {'Reviews':>10} {'Recall%':>10} "
          f"{'Mastery%':>10} {'Days':>10} {'Forgets':>10}")
    print("=" * 95)

    for name, metrics in results_dict.items():
        days_str = f"{metrics['avg_days_mastered']:.0f}" if metrics['avg_days_mastered'] else "N/A"
        print(f"{name:<25} {metrics['avg_cost']:>10.1f} {metrics['avg_reviews']:>10.1f} "
              f"{metrics['avg_recall_prob']*100:>9.1f}% {metrics['mastery_rate']*100:>9.1f}% "
              f"{days_str:>10} {metrics['avg_forgets']:>10.1f}")

    print("=" * 95)

    # Price of interpretability
    if 'Optimal (VI)' in results_dict:
        opt_cost = results_dict['Optimal (VI)']['avg_cost']
        print(f"\n=== Price of Interpretability ===")
        print(f"{'Policy':<25} {'Δ Cost':>10} {'% Overhead':>12}")
        print("-" * 50)
        for name, metrics in results_dict.items():
            if name == 'Optimal (VI)':
                continue
            delta = metrics['avg_cost'] - opt_cost
            pct = (delta / opt_cost) * 100 if opt_cost > 0 else 0
            print(f"{name:<25} {delta:>+10.1f} {pct:>+11.1f}%")


if __name__ == "__main__":
    print("Run main.py to see full simulation results.")
