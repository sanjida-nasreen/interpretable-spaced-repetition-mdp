"""
threshold_policies.py
=====================
Interpretable threshold-based scheduling policies.

These are simple rules that a human could follow without a lookup table.
We compare them against the optimal policy to measure the
"price of interpretability."

Each policy maps a state (halflife, difficulty) to a review interval.
"""

import numpy as np
from mdp import (
    N_STATES, TERMINAL_STATE, HALFLIFE_BINS, DIFFICULTY_LEVELS,
    N_HALFLIFE_BINS, N_DIFFICULTY, INTERVALS, state_index,
    state_from_index, is_terminal,
)


def nearest_interval(t: float) -> int:
    """Map a continuous interval to the nearest available discrete interval."""
    idx = np.argmin(np.abs(np.array(INTERVALS) - t))
    return idx


# ──────────────────────────────────────────────
# Policy 1: Fixed Interval
# ──────────────────────────────────────────────
def fixed_interval_policy(interval_days=7):
    """
    Always review after a fixed number of days.
    Rule: "Review every {interval_days} days, no matter what."

    This is the simplest possible policy — no adaptation at all.
    """
    policy = np.full(N_STATES, nearest_interval(interval_days), dtype=int)
    return policy


# ──────────────────────────────────────────────
# Policy 2: Halflife-Based Threshold
# ──────────────────────────────────────────────
def halflife_threshold_policy(target_recall=0.9):
    """
    Review when recall probability would drop to target_recall.
    Rule: "Wait until there's a {target_recall*100}% chance of remembering."

    Interval = -h * log2(target_recall)

    This is interpretable: stronger memories get longer intervals.
    """
    policy = np.zeros(N_STATES, dtype=int)
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            h = HALFLIFE_BINS[h_bin]
            # Wait until P(recall) = target_recall
            t = -h * np.log2(target_recall)
            t = max(1, t)
            policy[s] = nearest_interval(t)
    return policy


# ──────────────────────────────────────────────
# Policy 3: Difficulty-Aware Threshold
# ──────────────────────────────────────────────
def difficulty_aware_policy(base_recall=0.85):
    """
    Like halflife threshold, but harder items use a higher recall target.
    Rule: "Review at {base_recall}% recall for easy items,
           review earlier for harder items."

    target_recall = base_recall + 0.01 * (10 - difficulty)
    """
    policy = np.zeros(N_STATES, dtype=int)
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            h = HALFLIFE_BINS[h_bin]
            d = DIFFICULTY_LEVELS[d_idx]
            # Harder items → higher recall target → shorter interval
            target_recall = np.clip(base_recall + 0.01 * (10 - d), 0.5, 0.99)
            t = -h * np.log2(target_recall)
            t = max(1, t)
            policy[s] = nearest_interval(t)
    return policy


# ──────────────────────────────────────────────
# Policy 4: Simple Tiered Rules
# ──────────────────────────────────────────────
def tiered_policy():
    """
    Human-readable tiered rules:
      - If halflife < 2 days:   review in 1 day
      - If halflife < 8 days:   review in 3 days
      - If halflife < 32 days:  review in 7 days
      - If halflife < 128 days: review in 14 days
      - Else:                   review in 30 days
    """
    policy = np.zeros(N_STATES, dtype=int)
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            h = HALFLIFE_BINS[h_bin]
            if h < 2:
                policy[s] = nearest_interval(1)
            elif h < 8:
                policy[s] = nearest_interval(3)
            elif h < 32:
                policy[s] = nearest_interval(7)
            elif h < 128:
                policy[s] = nearest_interval(14)
            else:
                policy[s] = nearest_interval(30)
    return policy


# ──────────────────────────────────────────────
# Grid search for best threshold policy
# ──────────────────────────────────────────────
def optimize_threshold_recall(V_optimal, P, C, gamma=0.99):
    """
    Find the best target_recall parameter for the halflife threshold policy
    by grid search over the value function.

    Returns:
        best_recall: the recall threshold that minimizes total cost
        best_value: the expected cost under the best threshold
    """
    from value_iteration import value_iteration

    best_recall = 0.9
    best_cost = float('inf')

    for target_recall in np.arange(0.50, 0.99, 0.01):
        policy = halflife_threshold_policy(target_recall)
        # Evaluate this policy
        V = evaluate_policy(policy, P, C, gamma)
        # Average cost across starting states
        avg_cost = np.mean([V[state_index(0, d)] for d in range(N_DIFFICULTY)])
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_recall = target_recall

    return best_recall, best_cost


def evaluate_policy(policy, P, C, gamma=0.99, tol=1e-6, max_iter=5000):
    """
    Evaluate a fixed policy by solving V = C_π + γ * P_π * V.

    Args:
        policy: array of action indices for each state
        P: transition probabilities
        C: costs
        gamma: discount factor

    Returns:
        V: value function under this policy
    """
    V = np.zeros(N_STATES)

    for iteration in range(max_iter):
        V_old = V.copy()

        for s in range(N_STATES):
            if is_terminal(s):
                V[s] = 0.0
                continue
            a = policy[s]
            V[s] = C[a, s] + gamma * P[a, s, :] @ V

        delta = np.max(np.abs(V - V_old))
        if delta < tol:
            break

    return V


# ──────────────────────────────────────────────
# Collect all policies
# ──────────────────────────────────────────────
def get_all_policies():
    """Return a dict of all policies for comparison."""
    return {
        'Optimal (VI)': None,  # placeholder — set after value iteration
        'Fixed 7-day': fixed_interval_policy(7),
        'Fixed 3-day': fixed_interval_policy(3),
        'Threshold (p=0.9)': halflife_threshold_policy(0.9),
        'Threshold (p=0.8)': halflife_threshold_policy(0.8),
        'Threshold (p=0.7)': halflife_threshold_policy(0.7),
        'Difficulty-aware': difficulty_aware_policy(0.85),
        'Tiered rules': tiered_policy(),
    }


if __name__ == "__main__":
    print("=== Threshold Policies ===\n")

    policies = get_all_policies()
    for name, pol in policies.items():
        if pol is None:
            continue
        print(f"\n--- {name} ---")
        print(f"{'Halflife':>10}", end="")
        for d in DIFFICULTY_LEVELS:
            print(f"  d={d:>2}", end="")
        print()
        for h_bin in range(N_HALFLIFE_BINS):
            h = HALFLIFE_BINS[h_bin]
            print(f"{h:>10.1f}", end="")
            for d_idx in range(N_DIFFICULTY):
                s = state_index(h_bin, d_idx)
                print(f"  {INTERVALS[pol[s]]:>4}d", end="")
            print()
