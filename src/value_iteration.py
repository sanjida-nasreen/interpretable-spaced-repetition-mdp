"""
value_iteration.py
==================
Solves the spaced repetition MDP using value iteration.

This is the standard Bellman update for cost minimization:

  V(s) = min_a [ C(s,a) + γ * Σ_s' P(s'|s,a) * V(s') ]

For SSP problems, we want to minimize total cost to reach the terminal state.
The terminal state has V(terminal) = 0.

The optimal policy is:
  π*(s) = argmin_a [ C(s,a) + γ * Σ_s' P(s'|s,a) * V(s') ]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

from mdp import (
    N_STATES, N_ACTIONS, TERMINAL_STATE, GAMMA,
    build_transition_matrices, describe_state, describe_action,
    state_from_index, HALFLIFE_BINS, DIFFICULTY_LEVELS,
    N_HALFLIFE_BINS, N_DIFFICULTY, is_terminal, INTERVALS,
    state_index,
)


def value_iteration(P, C, gamma=GAMMA, tol=1e-6, max_iter=10000):
    """
    Run value iteration on the MDP.

    Args:
        P: transition probabilities (N_ACTIONS, N_STATES, N_STATES)
        C: expected costs (N_ACTIONS, N_STATES)
        gamma: discount factor
        tol: convergence threshold
        max_iter: max iterations

    Returns:
        V: value function (N_STATES,)
        policy: optimal action for each state (N_STATES,)
        history: list of max Bellman errors per iteration
    """
    V = np.zeros(N_STATES)
    policy = np.zeros(N_STATES, dtype=int)
    history = []

    for iteration in range(max_iter):
        V_old = V.copy()

        # Compute Q(s, a) = C(s,a) + γ * Σ P(s'|s,a) * V(s')
        # Shape: (N_ACTIONS, N_STATES)
        Q = C + gamma * np.einsum('asn,n->as', P, V)

        # Optimal value: V(s) = min_a Q(s, a)
        V = np.min(Q, axis=0)

        # Optimal policy: π(s) = argmin_a Q(s, a)
        policy = np.argmin(Q, axis=0)

        # Terminal state stays at 0
        V[TERMINAL_STATE] = 0.0

        # Check convergence
        delta = np.max(np.abs(V - V_old))
        history.append(delta)

        if delta < tol:
            print(f"Value iteration converged after {iteration + 1} iterations "
                  f"(Bellman error: {delta:.2e})")
            break
    else:
        print(f"Warning: Value iteration did not converge after {max_iter} iterations "
              f"(Bellman error: {delta:.2e})")

    return V, policy, history


def print_policy_table(policy, V):
    """Print the optimal policy as a readable table."""
    print("\n=== Optimal Policy ===")
    print(f"{'Halflife (days)':>16}", end="")
    for d in DIFFICULTY_LEVELS:
        print(f"  d={d:>2}", end="")
    print(f"  |  {'Value':>16}", end="")
    for d in DIFFICULTY_LEVELS:
        print(f"  d={d:>2}", end="")
    print()
    print("-" * 120)

    for h_bin in range(N_HALFLIFE_BINS):
        h = HALFLIFE_BINS[h_bin]
        print(f"{h:>16.1f}", end="")
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            a = policy[s]
            print(f"  {INTERVALS[a]:>4}d", end="")
        print(f"  |  ", end="")
        print(f"{'':>16}", end="")
        for d_idx in range(N_DIFFICULTY):
            s = state_index(h_bin, d_idx)
            print(f"  {V[s]:>4.0f}", end="")
        print()

    print(f"\nTerminal state value: {V[TERMINAL_STATE]:.2f}")


def check_monotonicity(policy):
    """
    Check if the optimal policy has monotone structure:
    1. Interval should INCREASE as halflife increases (stronger memory → wait longer)
    2. Interval should DECREASE as difficulty increases (harder items → review sooner)

    Returns:
        dict with monotonicity results
    """
    results = {
        'monotone_in_halflife': True,
        'monotone_in_difficulty': True,
        'violations_halflife': [],
        'violations_difficulty': [],
    }

    # Check monotonicity in halflife (for each fixed difficulty)
    for d_idx in range(N_DIFFICULTY):
        for h_bin in range(N_HALFLIFE_BINS - 1):
            s1 = state_index(h_bin, d_idx)
            s2 = state_index(h_bin + 1, d_idx)
            ivl1 = INTERVALS[policy[s1]]
            ivl2 = INTERVALS[policy[s2]]
            if ivl2 < ivl1:  # higher halflife should have >= interval
                results['monotone_in_halflife'] = False
                results['violations_halflife'].append(
                    (HALFLIFE_BINS[h_bin], HALFLIFE_BINS[h_bin + 1],
                     DIFFICULTY_LEVELS[d_idx], ivl1, ivl2)
                )

    # Check monotonicity in difficulty (for each fixed halflife)
    for h_bin in range(N_HALFLIFE_BINS):
        for d_idx in range(N_DIFFICULTY - 1):
            s1 = state_index(h_bin, d_idx)
            s2 = state_index(h_bin, d_idx + 1)
            ivl1 = INTERVALS[policy[s1]]
            ivl2 = INTERVALS[policy[s2]]
            if ivl2 > ivl1:  # higher difficulty should have <= interval
                results['monotone_in_difficulty'] = False
                results['violations_difficulty'].append(
                    (HALFLIFE_BINS[h_bin], DIFFICULTY_LEVELS[d_idx],
                     DIFFICULTY_LEVELS[d_idx + 1], ivl1, ivl2)
                )

    return results


def print_monotonicity_results(results):
    """Print monotonicity analysis results."""
    print("\n=== Monotonicity Analysis ===\n")

    print(f"Monotone in halflife (interval ↑ as memory ↑):  ", end="")
    if results['monotone_in_halflife']:
        print("YES ✓")
    else:
        n = len(results['violations_halflife'])
        print(f"NO ✗  ({n} violations)")
        for v in results['violations_halflife'][:5]:
            print(f"    h={v[0]:.1f}→{v[1]:.1f}, d={v[2]}: "
                  f"interval {v[3]}→{v[4]} (should not decrease)")

    print(f"Monotone in difficulty (interval ↓ as difficulty ↑): ", end="")
    if results['monotone_in_difficulty']:
        print("YES ✓")
    else:
        n = len(results['violations_difficulty'])
        print(f"NO ✗  ({n} violations)")
        for v in results['violations_difficulty'][:5]:
            print(f"    h={v[0]:.1f}, d={v[1]}→{v[2]}: "
                  f"interval {v[3]}→{v[4]} (should not increase)")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Building transition matrices...")
    P, C = build_transition_matrices()

    print(f"Running value iteration (γ={GAMMA})...")
    V, policy, history = value_iteration(P, C)

    print_policy_table(policy, V)

    mono = check_monotonicity(policy)
    print_monotonicity_results(mono)

    # Save results
    RESULTS.mkdir(exist_ok=True)
    np.save(RESULTS / "value_function.npy", V)
    np.save(RESULTS / "optimal_policy.npy", policy)
    np.save(RESULTS / "convergence_history.npy", np.array(history))
    print("\nResults saved to results/")
