"""
mdp.py
======
Spaced Repetition MDP Formulation (Stochastic Shortest Path).

This is a SIMPLIFIED version of the SSP-MMC formulation from Ye et al. (2022).

The original paper:
  - Uses ~152 logarithmic halflife bins × 18 difficulty levels = ~2700 states
  - Solves with value iteration in C++ over 200,000 iterations

Our simplified version:
  - Uses ~20 halflife bins × 5 difficulty levels = ~100 states
  - Solves with value iteration in Python (fast enough)
  - Keeps the same DHP memory model and cost structure

─── MDP Components ───

States:   (h_bin, d) where h_bin is discretized halflife, d is difficulty
          Plus one terminal "mastered" state

Actions:  Choose a review interval t ∈ {1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60} days
          (how long to wait before the next review)

Transitions:
  Given state (h, d) and chosen interval t:
    - P(recall) = 2^(-t / h)
    - With prob p:   recall → state (h', d)     where h' = recall_update(h)
    - With prob 1-p: forget → state (h'', d+1)  where h'' = forget_update(h)

Costs (SSP = minimize total cost):
  - Recall cost:  3  (reviewing takes effort even when successful)
  - Forget cost:  9  (re-learning is ~3x more expensive)
  - Terminal:     0  (reaching mastery ends the problem)

Objective:
  Find the policy π(s) → interval that MINIMIZES expected total cost
  to reach the terminal "mastered" state.
"""

import numpy as np
from memory_model import (
    recall_probability,
    next_halflife_recall,
    next_halflife_forget,
    starting_halflife,
)


# ──────────────────────────────────────────────
# MDP Configuration
# ──────────────────────────────────────────────

# Halflife bins (days) — logarithmically spaced
# Covers range from ~0.5 days to ~360 days
HALFLIFE_BINS = np.array([
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 360.0
])
N_HALFLIFE_BINS = len(HALFLIFE_BINS)

# Difficulty levels (simplified from 1-18 to 1-5)
DIFFICULTY_LEVELS = [1, 3, 5, 7, 10]
N_DIFFICULTY = len(DIFFICULTY_LEVELS)

# Target halflife for "mastered" state (days)
TARGET_HALFLIFE = 360.0

# Available review intervals (days)
INTERVALS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]

# Costs (from SSP-MMC paper)
RECALL_COST = 3.0   # cost of successful review
FORGET_COST = 9.0   # cost of failed review (relearning)

# Discount factor (set close to 1 for SSP-like behavior)
GAMMA = 0.99


# ──────────────────────────────────────────────
# State space utilities
# ──────────────────────────────────────────────

def halflife_to_bin(h: float) -> int:
    """Map a continuous halflife to the nearest bin index."""
    if h >= TARGET_HALFLIFE:
        return N_HALFLIFE_BINS - 1  # highest bin = near mastery
    idx = np.argmin(np.abs(HALFLIFE_BINS - h))
    return int(idx)


def state_index(h_bin: int, d_idx: int) -> int:
    """Convert (halflife_bin, difficulty_index) to flat state index."""
    return h_bin * N_DIFFICULTY + d_idx


def state_from_index(s: int):
    """Convert flat state index back to (h_bin, d_idx)."""
    h_bin = s // N_DIFFICULTY
    d_idx = s % N_DIFFICULTY
    return h_bin, d_idx


# Total states: regular states + 1 terminal "mastered" state
N_STATES = N_HALFLIFE_BINS * N_DIFFICULTY + 1
TERMINAL_STATE = N_STATES - 1
N_ACTIONS = len(INTERVALS)


def is_terminal(s: int) -> bool:
    """Check if state is the absorbing terminal (mastered) state."""
    return s == TERMINAL_STATE


# ──────────────────────────────────────────────
# Transition model
# ──────────────────────────────────────────────

def get_transitions(s: int, a_idx: int):
    """
    Compute transition probabilities and costs for state s and action a.

    Returns:
        list of (next_state, probability, cost) tuples
    """
    if is_terminal(s):
        # Terminal state is absorbing with zero cost
        return [(TERMINAL_STATE, 1.0, 0.0)]

    h_bin, d_idx = state_from_index(s)
    h = HALFLIFE_BINS[h_bin]
    d = DIFFICULTY_LEVELS[d_idx]
    t = INTERVALS[a_idx]  # chosen interval (days to wait)

    # Recall probability after waiting t days
    p = recall_probability(h, t)

    transitions = []

    # ── Outcome 1: Successful recall (prob = p) ──
    h_recall = next_halflife_recall(d, h, p)
    if h_recall >= TARGET_HALFLIFE:
        # Reached mastery!
        transitions.append((TERMINAL_STATE, p, RECALL_COST))
    else:
        h_bin_new = halflife_to_bin(h_recall)
        s_new = state_index(h_bin_new, d_idx)
        transitions.append((s_new, p, RECALL_COST))

    # ── Outcome 2: Forgetting (prob = 1-p) ──
    h_forget = next_halflife_forget(d, h, p)
    h_bin_new = halflife_to_bin(h_forget)
    # Difficulty increases on failure
    d_idx_new = min(d_idx + 1, N_DIFFICULTY - 1)
    s_new = state_index(h_bin_new, d_idx_new)
    transitions.append((s_new, 1.0 - p, FORGET_COST))

    return transitions


# ──────────────────────────────────────────────
# Build full transition matrices
# ──────────────────────────────────────────────

def build_transition_matrices():
    """
    Precompute transition probabilities and costs for all (state, action) pairs.

    Returns:
        P: array of shape (N_ACTIONS, N_STATES, N_STATES) — transition probs
        C: array of shape (N_ACTIONS, N_STATES) — expected immediate cost
    """
    P = np.zeros((N_ACTIONS, N_STATES, N_STATES))
    C = np.zeros((N_ACTIONS, N_STATES))

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            transitions = get_transitions(s, a)
            for (s_next, prob, cost) in transitions:
                P[a, s, s_next] += prob
                C[a, s] += prob * cost

    return P, C


# ──────────────────────────────────────────────
# Display utilities
# ──────────────────────────────────────────────

def describe_state(s: int) -> str:
    """Human-readable state description."""
    if is_terminal(s):
        return "MASTERED (terminal)"
    h_bin, d_idx = state_from_index(s)
    h = HALFLIFE_BINS[h_bin]
    d = DIFFICULTY_LEVELS[d_idx]
    return f"halflife={h:.1f} days, difficulty={d}"


def describe_action(a_idx: int) -> str:
    """Human-readable action description."""
    return f"wait {INTERVALS[a_idx]} days"


# ──────────────────────────────────────────────
# Quick summary
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MDP Configuration ===\n")
    print(f"Halflife bins ({N_HALFLIFE_BINS}): {list(HALFLIFE_BINS)}")
    print(f"Difficulty levels ({N_DIFFICULTY}): {DIFFICULTY_LEVELS}")
    print(f"Total states: {N_STATES} ({N_HALFLIFE_BINS}×{N_DIFFICULTY} + 1 terminal)")
    print(f"Actions ({N_ACTIONS}): wait {INTERVALS} days")
    print(f"Target halflife: {TARGET_HALFLIFE} days")
    print(f"Costs: recall={RECALL_COST}, forget={FORGET_COST}")
    print(f"Discount: γ={GAMMA}\n")

    # Show a few example transitions
    print("=== Example Transitions ===\n")
    for d_idx in [0, 2, 4]:
        for h_bin in [0, 5, 10, 15]:
            s = state_index(h_bin, d_idx)
            print(f"State: {describe_state(s)}")
            for a_idx in [0, 4, 8]:  # intervals 1, 7, 30
                trans = get_transitions(s, a_idx)
                print(f"  Action: {describe_action(a_idx)}")
                for (s_next, prob, cost) in trans:
                    print(f"    → {describe_state(s_next)} "
                          f"(p={prob:.3f}, cost={cost:.1f})")
            print()
