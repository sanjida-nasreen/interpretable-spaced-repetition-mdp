"""
memory_model.py
===============
DHP (Difficulty-Halflife-P_recall) memory model.

This implements the forgetting curve model from:
  Ye, Su, & Cao (2022) - SSP-MMC paper
  Su et al. (2023) - SSP-MMC-Plus paper

The key idea:
- Memory strength is captured by a single number: the "half-life" (h).
- Half-life = the number of days until P(recall) drops to 50%.
- Recall probability at elapsed time t: p = 2^(-t / h)
- After a successful review, h increases (memory gets stronger).
- After forgetting, h drops (memory weakens and difficulty goes up).

The parameters below are fitted from ~220 million MaiMemo learning logs.
We use them directly rather than re-fitting, since we want realistic
transition dynamics for our MDP.
"""

import numpy as np


def recall_probability(halflife: float, elapsed_days: float) -> float:
    """
    Exponential forgetting curve.
    P(recall) = 2^(-t / h)

    Args:
        halflife: current memory half-life in days
        elapsed_days: days since last review

    Returns:
        probability of successful recall (0 to 1)
    """
    if halflife <= 0 or elapsed_days < 0:
        return 0.0
    return 2.0 ** (-elapsed_days / halflife)


def next_halflife_recall(difficulty: int, halflife: float, p_recall: float) -> float:
    """
    Update half-life after SUCCESSFUL recall.
    Memory gets stronger — half-life increases.

    Formula from SSP-MMC (fitted DHP model):
    h' = h * (1 + e^3.81 * d^(-0.534) * h^(-0.127) * (1-p)^0.97)

    Args:
        difficulty: item difficulty (1 to 18)
        halflife: current half-life in days
        p_recall: recall probability at time of review

    Returns:
        new (larger) half-life
    """
    p_recall = np.clip(p_recall, 0.001, 0.999)
    return halflife * (
        1 + np.exp(3.81)
        * (difficulty ** -0.534)
        * (halflife ** -0.127)
        * ((1 - p_recall) ** 0.97)
    )


def next_halflife_forget(difficulty: int, halflife: float, p_recall: float) -> float:
    """
    Update half-life after FAILED recall (forgetting).
    Memory weakens — half-life drops significantly.

    Formula from SSP-MMC (fitted DHP model):
    h' = e^(-0.041) * d^(-0.041) * h^(0.377) * (1-p)^(-0.227)

    Args:
        difficulty: item difficulty (1 to 18)
        halflife: current half-life in days
        p_recall: recall probability at time of review

    Returns:
        new (smaller) half-life
    """
    p_recall = np.clip(p_recall, 0.001, 0.999)
    return (
        np.exp(-0.041)
        * (difficulty ** -0.041)
        * (halflife ** 0.377)
        * ((1 - p_recall) ** -0.227)
    )


def starting_halflife(difficulty: int) -> float:
    """
    Initial half-life when a learner first encounters an item.
    Harder items start with shorter half-lives.

    Formula: h0 = -1 / log2(max(0.925 - 0.05*d, 0.025))

    Args:
        difficulty: item difficulty (1 to 18)

    Returns:
        starting half-life in days
    """
    p = max(0.925 - 0.05 * difficulty, 0.025)
    return -1.0 / np.log2(p)


# ──────────────────────────────────────────────
# Quick demonstration of the model
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== DHP Memory Model Demo ===\n")

    d = 5  # medium difficulty
    h = starting_halflife(d)
    print(f"Difficulty: {d}")
    print(f"Starting half-life: {h:.2f} days\n")

    # Simulate a few reviews
    intervals = [1, 3, 7, 14, 30]
    for t in intervals:
        p = recall_probability(h, t)
        print(f"  After {t:2d} days: P(recall) = {p:.3f}", end="")
        if np.random.random() < p:
            h_new = next_halflife_recall(d, h, p)
            print(f"  → Recalled! h: {h:.1f} → {h_new:.1f}")
            h = h_new
        else:
            h_new = next_halflife_forget(d, h, p)
            print(f"  → Forgot.   h: {h:.1f} → {h_new:.1f}")
            h = h_new
            d = min(d + 2, 18)  # difficulty increases on failure
