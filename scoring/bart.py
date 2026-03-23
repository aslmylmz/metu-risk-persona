"""
BART Scoring Engine — NumPy Vectorized with Multi-Risk Profiles

Calculates psychometric metrics from raw BART event logs:
- Overall metrics: Average Pumps, Explosion Rate, Latency
- Color-based metrics: Performance by balloon color (purple/teal/orange)
- Learning metrics: Adaptation, risk adjustment, color discrimination
- Behavioral indices: Impulsivity, patience, consistency

Multi-Risk Profile System (Pymetrics-inspired):
- Purple balloons: Low risk (max 128 pumps) — rewards patience
- Teal balloons: Medium risk (max 32 pumps) — standard risk
- Orange balloons: High risk (max 8 pumps) — tests impulse control

Note: Uses neutral colors to avoid psychological bias (e.g., red = danger).

Explosion model (frontend):
    At each pump attempt k the balloon explodes with P(explode) = k / maxPumps
    (sequential independent Bernoulli trials with linearly increasing probability).
    This is NOT a pre-drawn uniform distribution — the optimal stopping point under
    this model is lower than maxPumps / 2.  For orange (N=8) the EV-maximising
    stop is ~2 pumps; for teal (N=32) ~6 pumps; for purple (N=128) ~12 pumps.
    (Optimal stops are approximately sqrt(N), derived from the EV-curve peak of
    the sequential Bernoulli model — not maxPumps/4 as previously documented.)

RNG-Truncation Robustness:
    All behavioral-intention metrics use COLLECTED (non-exploded) balloons only.
    On an exploded balloon the pump count is truncated by RNG — the participant
    may have intended to pump further.  Using collected balloons ensures we
    measure what the participant CHOSE, not what RNG allowed.  When too few
    collected balloons are available for a given color (< MIN_COLLECTED_FALLBACK),
    the engine falls back to all balloons with a session warning.

References:
    Lejuez et al. (2002). Evaluation of a behavioral measure of risk taking:
    The Balloon Analogue Risk Task (BART).

    Pymetrics Multi-Risk BART: Measures learning and adaptability through
    varying risk profiles.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from scipy import stats

from schemas.game_events import BARTMetrics, ColorMetrics, GameEvent

logger = logging.getLogger(__name__)


# ── Color Profile Constants ──────────────────────────────────────────────────

COLOR_PROFILES = {
    "purple": {"risk": "low", "max_pumps": 128},
    "teal": {"risk": "medium", "max_pumps": 32},
    "orange": {"risk": "high", "max_pumps": 8},
}

# Minimum collected (non-exploded) balloons per color before falling back
# to all balloons.  With 10 balloons per color, orange typically yields
# only 1-3 collected (P(survive) at 4 pumps ≈ 16%).  A threshold of 2
# ensures we have at least some variance estimate; below that, we fall
# back to all balloons (truncated but better than nothing).
MIN_COLLECTED_FALLBACK = 2


# ── EV Computation (Sequential Bernoulli Model) ─────────────────────────────


def _compute_ev(s: int, max_pumps: int) -> float:
    """
    Compute expected value of stopping after s pumps under the sequential
    Bernoulli explosion model: P(explode at pump k) = k / maxPumps.

    EV(s) = s × ∏(k=1 to s) (1 - k/N)

    Parameters
    ----------
    s : int
        Number of pumps before collecting.
    max_pumps : int
        Maximum pumps for this balloon color (N).

    Returns
    -------
    float
        Expected value (reward units = pump count × survival probability).
    """
    if s <= 0 or s > max_pumps:
        return 0.0
    survival = 1.0
    for k in range(1, s + 1):
        survival *= (1.0 - k / max_pumps)
        if survival <= 0:
            return 0.0
    return s * survival


def _compute_ev_optimal(max_pumps: int) -> tuple[int, float]:
    """
    Find the pump count that maximizes EV under P(explode at k) = k/N.

    Returns
    -------
    tuple[int, float]
        (optimal_stop, max_ev)
    """
    best_s = 0
    best_ev = 0.0
    for s in range(1, max_pumps + 1):
        ev = _compute_ev(s, max_pumps)
        if ev > best_ev:
            best_ev = ev
            best_s = s
        elif ev < best_ev * 0.5:
            # Past the peak and declining fast — stop searching
            break
    return best_s, best_ev


def _compute_survival_probability(s: int, max_pumps: int) -> float:
    """
    Compute probability of surviving s pumps: ∏(k=1 to s) (1 - k/N).
    """
    if s <= 0:
        return 1.0
    survival = 1.0
    for k in range(1, s + 1):
        survival *= (1.0 - k / max_pumps)
    return max(0.0, survival)


# Cache optimal stops so we don't recompute every call
_EV_OPTIMAL_CACHE: dict[int, tuple[int, float]] = {}


def _get_ev_optimal(max_pumps: int) -> tuple[int, float]:
    """Get cached EV-optimal stop for a given max_pumps."""
    if max_pumps not in _EV_OPTIMAL_CACHE:
        _EV_OPTIMAL_CACHE[max_pumps] = _compute_ev_optimal(max_pumps)
    return _EV_OPTIMAL_CACHE[max_pumps]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _segment_balloons(events: list[GameEvent]) -> list[list[GameEvent]]:
    """
    Split a flat event list into per-balloon segments.

    A new balloon starts after every 'collect' or 'explode' event.
    Returns a list of lists, one per balloon.
    """
    balloons: list[list[GameEvent]] = []
    current: list[GameEvent] = []

    for event in events:
        current.append(event)
        if event.type in ("collect", "explode"):
            balloons.append(current)
            current = []

    # Include any trailing pumps (incomplete final balloon)
    if current:
        balloons.append(current)

    return balloons


def _extract_balloon_color(balloon_events: list[GameEvent]) -> str:
    """
    Extract balloon color from event payload.

    Looks for 'color' or 'balloon_color' in the event payload.
    Defaults to 'teal' (medium risk) if not specified.
    """
    for event in balloon_events:
        if hasattr(event.payload, "color") and event.payload.color:
            return event.payload.color.lower()
        if hasattr(event.payload, "balloon_color") and event.payload.balloon_color:
            return event.payload.balloon_color.lower()

    # Fallback to teal (medium risk) if no color specified
    return "teal"


def _prefer_collected(
    collected: list[int],
    all_data: list[int],
    min_count: int = MIN_COLLECTED_FALLBACK,
) -> tuple[list[int], bool]:
    """
    Use collected (non-exploded) balloon data when available, else fall back.

    Parameters
    ----------
    collected : list[int]
        Pump counts from collected (non-exploded) balloons only.
    all_data : list[int]
        Pump counts from all balloons (including truncated/exploded).
    min_count : int
        Minimum collected balloons required; below this, fall back to all.

    Returns
    -------
    tuple[list[int], bool]
        (data_to_use, used_fallback)
    """
    if len(collected) >= min_count:
        return collected, False
    return all_data, True


def validate_bart_session(events: list[GameEvent]) -> dict[str, Any]:
    """
    Validate a BART session before scoring and flag potentially invalid data.

    Checks performed:
    1. Minimum balloon count  — < 15 -> invalid, 15-29 -> warning
    2. Color balance          — each color should have ~10 balloons
    3. Timestamp monotonicity — out-of-order timestamps indicate corruption
    4. Session speed          — < 30 s for a 30-balloon session is suspicious
    5. Pump uniformity        — near-zero std across all balloons suggests automation

    Parameters
    ----------
    events : list[GameEvent]
        Chronologically ordered BART events.

    Returns
    -------
    dict[str, Any]
        Keys:
          is_valid          (bool)
          warnings          (list[str])
          balloon_count     (int)
          color_distribution (dict[str, int])
    """
    if not events:
        return {
            "is_valid": False,
            "warnings": ["Empty event log"],
            "balloon_count": 0,
            "color_distribution": {},
        }

    warnings: list[str] = []
    is_valid = True

    balloons = _segment_balloons(events)
    balloon_count = len(balloons)

    # 1. Minimum balloon count
    if balloon_count < 15:
        warnings.append(
            f"Critically incomplete session: only {balloon_count}/30 balloons played"
        )
        is_valid = False
    elif balloon_count < 30:
        warnings.append(f"Incomplete session: {balloon_count}/30 balloons played")

    # 2. Color balance
    color_counts: dict[str, int] = defaultdict(int)
    for b in balloons:
        color = _extract_balloon_color(b)
        color_counts[color] += 1

    for color in ["purple", "teal", "orange"]:
        count = color_counts.get(color, 0)
        if count < 5:
            warnings.append(f"Too few {color} balloons: {count}/10 played")
        elif count < 10:
            warnings.append(f"Partial {color} balloons: {count}/10 played")

    # 3. Timestamp monotonicity
    for i in range(1, len(events)):
        if events[i].timestamp < events[i - 1].timestamp:
            warnings.append(
                f"Out-of-order timestamps at event index {i} "
                f"({events[i].timestamp:.1f} < {events[i-1].timestamp:.1f})"
            )
            is_valid = False
            break

    # 4. Session speed (suspicious if < 30 s for >= 15 balloons)
    total_time_ms = events[-1].timestamp - events[0].timestamp
    if balloon_count >= 15 and total_time_ms < 30_000:
        warnings.append(
            f"Session completed unusually fast: {total_time_ms / 1000:.1f}s "
            f"for {balloon_count} balloons"
        )

    # 5. Near-uniform pump counts (bot detection)
    pump_counts = [sum(1 for e in b if e.type == "pump") for b in balloons]
    if len(pump_counts) >= 10 and float(np.std(pump_counts)) < 0.5:
        warnings.append(
            "Suspicious: nearly identical pump counts across all balloons "
            "(possible automation or non-genuine engagement)"
        )

    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "balloon_count": balloon_count,
        "color_distribution": dict(color_counts),
    }


def _calculate_learning_rate(
    balloon_data: list[tuple[int, str, int, bool]],
) -> float:
    """
    Calculate learning rate using linear regression on pump counts over time.

    Uses COLLECTED (non-exploded) balloons only to avoid RNG truncation bias.
    On exploded balloons the pump count is cut short by RNG, which can fake
    a learning trend if explosions cluster in one half of the session.
    Falls back to all balloons per color if fewer than MIN_COLLECTED_FALLBACK
    collected balloons are available.

    For each color, fits a line to pump counts across trials to detect whether
    the participant adapts: pumping more on purple over time (exploitation) or
    pumping less on orange over time (risk avoidance).

    Note: With only 10 trials per color this regression is noisy — a single
    outlier trial can significantly shift the slope.  See half_split_learning_rate
    for a more robust alternative at this sample size.

    Parameters
    ----------
    balloon_data : list[tuple[int, str, int, bool]]
        List of (trial_number, color, pumps, exploded) tuples.

    Returns
    -------
    float
        Learning rate coefficient (-1 to 1). Positive = adaptive learning.
    """
    if len(balloon_data) < 3:
        return 0.0

    # Separate by color, keeping exploded flag
    color_trials_all: dict[str, list[tuple[int, int]]] = defaultdict(list)
    color_trials_collected: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for trial, color, pumps, exploded in balloon_data:
        color_trials_all[color].append((trial, pumps))
        if not exploded:
            color_trials_collected[color].append((trial, pumps))

    learning_slopes = []

    for color in color_trials_all:
        # Prefer collected, fall back to all if insufficient
        collected = color_trials_collected.get(color, [])
        all_trials = color_trials_all[color]
        trials = collected if len(collected) >= MIN_COLLECTED_FALLBACK else all_trials

        if len(trials) < 2:
            continue

        trial_nums = np.array([t[0] for t in trials])
        pump_counts = np.array([t[1] for t in trials])

        # Linear regression: pumps ~ trial_number
        if len(trial_nums) >= 2 and np.std(trial_nums) > 0:
            slope, _intercept, r_value, _p_value, _std_err = stats.linregress(
                trial_nums,
                pump_counts,
            )

            # Weight by R^2 (how well the trend fits)
            weighted_slope = slope * (r_value**2)

            # For orange balloons, negative slope is good (learning to reduce risk)
            # For purple balloons, positive slope is good (learning to maximize)
            if color == "orange":
                learning_slopes.append(-weighted_slope)
            elif color == "purple":
                learning_slopes.append(weighted_slope)
            else:  # teal (medium risk)
                # Decreasing pumps on teal = learning to be appropriately cautious.
                # Weight at 0.5 since teal is medium risk (less signal than orange/purple).
                learning_slopes.append(-weighted_slope * 0.5)

    if not learning_slopes:
        return 0.0

    mean_slope = float(np.mean(learning_slopes))
    if np.isnan(mean_slope):
        return 0.0
    return float(np.clip(mean_slope, -1.0, 1.0))


def _calculate_half_split_learning_rate(
    balloon_data: list[tuple[int, str, int, bool]],
) -> float:
    """
    Calculate learning rate by comparing first-half vs second-half trials per color.

    Uses COLLECTED (non-exploded) balloons only to avoid RNG truncation bias.
    If the first half of a color's balloons happen to explode early due to bad
    RNG, their pump counts are artificially lower, faking a "learning" signal
    in the second half.  Using collected-only removes this confound.
    Falls back to all balloons per color if fewer than 4 collected are available.

    More robust than regression-based learning_rate at N=10 per color because:
    - No single outlier trial can dominate the result.
    - Directly interpretable: a positive value means behavior improved in the
      second half relative to the first.

    Improvement direction per color:
    - Orange: pumping LESS in the second half = learning (delta negated).
    - Purple: pumping MORE in the second half = learning (delta kept).
    - Teal:   pumping less in the second half = cautious adaptation (delta negated, half-weight).

    Parameters
    ----------
    balloon_data : list[tuple[int, str, int, bool]]
        List of (trial_number, color, pumps, exploded) tuples.

    Returns
    -------
    float
        Learning rate (-1 to 1). Positive = improved adaptive behavior in second half.
    """
    if len(balloon_data) < 4:
        return 0.0

    color_trials_all: dict[str, list[tuple[int, int]]] = defaultdict(list)
    color_trials_collected: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for trial, color, pumps, exploded in balloon_data:
        color_trials_all[color].append((trial, pumps))
        if not exploded:
            color_trials_collected[color].append((trial, pumps))

    learning_scores = []

    for color in color_trials_all:
        # Prefer collected, fall back to all if insufficient
        collected = color_trials_collected.get(color, [])
        all_trials = color_trials_all[color]
        trials = collected if len(collected) >= 4 else all_trials

        if len(trials) < 4:  # Need at least 4 trials to form a meaningful split
            continue

        sorted_trials = sorted(trials, key=lambda x: x[0])
        half = len(sorted_trials) // 2

        first_half_mean = float(np.mean([t[1] for t in sorted_trials[:half]]))
        second_half_mean = float(np.mean([t[1] for t in sorted_trials[half:]]))
        overall_mean = float(np.mean([t[1] for t in sorted_trials]))

        if overall_mean == 0:
            continue

        # Relative change: positive = pumped more in second half
        delta = (second_half_mean - first_half_mean) / overall_mean

        if color == "orange":
            learning_scores.append(-delta)  # Less pumping = improvement
        elif color == "purple":
            learning_scores.append(delta)   # More pumping = improvement
        else:  # teal
            learning_scores.append(-delta * 0.5)  # Cautious reduction, half-weight

    if not learning_scores:
        return 0.0

    mean_learning = float(np.mean(learning_scores))
    if np.isnan(mean_learning):
        return 0.0
    return float(np.clip(mean_learning, -1.0, 1.0))


def _calculate_color_discrimination(
    color_pumps: dict[str, list[int]],
) -> float:
    """
    Calculate color discrimination index using effect size (Cohen's d).

    Measures how well the user discriminates between purple (safe) and orange (risky)
    balloons. Higher values indicate stronger behavioral differentiation.

    Expects collected-only pump data to avoid RNG truncation bias on orange.

    Parameters
    ----------
    color_pumps : dict[str, list[int]]
        Pump counts grouped by color (should be collected-only for accuracy).

    Returns
    -------
    float
        Discrimination index (0-1). 1 = perfect discrimination (d >= 2.0).
    """
    purple_pumps = color_pumps.get("purple", [])
    orange_pumps = color_pumps.get("orange", [])

    # Need at least 2 samples of each color for variance calculation
    if len(purple_pumps) < 2 or len(orange_pumps) < 2:
        return 0.0

    purple_arr = np.array(purple_pumps)
    orange_arr = np.array(orange_pumps)

    mean_diff = np.mean(purple_arr) - np.mean(orange_arr)
    pooled_std = np.sqrt(
        (np.var(purple_arr, ddof=1) + np.var(orange_arr, ddof=1)) / 2,
    )

    if pooled_std == 0:
        # If variance is zero, discrimination is perfect only if means differ
        return 1.0 if mean_diff > 0 else 0.0

    cohens_d = mean_diff / pooled_std

    # Normalize to [0, 1]: d >= 2.0 is considered very strong discrimination
    discrimination = np.clip(cohens_d / 2.0, 0.0, 1.0)

    if np.isnan(discrimination):
        return 0.0
    return float(discrimination)


def _calculate_risk_sensitivity(
    color_pumps: dict[str, list[int]],
) -> float:
    """
    Calculate risk sensitivity using Pearson correlation.

    Measures alignment between balloon risk capacity and pumping behavior.
    High correlation (r > 0.8) indicates the participant understands the risk
    model and adjusts behavior proportionally (purple > teal > orange).

    Expects collected-only pump data to avoid RNG truncation bias.

    Note: If the participant uses a flat strategy (same pumps for all colors),
    variance in risk_capacities will not correlate with behavior and r ~ 0.
    This is a known limitation — a flat strategy is ambiguous (risk-neutral or
    unresponsive), so a near-zero score should not be interpreted as a failure.

    Parameters
    ----------
    color_pumps : dict[str, list[int]]
        Pump counts separated by color (should be collected-only for accuracy).

    Returns
    -------
    float
        Correlation coefficient (-1 to 1).
    """
    risk_capacities = []
    user_pumps = []

    for color, pumps in color_pumps.items():
        if color not in COLOR_PROFILES:
            continue
        capacity = COLOR_PROFILES[color]["max_pumps"]
        for p in pumps:
            risk_capacities.append(capacity)
            user_pumps.append(p)

    if len(risk_capacities) < 3:
        return 0.0

    # Constant inputs yield undefined correlation (all pumps identical across colors)
    if np.std(user_pumps) == 0 or np.std(risk_capacities) == 0:
        return 0.0

    r, _p = stats.pearsonr(risk_capacities, user_pumps)

    if np.isnan(r):
        return 0.0
    return float(r)


def _calculate_risk_adjustment_score(
    color_pumps: dict[str, list[int]],
) -> float:
    """
    Calculate risk adjustment score based on EV-optimal behavior per color.

    Scores the participant on whether their average pumps are calibrated to the
    true EV-optimal stopping point for each balloon color.  The optimal stops are
    derived from the peak of the EV curve under the sequential Bernoulli model
    (P(explode at pump k) = k / maxPumps) and are approximately sqrt(N):

    - Purple (N=128): EV-optimal = 12 pumps
    - Teal   (N=32):  EV-optimal = 6 pumps
    - Orange (N=8):   EV-optimal = 2 pumps

    Each color is scored by absolute distance from its optimal stop, scaled so
    that score = 100 at the optimum and decreases linearly to 0 at the extremes
    (either 0 pumps or maxPumps for that color).  This correctly penalises both
    under-pumping AND over-pumping — the old asymmetric np.clip formulas rewarded
    maxing out purple and zeroing out orange, which is inconsistent with the
    EV-curve shape.

    Returns
    -------
    float
        Risk adjustment score (0-100). 100 = perfectly calibrated.
    """
    cp = color_pumps  # local alias to match caller convention

    optimal_stops = {"purple": 12.0, "teal": 6.0, "orange": 2.0}
    max_pumps_caps = {"purple": 128, "teal": 32, "orange": 8}
    scores = []

    for color in ["purple", "teal", "orange"]:
        if color in cp and len(cp[color]) > 0:
            mean_pumps = np.mean(cp[color])
            opt = optimal_stops[color]
            mx = max_pumps_caps[color]

            # Max possible distance from optimal (either down to 0, or up to max_pumps)
            max_dist = max(opt, mx - opt)

            # Score is 100 at optimal, scaling linearly down to 0 at the extremes
            score = np.clip(1.0 - abs(mean_pumps - opt) / max_dist, 0.0, 1.0) * 100.0
            scores.append(float(score))

    if not scores:
        return 0.0

    result = float(np.mean(scores))
    if np.isnan(result):
        return 0.0
    return result


def _compute_ev_ratio_score(
    color_pumps_collected: dict[str, list[int]],
    color_balloons: dict[str, int],
    min_collected: int = MIN_COLLECTED_FALLBACK,
) -> tuple[float, dict[str, float]]:
    """
    Compute EV-Ratio Risk Calibration Score (EV-weighted).

    For each color with sufficient COLLECTED data, computes:
        EV(round(mean_behavioral_pumps)) / EV(optimal)

    Colors where balloons existed but none were collected (all exploded)
    receive efficiency = 0.

    The overall score is a WEIGHTED average of per-color efficiencies,
    where each color's weight is its EV-optimal value. This means
    high-reward colors (purple, EV≈6.46) contribute more than low-reward
    colors (orange, EV≈1.31), reflecting actual reward potential.

    Weights: purple ≈ 60%, teal ≈ 28%, orange ≈ 12%.

    Parameters
    ----------
    color_pumps_collected : dict[str, list[int]]
        Pump counts per color — COLLECTED ONLY (not fallback).
    color_balloons : dict[str, int]
        Total balloons per color (including exploded).
    min_collected : int
        Minimum collected balloons required per color.

    Returns
    -------
    tuple[float, dict[str, float]]
        (overall_score, {color: efficiency})
        efficiency values are in [0, 1], overall_score in [0, 100].
    """
    per_color_efficiency: dict[str, float] = {}

    for color in ["purple", "teal", "orange"]:
        if color not in COLOR_PROFILES:
            continue
        total = color_balloons.get(color, 0)
        if total == 0:
            continue  # Color not in session

        pumps = color_pumps_collected.get(color, [])
        if len(pumps) < min_collected:
            # Balloons existed but insufficient collected — participant failed
            # to adapt. EV-efficiency = 0 (earned nothing from this risk level).
            per_color_efficiency[color] = 0.0
            continue

        max_p = COLOR_PROFILES[color]["max_pumps"]
        optimal_stop, optimal_ev = _get_ev_optimal(max_p)

        if optimal_ev <= 0:
            continue

        mean_pumps = float(np.mean(pumps))
        # Use floor and ceil to interpolate EV for non-integer mean
        s_low = max(0, int(np.floor(mean_pumps)))
        s_high = min(max_p, int(np.ceil(mean_pumps)))

        if s_low == s_high:
            participant_ev = _compute_ev(s_low, max_p)
        else:
            frac = mean_pumps - s_low
            ev_low = _compute_ev(s_low, max_p)
            ev_high = _compute_ev(s_high, max_p)
            participant_ev = ev_low + frac * (ev_high - ev_low)

        efficiency = min(1.0, participant_ev / optimal_ev)
        per_color_efficiency[color] = efficiency

    if not per_color_efficiency:
        return 0.0, {}

    # EV-weighted average: weight each color by its optimal EV value
    weighted_sum = 0.0
    weight_total = 0.0
    for color, eff in per_color_efficiency.items():
        max_p = COLOR_PROFILES[color]["max_pumps"]
        _, optimal_ev = _get_ev_optimal(max_p)
        weighted_sum += eff * optimal_ev
        weight_total += optimal_ev

    overall = (weighted_sum / weight_total) * 100.0 if weight_total > 0 else 0.0
    return overall, per_color_efficiency


def _compute_explosion_penalty(
    color_explosions: dict[str, int],
    color_balloons: dict[str, int],
) -> tuple[float, dict[str, float]]:
    """
    Compute explosion penalty: excess explosion rate vs expected at EV-optimal.

    For each color, the expected explosion rate at optimal play is:
        1 - ∏(k=1 to s*) (1 - k/N)
    where s* is the EV-optimal stop.

    Excess = max(0, observed_rate - expected_rate).
    Final penalty = mean of per-color excess rates.

    Returns
    -------
    tuple[float, dict[str, float]]
        (overall_penalty in [0,1], {color: excess_rate})
    """
    per_color_excess: dict[str, float] = {}

    for color in ["purple", "teal", "orange"]:
        if color not in COLOR_PROFILES:
            continue
        total = color_balloons.get(color, 0)
        if total == 0:
            continue

        explosions = color_explosions.get(color, 0)
        observed_rate = explosions / total

        max_p = COLOR_PROFILES[color]["max_pumps"]
        optimal_stop, _ = _get_ev_optimal(max_p)
        expected_rate = 1.0 - _compute_survival_probability(optimal_stop, max_p)

        excess = max(0.0, observed_rate - expected_rate)
        per_color_excess[color] = excess

    if not per_color_excess:
        return 0.0, {}

    overall = float(np.mean(list(per_color_excess.values())))
    return min(1.0, overall), per_color_excess


def _compute_ev_efficiency_differentiation(
    per_color_efficiency: dict[str, float],
    color_pumps_collected: dict[str, list[int]],
    color_balloons: dict[str, int],
) -> float | None:
    """
    Compute EV-efficiency differentiation: 1 - CV(per_color_efficiencies).

    Colors with sufficient collected data use their computed EV-efficiency.
    Colors where balloons EXISTED but NONE were collected (all exploded)
    receive efficiency = 0, since the participant earned nothing from that
    risk level — a clear failure to differentiate strategy.

    Returns None if fewer than 2 colors had any balloons at all.

    High score = participant achieves similar EV-efficiency across risk levels.
    """
    effective_efficiency: dict[str, float] = {}

    for color in ["purple", "teal", "orange"]:
        total = color_balloons.get(color, 0)
        if total == 0:
            continue  # Color not present in session

        collected = color_pumps_collected.get(color, [])

        if len(collected) >= MIN_COLLECTED_FALLBACK:
            # Use computed EV-efficiency
            if color in per_color_efficiency:
                effective_efficiency[color] = per_color_efficiency[color]
        else:
            # Balloons existed but insufficient collected — participant failed
            # to adapt to this risk level. EV-efficiency = 0 (earned nothing).
            effective_efficiency[color] = 0.0

    if len(effective_efficiency) < 2:
        return None

    values = list(effective_efficiency.values())
    mean_eff = float(np.mean(values))

    if mean_eff <= 0:
        return 0.0

    cv = float(np.std(values) / mean_eff)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def _detect_flat_strategy(
    color_pumps_all: dict[str, list[int]],
    color_explosions: dict[str, int],
    color_balloons: dict[str, int],
) -> bool:
    """
    Detect if participant uses an undifferentiated flat pumping strategy.

    A flat strategy is indicated by:
    1. Low CV of per-color RAW mean pumps (similar target across colors)
    2. Explosion rate increasing sharply with risk level (confirming the
       flat target exceeds safe capacity on riskier colors)

    The raw means (not collected-only) are used here because collected-only
    on orange would hide the flat target (truncated by explosions).
    """
    if len(color_pumps_all) < 2:
        return False

    # Compute raw means per color
    raw_means: dict[str, float] = {}
    for color in ["purple", "teal", "orange"]:
        pumps = color_pumps_all.get(color, [])
        if pumps:
            raw_means[color] = float(np.mean(pumps))

    if len(raw_means) < 2:
        return False

    # Check 1: Are raw means similar? (CV < 0.25 = quite flat)
    values = list(raw_means.values())
    mean_val = float(np.mean(values))
    if mean_val <= 0:
        return False

    cv = float(np.std(values) / mean_val)
    if cv > 0.25:
        return False  # Participant IS differentiating by raw pump count

    # Check 2: Does explosion rate increase with risk?
    # A flat strategy targeting X pumps should produce:
    # low explosion on purple (X << 128), moderate on teal, high on orange
    explosion_rates: dict[str, float] = {}
    for color in ["purple", "teal", "orange"]:
        total = color_balloons.get(color, 0)
        if total > 0:
            explosion_rates[color] = color_explosions.get(color, 0) / total

    # If orange explosion rate > 80% and purple < 40%, flat strategy confirmed
    orange_exp = explosion_rates.get("orange", 0)
    purple_exp = explosion_rates.get("purple", 0)

    return orange_exp > 0.8 and purple_exp < 0.5


def _calculate_consistency_breakdown(
    balloons: list[list[GameEvent]],
) -> tuple[float, float]:
    """
    Decompose response consistency into within-balloon and between-balloon components.

    The single global `response_consistency` CV cannot distinguish two very
    different participant profiles:
    - A strategically variable participant who pumps fast on some balloons and
      slow on others (high between-balloon CV, low within-balloon CV).
    - An erratic participant who is inconsistent even during a single balloon
      (high within-balloon CV regardless of between-balloon variation).

    Between-balloon CV uses COLLECTED (non-exploded) balloons only.
    Exploded balloons have truncated pump counts that introduce artificial
    variability not reflecting the participant's actual strategic consistency.
    Falls back to all balloons if fewer than 5 collected balloons available.

    Within-balloon CV is NOT affected by explosion truncation — it measures
    intra-pump latency timing, which is the same regardless of whether the
    balloon later explodes.

    Parameters
    ----------
    balloons : list[list[GameEvent]]
        Per-balloon event segments from _segment_balloons.

    Returns
    -------
    tuple[float, float]
        (within_balloon_cv, between_balloon_cv)

        within_balloon_cv  — mean CV of intra-pump latencies within each balloon
                             that had >= 3 pumps; 0.0 if no qualifying balloons.
        between_balloon_cv — CV of pump counts across collected balloons; 0.0 if
                             fewer than 2 balloons or zero mean.
    """
    # Within-balloon: average of per-balloon latency CVs (not affected by truncation)
    within_cvs: list[float] = []
    for balloon_events in balloons:
        pump_times = [e.timestamp for e in balloon_events if e.type == "pump"]
        if len(pump_times) >= 3:
            diffs = np.diff(pump_times)
            diffs = diffs[diffs < 2000.0]  # Filter outlier pauses > 2 s
            if len(diffs) >= 2 and np.mean(diffs) > 0:
                cv = float(np.std(diffs) / np.mean(diffs))
                within_cvs.append(cv)

    within_balloon_cv = float(np.mean(within_cvs)) if within_cvs else 0.0

    # Between-balloon: CV of pump counts WITHIN each color, then averaged.
    # This isolates genuine strategic inconsistency from appropriate
    # cross-color variation (pumping more on purple than orange is correct,
    # not inconsistent).
    # Uses collected-only balloons per color to avoid truncation variance.
    color_collected_pumps: dict[str, list[int]] = defaultdict(list)
    color_all_pumps: dict[str, list[int]] = defaultdict(list)

    for b in balloons:
        pumps = sum(1 for e in b if e.type == "pump")
        color = _extract_balloon_color(b)
        color_all_pumps[color].append(pumps)
        terminal = next(
            (e.type for e in reversed(b) if e.type in ("collect", "explode")),
            None,
        )
        if terminal != "explode":
            color_collected_pumps[color].append(pumps)

    per_color_cvs: list[float] = []
    for color in color_all_pumps:
        # Prefer collected, fall back to all if too few
        data = color_collected_pumps.get(color, [])
        if len(data) < 3:
            data = color_all_pumps[color]
        if len(data) < 2:
            continue
        arr = np.array(data, dtype=np.float64)
        mean_val = float(np.mean(arr))
        if mean_val > 0:
            per_color_cvs.append(float(np.std(arr) / mean_val))

    between_balloon_cv = float(np.mean(per_color_cvs)) if per_color_cvs else 0.0

    return within_balloon_cv, between_balloon_cv


def _generate_behavioral_profile(
    metrics: BARTMetrics,
) -> dict[str, Any]:
    """
    Generate narrative behavioral profile from EV-efficiency-based metrics.

    Uses scientifically grounded thresholds tied to the Bernoulli explosion
    model rather than arbitrary raw pump count cutoffs.

    Dimensions:
    - Risk Style         (risk_calibration_score, ev_efficiency, flat_strategy)
    - Adaptability       (half_split_learning_rate)
    - Consistency        (within_balloon_consistency, between_balloon_consistency)
    """
    profile: dict[str, Any] = {}

    # 1. Risk Style — use EV-based metrics instead of raw pump counts
    if metrics.flat_strategy_detected:
        risk_style = "Undifferentiated Risk Approach"
        risk_desc = (
            "You applied a similar pumping strategy across all balloon types regardless "
            "of their risk levels. While this can feel efficient, it misses opportunities "
            "on safe balloons and causes excessive losses on risky ones."
        )
        workplace = "May benefit from structured risk frameworks that make risk levels explicit."
    elif metrics.risk_calibration_score >= 80 and metrics.explosion_penalty < 0.1:
        risk_style = "Calibrated Risk Optimizer"
        risk_desc = (
            "You calibrated your risk-taking precisely to match actual danger levels. "
            "You pushed when it was safe and pulled back when risk was high — "
            "maximizing expected reward across all conditions."
        )
        workplace = "Strong fit for roles requiring nuanced risk assessment and optimization."
    elif metrics.explosion_penalty > 0.3:
        risk_style = "Aggressive Risk Taker"
        risk_desc = (
            "You pushed well past optimal stopping points, particularly on higher-risk "
            "balloons. This aggressive approach led to significantly more explosions "
            "than an optimal strategy would produce."
        )
        workplace = "Action-oriented energy that benefits from guardrails and risk frameworks."
    elif metrics.rng_normalized_pumps < 0.10 and metrics.explosion_penalty < 0.05:
        risk_style = "Conservative Safety-Seeker"
        risk_desc = (
            "You prioritized safety and certainty, stopping well before optimal "
            "on most balloons. You avoided losses but left significant reward on the table."
        )
        workplace = "Excellent for roles requiring risk mitigation and compliance."
    else:
        risk_style = "Balanced Explorer"
        risk_desc = (
            "You maintain a reasonable balance between safety and exploration. "
            "Your risk-taking is moderate with room for more precise calibration."
        )
        workplace = "Versatile fit for roles requiring balanced judgment."

    profile["risk_style"] = risk_style
    profile["description"] = risk_desc
    profile["workplace_implication"] = workplace

    # 2. Key Traits — EV-efficiency based
    traits = []

    if metrics.within_balloon_consistency < 0.2 and metrics.between_balloon_consistency < 0.4:
        traits.append("Highly Consistent")
    elif metrics.within_balloon_consistency > 0.6:
        traits.append("Erratic (within-balloon)")
    elif metrics.between_balloon_consistency > 1.0:
        traits.append("Strategically Variable")

    if metrics.half_split_learning_rate > 0.1:
        traits.append("Adaptive Learner")
    elif metrics.half_split_learning_rate < -0.1:
        traits.append("Risk-Averse Learner")

    # Impulsivity: only when we have real collected orange data
    if metrics.orange_avg_pumps is not None and metrics.orange_avg_pumps > 4.0:
        traits.append("Impulsive on High-Risk")

    # Patient Optimizer: purple EV-efficiency > 85% (actually near-optimal play)
    purple_eff = metrics.ev_optimal_stops.get("_purple_efficiency")
    if purple_eff is not None and purple_eff > 0.85:
        traits.append("Patient Optimizer")
    elif metrics.patience_index > 20:
        # Pumping 20+ on purple (optimal ~11) is over-pumping, not patience
        traits.append("Over-Pumper on Safe Balloons")

    if metrics.flat_strategy_detected:
        traits.append("Flat Strategy")

    if metrics.explosion_penalty > 0.3:
        traits.append("High Explosion Penalty")

    profile["dominant_traits"] = traits

    return profile


# ── Main Scoring Function ───────────────────────────────────────────────────


def score_bart(events: list[GameEvent]) -> BARTMetrics:
    """
    Score a BART session from raw events using NumPy vectorization.

    Multi-Risk Profile Analysis:
    - Calculates overall metrics (pumps, explosions, latency)
    - Breaks down performance by balloon color (purple/teal/orange)
    - Computes learning rate and adaptation metrics
    - Provides behavioral indices (impulsivity, patience, consistency)
    - Runs session validation and flags anomalies

    RNG-Truncation Robustness:
    All behavioral-intention metrics (impulsivity, patience, risk calibration,
    learning rate, color discrimination, rng_normalized_pumps, between-balloon
    consistency) use COLLECTED (non-exploded) balloons only.  Exploded balloons
    have their pump counts truncated by RNG, which does not reflect the
    participant's intended pumping strategy.

    Parameters
    ----------
    events : list[GameEvent]
        Chronologically ordered BART events (already validated).

    Returns
    -------
    BARTMetrics
        Computed psychometric metrics including color-based and learning metrics.

    Raises
    ------
    ValueError
        If event log is empty or contains no balloon data.
    """
    if not events:
        raise ValueError("Empty event log")

    # ── Session validation ────────────────────────────────────────────────────
    validation = validate_bart_session(events)
    session_valid = validation["is_valid"]
    session_warnings = list(validation["warnings"])

    balloons = _segment_balloons(events)

    if not balloons:
        raise ValueError("No balloon data found in event log")

    # DEBUG: Log color extraction
    balloon_colors = [_extract_balloon_color(b) for b in balloons]
    color_counts: dict[str, int] = {}
    for color in balloon_colors:
        color_counts[color] = color_counts.get(color, 0) + 1
    logger.info(
        "BART color distribution: %s (total %d balloons)", color_counts, len(balloons)
    )
    logger.debug("First 5 balloon colors: %s", balloon_colors[:5])

    # ── Data Collection ──────────────────────────────────────────────────────
    pump_counts: list[int] = []
    non_exploded_pumps: list[int] = []
    total_explosions = 0
    total_collections = 0

    # Color-based tracking: ALL balloons (descriptive metrics)
    color_pumps_all: dict[str, list[int]] = defaultdict(list)
    # Color-based tracking: COLLECTED only (behavioral-intention metrics)
    color_pumps_collected: dict[str, list[int]] = defaultdict(list)
    color_explosions: dict[str, int] = defaultdict(int)
    color_balloons: dict[str, int] = defaultdict(int)

    # Learning rate data: (trial_number, color, pumps, exploded)
    balloon_data: list[tuple[int, str, int, bool]] = []

    for trial_idx, balloon_events in enumerate(balloons):
        pumps = sum(1 for e in balloon_events if e.type == "pump")
        pump_counts.append(pumps)

        color = _extract_balloon_color(balloon_events)
        color_balloons[color] += 1

        terminal = next(
            (e.type for e in reversed(balloon_events) if e.type in ("collect", "explode")),
            None,
        )

        exploded = terminal == "explode"

        # Track ALL balloon pump counts (for descriptive metrics)
        color_pumps_all[color].append(pumps)

        if exploded:
            total_explosions += 1
            color_explosions[color] += 1
        else:
            # Collected or incomplete — reflects full behavioral intention
            total_collections += 1 if terminal == "collect" else 0
            non_exploded_pumps.append(pumps)
            color_pumps_collected[color].append(pumps)

        balloon_data.append((trial_idx, color, pumps, exploded))

    total_balloons = len(balloons)
    all_pumps_array = np.array(pump_counts, dtype=np.float64)
    total_pumps = int(np.sum(all_pumps_array))

    # ── Money collected ───────────────────────────────────────────────────────
    # Each pump on a collected balloon is worth $0.25.
    _money_pumps = 0
    money_collected = 0.0
    for evt in events:
        if evt.type == "pump":
            _money_pumps += 1
        elif evt.type == "collect":
            money_collected += _money_pumps * 0.25
            _money_pumps = 0
        elif evt.type == "explode":
            _money_pumps = 0

    # Theoretical optimal expected earnings (from EV-curve simulation):
    # 10 purple × EV(11,128)×0.25 + 10 teal × EV(5,32)×0.25 + 10 orange × EV(2,8)×0.25 ≈ $27.03
    _optimal_expected_earnings = 0.0
    for color_name, profile in COLOR_PROFILES.items():
        max_p = profile["max_pumps"]
        n_balloons = color_balloons.get(color_name, 10)
        optimal_stop, optimal_ev = _get_ev_optimal(max_p)
        _optimal_expected_earnings += n_balloons * optimal_ev * 0.25
    money_efficiency = min(1.0, money_collected / _optimal_expected_earnings) if _optimal_expected_earnings > 0 else 0.0

    # ── Resolve collected-vs-all per color ────────────────────────────────────
    # For each color, prefer collected (non-exploded) pump data for behavioral
    # metrics.  Fall back to all balloons if too few collected are available,
    # and emit a warning so downstream consumers know the metric is degraded.
    color_pumps_behavioral: dict[str, list[int]] = {}
    for color in COLOR_PROFILES:
        collected = color_pumps_collected.get(color, [])
        all_data = color_pumps_all.get(color, [])
        chosen, used_fallback = _prefer_collected(collected, all_data)
        color_pumps_behavioral[color] = chosen
        if used_fallback and len(all_data) > 0:
            session_warnings.append(
                f"RNG-truncation fallback: {color} has only {len(collected)} collected "
                f"balloon(s) (< {MIN_COLLECTED_FALLBACK}); using all {len(all_data)} "
                f"balloons (includes truncated pump counts from explosions)"
            )

    # ── Overall Metrics ──────────────────────────────────────────────────────

    # Average Pumps (all balloons) — descriptive, includes RNG-truncated counts.
    # Useful for comparison with other studies but NOT recommended for behavioral
    # intention measurement.  Use rng_normalized_pumps (collected-only) instead.
    avg_pumps_all_balloons = float(np.mean(all_pumps_array))

    # Average Pumps Adjusted (non-exploded only) — classic BART censoring correction.
    # Note: This metric excludes exploded balloons to avoid right-censoring bias
    # (we don't know how many more times an exploded balloon would have been pumped).
    if non_exploded_pumps:
        adjusted_array = np.array(non_exploded_pumps, dtype=np.float64)
        average_pumps_adjusted = float(np.mean(adjusted_array))
    else:
        # All balloons exploded — fall back to overall mean (no censoring to correct for)
        average_pumps_adjusted = avg_pumps_all_balloons

    # Explosion Rate
    explosion_rate = total_explosions / total_balloons if total_balloons > 0 else 0.0

    # Mean Inter-Pump Latency — computed per-balloon to exclude cross-balloon gaps.
    # Not affected by RNG truncation: the timing between pumps 1->2->3 is the same
    # regardless of whether the balloon later explodes.
    all_intra_latencies: list[float] = []
    for balloon_events in balloons:
        pump_times = [e.timestamp for e in balloon_events if e.type == "pump"]
        if len(pump_times) >= 2:
            diffs = np.diff(pump_times)
            all_intra_latencies.extend(diffs.tolist())

    intra_balloon_latencies = np.array(all_intra_latencies, dtype=np.float64)
    # Filter remaining within-balloon outliers (hesitation pauses > 2 seconds)
    if intra_balloon_latencies.size > 0:
        intra_balloon_latencies = intra_balloon_latencies[intra_balloon_latencies < 2000.0]

    if intra_balloon_latencies.size > 0:
        mean_latency = float(np.mean(intra_balloon_latencies))
    else:
        mean_latency = 0.0

    # ── EV-Based Metrics (scientifically rigorous, v3) ───────────────────────
    # Computed early so per-color results are available for ColorMetrics.

    # Compute dynamic EV-optimal stops
    ev_optimal_stops: dict[str, int] = {}
    for color, profile in COLOR_PROFILES.items():
        opt_stop, _ = _get_ev_optimal(profile["max_pumps"])
        ev_optimal_stops[color] = opt_stop

    # EV-Ratio Score: EV(participant) / EV(optimal) per color
    # Uses COLLECTED-ONLY data — never fallback/truncated data.
    ev_ratio_score, per_color_efficiency = _compute_ev_ratio_score(
        color_pumps_collected, color_balloons,
    )

    # Explosion Penalty: excess explosion rate vs expected at optimal
    explosion_penalty, per_color_excess = _compute_explosion_penalty(
        color_explosions, color_balloons,
    )

    # ── Color-Based Metrics (descriptive, uses ALL balloons) ─────────────────
    color_metrics_list: list[ColorMetrics] = []

    for color in ["purple", "teal", "orange"]:
        if color not in color_balloons:
            continue

        balloons_of_color = color_balloons[color]
        pumps_of_color = color_pumps_all.get(color, [])
        collected_of_color = color_pumps_collected.get(color, [])

        avg_pumps = float(np.mean(pumps_of_color)) if pumps_of_color else 0.0
        color_exp_rate = (
            color_explosions[color] / balloons_of_color if balloons_of_color > 0 else 0.0
        )

        # Behavioral avg: prefer collected-only, fallback to all
        behavioral_data, used_fb = _prefer_collected(collected_of_color, pumps_of_color)
        behavioral_avg = float(np.mean(behavioral_data)) if behavioral_data else 0.0

        color_ev_eff = per_color_efficiency.get(color)
        color_ev_optimal = ev_optimal_stops.get(color)
        color_excess_exp = per_color_excess.get(color)

        color_metrics_list.append(
            ColorMetrics(
                color=color,
                average_pumps=round(avg_pumps, 4),
                behavioral_avg_pumps=round(behavioral_avg, 4),
                explosion_rate=round(color_exp_rate, 4),
                total_balloons=balloons_of_color,
                collected_count=len(collected_of_color),
                risk_profile=COLOR_PROFILES[color]["risk"],
                used_fallback=used_fb,
                ev_efficiency=round(color_ev_eff, 4) if color_ev_eff is not None else None,
                ev_optimal_stop=color_ev_optimal,
                excess_explosion_rate=round(color_excess_exp, 4) if color_excess_exp is not None else None,
            ),
        )

    # ── Learning & Adaptation Metrics (use collected-only internally) ────────

    # Learning Rate — regression-based (preserved for backward compat; noisy at N=10)
    learning_rate = _calculate_learning_rate(balloon_data)

    # Half-Split Learning Rate — more robust at N=10 per color
    half_split_lr = _calculate_half_split_learning_rate(balloon_data)

    # Color Discrimination (LEGACY — Cohen's d, kept for backward compat)
    color_discrimination = _calculate_color_discrimination(color_pumps_behavioral)

    # Risk Adjustment Score (LEGACY — kept for backward compat)
    risk_adjustment = _calculate_risk_adjustment_score(color_pumps_behavioral)

    # Risk Sensitivity (Pearson r — kept for descriptive use)
    risk_sensitivity = _calculate_risk_sensitivity(color_pumps_behavioral)

    # Risk Calibration Score: combines EV-efficiency with explosion penalty
    risk_calibration_score = float(
        np.clip(ev_ratio_score * (1.0 - explosion_penalty), 0.0, 100.0)
    )

    # EV-Efficiency Differentiation (replaces Cohen's d)
    ev_efficiency_diff = _compute_ev_efficiency_differentiation(
        per_color_efficiency, color_pumps_collected, color_balloons,
    )

    # Flat Strategy Detection
    flat_strategy = _detect_flat_strategy(
        color_pumps_all, color_explosions, color_balloons,
    )

    # ── Behavioral Indices (use collected-only data) ─────────────────────────

    # Orange average pumps — None when insufficient collected data
    orange_collected_real = color_pumps_collected.get("orange", [])
    has_orange_data = len(orange_collected_real) >= MIN_COLLECTED_FALLBACK
    orange_avg_pumps: float | None = (
        float(np.mean(orange_collected_real)) if has_orange_data else None
    )

    # Response Consistency — global CV of all intra-balloon latencies.
    # Not affected by RNG truncation (measures timing, not pump counts).
    if intra_balloon_latencies.size > 1:
        cv = float(np.std(intra_balloon_latencies) / np.mean(intra_balloon_latencies))
        response_consistency = cv
    else:
        response_consistency = 0.0

    # Consistency breakdown (within-balloon vs between-balloon)
    within_balloon_cv, between_balloon_cv = _calculate_consistency_breakdown(balloons)

    # Impulsivity Index — composite feature combining multiple signals.
    # Always computable (never None), even when orange has 0 collected.
    #
    # Orange explosions primarily reflect poor color discrimination (the
    # optimal stop is only ~2 pumps, so even slight misjudgment causes
    # explosions). True impulsivity is better captured by timing behavior
    # and overall excess explosions across ALL colors.
    #
    # Components:
    #   1. Timing signal (weight 0.40):
    #      - Fast, reflexive pumping: 1 - clamp(mean_latency / 800, 0, 1)
    #      - Amplified by low within-balloon CV (consistent fast = reflexive)
    #   2. Excess explosion signal (weight 0.40):
    #      - Overall explosion_penalty captures over-pumping across ALL colors
    #   3. Orange risk-taking signal (weight 0.20):
    #      - Mild contribution — orange exploding is mostly discrimination
    #      - If orange collected: 1 - EV_efficiency (over-pumping past optimal)
    #      - If all exploded: orange_explosion_rate
    #
    # Range: [0, 1]. Higher = more impulsive.
    orange_total = color_balloons.get("orange", 0)
    orange_expl = color_explosions.get("orange", 0)

    # Component 1: Timing impulsivity (fast + consistent = reflexive)
    latency_signal = 1.0 - min(1.0, mean_latency / 800.0) if mean_latency > 0 else 0.0
    # Amplify if within-balloon timing is very consistent (low CV = autopilot)
    timing_impulsivity = min(1.0, latency_signal * (1.0 + 0.5 * max(0.0, 0.5 - within_balloon_cv)))

    # Component 2: Excess explosions across all colors
    explosion_signal = min(1.0, explosion_penalty * 2.0)  # Scale: 0.5 penalty → 1.0 signal

    # Component 3: Orange risk-taking (low weight — mostly discrimination)
    if has_orange_data and "orange" in per_color_efficiency:
        orange_signal = 1.0 - per_color_efficiency["orange"]
    elif orange_total > 0:
        orange_signal = orange_expl / orange_total
    else:
        orange_signal = 0.0

    # Weighted composite
    impulsivity_index = float(np.clip(
        0.40 * timing_impulsivity + 0.40 * explosion_signal + 0.20 * orange_signal,
        0.0, 1.0,
    ))

    # Patience Index — mean pumps on collected purple balloons
    purple_behavioral = color_pumps_behavioral.get("purple", [])
    patience_index = float(np.mean(purple_behavioral)) if purple_behavioral else 0.0

    # Patience Index Normalized
    purple_max = float(COLOR_PROFILES["purple"]["max_pumps"])
    patience_index_normalized = float(np.clip(patience_index / purple_max, 0.0, 1.0))

    # ── Composite Metrics ────────────────────────────────────────────────────

    # Adaptive Strategy Score — conditional weighting based on calibration quality.
    # If already well-calibrated (ev_ratio >= 80), learning matters less.
    # If poorly calibrated, learning signal is more informative.
    # Money efficiency gets a fixed 10% weight (outcome grounding).
    safe_hslr = 0.0 if np.isnan(half_split_lr) else half_split_lr
    safe_ev_diff = ev_efficiency_diff if ev_efficiency_diff is not None else 0.0
    safe_ev_ratio = ev_ratio_score / 100.0  # Normalize to [0, 1]

    W_MONEY = 0.10  # Fixed 10% for realized outcome
    if ev_ratio_score >= 80:
        w_learning, w_calibration, w_differentiation = 0.09, 0.45, 0.36
    else:
        w_learning, w_calibration, w_differentiation = 0.36, 0.27, 0.27

    learning_component = (safe_hslr + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    calibration_component = safe_ev_ratio           # [0, 1]
    differentiation_component = safe_ev_diff        # [0, 1]
    money_component = money_efficiency              # [0, 1]

    adaptive_strategy_score = (
        learning_component * w_learning
        + calibration_component * w_calibration
        + differentiation_component * w_differentiation
        + money_component * W_MONEY
    ) * 100.0
    adaptive_strategy_score = float(np.clip(adaptive_strategy_score, 0.0, 100.0))

    # RNG-Normalized Pumps — color-mean-then-average.
    # Average WITHIN each color first (per-color mean), then average across colors.
    # This gives equal weight to each risk level regardless of collection rates.
    # Uses collected-only data where available.
    per_color_normalized: list[float] = []
    for color in COLOR_PROFILES:
        collected = color_pumps_collected.get(color, [])
        if len(collected) >= MIN_COLLECTED_FALLBACK:
            cap = COLOR_PROFILES[color]["max_pumps"]
            color_mean = float(np.mean(collected)) / cap
            per_color_normalized.append(color_mean)

    rng_normalized_pumps = (
        float(np.mean(per_color_normalized)) if per_color_normalized else 0.0
    )

    # Store per-color efficiency in ev_optimal_stops dict for profile access
    _ev_stops_with_eff = dict(ev_optimal_stops)
    for c, eff in per_color_efficiency.items():
        _ev_stops_with_eff[f"_{c}_efficiency"] = eff

    # ── Logging ──────────────────────────────────────────────────────────────
    logger.info(
        "BART scored — balloons=%d pumps=%d explosions=%d "
        "avg_adjusted=%.2f avg_all=%.2f latency=%.1fms "
        "ev_ratio=%.1f explosion_penalty=%.3f risk_cal=%.1f "
        "adaptive_score=%.1f flat_strategy=%s "
        "patience=%.2f rng_norm=%.3f valid=%s warnings=%d",
        total_balloons,
        total_pumps,
        total_explosions,
        average_pumps_adjusted,
        avg_pumps_all_balloons,
        mean_latency,
        ev_ratio_score,
        explosion_penalty,
        risk_calibration_score,
        adaptive_strategy_score,
        flat_strategy,
        patience_index,
        rng_normalized_pumps,
        session_valid,
        len(session_warnings),
    )

    # ── Assemble metrics object ───────────────────────────────────────────────
    metrics_obj = BARTMetrics(
        # Core
        average_pumps_adjusted=round(average_pumps_adjusted, 4),
        explosion_rate=round(explosion_rate, 4),
        mean_latency_between_pumps=round(mean_latency, 4),
        total_balloons=total_balloons,
        total_pumps=total_pumps,
        total_explosions=total_explosions,
        total_collections=total_collections,
        # Color breakdown
        color_metrics=color_metrics_list,
        # Learning (legacy + robust)
        learning_rate=round(learning_rate, 4),
        half_split_learning_rate=round(half_split_lr, 4),
        # Legacy risk calibration (kept for backward compat)
        risk_adjustment_score=round(risk_adjustment, 4),
        color_discrimination_index=round(color_discrimination, 4) if not np.isnan(color_discrimination) else None,
        risk_sensitivity=round(risk_sensitivity, 4),
        # EV-based metrics (scientifically rigorous, v3)
        ev_ratio_score=round(ev_ratio_score, 4),
        explosion_penalty=round(explosion_penalty, 4),
        risk_calibration_score=round(risk_calibration_score, 4),
        ev_efficiency_differentiation=round(ev_efficiency_diff, 4) if ev_efficiency_diff is not None else None,
        flat_strategy_detected=flat_strategy,
        money_collected=round(money_collected, 2),
        money_efficiency=round(money_efficiency, 4),
        ev_optimal_stops=_ev_stops_with_eff,
        # Behavioral intention metrics (RNG-robust, collected-only)
        rng_normalized_pumps=round(rng_normalized_pumps, 4),
        avg_pumps_all_balloons=round(avg_pumps_all_balloons, 4),
        orange_avg_pumps=round(orange_avg_pumps, 4) if orange_avg_pumps is not None else None,
        impulsivity_index=round(impulsivity_index, 4),
        # Patience
        patience_index=round(patience_index, 4),
        patience_index_normalized=round(patience_index_normalized, 4),
        # Consistency
        response_consistency=round(response_consistency, 4),
        within_balloon_consistency=round(within_balloon_cv, 4),
        between_balloon_consistency=round(between_balloon_cv, 4),
        # Composite
        adaptive_strategy_score=round(adaptive_strategy_score, 4),
        # Session validity
        session_valid=session_valid,
        session_warnings=session_warnings,
        # Narrative profile (populated below)
        behavioral_profile={},
    )

    # Generate narrative profile using validated metrics
    profile = _generate_behavioral_profile(metrics_obj)
    metrics_obj.behavioral_profile = profile

    return metrics_obj
