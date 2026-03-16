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
    The true peaks of the EV curves are actually close to the square root of N.

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
    Calculate risk adjustment score based on appropriate behavior per color.

    Scores the participant on whether their average pumps are calibrated to
    each balloon's risk level.

    Expects collected-only pump data to avoid RNG truncation bias.
    Using all balloons would undercount orange pumps (most explode, truncating
    the count) and give undeserved credit for "low" orange pumping.

    Ideal behavior under the sequential Bernoulli explosion model
    (P(explode at pump k) = k / maxPumps):
    - Purple (N=128): EV-optimal ~ 12.0 pumps.
    - Teal   (N=32):  EV-optimal ~ 6.0 pumps.
    - Orange (N=8):   EV-optimal ~ 2.0 pumps.

    Note: The score evaluates the absolute distance from the EV-optimal point,
    scaled linearly down to 0 at the extreme distances (0 or max_pumps).

    Returns
    -------
    float
        Risk adjustment score (0-100). 100 = optimal risk calibration.
    """
    optimal_stops = {"purple": 12.0, "teal": 6.0, "orange": 2.0}
    max_pumps_caps = {"purple": 128, "teal": 32, "orange": 8}
    scores = []

    for color in ["purple", "teal", "orange"]:
        if color in color_pumps and len(color_pumps[color]) > 0:
            mean_pumps = np.mean(color_pumps[color])
            opt = optimal_stops[color]
            mx = max_pumps_caps[color]
            
            # Max possible distance from optimal (either down to 0, or up to max_pumps)
            max_dist = max(opt, mx - opt)
            
            # Score is 100 at optimal, scaling linearly down to 0 at the extremes
            score = float(np.clip(1.0 - abs(mean_pumps - opt) / max_dist, 0.0, 1.0) * 100.0)
            scores.append(score)

    if not scores:
        return 0.0

    result = float(np.mean(scores))
    if np.isnan(result):
        return 0.0
    return result


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

    # Between-balloon: CV of pump counts across COLLECTED balloons only.
    # Exploded balloons have truncated pump counts that add artificial variance.
    collected_pump_counts = []
    all_pump_counts = []
    for b in balloons:
        pumps = sum(1 for e in b if e.type == "pump")
        all_pump_counts.append(pumps)
        terminal = next(
            (e.type for e in reversed(b) if e.type in ("collect", "explode")),
            None,
        )
        if terminal != "explode":
            collected_pump_counts.append(pumps)

    # Fall back to all balloons if too few collected
    pump_counts_for_cv = (
        collected_pump_counts if len(collected_pump_counts) >= 5 else all_pump_counts
    )
    pump_arr = np.array(pump_counts_for_cv, dtype=np.float64)

    if len(pump_arr) >= 2 and np.mean(pump_arr) > 0:
        between_balloon_cv = float(np.std(pump_arr) / np.mean(pump_arr))
    else:
        between_balloon_cv = 0.0

    return within_balloon_cv, between_balloon_cv


def _generate_behavioral_profile(
    metrics: BARTMetrics,
) -> dict[str, Any]:
    """
    Generate Pymetrics-style narrative behavioral profile from metrics.

    Dimensions:
    - Risk Style         (rng_normalized_pumps vs explosion_rate)
    - Adaptability       (half_split_learning_rate vs risk_sensitivity)
    - Consistency        (within_balloon_consistency, between_balloon_consistency)
    """
    profile: dict[str, Any] = {}

    # 1. Risk Style — use rng_normalized_pumps (behavioral) rather than explosion_rate (outcome)
    if metrics.rng_normalized_pumps > 0.45:
        risk_style = "High-Stakes Risk Taker"
        risk_desc = (
            "You tend to push limits to the absolute maximum. "
            "While this can lead to high rewards, it often results in frequent failures."
        )
        workplace = "Best suited for R&D or crisis management where bold action is required."
    elif metrics.rng_normalized_pumps < 0.10 and metrics.average_pumps_adjusted < 10:
        risk_style = "Conservative Safety-Seeker"
        risk_desc = (
            "You prioritize safety and certainty over potential gains. "
            "You avoid unnecessary risks but may miss out on high-reward opportunities."
        )
        workplace = "Excellent for QA, compliance, or finance roles requiring risk mitigation."
    elif metrics.risk_sensitivity > 0.7:
        risk_style = "Strategic Risk Taker"
        risk_desc = (
            "You possess strong risk intuition. You take calculated risks "
            "only when the odds are in your favor and pull back when danger increases."
        )
        workplace = "Strong fit for trading, strategy, or leadership roles requiring calculated decisions."
    else:
        risk_style = "Balanced Explorer"
        risk_desc = (
            "You maintain a healthy balance between safety and exploration. "
            "You are willing to take risks but generally stay within reasonable bounds."
        )
        workplace = "Versatile fit for general management and operational roles."

    profile["risk_style"] = risk_style
    profile["description"] = risk_desc
    profile["workplace_implication"] = workplace

    # 2. Key Traits
    traits = []

    if metrics.within_balloon_consistency < 0.2 and metrics.between_balloon_consistency < 0.4:
        traits.append("Highly Consistent")
    elif metrics.within_balloon_consistency > 0.6:
        traits.append("Erratic (within-balloon)")
    elif metrics.between_balloon_consistency > 1.0:
        traits.append("Strategically Variable")

    # Use half_split_learning_rate (more robust) for the trait label
    if metrics.half_split_learning_rate > 0.1:
        traits.append("Adaptive Learner")
    elif metrics.half_split_learning_rate < -0.1:
        traits.append("Risk-Averse Learner")

    # Impulsivity: orange avg pumps > 4 (half of orange max=8) = pumping past
    # the EV-optimal region and into clearly above-average risk on high-risk balloons.
    if metrics.orange_avg_pumps > 4.0:
        traits.append("Impulsive")

    # Patience: patience_index > 40 pumps on purple (31% of max=128)
    if metrics.patience_index > 40:
        traits.append("Patient Optimizer")

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
            ),
        )

    # ── Learning & Adaptation Metrics (use collected-only internally) ────────

    # Learning Rate — regression-based (preserved for backward compat; noisy at N=10)
    # Now uses collected-only balloons internally to avoid RNG truncation bias.
    learning_rate = _calculate_learning_rate(balloon_data)

    # Half-Split Learning Rate — more robust at N=10 per color
    # Now uses collected-only balloons internally to avoid RNG truncation bias.
    half_split_lr = _calculate_half_split_learning_rate(balloon_data)

    # Color Discrimination (purple vs orange behavior differentiation, Cohen's d)
    # Uses collected-only pump data via color_pumps_behavioral.
    color_discrimination = _calculate_color_discrimination(color_pumps_behavioral)

    # Risk Adjustment Score (appropriate behavior per color, 0-100)
    # Uses collected-only pump data via color_pumps_behavioral.
    risk_adjustment = _calculate_risk_adjustment_score(color_pumps_behavioral)

    # Risk Sensitivity (Pearson r between risk capacity and pumping behavior)
    # Uses collected-only pump data via color_pumps_behavioral.
    risk_sensitivity = _calculate_risk_sensitivity(color_pumps_behavioral)

    # ── Behavioral Indices (use collected-only data) ─────────────────────────

    # Orange average pumps — uses collected (non-exploded) balloons to reflect
    # actual behavioral intention.  Exploded orange balloons have truncated counts
    # (P(survive 4 pumps on orange) ~ 16%, so most orange balloons explode).
    orange_behavioral = color_pumps_behavioral.get("orange", [])
    orange_avg_pumps = float(np.mean(orange_behavioral)) if orange_behavioral else 0.0

    # Impulsivity Index — behavioral, normalized to orange max capacity.
    # Uses collected-only orange balloons: the pump count at which the participant
    # CHOSE to stop, not where RNG terminated them.
    # Range [0, 1]: 0 = no pumping, 1 = always pumped to the explosion ceiling.
    orange_max = float(COLOR_PROFILES["orange"]["max_pumps"])
    impulsivity_index = float(np.clip(orange_avg_pumps / orange_max, 0.0, 1.0))

    # Patience Index — mean pumps on collected purple balloons.
    # Uses collected-only to reflect actual chosen stopping point, not
    # RNG-truncated pump counts from exploded purple balloons.
    purple_behavioral = color_pumps_behavioral.get("purple", [])
    patience_index = float(np.mean(purple_behavioral)) if purple_behavioral else 0.0

    # Patience Index Normalized — patience_index / purple max capacity (128).
    purple_max = float(COLOR_PROFILES["purple"]["max_pumps"])
    patience_index_normalized = float(np.clip(patience_index / purple_max, 0.0, 1.0))

    # Response Consistency — global CV of all intra-balloon latencies.
    # Not affected by RNG truncation (measures timing, not pump counts).
    # Note: High CV can mean erratic timing OR deliberate bimodal strategy
    # (fast pumps on safe balloons, slow deliberate pumps on risky ones).
    # See within_balloon_consistency and between_balloon_consistency for a
    # decomposed view that distinguishes these two cases.
    if intra_balloon_latencies.size > 1:
        cv = float(np.std(intra_balloon_latencies) / np.mean(intra_balloon_latencies))
        response_consistency = cv
    else:
        response_consistency = 0.0

    # Consistency breakdown (within-balloon vs between-balloon)
    # Between-balloon CV now uses collected-only balloons internally.
    within_balloon_cv, between_balloon_cv = _calculate_consistency_breakdown(balloons)

    # ── Composite Metrics ────────────────────────────────────────────────────

    # Adaptive Strategy Score — composite of learning, discrimination, and risk adjustment.
    # All three sub-metrics now use collected-only data, making the composite
    # robust to RNG truncation bias.
    # Design decision: Equal weighting (33.33% each).
    # Rationale: No theoretical basis to privilege one sub-metric over another.
    # Components:
    #   1. half_split_learning_rate: [-1, 1] -> [0, 33.33]
    #   2. color_discrimination:     [0, 1]  -> [0, 33.33]
    #   3. risk_adjustment:          [0, 100] -> [0, 33.33]
    safe_hslr = 0.0 if np.isnan(half_split_lr) else half_split_lr
    safe_color_discrimination = 0.0 if np.isnan(color_discrimination) else color_discrimination
    safe_risk_adjustment = 0.0 if np.isnan(risk_adjustment) else risk_adjustment

    adaptive_strategy_score = (
        (safe_hslr + 1.0) / 2.0 * 33.33          # Normalize [-1, 1] -> [0, 33.33]
        + safe_color_discrimination * 33.33        # [0, 1] -> [0, 33.33]
        + (safe_risk_adjustment / 100.0) * 33.33   # [0, 100] -> [0, 33.33]
    )
    adaptive_strategy_score = float(np.clip(adaptive_strategy_score, 0.0, 100.0))

    # RNG-Normalized Pumps — mean(pumps / color_max_pumps) across COLLECTED balloons.
    # Uses collected-only data so the fraction reflects the participant's CHOSEN
    # stopping point, not an RNG-truncated pump count.
    # Captures behavioral intention as a fraction of each balloon's capacity,
    # making it independent of which color appeared most.
    normalized_pump_list: list[float] = []
    for color, pumps_list in color_pumps_behavioral.items():
        if color in COLOR_PROFILES:
            cap = COLOR_PROFILES[color]["max_pumps"]
            for p in pumps_list:
                normalized_pump_list.append(p / cap)

    rng_normalized_pumps = (
        float(np.mean(normalized_pump_list)) if normalized_pump_list else 0.0
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    logger.info(
        "BART scored — balloons=%d pumps=%d explosions=%d "
        "avg_adjusted=%.2f avg_all=%.2f latency=%.1fms "
        "learning_rate=%.3f half_split_lr=%.3f "
        "discrimination=%.3f risk_adj=%.1f adaptive_score=%.1f "
        "sensitivity=%.2f orange_avg=%.2f impulsivity=%.3f "
        "patience=%.2f(norm=%.3f) rng_norm=%.3f valid=%s warnings=%d",
        total_balloons,
        total_pumps,
        total_explosions,
        average_pumps_adjusted,
        avg_pumps_all_balloons,
        mean_latency,
        learning_rate,
        half_split_lr,
        color_discrimination,
        risk_adjustment,
        adaptive_strategy_score,
        risk_sensitivity,
        orange_avg_pumps,
        impulsivity_index,
        patience_index,
        patience_index_normalized,
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
        # Learning
        learning_rate=round(learning_rate, 4),
        half_split_learning_rate=round(half_split_lr, 4),
        # Risk calibration
        risk_adjustment_score=round(risk_adjustment, 4),
        color_discrimination_index=round(color_discrimination, 4),
        risk_sensitivity=round(risk_sensitivity, 4),
        # Behavioral intention metrics (RNG-robust, collected-only)
        rng_normalized_pumps=round(rng_normalized_pumps, 4),
        avg_pumps_all_balloons=round(avg_pumps_all_balloons, 4),
        orange_avg_pumps=round(orange_avg_pumps, 4),
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
