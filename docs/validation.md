# Session Validation

Remote behavioral data is vulnerable to bots, distraction, and superficial
engagement. The engine embeds a validation pipeline that runs automatically
inside {py:func}`~scoring.bart.score_bart` and is also callable on its own via
{py:func}`~scoring.bart.validate_bart_session`.

Validation never raises on a *playable* session; instead it sets a boolean
`is_valid` and accumulates human-readable `warnings`. Only structurally broken
input (an empty log, or no balloons) raises.

## What `validate_bart_session` returns

```python
{
    "is_valid": bool,                  # False if any hard check fails
    "warnings": list[str],             # all soft + hard findings
    "balloon_count": int,
    "color_distribution": dict,        # {"purple": 10, "teal": 10, "orange": 10}
}
```

## Checks

```{list-table}
:header-rows: 1
:widths: 26 18 56

* - Check
  - Effect
  - Rule
* - **Completeness**
  - invalidates
  - Fewer than 15 of 30 balloons → `is_valid = False`.
* - **Attrition**
  - warns
  - 15–29 balloons → retained but flagged incomplete.
* - **Color balance**
  - warns
  - Fewer than 5 balloons of a color → "too few"; 5–9 → "partial" (target is 10 each).
* - **Monotonicity**
  - invalidates
  - Any out-of-order timestamp → `is_valid = False` (browser lag, desync, or tampering).
* - **Pacing**
  - warns
  - A ≥15-balloon session finished in under 30 s.
* - **Pump uniformity**
  - warns
  - Across ≥10 balloons, pump-count standard deviation < 0.5 → possible automated play.
```

## Auto-repeat detection

Separately from `validate_bart_session`, the scorer detects **OS key-repeat**:
holding the pump key produces a burst of pumps at ~30–80 ms intervals rather
than discrete decisions. {py:func}`~scoring.bart._is_autorepeat_balloon` flags a
balloon when the median inter-pump interval (over intervals < 2000 ms) is below
80 ms. Auto-repeat balloons are:

- **excluded** from all behavioral-intention metrics (pumps, learning,
  consistency, latency), and
- reported in `session_warnings` with the count of affected balloons.

They still count toward `total_balloons`, `explosion_rate`, and `money_collected`.

## The RNG fallback

When a color has fewer than {py:data}`~scoring.bart.MIN_COLLECTED_FALLBACK` (2)
collected balloons, the engine cannot estimate behavioral intention from
collected trials alone. It then falls back to using all trials of that color for
that color's behavioral mean and emits an `RNG fallback` warning. The
EV-referenced efficiency for such a color is instead set to zero, because the
participant secured essentially no payoff at that hazard level.

## Interpreting validity

`session_valid = False` does not necessarily mean the data is worthless — it
means at least one hard check failed (too few balloons, or non-monotonic
timestamps). Treat `session_warnings` as a triage list: decide per study whether
to exclude, retain, or manually review each flagged session. The reference study
retained 15–29-balloon sessions with warnings and excluded sub-15 sessions
entirely.
