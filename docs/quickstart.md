# Quick Start

This page scores a single BART session end to end.

## 1. Build an event log

The scoring engine consumes a flat, chronologically ordered list of
{py:class}`~scoring.schemas.GameEvent` objects. Each event has a monotonic
`timestamp` (milliseconds, e.g. from `performance.now()`), a `type` (`"pump"`,
`"collect"`, or `"explode"`), and a `payload` carrying at least the balloon
`color`.

A balloon is the run of `pump` events up to and including its terminal
`collect` or `explode` event.

```python
from scoring.schemas import GameEvent, EventPayload
from scoring.bart import score_bart

events = [
    # A purple balloon collected after 3 pumps
    GameEvent(timestamp=100, type="pump",    payload=EventPayload(color="purple")),
    GameEvent(timestamp=400, type="pump",    payload=EventPayload(color="purple")),
    GameEvent(timestamp=750, type="pump",    payload=EventPayload(color="purple")),
    GameEvent(timestamp=950, type="collect", payload=EventPayload(color="purple")),

    # An orange balloon that burst on the 3rd pump
    GameEvent(timestamp=1200, type="pump",    payload=EventPayload(color="orange")),
    GameEvent(timestamp=1500, type="pump",    payload=EventPayload(color="orange")),
    GameEvent(timestamp=1800, type="pump",    payload=EventPayload(color="orange")),
    GameEvent(timestamp=1850, type="explode", payload=EventPayload(color="orange")),
    # ... more balloons
]
```

## 2. Score it

```python
metrics = score_bart(events)

print(f"EV ratio score:    {metrics.ev_ratio_score:.1f} / 100")
print(f"Adaptive strategy: {metrics.adaptive_strategy_score:.1f} / 100")
print(f"Money collected:   ${metrics.money_collected:.2f}")
print(f"RNG-norm pumps:    {metrics.rng_normalized_pumps:.2f}")
print(f"Risk style:        {metrics.behavioral_profile['risk_style']}")
```

{py:func}`~scoring.bart.score_bart` returns a {py:class}`~scoring.schemas.BARTMetrics`
object — a pydantic model whose fields are the full metric set described in the
[Metrics Reference](metrics_reference.md). Convert it to a dictionary or JSON
with the usual pydantic methods (`metrics.model_dump()` / `metrics.model_dump_json()`).

```{admonition} A full 30-balloon session
:class: note

A complete, valid session has 30 balloons (10 purple, 10 teal, 10 orange). With
fewer than 15 balloons the session is marked invalid; see [Validation](validation.md)
for the full set of checks. The two-balloon example above will score, but its
`session_valid` flag will be `False` and its `session_warnings` list will explain
why.
```

## 3. Validate before scoring (optional)

{py:func}`~scoring.bart.score_bart` always runs validation internally and
records the outcome on `metrics.session_valid` / `metrics.session_warnings`. If
you want to inspect validity *before* scoring — for example to reject a session
at intake — call {py:func}`~scoring.bart.validate_bart_session` directly:

```python
from scoring.bart import validate_bart_session

report = validate_bart_session(events)
if not report["is_valid"]:
    print("Rejected:", report["warnings"])
```

## 4. Reading the result

A handful of fields carry most of the interpretive weight:

- **`ev_ratio_score`** (0–100) — how close the participant came to
  expected-value-optimal play, weighted across the three hazard levels. This is
  the engine's primary calibration measure.
- **`explosion_penalty`** (0–1) — how much the participant over-pumped, measured
  as excess burst rate beyond what optimal play would produce. Reported
  *separately* from calibration to avoid double-penalizing.
- **`rng_normalized_pumps`** (≥0) — mean stop as a fraction of the EV-optimal
  stop; `1.0` is optimal, `<1` conservative, `>1` over-pumping.
- **`behavioral_profile`** — a narrative classification (`risk_style`,
  `description`, `dominant_traits`); see [Behavioral profiles](scoring_engine.md#behavioral-profiles).

See [The scoring engine](scoring_engine.md) for how each number is computed.
