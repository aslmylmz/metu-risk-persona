"""
Game Event Schemas — Re-export + Game-Specific Validators

Usage:
    from schemas.game_events import (
        GameSession, GameEvent, BARTMetrics, NBackMetrics,
        validate_bart_events, validate_nback_events,
    )
"""

from __future__ import annotations

from schemas import (
    AssessmentResponse,
    BARTMetrics,
    ColorMetrics,
    EventPayload,
    GameEvent,
    GameSession,
    GameType,
    NBackMetrics,
    NormalizedScore,
    StroopMetrics,
)

__all__ = [
    "AssessmentResponse",
    "BARTMetrics",
    "ColorMetrics",
    "EventPayload",
    "GameEvent",
    "GameSession",
    "GameType",
    "NBackMetrics",
    "NormalizedScore",
    "StroopMetrics",
    "validate_bart_events",
    "validate_nback_events",
    "validate_stroop_events",
]


# ── BART Event Type Registry ────────────────────────────────────────────────

BART_VALID_EVENT_TYPES = frozenset({"pump", "collect", "explode"})


def validate_bart_events(events: list[GameEvent]) -> list[GameEvent]:
    """
    BART-specific event validation.

    Rules:
    1. Only 'pump', 'collect', and 'explode' event types are allowed.
    2. Every balloon must end with exactly one terminal event
       ('collect' or 'explode') — never both, never neither.
    3. Each balloon must have at least one 'pump' before a terminal event.
    """
    for i, event in enumerate(events):
        if event.type not in BART_VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid BART event type '{event.type}' at index {i}. "
                f"Allowed: {sorted(BART_VALID_EVENT_TYPES)}"
            )

    current_balloon_pumps = 0
    balloon_index = 1

    for i, event in enumerate(events):
        if event.type == "pump":
            current_balloon_pumps += 1
        elif event.type in ("collect", "explode"):
            if current_balloon_pumps == 0:
                raise ValueError(
                    f"Balloon {balloon_index} has a terminal event "
                    f"('{event.type}') at index {i} with zero pumps."
                )
            current_balloon_pumps = 0
            balloon_index += 1

    return events


# ── N-Back Event Type Registry ──────────────────────────────────────────────

NBACK_VALID_EVENT_TYPES = frozenset({"stimulus", "response"})


def validate_nback_events(events: list[GameEvent]) -> list[GameEvent]:
    """
    N-Back-specific event validation.

    Rules:
    1. Only 'stimulus' and 'response' event types are allowed.
    2. 'stimulus' events must have a non-empty `stimulus` field in payload.
    3. 'response' events must have a `response` field of 'match' or 'no_response'.
    4. Every 'response' must follow a 'stimulus' (no orphan responses).
    5. There must be at least one 'stimulus' event.

    Parameters
    ----------
    events : list[GameEvent]
        Chronologically ordered events (generic validation already passed).

    Returns
    -------
    list[GameEvent]
        The validated events (unchanged).

    Raises
    ------
    ValueError
        If any N-Back-specific invariant is violated.
    """
    # Rule 1: Only valid N-Back event types
    for i, event in enumerate(events):
        if event.type not in NBACK_VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid N-Back event type '{event.type}' at index {i}. "
                f"Allowed: {sorted(NBACK_VALID_EVENT_TYPES)}"
            )

    # Rule 5: At least one stimulus
    stimulus_count = sum(1 for e in events if e.type == "stimulus")
    if stimulus_count == 0:
        raise ValueError("N-Back event log must contain at least one 'stimulus' event.")

    # Rule 2: Stimulus events must have the stimulus character
    for i, event in enumerate(events):
        if event.type == "stimulus":
            if not event.payload.stimulus:
                raise ValueError(
                    f"Stimulus event at index {i} is missing the 'stimulus' field."
                )

    # Rule 3 & 4: Response events validation
    last_was_stimulus = False
    for i, event in enumerate(events):
        if event.type == "stimulus":
            last_was_stimulus = True
        elif event.type == "response":
            if not last_was_stimulus:
                raise ValueError(
                    f"Response event at index {i} has no preceding stimulus."
                )
            # Response is simply the user pressing the key — payload.response
            # can be 'match' or absent (indicating they pressed the key)

    return events


# ── Stroop Event Type Registry ─────────────────────────────────────────────

STROOP_VALID_EVENT_TYPES = frozenset({"stimulus", "response"})


def validate_stroop_events(events: list[GameEvent]) -> list[GameEvent]:
    """
    Stroop/Go-No-Go-specific event validation.

    Rules:
    1. Only 'stimulus' and 'response' event types are allowed.
    2. 'stimulus' events must have word, ink_color, and is_go_trial fields.
    3. 'response' events must have a response field.
    4. Every 'response' must follow a 'stimulus' (no orphan responses).
    5. There must be at least one 'stimulus' event.

    Parameters
    ----------
    events : list[GameEvent]
        Chronologically ordered events (generic validation already passed).

    Returns
    -------
    list[GameEvent]
        The validated events (unchanged).

    Raises
    ------
    ValueError
        If any Stroop-specific invariant is violated.
    """
    # Rule 1: Only valid Stroop event types
    for i, event in enumerate(events):
        if event.type not in STROOP_VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid Stroop event type '{event.type}' at index {i}. "
                f"Allowed: {sorted(STROOP_VALID_EVENT_TYPES)}"
            )

    # Rule 5: At least one stimulus
    stimulus_count = sum(1 for e in events if e.type == "stimulus")
    if stimulus_count == 0:
        raise ValueError("Stroop event log must contain at least one 'stimulus' event.")

    # Rule 2: Stimulus events must have required fields
    for i, event in enumerate(events):
        if event.type == "stimulus":
            extra = event.payload.model_extra or {}
            word = extra.get("word")
            ink_color = extra.get("ink_color")
            is_go_trial = extra.get("is_go_trial")

            if not word:
                raise ValueError(
                    f"Stimulus event at index {i} is missing the 'word' field."
                )
            if not ink_color:
                raise ValueError(
                    f"Stimulus event at index {i} is missing the 'ink_color' field."
                )
            if is_go_trial is None:
                raise ValueError(
                    f"Stimulus event at index {i} is missing the 'is_go_trial' field."
                )

    # Rule 3 & 4: Response events validation
    last_was_stimulus = False
    for i, event in enumerate(events):
        if event.type == "stimulus":
            last_was_stimulus = True
        elif event.type == "response":
            if not last_was_stimulus:
                raise ValueError(
                    f"Response event at index {i} has no preceding stimulus."
                )
            # Response is the user's color selection
            extra = event.payload.model_extra or {}
            response = event.payload.response or extra.get("response")
            if not response:
                raise ValueError(
                    f"Response event at index {i} is missing the 'response' field."
                )

    return events
