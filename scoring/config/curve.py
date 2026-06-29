"""Survival / EV curve and the numeric optimum derived from a hazard vector.

This is the artifact shared by the task and the scoring engine: the task bursts
from ``hazard`` while scoring references ``optimum``/``ev``. The optimum is found
by a full numeric scan (never a closed form), so non-monotone families work too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BalloonCurve:
    """Precomputed curves for one color, indexed by stop ``s``.

    ``hazard[i]`` is h(i+1); ``survival[s]`` is S(s) with S(0)=1; ``ev[s]`` is
    EV(s) = reward * s * S(s). ``optimum`` is argmax_{1<=s<=N} EV(s) (smallest s
    on ties), ``optimal_ev`` is EV at that stop.
    """

    hazard: tuple[float, ...]
    survival: tuple[float, ...]
    ev: tuple[float, ...]
    optimum: int
    optimal_ev: float


def balloon_curve(hazard: list[float], reward_per_pump: float) -> BalloonCurve:
    """Build the survival/EV curve for a hazard vector and find the numeric optimum.

    ``reward_per_pump`` scales EV uniformly and therefore does not move the
    optimum; it is applied so ``optimal_ev`` is in the configured currency units.
    """
    if any(not math.isfinite(h) for h in hazard):
        raise ValueError("hazard vector contains non-finite values (NaN/inf)")
    n = len(hazard)
    survival = [1.0]
    for h in hazard:
        survival.append(survival[-1] * (1.0 - h))
    ev = [reward_per_pump * s * survival[s] for s in range(n + 1)]
    # max() returns the first maximal element; scanning ascending s -> smallest
    # s on ties, matching the established engine's strict-greater scan.
    optimum = max(range(1, n + 1), key=lambda s: ev[s])
    return BalloonCurve(
        hazard=tuple(hazard),
        survival=tuple(survival),
        ev=tuple(ev),
        optimum=optimum,
        optimal_ev=ev[optimum],
    )
