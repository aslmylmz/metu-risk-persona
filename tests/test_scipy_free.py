"""The scoring package must run with no scipy installed.

scipy is the riskiest native dependency to PyInstaller-freeze (SPEC §3, §18), so
the frozen sidecar ships without it. This test simulates scipy being absent and
exercises every code path that used to call it — the lognormal hazard, the
learning-rate regression, and the risk-sensitivity correlation.
"""

from __future__ import annotations

import importlib
import sys

from scoring.schemas import EventPayload, GameEvent


def _session():
    """A varied multi-color, multi-trial session of collected balloons.

    Pumps vary within and across colors so the learning-rate regression and the
    risk-sensitivity correlation both run on non-degenerate data.
    """
    plan = {
        "purple": [9, 11, 10, 13, 12, 14, 11, 12, 13, 15],
        "teal": [4, 5, 6, 5, 7, 6, 5, 6, 4, 5],
        "orange": [1, 2, 3, 2, 2, 3, 1, 2, 2, 3],
    }
    events: list[GameEvent] = []
    t = 0.0
    for color, pump_seq in plan.items():
        for pumps in pump_seq:
            for _ in range(pumps):
                t += 300.0
                events.append(
                    GameEvent(timestamp=t, type="pump", payload=EventPayload(color=color))
                )
            t += 200.0
            events.append(
                GameEvent(timestamp=t, type="collect", payload=EventPayload(color=color))
            )
    return events


def test_engine_imports_and_scores_without_scipy(monkeypatch):
    # Make any `import scipy` (or `from scipy import ...`) raise ImportError.
    monkeypatch.setitem(sys.modules, "scipy", None)
    # Drop already-imported scoring modules so they re-import under the block.
    for name in list(sys.modules):
        if name == "scoring" or name.startswith("scoring."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    bart = importlib.import_module("scoring.bart")
    hazards = importlib.import_module("scoring.config.hazards")

    # Lognormal hazard (formerly scipy.stats.lognorm).
    vec = hazards.LognormalHazard(mu=3.0, sigma=0.5).hazard_vector(60)
    assert len(vec) == 60
    assert all(0.0 <= h <= 1.0 for h in vec)

    # score_bart drives the linregress + pearsonr paths.
    metrics = bart.score_bart(_session())
    assert metrics.total_balloons == 30
    assert metrics.total_collections == 30
