# 16 — Run mode: config-driven task + consent/ID/debrief + per-study scoring

**Phase 3 · SPEC §7.2, §8, §11, §13, §17 · depends on: 13, 14**

## Context

The closing Phase-3 slice — it satisfies the SPEC §17 acceptance. Three coupled
changes turn the hardcoded demo into the configured study:

1. **Make the task config-driven.** [BartGame.tsx](../../../app/src/BartGame.tsx)
   currently hardcodes `RISK_PROFILES` (8/32/128), `TOTAL_BALLOONS = 30` (10 each),
   `$0.25`/pump, **unseeded** `Math.random()` bursting with `k / maxPumps` (linear
   only), and Turkish-only strings. Drive it instead from the active `TaskConfig`:
   per-color `max_pumps`/`trials`, the **precomputed hazard vector** (from `/preview`
   `curves[color].hazard`, so client bursting and the scorer share the exact
   landscape — SPEC §5), `reward_per_pump`, `seed` (reproducible bursts — SPEC §7.2),
   and `language`.
2. **Wrap it in the Run flow:** consent → participant-ID entry → task → debrief.
   **Reuse the existing results screen as the debrief** (decision this session;
   engagement-only, no LLM).
3. **Per-study scoring/persistence:** thread the active config into `/score` and
   `/write-output` so the persisted `*_metrics.json` + `*_config.json` reflect the
   actual study. `score_bart(events, config=...)` already accepts a config
   ([scoring/bart.py](../../../scoring/bart.py)), so this is wiring, not an engine
   change.

## Scope

- [ ] New `app/src/run/RunFlow.tsx`: consent screen → participant-ID entry →
  `BartGame` (driven by the active config) → debrief; supplies a real `candidateId`;
  honors `config.language` for its own strings.
- [ ] New `app/src/lib/i18n.ts` (or `run/strings.ts`): extract the participant-facing
  **tr/en string table** out of `BartGame.tsx` (decision this session: extract tr/en)
  — pump/collect/explode/feedback + consent/debrief labels; select by
  `config.language`.
- [ ] New `app/src/run/sequence.ts` (pure): build the **seeded**, shuffled balloon
  sequence from a config (`trials` balloons per color); expose a small seeded PRNG
  (e.g. mulberry32 from `config.seed`) and the per-pump burst test `u <
  hazard[k-1]`. Fully unit-testable; same seed ⇒ same sequence.
- [ ] [BartGame.tsx](../../../app/src/BartGame.tsx): accept the active `TaskConfig`
  (or a derived per-color burst table) as a prop; build the sequence from
  `app/src/run/sequence.ts` (replace `generateSessionConfig`/`TOTAL_BALLOONS`); burst
  against the config hazard vector with the seeded PRNG (replace
  `Math.random()`/`k/maxPumps`); take `reward_per_pump`, color hexes/labels from
  `config.colors`, and strings from the i18n table.
- [ ] [api.ts](../../../app/src/lib/api.ts): `submitSession`/`persistSession` send the
  active config — `{ session, config }` — to `/score` and `/write-output`.
- [ ] [app/sidecar/models.py](../../../app/sidecar/models.py): add
  `ScoreRequest { session: GameSession; config: TaskConfig | None = None }` (mirrors
  `WriteOutputRequest`). [app/sidecar/app.py](../../../app/sidecar/app.py): `/score`
  and `/write-output` pass `req.config or DEFAULT_TASK_CONFIG` into `score_bart`.
- [ ] Tests: `app/src/run/sequence.test.ts` — same seed ⇒ identical sequence + burst
  draws; trials/colors honored; reward applied. `tests/test_sidecar.py` — `/score`
  with a non-default `config` scores against that config's optima (result differs
  from the default), and `/write-output` snapshots the supplied config.

## Acceptance

- **Phase 3 rollup (SPEC §17):** a non-coder changes the hazard family + params in
  Study Setup, **sees the optimum update**, **saves a study**, switches to Run, and
  **runs it** — the task uses the configured colors / `max_pumps` / `trials` / hazard
  / reward / language, and a seeded run **replays identically**.
- The persisted `*_events.jsonl` / `*_metrics.json` / `*_config.json` reflect **that**
  study (metrics scored against its optima; the config snapshot is the run's config).
- `npm test`, `tsc --noEmit`, `vite build`, and `pytest` all stay green.
