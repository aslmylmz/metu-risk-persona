# 13 — Config-UX app shell: mode routing + active-config store + TS config types

**Phase 3 · SPEC §6, §11, §17 · depends on: Phase 2**

## Context

Phase 3 turns the SPA into a two-mode instrument: **Study Setup** (researcher) and
**Run** (participant). Today [App.tsx](../../../app/src/App.tsx) renders
[BartGame.tsx](../../../app/src/BartGame.tsx) directly with a hardcoded study. This
issue stands up the foundation the other three Phase-3 slices hang off — nothing
user-visible beyond a mode switch — so 14/15/16 stay focused:

- a top-level **mode** (`"setup" | "run"`) held in `App` state (plain React state;
  no router dependency is added);
- a single in-memory **active `TaskConfig`** seeded from the validated default study;
- a hand-written **TypeScript mirror** of the config schema (decision this session:
  hand-written TS types, with the sidecar's `/validate-config` kept as the authority
  on save — the pydantic models in [scoring/config/](../../../scoring/config/) stay
  the source of truth, the TS types are a typing convenience).

Field names must match the pydantic JSON exactly (snake_case: `schema_version`,
`reward_per_pump`, `max_pumps`, `display_hex`, `output_dir`, …) so a config object
serializes straight to `study.json` and is accepted verbatim by `/validate-config`
and `/preview`.

## Scope

- [ ] New `app/src/lib/config.ts`: TS types mirroring
  [scoring/config](../../../scoring/config/) — a `HazardSpec` discriminated union over
  the 11 families in [hazards.py](../../../scoring/config/hazards.py) (each family's
  `family` literal + its params), `ColorProfile`, and `TaskConfig`. Export a
  `DEFAULT_STUDY: TaskConfig` mirroring `DEFAULT_TASK_CONFIG`
  ([task_config.py](../../../scoring/config/task_config.py)): 128/32/8 linear,
  `reward_per_pump=0.25`, 10 trials each, `language="en"`.
- [ ] [App.tsx](../../../app/src/App.tsx): hold `mode` (`"setup" | "run"`) and the
  active `TaskConfig` (initialized to `DEFAULT_STUDY`); render a Study-Setup
  placeholder vs the Run wrapper; keep the existing **F11 fullscreen** handler.
- [ ] Keep `candidateId` plumbing intact — `BartGame` still takes it; Run mode
  supplies a real value in issue 16. The placeholder may keep `"anonymous"`.
- [ ] Tests: `app/src/lib/config.test.ts` — `DEFAULT_STUDY` has the expected shape
  (three colors, caps `128/32/8`, 10 trials, reward `0.25`, `linear` family,
  `language="en"`); a guard asserting each of the 11 family `family` literals is
  representable in `HazardSpec` (catches drift against `hazards.py`).

## Acceptance

- The app boots into **Study Setup** mode with the active config defaulting to the
  linear 128/32/8 study; a control toggles to **Run** mode and back.
- A `study.json` serialized from `DEFAULT_STUDY` is accepted by `/validate-config`
  (snake_case field parity) — spot-checked or asserted via the shape guard.
- `npm test`, `tsc --noEmit`, and `vite build` stay green.
