# 15 — Live EV-curve + optimum preview (hand-rolled SVG)

**Phase 3 · SPEC §7.3, §11, §17 · depends on: 13 (parallel to 14)**

## Context

The headline Study-Setup feature (SPEC §7.3): as the researcher edits family +
params, the screen calls [`/preview`](../../../app/sidecar/app.py) and plots, per
color, the hazard `h(k)`, the survival `S(s)`, the EV curve, and the **marked
numeric optimum** `s*`. Researchers *see* their design before running a participant.

Rendering is **hand-rolled inline SVG** (decision this session: no charting
dependency — keeps the fully-offline/JOSS posture and the bundle minimal;
[package.json](../../../app/package.json) stays dependency-light). `/preview` already
returns everything needed straight from `TaskConfig.curves`
([models.py](../../../app/sidecar/models.py) `CurvePreview`:
`hazard`/`survival`/`ev`/`optimum`/`optimal_ev`), so this issue is purely a
client-side transform + render. The numeric optimum readout is the acceptance hook
("see the optimum update").

## Scope

- [ ] [api.ts](../../../app/src/lib/api.ts): add `preview(config):
  Promise<PreviewResponse>` hitting `/preview`, typed to the sidecar's `CurvePreview`
  shape.
- [ ] New `app/src/setup/evGeometry.ts` — a **pure** transform: given a
  `CurvePreview` (+ plot width/height/padding), produce SVG polyline point strings
  for hazard/survival/EV (scaled into the plot box) and the optimum marker
  coordinates. No DOM/React — fully unit-testable.
- [ ] New `app/src/setup/EvPreview.tsx` — renders the SVG per color from
  `evGeometry`, shows the numeric `optimum` / `optimal_ev`; refetches `/preview` on
  config change, **debounced** (~150–250 ms), with a stale-response guard (ignore
  out-of-order responses) and a non-blocking error state (an invalid intermediate
  config just leaves the last-good plot in place).
- [ ] Tests: `app/src/setup/evGeometry.test.ts` — vectors map to the expected scaled
  coordinates; the optimum marker lands at the right index; flat/degenerate vectors
  degrade gracefully. The fetch/debounce wiring is build + smoke verified.

## Acceptance

- Editing the family or any parameter updates the plotted hazard/survival/EV curves
  and the **marked + numeric** optimum **live**, per color, with no remote requests
  (only the local sidecar).
- An invalid intermediate config does not crash the preview (last-good plot stays;
  error noted).
- `npm test`, `tsc --noEmit`, and `vite build` stay green.
