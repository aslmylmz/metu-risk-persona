# 21 — Researcher quickstart guide (Sphinx / MyST → Read the Docs)

**Phase 4 · SPEC §12, §15, §17 · depends on: 18, 19, 20**

## Context

The end user of this instrument is a researcher with no coding background. The
quickstart is the **first document they read**: it must get them from "I downloaded
a file" to "I have participant data" without touching a terminal.

The project already has a Sphinx + MyST documentation setup
([docs/](../../../docs/), builds to Read the Docs via
[.readthedocs.yaml](../../../.readthedocs.yaml)). The quickstart lives there so it
is searchable, versioned, and compiled alongside the existing docs.

## Scope

- [ ] New `docs/standalone/quickstart.md` (MyST Markdown, Sphinx-compilable):
  1. **Download** — where to get the latest installer: GitHub Releases page for
     tagged versions; CI artifact for development builds.
  2. **Install** — run the NSIS installer; per-user, no admin needed. Link to
     [SMARTSCREEN.md](../../../docs/standalone/SMARTSCREEN.md) for the unsigned-app
     bypass.
  3. **Create a study** — open the app → Study Setup mode. Walk through: pick a
     hazard family, set parameters per color, observe the live EV-curve preview,
     save the study as `study.json`.
  4. **Run participants** — switch to Run mode. Consent screen → participant ID →
     balloon task → debrief. One participant per run; relaunch for the next.
  5. **Collect data** — where the output files land
     (`%LOCALAPPDATA%/com.metu.bart/sessions/` by default, or the study's configured
     `output_dir`). What each file contains:
     - `*_events.jsonl` — raw pump-level event log.
     - `*_metrics.json` — computed behavioral metrics.
     - `*_config.json` — snapshot of the study config used.
  6. **Customize** — how to load a different `study.json`; brief explanation of key
     parameters (hazard families with a link to the family table in SPEC §7.2,
     `max_pumps`, `reward_per_pump`, language, RNG seed).
  7. **Troubleshooting** — SmartScreen (→ `SMARTSCREEN.md`); WebView2 on older
     Windows 10 (the installer embeds the bootstrapper, but note the requirement);
     antivirus false positives.
- [ ] Wire `docs/standalone/quickstart.md` into the Sphinx toctree (update the
  appropriate `index.rst` or `index.md`).
- [ ] Update [docs/standalone/issues/README.md](./README.md): add the Phase 4 issue
  table (17–21) and acceptance criteria following the existing format.

## Acceptance

- `make -C docs html` (or the Read the Docs build) includes the quickstart page
  with no Sphinx warnings.
- A researcher with no coding background can follow the guide end-to-end: download →
  install → configure → run → collect data.
- The quickstart links to `SMARTSCREEN.md` and the SPEC §7.2 family table where
  appropriate.
- The issues README reflects Phase 4 (17–21).
