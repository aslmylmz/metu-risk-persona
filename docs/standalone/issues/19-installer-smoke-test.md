# 19 — Installer smoke test + manual verification checklist

**Phase 4 · SPEC §12, §15 · depends on: 18**

## Context

Issue 18 produces an NSIS installer artifact. This issue verifies it works:

1. A **lightweight CI job** that silently installs and checks the file layout —
   enough to catch packaging regressions (missing sidecar, wrong paths) without
   attempting fragile GUI automation.
2. A **manual verification checklist** (`VERIFY-WINDOWS.md`) for pre-release QA on
   a real Windows VM or lab machine — the only way to confirm the full flow
   (SmartScreen bypass → window opens → session runs → data written).

## Scope

- [ ] Extend `.github/workflows/windows-release.yml` — new **Job 3
  `smoke-install`** (`windows-latest`, `needs: build-installer`):
  - Download the `bart-installer-windows` artifact.
  - Run the NSIS installer with `/S` (silent) + `/D=<temp dir>` (install dir).
  - Assert the app executable exists at the install dir.
  - Assert `bart-sidecar-x86_64-pc-windows-msvc.exe` exists alongside it (Tauri
    copies `externalBin` entries next to the main binary).
  - (Best-effort, `continue-on-error: true`) Launch the app, wait up to 30 s for
    the sidecar's `PORT=<n>` on stdout, hit `/healthz`; kill the process. CI runners
    may not have a display — if the launch fails, the file-layout checks above are
    enough.
- [ ] New `docs/standalone/VERIFY-WINDOWS.md` — manual pre-release checklist:
  1. Download the CI installer artifact.
  2. Install on a clean Windows 11 VM (or lab machine).
  3. Bypass SmartScreen (link to issue 20's `SMARTSCREEN.md`).
  4. Launch — confirm the sidecar starts (`bart-sidecar.exe` in Task Manager).
  5. Run one full session: consent → participant ID → pump balloons → collect →
     debrief.
  6. Verify output files are written under the sessions directory
     (`%LOCALAPPDATA%/com.metu.bart/sessions/`).
  7. Verify the `*_config.json` snapshot matches the study that was run.
  8. Uninstall via Windows Settings → Apps.

## Acceptance

- CI Job 3 passes (file-layout assertions green).
- `VERIFY-WINDOWS.md` documents the end-to-end manual procedure.
- A manual test following the checklist succeeds on a Windows 11 VM before the
  first tagged release.
