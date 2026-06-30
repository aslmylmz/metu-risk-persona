# 17 — Windows CI: full sidecar freeze + smoke test

**Phase 4 · SPEC §9, §12, §18 · depends on: Phase 3**

## Context

Phase 1 (issue 07) proved `hello-score` freezes and runs on Windows. Phase 2
(issue 09) froze the **full** sidecar locally on macOS. This issue closes the
loop: freeze the real `bart-sidecar` on a `windows-latest` CI runner and smoke-test
it, producing the binary that Tauri will bundle in issue 18.

The existing [sidecar-windows.yml](../../../.github/workflows/sidecar-windows.yml)
already freezes `hello-score` and runs `pytest` on Windows. This issue extends it
to also freeze `bart-sidecar` via
[sidecar.spec](../../../app/sidecar/sidecar.spec) and verify `/healthz` + `/score`
on the frozen `.exe`.

Tauri v2's `externalBin` expects a platform-triple suffix. The frozen binary must
be renamed to `bart-sidecar-x86_64-pc-windows-msvc.exe` so the Tauri bundle
(issue 18) resolves it.

## Scope

- [ ] Extend
  [sidecar-windows.yml](../../../.github/workflows/sidecar-windows.yml):
  - New step after the existing hello-score freeze:
    `pyinstaller app/sidecar/sidecar.spec --distpath dist --workpath build --noconfirm`.
  - New smoke step (PowerShell): spawn `dist/bart-sidecar.exe` in the background,
    capture the `PORT=<n>` line, `GET /healthz` and assert `200 + status:ok`,
    `POST /score` with a sample session and assert non-empty `raw_metrics`, kill the
    process.
  - Rename `dist/bart-sidecar.exe` →
    `dist/bart-sidecar-x86_64-pc-windows-msvc.exe`.
  - `actions/upload-artifact@v4`: upload
    `dist/bart-sidecar-x86_64-pc-windows-msvc.exe` as artifact
    `sidecar-windows-exe`.
- [ ] Keep the existing `hello-score` steps intact (regression guard).

## Acceptance

- The workflow is **green** on `windows-latest`.
- A `bart-sidecar-x86_64-pc-windows-msvc.exe` CI artifact is uploaded.
- The frozen sidecar answers `/healthz` with `status: ok` and scores a sample
  session via `/score` (smoke test).
- `pytest` and the existing `hello-score` assertions still pass.

> Verified in CI only — macOS dev machines cannot produce/run the Windows binary
> (SPEC §12).
