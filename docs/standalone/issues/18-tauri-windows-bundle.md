# 18 — Tauri Windows bundle + NSIS per-user installer

**Phase 4 · SPEC §12, §14 · depends on: 17**

## Context

Issue 17 proves the full sidecar freezes and runs on Windows. This issue wires the
**Tauri build** on Windows CI, producing a per-user NSIS installer (`.exe`) that
bundles:

- the Vite SPA (compiled frontend),
- the Rust shell,
- the frozen `bart-sidecar-x86_64-pc-windows-msvc.exe` via Tauri's `externalBin`,
- the **WebView2 offline bootstrapper** — embedded so the installer never reaches
  out to the network, honoring the 100 %-offline-secure mandate (SPEC §4).

**Decisions locked for this issue:**

- **NSIS** (not WiX). Per-user install, no admin rights required.
- **Unsigned** — SmartScreen bypass is documented in issue 20.
- **Tag-triggered** — the workflow fires on `v*` tags only (plus `workflow_dispatch`
  for manual runs). Dev pushes do **not** trigger a Tauri build; the lighter
  `sidecar-windows.yml` (issue 17) covers push-triggered sidecar verification.

## Scope

- [ ] New `.github/workflows/windows-release.yml` — tag-triggered, two jobs:
  - **Job 1 `freeze-sidecar`** (`windows-latest`):
    - Checkout, `actions/setup-python@v5` (3.12).
    - `pip install -e ".[sidecar,build,dev]"`.
    - `pyinstaller app/sidecar/sidecar.spec`.
    - Rename → `bart-sidecar-x86_64-pc-windows-msvc.exe`.
    - Upload as inter-job artifact `sidecar-windows-exe`.
  - **Job 2 `build-installer`** (`windows-latest`, `needs: freeze-sidecar`):
    - Checkout, download `sidecar-windows-exe` into `app/src-tauri/binaries/`.
    - `actions/setup-node@v4` (20), install Rust (stable, `dtolnay/rust-toolchain`).
    - `npm ci --prefix app`.
    - `npm run --prefix app tauri build`.
    - Upload the NSIS installer from
      `app/src-tauri/target/release/bundle/nsis/*.exe` as artifact
      `bart-installer-windows`.
- [ ] [tauri.conf.json](../../../app/src-tauri/tauri.conf.json):
  - `bundle.targets` → `["nsis"]` (was `"all"`).
  - Add `bundle.windows.webviewInstallMode`:
    `{ "type": "offlineInstaller", "path": "./WebView2RuntimeInstaller.exe" }` —
    the CI step downloads the offline bootstrapper from Microsoft once and places it
    so Tauri embeds it (no runtime network call).
  - NSIS config: `installMode: "currentUser"` (per-user, no admin).
- [ ] [lib.rs](../../../app/src-tauri/src/lib.rs) — release `sidecar_command()`
  (line 49–56): append `std::env::consts::EXE_SUFFIX` to the binary path so the
  shell finds `bart-sidecar.exe` on Windows.

## Acceptance

- A `v*` tag push produces a downloadable NSIS installer artifact in GitHub Actions.
- The installer embeds the WebView2 offline bootstrapper (no network during install).
- `tauri dev` on macOS still works (the `#[cfg(debug_assertions)]` path is
  unchanged).
- `npm test`, `tsc --noEmit`, `vite build`, and `pytest` stay green locally.

> The installer is verified by issue 19 (smoke install) and manual testing on a
> Windows VM (see `docs/standalone/VERIFY-WINDOWS.md`).
