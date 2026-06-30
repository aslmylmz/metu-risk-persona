# 20 — SmartScreen bypass documentation

**Phase 4 · SPEC §14, §18 · depends on: — (parallel, docs only)**

## Context

The installer from issue 18 is **unsigned** (SPEC §20 decision: ship unsigned for
now). On Windows, Microsoft Defender SmartScreen blocks executables that lack a
code-signing certificate or a reputation history: the user sees "Windows protected
your PC." For a researcher in a lab, this is a one-time obstacle — but without
documentation it looks alarming.

Additionally, the frozen sidecar (`bart-sidecar.exe`) is a PyInstaller one-file
binary; some AV engines flag these as "generic packer" false positives.

This issue creates a self-contained reference doc that the quickstart (issue 21),
`VERIFY-WINDOWS.md` (issue 19), and the GitHub Releases page can all link to.

## Scope

- [ ] New `docs/standalone/SMARTSCREEN.md`:
  - **What is SmartScreen** — non-technical, one-paragraph explanation.
  - **Why the warning appears** — the app is unsigned research software; this is
    normal and expected.
  - **How to bypass** — step-by-step:
    1. Click "More info" on the blue warning dialog.
    2. Click "Run anyway."
  - **IT administrator notes** — for managed lab machines:
    - Group Policy: how to whitelist by publisher or file hash.
    - Intune: `AllowSmartScreen` and app-control policies.
  - **Antivirus false positives** — PyInstaller-frozen binaries can trigger generic
    heuristic detections. How to add an exclusion in Windows Defender and common
    third-party AV.
  - **Future: code signing** — brief note that an OV code-signing certificate would
    remove the warning; link to SPEC §14/§20 for the decision record.

## Acceptance

- A non-technical researcher can follow the doc to bypass SmartScreen on first
  launch.
- IT admin guidance is specific enough to configure an exemption in Group Policy or
  Intune.
- No code changes — docs only.
