# No outbound network calls — total offline isolation

The instrument makes zero outbound network connections. Neither the Tauri webview nor the Python sidecar may contact any remote host, ever. This is a hard constraint driven by clinical IRB approvals and university data-privacy requirements.

## Enforcement at three layers

1. **Webview CSP** — `tauri.conf.json` sets a strict Content-Security-Policy restricting `connect-src`, `script-src`, and all fetch-capable directives to `'self'` and the loopback sidecar origin (`http://127.0.0.1:{port}`). No CDN fonts, no remote analytics scripts, no external images.

2. **Sidecar binds loopback only** — the Uvicorn process binds to `127.0.0.1`, never `0.0.0.0`. It is unreachable from other machines on the network.

3. **No outbound calls in Python** — the sidecar's dependency tree must not import `requests`, `urllib.request`, `httpx`, or any library that initiates outbound connections. No update checks, no crash reporting, no telemetry. The `test_scipy_free.py` pattern (monkeypatching imports to raise) can be extended to enforce this in CI.

## Why record this

Most desktop applications check for updates, report crashes, or load remote assets. A future contributor will naturally want to add one of these. This ADR exists to make the constraint visible: the "no network" guarantee is deliberate, not an oversight, and relaxing it requires re-evaluating IRB and data-privacy commitments before writing code.

## Considered alternative

Allow opt-in update checks with a user-visible toggle. Rejected because: (a) clinical IRB protocols typically prohibit any network capability in the approved software, not just default-off features, and (b) the presence of networking code — even disabled — complicates the security audit that locked-down institutions require before installation.
