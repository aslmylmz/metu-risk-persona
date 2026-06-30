# 22 — Sidecar CORS middleware + `localhost` → `127.0.0.1` fix

**Bugfix · depends on: 08, 15, 16**

## Context

Both the **EV preview** and the **Run task** fail with `Failed to fetch` on Windows
(Tauri WebView2) and in any plain browser during `npm run dev`. The root cause is
that the sidecar ([app.py](../../../app/sidecar/app.py)) has **no CORS middleware**:

1. The webview / browser origin (`http://localhost:5173` in dev, or a Tauri custom
   origin) differs from the sidecar's origin (`http://127.0.0.1:<port>`). Different
   origins → the browser sends a CORS preflight (`OPTIONS`).
2. The sidecar returns no `Access-Control-Allow-Origin` header → browser refuses the
   response.
3. `fetch()` throws `TypeError: Failed to fetch` →
   [`EvPreview.tsx`](../../../app/src/setup/EvPreview.tsx) shows "Preview not updated:
   Failed to fetch" and [`RunFlow.tsx`](../../../app/src/run/RunFlow.tsx) fails at the
   `"loading"` phase with the same error, blocking the task from starting.

This is invisible on macOS Tauri (WebKit relaxes CORS for loopback) but breaks on
Windows WebView2 (Chromium-based, stricter CORS) and any regular browser.

A secondary issue: [`api.ts`](../../../app/src/lib/api.ts) `DEFAULT_API_URL` uses
`http://localhost:8000`. On Windows, `localhost` can resolve to `::1` (IPv6) while the
sidecar only binds IPv4 `127.0.0.1`, causing a connection failure even without the
CORS issue.

## Scope

- [ ] [app/sidecar/app.py](../../../app/sidecar/app.py): add FastAPI
  `CORSMiddleware` with `allow_origins=["*"]`, `allow_methods=["*"]`,
  `allow_headers=["*"]`. Safe because the sidecar is strictly loopback-only (bound to
  `127.0.0.1`, not network-accessible).
- [ ] [app/src/lib/api.ts](../../../app/src/lib/api.ts): change `DEFAULT_API_URL`
  from `http://localhost:8000` to `http://127.0.0.1:8000` to avoid IPv6 resolution
  mismatches on Windows.
- [ ] [tests/test_sidecar.py](../../../tests/test_sidecar.py): add a test that sends
  an `OPTIONS` preflight to `/preview` and verifies `Access-Control-Allow-Origin` is
  present in the response headers.

## Acceptance

- Opening `http://localhost:5173` in a regular browser (with the sidecar running)
  shows the EV preview curves — no "Failed to fetch".
- The Run flow (consent → ID → loading → task) completes without errors.
- Existing tests stay green: `npm test`, `tsc --noEmit`, `vite build`, `pytest`.
