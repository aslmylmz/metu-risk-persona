# 23 — RunFlow error recovery: retry instead of re-enter ID

**UX fix · depends on: 16, 22**

## Context

When the [`RunFlow`](../../../app/src/run/RunFlow.tsx) `/preview` fetch fails (issue
22 is the root cause), the error phase shows a button labelled "Start task"
(`t.idContinue`) that sends the user back to the participant-ID entry step. This
creates a frustrating loop: the user re-enters their ID, the loading phase fires the
same failing fetch, and the same error appears — with no indication that something
is structurally wrong vs. a transient hiccup.

Even with issue 22 fixed, transient network issues (sidecar not ready, slow startup)
can still land the user here, so the error recovery should be useful on its own.

## Scope

- [ ] [app/src/run/RunFlow.tsx](../../../app/src/run/RunFlow.tsx): in the `"error"`
  phase, change the button from `onClick={() => setPhase("id")}` /
  `{t.idContinue}` to `onClick={() => setPhase("loading")}` / `"Retry"` so the
  user can retry the `/preview` fetch without re-entering their participant ID.
- [ ] [app/src/lib/i18n.ts](../../../app/src/lib/i18n.ts): add a `retry` string
  (`"Retry"` / `"Tekrar Dene"`) to the `TaskStrings` interface and both locale
  tables, so the button label is properly localized.

## Acceptance

- When the `/preview` fetch fails, the error screen shows a "Retry" button that
  retries loading (not re-entering the ID).
- The participant ID entered before the error is preserved across retries.
- `npm test`, `tsc --noEmit`, and `vite build` stay green.
