import { useEffect, useState } from "react";

import BartGame from "./BartGame";
import { DEFAULT_STUDY, type TaskConfig } from "./lib/config";
import { toggleFullscreen } from "./lib/desktop";

type Mode = "setup" | "run";

// Phase 3 app shell: two modes in the SPA — Study Setup (researcher) and Run
// (participant). This issue stands up the mode switch and the in-memory active
// TaskConfig (seeded from the validated default study). The real Study-Setup form
// (issue 14), live EV preview (15), and config-driven Run flow (16) replace the
// placeholders below; the active config is the store they read/mutate. F11 toggles
// kiosk/fullscreen (SPEC §10); outside Tauri it is a harmless no-op.
export function App() {
  const [mode, setMode] = useState<Mode>("setup");
  const [config] = useState<TaskConfig>(DEFAULT_STUDY);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "F11") {
        e.preventDefault();
        void toggleFullscreen().catch((err) =>
          console.error("Fullscreen toggle failed:", err),
        );
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  if (mode === "run") {
    return (
      <div>
        <button type="button" onClick={() => setMode("setup")}>
          ← Back to setup
        </button>
        <BartGame candidateId="anonymous" />
      </div>
    );
  }

  // Study Setup placeholder — issue 14 replaces this with the real form + issue 15's
  // live EV preview. The active study defaults to the validated 128/32/8 linear config.
  return (
    <div>
      <h1>Study Setup</h1>
      <p>Active study: {config.title}</p>
      <button type="button" onClick={() => setMode("run")}>
        Start run →
      </button>
    </div>
  );
}
