/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Vite SPA + Vitest. The task runs fully offline inside the Tauri webview later;
// the Python sidecar becomes the scoring endpoint (configurable via VITE_API_URL).
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
