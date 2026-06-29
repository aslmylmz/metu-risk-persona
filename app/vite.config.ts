/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Vite SPA + Vitest. The task runs fully offline inside the Tauri webview; the
// Python sidecar becomes the scoring endpoint (its port handed over at runtime).
// The server block keeps `tauri dev` deterministic (fixed port it can point at).
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  // Expose TAURI_* alongside VITE_* to the client without leaking other env.
  envPrefix: ["VITE_", "TAURI_"],
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
