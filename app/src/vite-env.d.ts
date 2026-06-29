/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the scoring endpoint (Python sidecar). Defaults to localhost:8000. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
