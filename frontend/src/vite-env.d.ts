/// <reference types="vite/client" />

declare module "vite/client" {
  // Ambient fallback
}

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  [key: string]: any;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
