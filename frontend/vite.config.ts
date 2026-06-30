import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend target for the dev proxy. Inside Docker Compose this is set to
// http://api:8000 (the API service); on a bare host it defaults to
// http://localhost:8000.
const PROXY_TARGET = process.env.VITE_PROXY_TARGET || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    proxy: {
      "/v1": {
        target: PROXY_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
