import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/health": "http://127.0.0.1:8020",
      "/teams": "http://127.0.0.1:8020",
      "/games": "http://127.0.0.1:8020",
      "/hub": "http://127.0.0.1:8020",
      "/manifest": "http://127.0.0.1:8020",
      "/retrain": "http://127.0.0.1:8020",
      "/refresh-odds": "http://127.0.0.1:8020",
      "/calibration": "http://127.0.0.1:8020",
      "/season": "http://127.0.0.1:8020",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
    // Tip-off times render in the viewer's zone; pin one so tests are stable.
    env: { TZ: "America/Chicago" },
  },
});