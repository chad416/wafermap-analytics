/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the API runs on :8000; the dev server proxies /api so the browser
// sees a single origin (same as production, where nginx does the proxying).
const apiTarget = process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000";
const proxy = { "/api": apiTarget, "/docs": apiTarget, "/openapi.json": apiTarget };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
