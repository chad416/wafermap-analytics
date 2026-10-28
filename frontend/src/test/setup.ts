import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());

// jsdom has no canvas implementation; WaferMap already handles a null 2-D context.
Object.defineProperty(HTMLCanvasElement.prototype, "getContext", { value: () => null });
