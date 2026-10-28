import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PredictResponse } from "../api/types";
import { ClassifyPage } from "./ClassifyPage";

afterEach(() => vi.restoreAllMocks());

const response: PredictResponse = {
  model_name: "cnn",
  model_version: "abc123",
  label: "Edge-Ring",
  confidence: 0.97,
  probabilities: { "Edge-Ring": 0.97, none: 0.02, "Edge-Loc": 0.01 },
  stats: { rows: 3, cols: 3, n_die: 9, n_fail: 8, yield_pct: 11.1 },
  latency_ms: 3.2,
  wafer_id: null,
  grid: [
    [2, 2, 2],
    [2, 1, 2],
    [2, 2, 2],
  ],
};

describe("ClassifyPage", () => {
  it("uploads a file and shows the live classification", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify(response)));
    render(
      <MemoryRouter>
        <ClassifyPage />
      </MemoryRouter>,
    );
    await userEvent.upload(
      screen.getByLabelText("wafer map file"),
      new File(["2,2,2\n2,1,2\n2,2,2"], "ring.csv"),
    );

    expect(await screen.findByText(/Prediction:/)).toBeInTheDocument();
    expect(screen.getAllByText("Edge-Ring").length).toBeGreaterThan(0);
    expect(screen.getByText("97.0 %")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/predict/upload?model=cnn&save=false",
      expect.anything(),
    );
  });

  it("shows server errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "wafer map has 4 dies, need at least 100" }), {
        status: 422,
      }),
    );
    render(
      <MemoryRouter>
        <ClassifyPage />
      </MemoryRouter>,
    );
    await userEvent.upload(
      screen.getByLabelText("wafer map file"),
      new File(["1,1\n1,1"], "tiny.csv"),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("need at least 100");
  });
});
