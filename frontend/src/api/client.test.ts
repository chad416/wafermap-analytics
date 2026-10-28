import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, query } from "./client";

afterEach(() => vi.restoreAllMocks());

describe("query", () => {
  it("drops empty values", () => {
    expect(query({ a: 1, b: "", c: undefined, d: null, e: false })).toBe("?a=1&e=false");
    expect(query({})).toBe("");
  });
});

describe("api", () => {
  it("surfaces FastAPI validation messages", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ detail: [{ msg: "Input should be less than or equal to 200" }] }),
        {
          status: 422,
        },
      ),
    );
    const err = await api.lots({ page_size: 500 }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(422);
    expect((err as ApiError).message).toContain("less than or equal to 200");
  });

  it("posts uploads as multipart form data with query options", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ label: "Center" })));
    await api.predictUpload(new File(["1,1"], "w.csv"), { model: "baseline", save: true });
    const [url, init] = spy.mock.calls[0]!;
    expect(url).toBe("/api/v1/predict/upload?model=baseline&save=true");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeInstanceOf(FormData);
  });
});
