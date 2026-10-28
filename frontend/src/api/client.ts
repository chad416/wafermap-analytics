import type {
  LabelSource,
  Lot,
  LotDetail,
  ModelInfo,
  ModelName,
  Page,
  PatternCount,
  PredictResponse,
  Summary,
  WaferDetail,
  WaferMap,
  WaferSummary,
  YieldTrendPoint,
} from "./types";

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | boolean | null | undefined>;

/** Build a query string, dropping empty values so defaults stay server-side. */
export function query(params: Params): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  const s = qs.toString();
  return s ? `?${s}` : "";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export const api = {
  summary: (signal?: AbortSignal) => request<Summary>("/stats/summary", { signal }),
  models: (signal?: AbortSignal) => request<ModelInfo[]>("/models", { signal }),
  yieldTrend: (
    p: { bucket_lots?: number; pattern?: string; source?: LabelSource },
    signal?: AbortSignal,
  ) => request<YieldTrendPoint[]>(`/stats/yield-trend${query(p)}`, { signal }),
  patterns: (p: { source?: LabelSource; lot?: string }, signal?: AbortSignal) =>
    request<PatternCount[]>(`/stats/patterns${query(p)}`, { signal }),
  lots: (p: Params, signal?: AbortSignal) => request<Page<Lot>>(`/lots${query(p)}`, { signal }),
  lot: (name: string, signal?: AbortSignal) =>
    request<LotDetail>(`/lots/${encodeURIComponent(name)}`, { signal }),
  wafers: (p: Params, signal?: AbortSignal) =>
    request<Page<WaferSummary>>(`/wafers${query(p)}`, { signal }),
  wafer: (id: number, signal?: AbortSignal) => request<WaferDetail>(`/wafers/${id}`, { signal }),
  waferMap: (id: number, signal?: AbortSignal) =>
    request<WaferMap>(`/wafers/${id}/map`, { signal }),
  predictGrid: (grid: number[][], model?: ModelName) =>
    request<PredictResponse>(`/predict${query({ model })}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid }),
    }),
  predictUpload: (file: File, opts: { model?: ModelName; save?: boolean }) => {
    const form = new FormData();
    form.append("file", file);
    return request<PredictResponse>(`/predict/upload${query(opts)}`, {
      method: "POST",
      body: form,
    });
  },
};
