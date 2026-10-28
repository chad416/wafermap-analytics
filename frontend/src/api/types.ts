// Mirrors backend/src/wafermap/api/schemas.py (OpenAPI at /docs).

export const PATTERNS = [
  "Center",
  "Donut",
  "Edge-Loc",
  "Edge-Ring",
  "Loc",
  "Near-full",
  "Random",
  "Scratch",
  "none",
] as const;
export type Pattern = (typeof PATTERNS)[number];
export type LabelSource = "true" | "predicted";
export type ModelName = "cnn" | "baseline";

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface Prediction {
  model_name: string;
  model_version: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface WaferSummary {
  id: number;
  lot_name: string;
  wafer_index: number;
  rows: number;
  cols: number;
  n_die: number;
  n_fail: number;
  yield_pct: number;
  true_label: string | null;
  split: string | null;
  source: string;
  prediction: Prediction | null;
}

export interface WaferDetail extends WaferSummary {
  predictions: Prediction[];
}

export interface WaferMap {
  id: number;
  rows: number;
  cols: number;
  grid: number[][];
}

export interface Lot {
  name: string;
  lot_number: number;
  source: string;
  wafer_count: number;
  mean_yield: number | null;
  pattern_counts: Record<string, number>;
  top_defect: string | null;
}

export interface LotDetail extends Lot {
  wafers: WaferSummary[];
}

export interface YieldTrendPoint {
  bucket: number;
  lot_number_from: number;
  lot_number_to: number;
  lots: number;
  wafers: number;
  mean_yield: number;
  min_yield: number;
}

export interface PatternCount {
  pattern: string;
  count: number;
  share_pct: number;
  mean_yield: number | null;
}

export interface Summary {
  lots: number;
  wafers: number;
  labelled_wafers: number;
  predicted_wafers: number;
  uploaded_wafers: number;
  mean_yield: number | null;
  predicted_defect_share_pct: number | null;
  default_model: string;
}

export interface ModelInfo {
  name: string;
  kind: string;
  version: string;
  val_macro_f1: number | null;
  test_macro_f1: number | null;
}

export interface PredictResponse {
  model_name: string;
  model_version: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  stats: { rows: number; cols: number; n_die: number; n_fail: number; yield_pct: number };
  latency_ms: number;
  wafer_id: number | null;
  grid: number[][] | null;
}
