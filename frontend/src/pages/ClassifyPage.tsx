import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { ModelName, PredictResponse } from "../api/types";
import { ErrorBox, Loading, PatternBadge } from "../components/common";
import { fmtPct } from "../lib/display";
import { ProbabilityBars } from "../components/ProbabilityBars";
import { BinLegend, WaferMap } from "../components/WaferMap";

const SAMPLES = ["center", "edge-ring", "scratch", "donut", "clean"];

export function ClassifyPage() {
  const [model, setModel] = useState<ModelName>("cnn");
  const [save, setSave] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  async function classify(file: File) {
    setBusy(true);
    setError(null);
    setFileName(file.name);
    try {
      setResult(await api.predictUpload(file, { model, save }));
    } catch (e) {
      setResult(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadSample(name: string) {
    const res = await fetch(`/samples/${name}.csv`);
    const blob = await res.blob();
    await classify(new File([blob], `${name}.csv`, { type: "text/csv" }));
  }

  return (
    <>
      <h1>Classify a wafer map</h1>
      <p className="muted">
        Upload a die-bin grid (<code>0</code> = no die, <code>1</code> = pass, <code>2</code> =
        fail) as <code>.csv</code>, <code>.txt</code>, <code>.json</code> (2-D list) or{" "}
        <code>.npy</code>. The map is classified server-side in real time.
      </p>
      <div className="toolbar">
        <label>
          Model{" "}
          <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
            <option value="cnn">CNN (ONNX Runtime)</option>
            <option value="baseline">LightGBM baseline</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)} /> save
          to lot “uploads”
        </label>
      </div>
      <label
        className={`dropzone${dragging ? " dropzone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) void classify(f);
        }}
      >
        <input
          type="file"
          accept=".csv,.txt,.json,.npy"
          aria-label="wafer map file"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void classify(f);
          }}
        />
        <span>Drop a wafer-map file here, or click to choose one</span>
      </label>
      <p className="samples">
        Synthetic samples:{" "}
        {SAMPLES.map((s) => (
          <button key={s} className="link" onClick={() => void loadSample(s)}>
            {s}
          </button>
        ))}
      </p>
      {busy && <Loading />}
      {error && <ErrorBox message={error} />}
      {result && result.grid && (
        <section className="grid-2">
          <div className="card center">
            <WaferMap grid={result.grid} size={380} interactive label={fileName ?? "upload"} />
            <BinLegend />
          </div>
          <div className="card">
            <h2>
              Prediction: <PatternBadge label={result.label} />
            </h2>
            <p>
              Confidence <strong>{(result.confidence * 100).toFixed(1)} %</strong> · yield{" "}
              {fmtPct(result.stats.yield_pct, 2)} · {result.stats.rows} × {result.stats.cols} grid
            </p>
            <ProbabilityBars probabilities={result.probabilities} />
            <p className="muted">
              {result.model_name} <code>{result.model_version}</code> ·{" "}
              {result.latency_ms.toFixed(1)} ms server-side
            </p>
            {result.wafer_id !== null && (
              <p>
                Saved as <Link to={`/wafers/${result.wafer_id}`}>wafer {result.wafer_id}</Link>.
              </p>
            )}
          </div>
        </section>
      )}
    </>
  );
}
