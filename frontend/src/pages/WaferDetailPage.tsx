import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { PredictResponse } from "../api/types";
import { ErrorBox, Loading, PatternBadge, StatCard } from "../components/common";
import { fmtPct } from "../lib/display";
import { ProbabilityBars } from "../components/ProbabilityBars";
import { BinLegend, WaferMap } from "../components/WaferMap";
import { useApi } from "../hooks/useApi";

export function WaferDetailPage() {
  const id = Number(useParams().waferId);
  // keyed by id so per-wafer state (the on-demand baseline result) resets on navigation
  return <WaferDetail key={id} id={id} />;
}

function WaferDetail({ id }: { id: number }) {
  const wafer = useApi((s) => api.wafer(id, s), [id]);
  const map = useApi((s) => api.waferMap(id, s), [id]);
  const [baseline, setBaseline] = useState<PredictResponse | string | null>(null);

  if (wafer.error) return <ErrorBox message={wafer.error} />;
  if (!wafer.data) return <Loading />;
  const w = wafer.data;

  async function runBaseline() {
    if (!map.data) return;
    setBaseline("running");
    try {
      setBaseline(await api.predictGrid(map.data.grid, "baseline"));
    } catch (e) {
      setBaseline(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <>
      <p className="crumbs">
        <Link to="/lots">Lots</Link> /{" "}
        <Link to={`/lots/${encodeURIComponent(w.lot_name)}`}>{w.lot_name}</Link> / wafer{" "}
        {w.wafer_index}
      </p>
      <h1>
        {w.lot_name} · wafer {w.wafer_index}
      </h1>
      <section className="stats">
        <StatCard
          label="Yield"
          value={fmtPct(w.yield_pct, 2)}
          hint={`${(w.n_die - w.n_fail).toLocaleString()} / ${w.n_die.toLocaleString()} dies pass`}
        />
        <StatCard label="Grid" value={`${w.rows} × ${w.cols}`} />
        <StatCard
          label="Expert label"
          value={<PatternBadge label={w.true_label} />}
          hint={w.split ? `${w.split} split` : undefined}
        />
        <StatCard
          label="CNN prediction"
          value={<PatternBadge label={w.prediction?.label ?? null} />}
          hint={
            w.prediction
              ? `confidence ${(w.prediction.confidence * 100).toFixed(1)} %`
              : "not predicted"
          }
        />
      </section>
      <section className="grid-2">
        <div className="card center">
          {map.error ? (
            <ErrorBox message={map.error} />
          ) : map.data ? (
            <WaferMap
              grid={map.data.grid}
              size={420}
              interactive
              label={`${w.lot_name} wafer ${w.wafer_index}`}
            />
          ) : (
            <Loading />
          )}
          <BinLegend />
        </div>
        <div className="card">
          <h2>Class probabilities</h2>
          {w.predictions.map((p) => (
            <div key={p.model_name} className="model-block">
              <h3>
                {p.model_name} <code>{p.model_version}</code>
              </h3>
              <ProbabilityBars probabilities={p.probabilities} top={5} />
            </div>
          ))}
          <div className="model-block">
            <h3>baseline (LightGBM, on demand)</h3>
            {baseline === null && (
              <button onClick={runBaseline} disabled={!map.data}>
                Classify with baseline
              </button>
            )}
            {baseline === "running" && <Loading />}
            {typeof baseline === "string" && baseline !== "running" && (
              <ErrorBox message={baseline} />
            )}
            {baseline && typeof baseline === "object" && (
              <>
                <ProbabilityBars probabilities={baseline.probabilities} top={5} />
                <p className="muted">{baseline.latency_ms.toFixed(1)} ms server-side</p>
              </>
            )}
          </div>
        </div>
      </section>
    </>
  );
}
