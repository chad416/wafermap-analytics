import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { LabelSource } from "../api/types";
import { PatternChart, YieldTrendChart } from "../components/charts";
import { ErrorBox, Loading, PatternSelect, StatCard } from "../components/common";
import { fmtPct } from "../lib/display";
import { useApi } from "../hooks/useApi";

export function Dashboard() {
  const [params, setParams] = useSearchParams();
  const pattern = params.get("pattern") ?? "";
  const source = (params.get("source") as LabelSource | null) ?? "predicted";
  const bucket = Number(params.get("bucket") ?? 2000);

  const summary = useApi((s) => api.summary(s), []);
  const models = useApi((s) => api.models(s), []);
  const trend = useApi(
    (s) => api.yieldTrend({ bucket_lots: bucket, pattern, source }, s),
    [bucket, pattern, source],
  );
  const dist = useApi((s) => api.patterns({ source }, s), [source]);

  function set(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  }

  const s = summary.data;
  const cnn = models.data?.find((m) => m.name === "cnn");
  return (
    <>
      <h1>Fab overview</h1>
      {summary.error && <ErrorBox message={summary.error} />}
      <section className="stats">
        <StatCard label="Lots" value={s ? s.lots.toLocaleString() : "…"} />
        <StatCard
          label="Wafers"
          value={s ? s.wafers.toLocaleString() : "…"}
          hint={s ? `${s.labelled_wafers.toLocaleString()} with expert labels` : undefined}
        />
        <StatCard label="Mean wafer yield" value={fmtPct(s?.mean_yield, 2)} />
        <StatCard
          label="Wafers with a defect pattern"
          value={fmtPct(s?.predicted_defect_share_pct, 1)}
          hint={s ? `predicted by ${s.default_model}` : undefined}
        />
        <StatCard
          label="CNN test macro-F1"
          value={cnn?.test_macro_f1?.toFixed(3) ?? "–"}
          hint={cnn ? `model ${cnn.version}` : undefined}
        />
      </section>

      <div className="toolbar">
        <label>
          Labels{" "}
          <select value={source} onChange={(e) => set("source", e.target.value)}>
            <option value="predicted">model predictions</option>
            <option value="true">expert labels (labelled wafers only)</option>
          </select>
        </label>
        <label>
          Pattern <PatternSelect value={pattern} onChange={(v) => set("pattern", v)} />
        </label>
        <label>
          Bucket{" "}
          <select value={bucket} onChange={(e) => set("bucket", e.target.value)}>
            {[500, 1000, 2000, 5000].map((b) => (
              <option key={b} value={b}>
                {b.toLocaleString()} lots
              </option>
            ))}
          </select>
        </label>
      </div>

      <section className="grid-2">
        <div className="card">
          <h2>Yield trend {pattern && <small>· wafers with {pattern}</small>}</h2>
          <p className="muted">
            Mean wafer yield per bucket of consecutive lot numbers. WM-811K has no timestamps, so
            lot order stands in for production sequence.
          </p>
          {trend.loading && !trend.data ? (
            <Loading />
          ) : trend.error ? (
            <ErrorBox message={trend.error} />
          ) : (
            <YieldTrendChart data={trend.data ?? []} />
          )}
        </div>
        <div className="card">
          <h2>Defect-pattern distribution</h2>
          <p className="muted">
            Wafers per pattern (log scale).{" "}
            {source === "predicted"
              ? "All wafers, CNN predictions."
              : "Labelled wafers, expert labels."}
          </p>
          {dist.loading && !dist.data ? (
            <Loading />
          ) : dist.error ? (
            <ErrorBox message={dist.error} />
          ) : (
            <PatternChart data={dist.data ?? []} metric="count" />
          )}
        </div>
        <div className="card">
          <h2>Mean yield by pattern</h2>
          <p className="muted">How much each failure mechanism costs in yield.</p>
          {dist.data && <PatternChart data={dist.data} metric="mean_yield" />}
        </div>
        <div className="card">
          <h2>Models</h2>
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Runtime</th>
                <th>Version</th>
                <th>Val macro-F1</th>
                <th>Test macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {models.data?.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td>{m.kind}</td>
                  <td>
                    <code>{m.version}</code>
                  </td>
                  <td>{m.val_macro_f1?.toFixed(3)}</td>
                  <td>{m.test_macro_f1?.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">
            Scores on held-out, lot-grouped splits (see README for the protocol).
          </p>
        </div>
      </section>
    </>
  );
}
