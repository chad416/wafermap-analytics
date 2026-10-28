import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { ErrorBox, Loading, Pagination, PatternBadge, PatternSelect } from "../components/common";
import { fmtPct } from "../lib/display";
import { useApi } from "../hooks/useApi";

const SORTS = {
  lot_number: "Lot number",
  mean_yield: "Mean yield",
  wafer_count: "Wafer count",
} as const;

export function LotsPage() {
  const [params, setParams] = useSearchParams();
  const page = Number(params.get("page") ?? 1);
  const pattern = params.get("pattern") ?? "";
  const q = params.get("q") ?? "";
  const sort = params.get("sort") ?? "lot_number";
  const descending = params.get("desc") === "1";
  const [search, setSearch] = useState(q);

  const lots = useApi(
    (s) => api.lots({ page, page_size: 20, pattern, q, sort, descending }, s),
    [page, pattern, q, sort, descending],
  );

  function update(changes: Record<string, string>) {
    const next = new URLSearchParams(params);
    for (const [k, v] of Object.entries(changes)) {
      if (v) next.set(k, v);
      else next.delete(k);
    }
    if (!("page" in changes)) next.delete("page");
    setParams(next);
  }

  // debounce the free-text search
  useEffect(() => {
    if (search === q) return;
    const t = setTimeout(() => update({ q: search }), 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  return (
    <>
      <h1>Lots</h1>
      <div className="toolbar">
        <input
          type="search"
          placeholder="Search lot name…"
          aria-label="search lots"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <label>
          Contains pattern{" "}
          <PatternSelect value={pattern} onChange={(v) => update({ pattern: v })} allLabel="any" />
        </label>
        <label>
          Sort{" "}
          <select value={sort} onChange={(e) => update({ sort: e.target.value })}>
            {Object.entries(SORTS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={descending}
            onChange={(e) => update({ desc: e.target.checked ? "1" : "" })}
          />{" "}
          descending
        </label>
      </div>
      {lots.error && <ErrorBox message={lots.error} />}
      {lots.loading && !lots.data && <Loading />}
      {lots.data && (
        <div className="card">
          <table className="table">
            <thead>
              <tr>
                <th>Lot</th>
                <th className="num">Wafers</th>
                <th className="num">Mean yield</th>
                <th>Most frequent defect</th>
                <th className="num">Defect wafers</th>
              </tr>
            </thead>
            <tbody className={lots.loading ? "is-loading" : undefined}>
              {lots.data.items.map((lot) => {
                const defects = Object.entries(lot.pattern_counts)
                  .filter(([k]) => k !== "none")
                  .reduce((a, [, n]) => a + n, 0);
                return (
                  <tr key={lot.name}>
                    <td>
                      <Link to={`/lots/${encodeURIComponent(lot.name)}`}>{lot.name}</Link>
                    </td>
                    <td className="num">{lot.wafer_count}</td>
                    <td className="num">{fmtPct(lot.mean_yield, 2)}</td>
                    <td>
                      {lot.top_defect ? (
                        <PatternBadge label={lot.top_defect} />
                      ) : (
                        <span className="muted">none</span>
                      )}
                    </td>
                    <td className="num">{defects}</td>
                  </tr>
                );
              })}
              {lots.data.items.length === 0 && (
                <tr>
                  <td colSpan={5} className="muted">
                    No lots match. If the database is empty, run the ingest step (see README).
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <Pagination
            page={lots.data.page}
            pages={lots.data.pages}
            total={lots.data.total}
            onPage={(p) => update({ page: String(p) })}
          />
        </div>
      )}
    </>
  );
}
