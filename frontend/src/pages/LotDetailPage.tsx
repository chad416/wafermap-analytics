import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { WaferSummary } from "../api/types";
import { ErrorBox, Loading, PatternBadge, StatCard } from "../components/common";
import { fmtPct } from "../lib/display";
import { WaferMap } from "../components/WaferMap";
import { useApi } from "../hooks/useApi";

function WaferThumb({ w }: { w: WaferSummary }) {
  const map = useApi((s) => api.waferMap(w.id, s), [w.id]);
  const predicted = w.prediction?.label ?? null;
  const disagree = w.true_label !== null && predicted !== null && w.true_label !== predicted;
  return (
    <Link to={`/wafers/${w.id}`} className={`thumb${disagree ? " thumb--disagree" : ""}`}>
      <div className="thumb__map">
        {map.data ? (
          <WaferMap grid={map.data.grid} size={132} gaps={false} label={`wafer ${w.wafer_index}`} />
        ) : (
          <div className="thumb__placeholder" />
        )}
      </div>
      <div className="thumb__meta">
        <strong>#{w.wafer_index}</strong> <span className="muted">{fmtPct(w.yield_pct, 1)}</span>
      </div>
      <PatternBadge label={predicted} />
      {w.true_label && (
        <div className="thumb__truth" title="expert label">
          label: {w.true_label}
        </div>
      )}
    </Link>
  );
}

export function LotDetailPage() {
  const { lotName = "" } = useParams();
  const lot = useApi((s) => api.lot(lotName, s), [lotName]);

  if (lot.error) return <ErrorBox message={lot.error} />;
  if (!lot.data) return <Loading />;
  const l = lot.data;
  const labelled = l.wafers.filter((w) => w.true_label !== null);
  const agree = labelled.filter((w) => w.prediction?.label === w.true_label).length;
  return (
    <>
      <p className="crumbs">
        <Link to="/lots">Lots</Link> / {l.name}
      </p>
      <h1>{l.name}</h1>
      <section className="stats">
        <StatCard label="Wafers" value={l.wafer_count} />
        <StatCard label="Mean yield" value={fmtPct(l.mean_yield, 2)} />
        <StatCard label="Most frequent defect" value={l.top_defect ?? "none"} />
        <StatCard
          label="Model vs expert labels"
          value={labelled.length ? `${agree} / ${labelled.length}` : "–"}
          hint={
            labelled.length
              ? "labelled wafers where the CNN agrees"
              : "no labelled wafers in this lot"
          }
        />
      </section>
      <div className="card">
        <h2>Wafers</h2>
        <p className="muted">
          Badge = CNN prediction; red outline = prediction disagrees with the expert label.
        </p>
        <div className="thumbs">
          {l.wafers.map((w) => (
            <WaferThumb key={w.id} w={w} />
          ))}
        </div>
      </div>
    </>
  );
}
