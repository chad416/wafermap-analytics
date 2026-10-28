import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PatternCount, YieldTrendPoint } from "../api/types";
import { PATTERN_COLORS } from "../lib/display";

export function YieldTrendChart({ data }: { data: YieldTrendPoint[] }) {
  const rows = data.map((d) => ({ ...d, label: `${d.lot_number_from}–${d.lot_number_to}` }));
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke="#e5e7eb" vertical={false} />
        <XAxis
          dataKey="lot_number_from"
          tick={{ fontSize: 12 }}
          label={{ value: "lot number", position: "insideBottom", offset: -4, fontSize: 12 }}
        />
        <YAxis domain={["auto", "auto"]} tick={{ fontSize: 12 }} unit="%" width={52} />
        <Tooltip
          formatter={(v) => [`${Number(v).toFixed(2)} %`, "mean yield"]}
          labelFormatter={(_, p) => {
            const d = p?.[0]?.payload as (YieldTrendPoint & { label: string }) | undefined;
            return d ? `lots ${d.label} · ${d.wafers.toLocaleString()} wafers` : "";
          }}
        />
        <Line
          type="monotone"
          dataKey="mean_yield"
          stroke="#2563eb"
          strokeWidth={2}
          dot={{ r: 2 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PatternChart({
  data,
  metric,
}: {
  data: PatternCount[];
  metric: "count" | "mean_yield";
}) {
  const rows = data.filter((d) => metric === "count" || d.mean_yield !== null);
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 28, left: 0 }}>
        <CartesianGrid stroke="#e5e7eb" vertical={false} />
        <XAxis
          dataKey="pattern"
          tick={{ fontSize: 11 }}
          angle={-30}
          textAnchor="end"
          interval={0}
        />
        <YAxis
          tick={{ fontSize: 12 }}
          width={56}
          scale={metric === "count" ? "log" : "auto"}
          domain={metric === "count" ? [1, "auto"] : [0, 100]}
          allowDataOverflow
          unit={metric === "mean_yield" ? "%" : undefined}
        />
        <Tooltip
          formatter={(v, _n, p) => {
            const d = p.payload as PatternCount;
            return metric === "count"
              ? [`${Number(v).toLocaleString()} wafers (${d.share_pct.toFixed(2)} %)`, "count"]
              : [`${Number(v).toFixed(1)} %`, "mean yield"];
          }}
        />
        <Bar dataKey={metric} isAnimationActive={false}>
          {rows.map((d) => (
            <Cell key={d.pattern} fill={PATTERN_COLORS[d.pattern] ?? "#64748b"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
