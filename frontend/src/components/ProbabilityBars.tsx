import { PATTERN_COLORS } from "../lib/display";

/** Class probabilities as horizontal bars, highest first. */
export function ProbabilityBars({
  probabilities,
  top = 9,
}: {
  probabilities: Record<string, number>;
  top?: number;
}) {
  const rows = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .slice(0, top);
  return (
    <ul className="probs" aria-label="class probabilities">
      {rows.map(([label, p]) => (
        <li key={label}>
          <span className="probs__label">{label}</span>
          <span className="probs__track">
            <span
              className="probs__bar"
              style={{
                width: `${(p * 100).toFixed(1)}%`,
                background: PATTERN_COLORS[label] ?? "#64748b",
              }}
            />
          </span>
          <span className="probs__value">{(p * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}
