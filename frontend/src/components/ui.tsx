/** Shared presentational components.
 *
 * Everything here is deliberately small and unopinionated about data. The tone
 * vocabulary (good / warning / serious / critical) maps onto the reserved status
 * palette and is always paired with a word or an icon, never carried by colour
 * alone -- which is what makes a red badge legible to a reader who cannot see it
 * as red.
 */

import type { ReactNode } from "react";
import { DASH, num, type Tone } from "../lib/format";
import { ApiError } from "../lib/api";

/* ----------------------------------------------------------------- badge -- */
export function Badge({
  tone = "neutral",
  children,
  icon,
}: {
  tone?: Tone;
  children: ReactNode;
  icon?: string;
}) {
  const cls = tone === "neutral" ? "badge" : `badge badge--${tone}`;
  return (
    <span className={cls}>
      {icon && <span aria-hidden="true">{icon}</span>}
      {children}
    </span>
  );
}

export function PassFail({ passed, labels = ["Pass", "Fail"] }: { passed: boolean; labels?: [string, string] | string[] }) {
  return (
    <Badge tone={passed ? "good" : "critical"} icon={passed ? "✓" : "✕"}>
      {passed ? labels[0] : labels[1]}
    </Badge>
  );
}

/* ------------------------------------------------------------- stat tile -- */
export function Stat({
  label,
  value,
  unit,
  note,
  tone,
  hero = false,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  note?: ReactNode;
  tone?: Tone;
  hero?: boolean;
}) {
  const color =
    tone && tone !== "neutral"
      ? tone === "accent"
        ? "var(--accent, var(--series-1))"
        : `var(--status-${tone})`
      : undefined;
  return (
    <div className="stat">
      <div className="stat__label">{label}</div>
      <div className={hero ? "stat__value stat__value--hero" : "stat__value"} style={{ color }}>
        {value}
        {unit && <span className="stat__unit">{unit}</span>}
      </div>
      {note && <div className="stat__note">{note}</div>}
    </div>
  );
}

/* -------------------------------------------------------------- callout --- */
export function Callout({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "good" | "warning" | "critical";
  title?: ReactNode;
  children: ReactNode;
}) {
  const icon = { info: "ℹ", good: "✓", warning: "⚠", critical: "✕" }[tone];
  return (
    <div className={`callout callout--${tone}`}>
      {title && (
        <div className="callout__title">
          <span aria-hidden="true">{icon}</span>
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

/* -------------------------------------------------------------- section --- */
export function Section({
  title,
  description,
  aside,
  children,
}: {
  title: string;
  description?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="section">
      <div className="section__head">
        <h2 className="section__title">{title}</h2>
        {aside && <div className="section__aside">{aside}</div>}
      </div>
      {description && <div className="section__desc">{description}</div>}
      {children}
    </section>
  );
}

/* ---------------------------------------------------------------- table --- */
export interface Column<T> {
  key: string;
  header: string;
  numeric?: boolean;
  width?: string;
  render: (row: T, index: number) => ReactNode;
}

export function DataTable<T>({
  columns,
  rows,
  empty = "Nothing to show.",
  maxHeight,
  rowKey,
}: {
  columns: Column<T>[];
  rows: T[];
  empty?: string;
  maxHeight?: number;
  rowKey?: (row: T, i: number) => string;
}) {
  if (!rows.length) {
    return <div className="state">{empty}</div>;
  }
  return (
    <div className="tablewrap" style={maxHeight ? { maxHeight, overflowY: "auto" } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.numeric ? "num" : undefined} style={{ width: c.width }}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={rowKey ? rowKey(row, i) : i}>
              {columns.map((c) => (
                <td key={c.key} className={c.numeric ? "num" : undefined}>
                  {c.render(row, i)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* --------------------------------------------------------------- states --- */
export function Loading({ what = "data" }: { what?: string }) {
  return (
    <div className="state">
      <div className="spinner" />
      Loading {what}…
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const untrained = error.status === 503;
  return (
    <div className="card">
      <Callout tone={untrained ? "warning" : "critical"} title={untrained ? "Not trained yet" : "Could not load"}>
        <p style={{ marginBottom: 8 }}>{error.message}</p>
        {error.hint && <p className="small muted" style={{ marginBottom: 8 }}>{error.hint}</p>}
        {untrained && (
          <pre className="pre" style={{ marginTop: 8 }}>
{`cd ds-fullstack-portfolio
python scripts/fetch_data.py     # once, to download the datasets
python scripts/train_all.py      # produces every artifact this page reads`}
          </pre>
        )}
        {onRetry && (
          <button className="btn btn--ghost btn--sm" style={{ marginTop: 10 }} onClick={onRetry}>
            Retry
          </button>
        )}
      </Callout>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="state">{children}</div>;
}

/* --------------------------------------------------------------- inputs --- */
export function SliderField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v: number) => String(v),
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: ReactNode;
}) {
  return (
    <div className="field">
      <label className="field__label">
        <span>{label}</span>
        <span className="field__value">{format(value)}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
      />
      {hint && <div className="field__hint">{hint}</div>}
    </div>
  );
}

export function SelectField({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: string | number;
  options: { value: string | number; label: string }[];
  onChange: (v: string) => void;
  hint?: ReactNode;
}) {
  return (
    <div className="field">
      <label className="field__label">
        <span>{label}</span>
      </label>
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label={label}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {hint && <div className="field__hint">{hint}</div>}
    </div>
  );
}

export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: ReactNode;
}) {
  return (
    <div className="field">
      <label className="field__label">
        <span>{label}</span>
      </label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
      />
      {hint && <div className="field__hint">{hint}</div>}
    </div>
  );
}

/* ----------------------------------------------------------- key/values --- */
export function KeyValues({
  entries,
}: {
  entries: [string, ReactNode][];
}) {
  return (
    <div className="kv">
      {entries.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <div className="kv__k">{k}</div>
          <div className="kv__v">{v ?? DASH}</div>
        </div>
      ))}
    </div>
  );
}

export function MetricList({
  metrics,
  highlight = [],
}: {
  metrics: Record<string, unknown>;
  highlight?: string[];
}) {
  const entries = Object.entries(metrics).filter(
    ([k, v]) => !k.startsWith("_") && (typeof v === "number" || typeof v === "boolean" || v === null),
  );
  return (
    <div className="kv">
      {entries.map(([k, v]) => (
        <div key={k} style={{ display: "contents" }}>
          <div className="kv__k">{k.replace(/_/g, " ")}</div>
          <div
            className="kv__v tnum"
            style={highlight.includes(k) ? { color: "var(--accent, var(--series-1))", fontWeight: 620 } : undefined}
          >
            {typeof v === "boolean" ? (v ? "yes" : "no") : typeof v === "number" ? num(v, 4) : DASH}
          </div>
        </div>
      ))}
    </div>
  );
}

/** A labelled 0-1 score with its evidence, used by the quality scorecard and
 *  the audit. The bar is a magnitude encoding, so it uses the sequential ramp
 *  rather than a categorical hue, and the number is always printed beside it --
 *  a bar length alone is not readable to two decimal places. */
export function ScoreBarRow({
  label,
  score,
  detail,
  format = (v: number) => `${(v * 100).toFixed(1)}%`,
}: {
  label: string;
  score: number;
  detail?: ReactNode;
  format?: (v: number) => string;
}) {
  const tone: Tone = score >= 0.95 ? "good" : score >= 0.85 ? "warning" : score >= 0.6 ? "serious" : "critical";
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 12,
          marginBottom: 5,
        }}
      >
        <span style={{ fontWeight: 580, fontSize: "var(--text-sm)" }}>{label}</span>
        <span className="tnum mono" style={{ color: `var(--status-${tone})`, fontWeight: 640 }}>
          {format(score)}
        </span>
      </div>
      <div
        style={{
          background: "var(--surface-3)",
          borderRadius: 4,
          height: 6,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.max(0, Math.min(1, score)) * 100}%`,
            height: "100%",
            background: `var(--status-${tone})`,
            borderRadius: 4,
          }}
        />
      </div>
      {detail && (
        <div className="small muted" style={{ marginTop: 4, lineHeight: 1.45 }}>
          {detail}
        </div>
      )}
    </div>
  );
}

/** A word-wrapped list of short strings as chips. */
export function TagList({ items, tone }: { items: string[]; tone?: Tone }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {items.map((t) => (
        <Badge key={t} tone={tone}>
          {t}
        </Badge>
      ))}
    </div>
  );
}
