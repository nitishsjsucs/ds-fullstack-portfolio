/** Number and text formatting.
 *
 * One rule runs through this file: a missing value renders as an em dash, never
 * as "0", "NaN" or "null". The backend already maps NaN and infinity to null, so
 * the frontend has exactly one missing sentinel to handle.
 */

export const DASH = "—";

export function num(v: unknown, digits = 3): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) {
    return DASH;
  }
  return v.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function int(v: unknown): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) {
    return DASH;
  }
  return Math.round(v).toLocaleString();
}

export function compact(v: unknown): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) {
    return DASH;
  }
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(v / 1e3).toFixed(0)}k`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

export function pct(v: unknown, digits = 1): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) {
    return DASH;
  }
  return `${(v * 100).toFixed(digits)}%`;
}

export function signed(v: unknown, digits = 3): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) {
    return DASH;
  }
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export function usd(v: unknown, digits = 0): string {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) {
    return DASH;
  }
  return v.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function titleCase(s: string): string {
  return s.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function humanKey(s: string): string {
  return s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

export function shortDate(s: string | null | undefined): string {
  if (!s) return DASH;
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return String(s).slice(0, 19).replace("T", " ");
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function truncate(s: string, n = 90): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

/** Map a 0-1 score to the status vocabulary. Higher is better. */
export function gradeTone(score: number | null | undefined): Tone {
  if (score === null || score === undefined) return "neutral";
  if (score >= 0.9) return "good";
  if (score >= 0.75) return "warning";
  if (score >= 0.5) return "serious";
  return "critical";
}

export type Tone = "good" | "warning" | "serious" | "critical" | "neutral" | "accent";

export const SERIES = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
  "var(--series-7)",
  "var(--series-8)",
] as const;

/** Categorical colour by slot index. Never cycles past the eighth slot --
 *  callers with more series must fold the tail into "Other" or facet. */
export function seriesColor(i: number): string {
  return SERIES[Math.min(i, SERIES.length - 1)];
}

/** Sequential blue ramp for magnitude encoding, t in [0, 1]. */
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

export function sequential(t: number): string {
  if (!Number.isFinite(t)) return "var(--surface-2)";
  const clamped = Math.max(0, Math.min(1, t));
  return SEQ[Math.min(SEQ.length - 1, Math.floor(clamped * SEQ.length))];
}

/** Diverging ramp: two poles with a neutral midpoint, t in [-1, 1]. */
export function diverging(t: number): string {
  if (!Number.isFinite(t)) return "var(--div-mid)";
  const clamped = Math.max(-1, Math.min(1, t));
  if (Math.abs(clamped) < 0.06) return "var(--div-mid)";
  const mag = Math.abs(clamped);
  const ramp = clamped > 0
    ? ["#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
    : ["#c0392b", "#e66767", "#ef8f8f", "#f6bcbc"];
  return ramp[Math.min(ramp.length - 1, Math.floor((1 - mag) * ramp.length))];
}
