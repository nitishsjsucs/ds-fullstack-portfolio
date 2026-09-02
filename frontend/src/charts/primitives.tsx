/** Chart scaffolding: scales, axes, frames and the hover layer.
 *
 * Every chart in this app is hand-drawn SVG rather than a charting library. That
 * is a deliberate trade: it costs more code, and it buys exact control over the
 * things the visualisation method makes non-negotiable -- 2px strokes, 4px
 * rounded data-ends anchored to the baseline, a 2px surface gap between adjacent
 * fills, recessive grid lines, and a tooltip on every mark. Those are hard to
 * enforce through a library's theming API and trivial to enforce here.
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

/* --------------------------------------------------------------- scales -- */
export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
  invert: (px: number) => number;
  ticks: (count?: number) => number[];
}

export function linearScale(
  domain: [number, number],
  range: [number, number],
): Scale {
  let [d0, d1] = domain;
  if (!Number.isFinite(d0) || !Number.isFinite(d1)) [d0, d1] = [0, 1];
  if (d0 === d1) {
    // A constant series still needs a drawable band, otherwise every mark lands
    // on one pixel and the axis reads as broken.
    const pad = Math.abs(d0) * 0.1 || 1;
    d0 -= pad;
    d1 += pad;
  }
  const [r0, r1] = range;
  const scale = ((v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  scale.domain = [d0, d1];
  scale.range = range;
  scale.invert = (px: number) => d0 + ((px - r0) / (r1 - r0)) * (d1 - d0);
  scale.ticks = (count = 5) => niceTicks(d0, d1, count);
  return scale;
}

/** Round tick values to 1/2/5 x 10^n so axis labels read cleanly. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(rawStep));
  const norm = rawStep / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toFixed(10)));
  }
  return out;
}

/** Pick tick decimals from the span, not from the magnitude.
 *
 * A default formatter that rounds to one decimal turns an axis spanning
 * 0.907–0.931 into six ticks all reading "0.9", which is worse than no axis at
 * all. This chooses enough precision to make adjacent ticks distinguishable.
 */
export function autoFormat(domain: [number, number]): (v: number) => string {
  const span = Math.abs(domain[1] - domain[0]);
  if (!Number.isFinite(span) || span === 0) return (v) => String(v);
  if (span >= 1000) return (v) => (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "k";
  if (span >= 100) return (v) => v.toFixed(0);
  if (span >= 10) return (v) => v.toFixed(0);
  if (span >= 1) return (v) => v.toFixed(1);
  if (span >= 0.1) return (v) => v.toFixed(2);
  if (span >= 0.01) return (v) => v.toFixed(3);
  return (v) => v.toFixed(4);
}

export function extent(values: number[], pad = 0): [number, number] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (!finite.length) return [0, 1];
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (pad > 0) {
    const span = (max - min) || Math.abs(max) || 1;
    min -= span * pad;
    max += span * pad;
  }
  return [min, max];
}

/* --------------------------------------------------------------- layout -- */
export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const DEFAULT_MARGIN: Margin = { top: 12, right: 16, bottom: 34, left: 52 };

export interface Plot {
  width: number;
  height: number;
  margin: Margin;
  innerWidth: number;
  innerHeight: number;
}

/** Compute the plot box. Deliberately a plain function, not a hook.
 *
 * Charts call this inside `Responsive`'s render callback, which only runs once
 * the container has been measured. A hook there would be called conditionally --
 * zero hooks on the first render, one on the second -- which React rejects with
 * "Rendered more hooks than during the previous render" and which unmounts the
 * whole tree. The computation is four subtractions; memoising it was never worth
 * the constraint.
 */
export function makePlot(
  width: number,
  height: number,
  margin: Partial<Margin> = {},
): Plot {
  const m = { ...DEFAULT_MARGIN, ...margin };
  return {
    width,
    height,
    margin: m,
    innerWidth: Math.max(1, width - m.left - m.right),
    innerHeight: Math.max(1, height - m.top - m.bottom),
  };
}

/* -------------------------------------------------------------- tooltip -- */
interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

const TooltipCtx = createContext<{
  show: (x: number, y: number, content: ReactNode) => void;
  hide: () => void;
}>({ show: () => {}, hide: () => {} });

export function useTooltipTarget() {
  return useContext(TooltipCtx);
}

/** Wraps a chart and renders the floating tooltip above it. */
export function TooltipHost({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TooltipState | null>(null);
  const hostRef = useRef<HTMLDivElement>(null);

  const show = useCallback((x: number, y: number, content: ReactNode) => {
    setTip({ x, y, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const ctx = useMemo(() => ({ show, hide }), [show, hide]);

  // Flip the tooltip to the left of the cursor when it would overflow the host,
  // so it never gets clipped at the right edge of a narrow card.
  const hostWidth = hostRef.current?.clientWidth ?? 0;
  const flip = tip !== null && hostWidth > 0 && tip.x > hostWidth - 190;

  return (
    <TooltipCtx.Provider value={ctx}>
      <div ref={hostRef} style={{ position: "relative" }}>
        {children}
        {tip && (
          <div
            className="tooltip"
            style={{
              left: flip ? undefined : tip.x + 12,
              right: flip ? hostWidth - tip.x + 12 : undefined,
              top: Math.max(0, tip.y - 12),
            }}
          >
            {tip.content}
          </div>
        )}
      </div>
    </TooltipCtx.Provider>
  );
}

export function TipRow({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="tooltip__row">
      <span className="tooltip__key">{k}</span>
      <span>{v}</span>
    </div>
  );
}

/* ----------------------------------------------------------------- axes -- */
export function GridY({
  scale,
  plot,
  count = 5,
}: {
  scale: Scale;
  plot: Plot;
  count?: number;
}) {
  return (
    <g className="chart__grid">
      {scale.ticks(count).map((t) => (
        <line key={t} x1={0} x2={plot.innerWidth} y1={scale(t)} y2={scale(t)} />
      ))}
    </g>
  );
}

export function AxisY({
  scale,
  plot,
  count = 5,
  format = (v: number) => String(v),
  label,
}: {
  scale: Scale;
  plot: Plot;
  count?: number;
  format?: (v: number) => string;
  label?: string;
}) {
  return (
    <g className="chart__axis">
      {scale.ticks(count).map((t) => (
        <text
          key={t}
          className="chart__tick"
          x={-8}
          y={scale(t)}
          dy="0.32em"
          textAnchor="end"
        >
          {format(t)}
        </text>
      ))}
      {label && (
        <text
          className="chart__axislabel"
          transform={`translate(${-plot.margin.left + 12}, ${plot.innerHeight / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {label}
        </text>
      )}
    </g>
  );
}

export function AxisX({
  scale,
  plot,
  count = 6,
  format = (v: number) => String(v),
  label,
}: {
  scale: Scale;
  plot: Plot;
  count?: number;
  format?: (v: number) => string;
  label?: string;
}) {
  return (
    <g className="chart__axis" transform={`translate(0, ${plot.innerHeight})`}>
      <line x1={0} x2={plot.innerWidth} y1={0} y2={0} />
      {scale.ticks(count).map((t) => (
        <text
          key={t}
          className="chart__tick"
          x={scale(t)}
          y={16}
          textAnchor="middle"
        >
          {format(t)}
        </text>
      ))}
      {label && (
        <text
          className="chart__axislabel"
          x={plot.innerWidth / 2}
          y={30}
          textAnchor="middle"
        >
          {label}
        </text>
      )}
    </g>
  );
}

export function AxisXCategorical({
  labels,
  plot,
  every = 1,
  rotate = false,
}: {
  labels: string[];
  plot: Plot;
  every?: number;
  rotate?: boolean;
}) {
  const step = plot.innerWidth / Math.max(1, labels.length);
  return (
    <g className="chart__axis" transform={`translate(0, ${plot.innerHeight})`}>
      <line x1={0} x2={plot.innerWidth} y1={0} y2={0} />
      {labels.map((l, i) =>
        i % every === 0 ? (
          <text
            key={`${l}-${i}`}
            className="chart__tick"
            x={step * (i + 0.5)}
            y={rotate ? 10 : 16}
            textAnchor={rotate ? "end" : "middle"}
            transform={rotate ? `rotate(-38, ${step * (i + 0.5)}, 10)` : undefined}
          >
            {l}
          </text>
        ) : null,
      )}
    </g>
  );
}

/* ---------------------------------------------------------------- frame -- */
export function ChartFrame({
  title,
  subtitle,
  footnote,
  legend,
  children,
  aside,
}: {
  title?: string;
  subtitle?: string;
  footnote?: ReactNode;
  legend?: { label: string; color: string; shape?: "square" | "line" }[];
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="chartframe">
      {(title || subtitle || aside) && (
        <div className="chartframe__head">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
            }}
          >
            {title && <div className="chartframe__title">{title}</div>}
            {aside}
          </div>
          {subtitle && <div className="chartframe__sub">{subtitle}</div>}
        </div>
      )}
      {legend && legend.length > 1 && (
        <div className="legend">
          {legend.map((l) => (
            <span className="legend__item" key={l.label}>
              <span
                className="legend__swatch"
                style={{
                  background: l.color,
                  height: l.shape === "line" ? 2 : 10,
                  borderRadius: l.shape === "line" ? 1 : 2,
                }}
              />
              {l.label}
            </span>
          ))}
        </div>
      )}
      <TooltipHost>{children}</TooltipHost>
      {footnote && <div className="chartframe__foot">{footnote}</div>}
    </div>
  );
}

/** A sized container that hands its measured width to a render function.
 *
 * Charts need a concrete pixel width to lay out an SVG, but the cards they sit
 * in are fluid. This measures the container and re-renders on resize, so the
 * same chart works in a full-width section and in a two-column grid.
 */
export function Responsive({
  height,
  children,
  minWidth = 240,
}: {
  height: number;
  minWidth?: number;
  children: (width: number) => ReactNode;
}) {
  const [width, setWidth] = useState(0);
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback((node: HTMLDivElement | null) => {
    observer.current?.disconnect();
    if (!node) return;
    const measure = () => setWidth(node.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(node);
    observer.current = ro;
  }, []);

  return (
    <div ref={ref} style={{ width: "100%", minHeight: height }}>
      {width >= minWidth ? children(width) : null}
    </div>
  );
}

/** Rounded-top bar path: 4px radius on the data end, square on the baseline. */
export function barPathV(
  x: number,
  y: number,
  w: number,
  h: number,
  r = 4,
): string {
  const radius = Math.min(r, w / 2, Math.max(0, h));
  if (h <= 0.5) return "";
  return [
    `M${x},${y + h}`,
    `L${x},${y + radius}`,
    `Q${x},${y} ${x + radius},${y}`,
    `L${x + w - radius},${y}`,
    `Q${x + w},${y} ${x + w},${y + radius}`,
    `L${x + w},${y + h}`,
    "Z",
  ].join(" ");
}

/** Rounded-right bar path for horizontal bars. */
export function barPathH(
  x: number,
  y: number,
  w: number,
  h: number,
  r = 4,
): string {
  const radius = Math.min(r, h / 2, Math.max(0, w));
  if (w <= 0.5) return "";
  return [
    `M${x},${y}`,
    `L${x + w - radius},${y}`,
    `Q${x + w},${y} ${x + w},${y + radius}`,
    `L${x + w},${y + h - radius}`,
    `Q${x + w},${y + h} ${x + w - radius},${y + h}`,
    `L${x},${y + h}`,
    "Z",
  ].join(" ");
}

export function linePath(points: { x: number; y: number }[]): string {
  return points
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
    .join(" ");
}

export function areaPath(
  points: { x: number; y0: number; y1: number }[],
): string {
  const valid = points.filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.y0) && Number.isFinite(p.y1),
  );
  if (!valid.length) return "";
  const top = valid.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y1.toFixed(2)}`);
  const bottom = [...valid]
    .reverse()
    .map((p) => `L${p.x.toFixed(2)},${p.y0.toFixed(2)}`);
  return `${top.join(" ")} ${bottom.join(" ")} Z`;
}
