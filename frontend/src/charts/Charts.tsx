/** The chart kit.
 *
 * Each component follows the same shape: take tidy data, compute scales, draw
 * recessive chrome first, then the marks, then the hover layer. Mark geometry is
 * fixed by the visualisation method -- 2px strokes, 4px rounded data-ends, >=8px
 * point markers, a 2px surface gap between adjacent fills -- so callers choose
 * data and colour, never line weight.
 */

import { Fragment, type ReactNode } from "react";
import { compact, num, pct, sequential, seriesColor, diverging } from "../lib/format";
import {
  AxisX,
  AxisXCategorical,
  AxisY,
  ChartFrame,
  GridY,
  Responsive,
  TipRow,
  areaPath,
  barPathH,
  barPathV,
  extent,
  autoFormat,
  linePath,
  linearScale,
  makePlot,
  useTooltipTarget,
  type Margin,
} from "./primitives";

const GAP = 2; // the surface gap between adjacent fills

/* ============================================================ bar charts == */
export interface BarDatum {
  label: string;
  value: number;
  color?: string;
  note?: string;
}

export function BarChartH({
  data,
  height,
  format = (v: number) => num(v, 3),
  valueLabel = "Value",
  labelWidth = 168,
  showValues = true,
  domain,
}: {
  data: BarDatum[];
  height?: number;
  format?: (v: number) => string;
  valueLabel?: string;
  labelWidth?: number;
  showValues?: boolean;
  domain?: [number, number];
}) {
  const tip = useTooltipTarget();
  const rowH = 26;
  const h = height ?? Math.max(80, data.length * rowH + 34);
  const margin: Partial<Margin> = { left: labelWidth, right: showValues ? 56 : 16, top: 6, bottom: 26 };

  return (
    <Responsive height={h}>
      {(width) => {
        const plot = makePlot(width, h, margin);
        const values = data.map((d) => d.value);
        const lo = Math.min(0, ...values);
        const x = linearScale(domain ?? [lo, Math.max(...values, 0) || 1], [0, plot.innerWidth]);
        const zero = x(0);
        const band = plot.innerHeight / Math.max(1, data.length);
        const barH = Math.max(6, Math.min(20, band - GAP * 2));

        return (
          <svg className="chart" width={width} height={h} role="img"
               aria-label={`Horizontal bar chart of ${valueLabel}`}>
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <g className="chart__grid">
                {x.ticks(4).map((t) => (
                  <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={plot.innerHeight} />
                ))}
              </g>
              {data.map((d, i) => {
                const y = band * i + (band - barH) / 2;
                const w = Math.abs(x(d.value) - zero);
                const bx = d.value >= 0 ? zero : zero - w;
                const color = d.color ?? seriesColor(0);
                return (
                  <g key={`${d.label}-${i}`}>
                    <text
                      className="chart__tick"
                      x={-10}
                      y={y + barH / 2}
                      dy="0.32em"
                      textAnchor="end"
                      style={{ fill: "var(--ink-secondary)" }}
                    >
                      {d.label.length > 26 ? `${d.label.slice(0, 25)}…` : d.label}
                    </text>
                    <path
                      d={
                        d.value >= 0
                          ? barPathH(bx, y, w, barH)
                          : barPathH(bx, y, w, barH)
                      }
                      fill={color}
                      onMouseMove={(e) =>
                        tip.show(
                          e.nativeEvent.offsetX,
                          e.nativeEvent.offsetY,
                          <>
                            <span className="tooltip__title">{d.label}</span>
                            <TipRow k={valueLabel} v={format(d.value)} />
                            {d.note && (
                              <div style={{ marginTop: 4, color: "var(--ink-secondary)" }}>
                                {d.note}
                              </div>
                            )}
                          </>,
                        )
                      }
                      onMouseLeave={tip.hide}
                    />
                    {showValues && (
                      <text
                        className="chart__label"
                        x={(d.value >= 0 ? bx + w : bx) + (d.value >= 0 ? 7 : -7)}
                        y={y + barH / 2}
                        dy="0.32em"
                        textAnchor={d.value >= 0 ? "start" : "end"}
                      >
                        {format(d.value)}
                      </text>
                    )}
                  </g>
                );
              })}
              <line
                x1={zero}
                x2={zero}
                y1={0}
                y2={plot.innerHeight}
                stroke="var(--axis)"
                strokeWidth={1}
              />
              <AxisX scale={x} plot={plot} count={4} format={(v) => compact(v)} />
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

export function BarChartV({
  data,
  height = 220,
  format = (v: number) => num(v, 2),
  valueLabel = "Value",
  yLabel,
  rotateLabels = false,
  labelEvery = 1,
}: {
  data: BarDatum[];
  height?: number;
  format?: (v: number) => string;
  valueLabel?: string;
  yLabel?: string;
  rotateLabels?: boolean;
  labelEvery?: number;
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        const plot = makePlot(width, height, {
          bottom: rotateLabels ? 62 : 34,
          left: yLabel ? 60 : 48,
        });
        const values = data.map((d) => d.value);
        const y = linearScale(
          [Math.min(0, ...values), Math.max(...values, 0) || 1],
          [plot.innerHeight, 0],
        );
        const band = plot.innerWidth / Math.max(1, data.length);
        const barW = Math.max(2, band - GAP * 2);

        return (
          <svg className="chart" width={width} height={height} role="img"
               aria-label={`Bar chart of ${valueLabel}`}>
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <GridY scale={y} plot={plot} />
              <AxisY scale={y} plot={plot} format={(v) => compact(v)} label={yLabel} />
              {data.map((d, i) => {
                const top = y(Math.max(0, d.value));
                const h = Math.abs(y(d.value) - y(0));
                return (
                  <path
                    key={`${d.label}-${i}`}
                    d={barPathV(band * i + GAP, top, barW, h)}
                    fill={d.color ?? seriesColor(0)}
                    onMouseMove={(e) =>
                      tip.show(
                        e.nativeEvent.offsetX,
                        e.nativeEvent.offsetY,
                        <>
                          <span className="tooltip__title">{d.label}</span>
                          <TipRow k={valueLabel} v={format(d.value)} />
                          {d.note && (
                            <div style={{ marginTop: 4, color: "var(--ink-secondary)" }}>
                              {d.note}
                            </div>
                          )}
                        </>,
                      )
                    }
                    onMouseLeave={tip.hide}
                  />
                );
              })}
              <AxisXCategorical
                labels={data.map((d) => d.label)}
                plot={plot}
                rotate={rotateLabels}
                every={labelEvery}
              />
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/** Grouped bars: one group per category, one bar per series. */
export function GroupedBars({
  categories,
  series,
  height = 240,
  format = (v: number) => num(v, 3),
  yLabel,
  rotateLabels = false,
}: {
  categories: string[];
  series: { name: string; values: (number | null)[]; color?: string }[];
  height?: number;
  format?: (v: number) => string;
  yLabel?: string;
  rotateLabels?: boolean;
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        const plot = makePlot(width, height, {
          bottom: rotateLabels ? 66 : 34,
          left: yLabel ? 60 : 48,
        });
        const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
        const y = linearScale([Math.min(0, ...all), Math.max(...all, 0) || 1], [plot.innerHeight, 0]);
        const band = plot.innerWidth / Math.max(1, categories.length);
        const groupW = band - GAP * 3;
        const barW = Math.max(2, groupW / series.length - GAP);

        return (
          <svg className="chart" width={width} height={height} role="img" aria-label="Grouped bar chart">
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <GridY scale={y} plot={plot} />
              <AxisY scale={y} plot={plot} format={(v) => compact(v)} label={yLabel} />
              {categories.map((cat, ci) =>
                series.map((s, si) => {
                  const v = s.values[ci];
                  if (v === null || v === undefined || !Number.isFinite(v)) return null;
                  const x = band * ci + GAP * 1.5 + si * (barW + GAP);
                  const top = y(Math.max(0, v));
                  const h = Math.abs(y(v) - y(0));
                  return (
                    <path
                      key={`${cat}-${s.name}`}
                      d={barPathV(x, top, barW, h)}
                      fill={s.color ?? seriesColor(si)}
                      onMouseMove={(e) =>
                        tip.show(
                          e.nativeEvent.offsetX,
                          e.nativeEvent.offsetY,
                          <>
                            <span className="tooltip__title">{cat}</span>
                            <TipRow k={s.name} v={format(v)} />
                          </>,
                        )
                      }
                      onMouseLeave={tip.hide}
                    />
                  );
                }),
              )}
              <AxisXCategorical labels={categories} plot={plot} rotate={rotateLabels} />
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/* ============================================================== line ==== */
export interface LineSeries {
  name: string;
  points: { x: number; y: number | null }[];
  color?: string;
  dashed?: boolean;
}

export function LineChart({
  series,
  height = 260,
  xLabel,
  yLabel,
  xFormat = (v: number) => compact(v),
  yFormat,
  tipFormat = (v: number) => num(v, 3),
  band,
  yDomain,
  markers = false,
  reference,
}: {
  series: LineSeries[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  tipFormat?: (v: number) => string;
  band?: { points: { x: number; lo: number; hi: number }[]; color?: string; label?: string };
  yDomain?: [number, number];
  markers?: boolean;
  reference?: { y: number; label: string; color?: string };
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        const plot = makePlot(width, height, { left: yLabel ? 62 : 52, bottom: xLabel ? 44 : 34 });
        const xs = series.flatMap((s) => s.points.map((p) => p.x));
        const ys = series
          .flatMap((s) => s.points.map((p) => p.y))
          .filter((v): v is number => v !== null && Number.isFinite(v));
        const bandYs = band ? band.points.flatMap((p) => [p.lo, p.hi]) : [];
        const refYs = reference ? [reference.y] : [];
        const x = linearScale(extent(xs), [0, plot.innerWidth]);
        const y = linearScale(
          yDomain ?? extent([...ys, ...bandYs, ...refYs], 0.08),
          [plot.innerHeight, 0],
        );

        const onMove = (e: React.MouseEvent<SVGRectElement>) => {
          const px = e.nativeEvent.offsetX - plot.margin.left;
          const xv = x.invert(px);
          const rows = series
            .map((s) => {
              let best = s.points[0];
              let bestD = Infinity;
              for (const p of s.points) {
                const d = Math.abs(p.x - xv);
                if (d < bestD) {
                  bestD = d;
                  best = p;
                }
              }
              return { name: s.name, point: best, color: s.color };
            })
            .filter((r) => r.point && r.point.y !== null);
          if (!rows.length) return;
          tip.show(
            e.nativeEvent.offsetX,
            e.nativeEvent.offsetY,
            <>
              <span className="tooltip__title">{xFormat(rows[0].point.x)}</span>
              {rows.map((r, i) => (
                <TipRow
                  key={r.name}
                  k={r.name}
                  v={
                    <span style={{ color: r.color ?? seriesColor(i) }}>
                      {tipFormat(r.point.y as number)}
                    </span>
                  }
                />
              ))}
            </>,
          );
        };

        // Without an explicit formatter, derive the precision from the axis span
        // so a range like 0.907-0.931 does not render six ticks all reading "0.9".
        const fmtY = yFormat ?? autoFormat(y.domain);

        return (
          <svg className="chart" width={width} height={height} role="img"
               aria-label={`Line chart${yLabel ? ` of ${yLabel}` : ""}`}>
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <GridY scale={y} plot={plot} />
              <AxisY scale={y} plot={plot} format={fmtY} label={yLabel} />
              <AxisX scale={x} plot={plot} format={xFormat} label={xLabel} />

              {band && (
                <path
                  d={areaPath(band.points.map((p) => ({ x: x(p.x), y0: y(p.lo), y1: y(p.hi) })))}
                  fill={band.color ?? seriesColor(0)}
                  opacity={0.16}
                />
              )}

              {reference && (
                <g>
                  <line
                    x1={0}
                    x2={plot.innerWidth}
                    y1={y(reference.y)}
                    y2={y(reference.y)}
                    stroke={reference.color ?? "var(--ink-muted)"}
                    strokeWidth={1.5}
                    strokeDasharray="5 4"
                  />
                  <text
                    className="chart__tick"
                    x={plot.innerWidth - 4}
                    y={y(reference.y) - 5}
                    textAnchor="end"
                    style={{ fill: reference.color ?? "var(--ink-muted)", fontWeight: 600 }}
                  >
                    {reference.label}
                  </text>
                </g>
              )}

              {series.map((s, i) => {
                const pts = s.points
                  .filter((p) => p.y !== null && Number.isFinite(p.y))
                  .map((p) => ({ x: x(p.x), y: y(p.y as number) }));
                const color = s.color ?? seriesColor(i);
                return (
                  <Fragment key={s.name}>
                    <path
                      d={linePath(pts)}
                      fill="none"
                      stroke={color}
                      strokeWidth={2}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                      strokeDasharray={s.dashed ? "6 4" : undefined}
                    />
                    {markers &&
                      pts.map((p, j) => (
                        <circle
                          key={j}
                          cx={p.x}
                          cy={p.y}
                          r={4}
                          fill={color}
                          stroke="var(--surface-1)"
                          strokeWidth={2}
                        />
                      ))}
                  </Fragment>
                );
              })}

              <rect
                x={0}
                y={0}
                width={plot.innerWidth}
                height={plot.innerHeight}
                fill="transparent"
                onMouseMove={onMove}
                onMouseLeave={tip.hide}
              />
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/* ============================================================ scatter ==== */
export function ScatterPlot({
  points,
  height = 280,
  xLabel,
  yLabel,
  colorOf,
  identity,
  diagonal = false,
  tipContent,
}: {
  points: { x: number; y: number; group?: string | number; meta?: Record<string, unknown> }[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  colorOf?: (p: { group?: string | number }) => string;
  identity?: boolean;
  diagonal?: boolean;
  tipContent?: (p: { x: number; y: number; group?: string | number; meta?: Record<string, unknown> }) => ReactNode;
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        const plot = makePlot(width, height, { left: 58, bottom: 42 });
        const xs = points.map((p) => p.x);
        const ys = points.map((p) => p.y);
        const dom: [number, number] = identity
          ? [Math.min(...xs, ...ys), Math.max(...xs, ...ys)]
          : [0, 1];
        const x = linearScale(identity ? dom : extent(xs, 0.05), [0, plot.innerWidth]);
        const y = linearScale(identity ? dom : extent(ys, 0.05), [plot.innerHeight, 0]);

        return (
          <svg className="chart" width={width} height={height} role="img" aria-label="Scatter plot">
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <GridY scale={y} plot={plot} />
              <AxisY scale={y} plot={plot} format={(v) => compact(v)} label={yLabel} />
              <AxisX scale={x} plot={plot} format={(v) => compact(v)} label={xLabel} />
              {(identity || diagonal) && (
                <line
                  x1={x(dom[0])}
                  y1={y(dom[0])}
                  x2={x(dom[1])}
                  y2={y(dom[1])}
                  stroke="var(--ink-muted)"
                  strokeWidth={1.5}
                  strokeDasharray="5 4"
                />
              )}
              {points.map((p, i) => (
                <circle
                  key={i}
                  cx={x(p.x)}
                  cy={y(p.y)}
                  r={4}
                  fill={colorOf ? colorOf(p) : seriesColor(0)}
                  opacity={0.62}
                  onMouseMove={(e) =>
                    tip.show(
                      e.nativeEvent.offsetX,
                      e.nativeEvent.offsetY,
                      tipContent ? (
                        tipContent(p)
                      ) : (
                        <>
                          <TipRow k={xLabel ?? "x"} v={num(p.x, 2)} />
                          <TipRow k={yLabel ?? "y"} v={num(p.y, 2)} />
                        </>
                      ),
                    )
                  }
                  onMouseLeave={tip.hide}
                />
              ))}
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/* ============================================================ histogram == */
export function Histogram({
  bins,
  height = 190,
  color,
  markers,
  xFormat = (v: number) => compact(v),
  xLabel,
}: {
  bins: { bin_start: number; bin_end: number; count: number }[];
  height?: number;
  color?: string;
  markers?: { value: number; label: string; color?: string }[];
  xFormat?: (v: number) => string;
  xLabel?: string;
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        const plot = makePlot(width, height, { left: 48, bottom: xLabel ? 44 : 32, right: 12 });
        const lo = bins.length ? bins[0].bin_start : 0;
        const hi = bins.length ? bins[bins.length - 1].bin_end : 1;
        const x = linearScale([lo, hi], [0, plot.innerWidth]);
        const y = linearScale([0, Math.max(...bins.map((b) => b.count), 1)], [plot.innerHeight, 0]);

        return (
          <svg className="chart" width={width} height={height} role="img" aria-label="Histogram">
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <GridY scale={y} plot={plot} count={4} />
              <AxisY scale={y} plot={plot} count={4} format={(v) => compact(v)} />
              {bins.map((b, i) => {
                const bx = x(b.bin_start);
                const bw = Math.max(1, x(b.bin_end) - x(b.bin_start) - GAP);
                const by = y(b.count);
                return (
                  <path
                    key={i}
                    d={barPathV(bx, by, bw, plot.innerHeight - by, 3)}
                    fill={color ?? seriesColor(0)}
                    onMouseMove={(e) =>
                      tip.show(
                        e.nativeEvent.offsetX,
                        e.nativeEvent.offsetY,
                        <>
                          <span className="tooltip__title">
                            {xFormat(b.bin_start)} – {xFormat(b.bin_end)}
                          </span>
                          <TipRow k="Count" v={compact(b.count)} />
                        </>,
                      )
                    }
                    onMouseLeave={tip.hide}
                  />
                );
              })}
              {markers?.map((m) => (
                <g key={m.label}>
                  <line
                    x1={x(m.value)}
                    x2={x(m.value)}
                    y1={0}
                    y2={plot.innerHeight}
                    stroke={m.color ?? "var(--ink-primary)"}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                  />
                  <text
                    className="chart__tick"
                    x={x(m.value)}
                    y={-1}
                    textAnchor="middle"
                    style={{ fill: m.color ?? "var(--ink-secondary)", fontWeight: 600 }}
                  >
                    {m.label}
                  </text>
                </g>
              ))}
              <AxisX scale={x} plot={plot} count={5} format={xFormat} label={xLabel} />
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/** Two-class histogram: the separation plot for a classifier's scores. */
export function SeparationHistogram({
  bins,
  height = 200,
  threshold,
}: {
  bins: { bin_start: number; bin_end: number; negatives: number; positives: number }[];
  height?: number;
  threshold?: number;
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        const plot = makePlot(width, height, { left: 50, bottom: 40 });
        const x = linearScale([0, 1], [0, plot.innerWidth]);
        const max = Math.max(...bins.map((b) => Math.max(b.negatives, b.positives)), 1);
        const y = linearScale([0, max], [plot.innerHeight, 0]);
        const bw = plot.innerWidth / Math.max(1, bins.length);
        const half = Math.max(1.5, bw / 2 - GAP);

        return (
          <svg className="chart" width={width} height={height} role="img"
               aria-label="Predicted score distribution by true class">
            <g transform={`translate(${plot.margin.left}, ${plot.margin.top})`}>
              <GridY scale={y} plot={plot} count={4} />
              <AxisY scale={y} plot={plot} count={4} format={(v) => compact(v)} />
              {bins.map((b, i) => (
                <g
                  key={i}
                  onMouseMove={(e) =>
                    tip.show(
                      e.nativeEvent.offsetX,
                      e.nativeEvent.offsetY,
                      <>
                        <span className="tooltip__title">
                          score {b.bin_start.toFixed(2)}–{b.bin_end.toFixed(2)}
                        </span>
                        <TipRow k="Negatives" v={compact(b.negatives)} />
                        <TipRow k="Positives" v={compact(b.positives)} />
                      </>,
                    )
                  }
                  onMouseLeave={tip.hide}
                >
                  <path
                    d={barPathV(bw * i + GAP / 2, y(b.negatives), half, plot.innerHeight - y(b.negatives), 3)}
                    fill={seriesColor(0)}
                  />
                  <path
                    d={barPathV(bw * i + half + GAP * 1.5, y(b.positives), half, plot.innerHeight - y(b.positives), 3)}
                    fill={seriesColor(1)}
                  />
                </g>
              ))}
              {threshold !== undefined && (
                <g>
                  <line
                    x1={x(threshold)}
                    x2={x(threshold)}
                    y1={0}
                    y2={plot.innerHeight}
                    stroke="var(--ink-primary)"
                    strokeWidth={2}
                    strokeDasharray="4 3"
                  />
                  <text className="chart__label" x={x(threshold)} y={-2} textAnchor="middle">
                    threshold {threshold.toFixed(2)}
                  </text>
                </g>
              )}
              <AxisX scale={x} plot={plot} count={5} format={(v) => v.toFixed(1)}
                     label="Predicted probability" />
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/* ============================================================== heatmap == */
export function Heatmap({
  rows,
  cols,
  matrix,
  height,
  valueFormat = (v: number) => num(v, 2),
  colorOf = (t: number) => sequential(t),
  title,
  cellLabels = false,
}: {
  rows: string[];
  cols: (string | number)[];
  matrix: (number | null)[][];
  height?: number;
  valueFormat?: (v: number) => string;
  colorOf?: (t: number, v: number) => string;
  title?: string;
  cellLabels?: boolean;
}) {
  const tip = useTooltipTarget();
  const flat = matrix.flat().filter((v): v is number => v !== null && Number.isFinite(v));
  const min = Math.min(...flat, 0);
  const max = Math.max(...flat, 1);
  const h = height ?? Math.max(150, rows.length * 26 + 48);

  return (
    <Responsive height={h}>
      {(width) => {
        const left = 62;
        const top = 20;
        const cellW = (width - left - 8) / Math.max(1, cols.length);
        const cellH = (h - top - 24) / Math.max(1, rows.length);
        return (
          <svg className="chart" width={width} height={h} role="img"
               aria-label={title ?? "Heat map"}>
            <g transform={`translate(${left}, ${top})`}>
              {cols.map((c, j) =>
                j % Math.ceil(cols.length / 12) === 0 ? (
                  <text key={`c${j}`} className="chart__tick" x={cellW * (j + 0.5)} y={-6}
                        textAnchor="middle">
                    {c}
                  </text>
                ) : null,
              )}
              {rows.map((r, i) => (
                <text key={`r${i}`} className="chart__tick" x={-8} y={cellH * (i + 0.5)}
                      dy="0.32em" textAnchor="end" style={{ fill: "var(--ink-secondary)" }}>
                  {r}
                </text>
              ))}
              {matrix.map((row, i) =>
                row.map((v, j) => {
                  const t = v === null ? 0 : (v - min) / (max - min || 1);
                  return (
                    <g key={`${i}-${j}`}>
                      <rect
                        x={cellW * j + 1}
                        y={cellH * i + 1}
                        width={Math.max(1, cellW - GAP)}
                        height={Math.max(1, cellH - GAP)}
                        rx={2}
                        fill={v === null ? "var(--surface-2)" : colorOf(t, v)}
                        onMouseMove={(e) =>
                          tip.show(
                            e.nativeEvent.offsetX,
                            e.nativeEvent.offsetY,
                            <>
                              <span className="tooltip__title">
                                {r_label(rows, i)} · {cols[j]}
                              </span>
                              <TipRow k="Value" v={v === null ? "—" : valueFormat(v)} />
                            </>,
                          )
                        }
                        onMouseLeave={tip.hide}
                      />
                      {cellLabels && v !== null && cellW > 42 && (
                        <text
                          className="chart__tick"
                          x={cellW * (j + 0.5)}
                          y={cellH * (i + 0.5)}
                          dy="0.32em"
                          textAnchor="middle"
                          style={{ fill: t > 0.55 ? "#fff" : "var(--ink-primary)", fontWeight: 600 }}
                        >
                          {valueFormat(v)}
                        </text>
                      )}
                    </g>
                  );
                }),
              )}
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

function r_label(rows: string[], i: number): string {
  return rows[i] ?? String(i);
}

/** Correlation matrix uses the diverging ramp: sign is the point. */
export function CorrelationMatrix({
  columns,
  matrix,
  height,
}: {
  columns: string[];
  matrix: number[][];
  height?: number;
}) {
  return (
    <Heatmap
      rows={columns}
      cols={columns}
      matrix={matrix}
      height={height ?? Math.max(200, columns.length * 24 + 60)}
      colorOf={(_t, v) => diverging(v)}
      valueFormat={(v) => v.toFixed(2)}
      cellLabels
      title="Correlation matrix"
    />
  );
}

/* ============================================================ waterfall == */
export function Waterfall({
  base,
  contributions,
  prediction,
  height,
  format = (v: number) => num(v, 3),
  unit = "",
}: {
  base: number;
  contributions: { feature: string; value?: unknown; shap: number }[];
  prediction: number;
  height?: number;
  format?: (v: number) => string;
  unit?: string;
}) {
  const tip = useTooltipTarget();
  const rowH = 26;
  const h = height ?? contributions.length * rowH + 70;

  return (
    <Responsive height={h}>
      {(width) => {
        const left = 178;
        const right = 74;
        const inner = Math.max(60, width - left - right);
        // Running total from the base value, so each bar starts where the last
        // one ended -- that is what makes the additivity of SHAP visible.
        let running = base;
        const steps = contributions.map((c) => {
          const from = running;
          running += c.shap;
          return { ...c, from, to: running };
        });
        const all = [base, prediction, ...steps.flatMap((s) => [s.from, s.to])];
        const x = linearScale(extent(all, 0.1), [0, inner]);

        return (
          <svg className="chart" width={width} height={h} role="img"
               aria-label="SHAP waterfall">
            <g transform={`translate(${left}, 26)`}>
              <line x1={x(base)} x2={x(base)} y1={-16} y2={steps.length * rowH + 4}
                    stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="3 3" />
              <text className="chart__tick" x={x(base)} y={-20} textAnchor="middle">
                base {format(base)}
              </text>
              {steps.map((s, i) => {
                const y = i * rowH + 4;
                const bh = rowH - 9;
                const x0 = Math.min(x(s.from), x(s.to));
                const w = Math.abs(x(s.to) - x(s.from));
                const positive = s.shap >= 0;
                return (
                  <g key={`${s.feature}-${i}`}>
                    <text className="chart__tick" x={-10} y={y + bh / 2} dy="0.32em"
                          textAnchor="end" style={{ fill: "var(--ink-secondary)" }}>
                      {s.feature.length > 24 ? `${s.feature.slice(0, 23)}…` : s.feature}
                    </text>
                    <rect
                      x={x0}
                      y={y}
                      width={Math.max(1.5, w)}
                      height={bh}
                      rx={3}
                      fill={positive ? seriesColor(0) : seriesColor(7)}
                      onMouseMove={(e) =>
                        tip.show(
                          e.nativeEvent.offsetX,
                          e.nativeEvent.offsetY,
                          <>
                            <span className="tooltip__title">{s.feature}</span>
                            {s.value !== undefined && (
                              <TipRow k="Value" v={String(s.value)} />
                            )}
                            <TipRow k="Contribution" v={`${s.shap >= 0 ? "+" : ""}${format(s.shap)}`} />
                            <TipRow k="Running total" v={format(s.to)} />
                          </>,
                        )
                      }
                      onMouseLeave={tip.hide}
                    />
                    <text
                      className="chart__label"
                      x={positive ? x0 + w + 6 : x0 - 6}
                      y={y + bh / 2}
                      dy="0.32em"
                      textAnchor={positive ? "start" : "end"}
                    >
                      {s.shap >= 0 ? "+" : ""}
                      {format(s.shap)}
                    </text>
                  </g>
                );
              })}
              <line
                x1={x(prediction)}
                x2={x(prediction)}
                y1={-16}
                y2={steps.length * rowH + 16}
                stroke="var(--ink-primary)"
                strokeWidth={2}
              />
              <text className="chart__label" x={x(prediction)} y={steps.length * rowH + 30}
                    textAnchor="middle">
                prediction {format(prediction)}{unit}
              </text>
            </g>
          </svg>
        );
      }}
    </Responsive>
  );
}

/* ============================================================== network == */
export function NetworkGraph({
  nodes,
  links,
  height = 400,
}: {
  nodes: { id: string; degree: number; support?: number }[];
  links: { source: string; target: string; lift: number; confidence?: number }[];
  height?: number;
}) {
  const tip = useTooltipTarget();
  return (
    <Responsive height={height}>
      {(width) => {
        // A deterministic radial layout rather than a physics simulation: it is
        // reproducible across renders (so a screenshot always matches), it never
        // jitters, and for a filtered rule set the ordering by degree is more
        // legible than whatever a force layout settles into.
        const cx = width / 2;
        const cy = height / 2;
        const ordered = [...nodes].sort((a, b) => b.degree - a.degree);
        const positions = new Map<string, { x: number; y: number; r: number }>();
        const ringSize = [1, 8, 16, 28];
        let index = 0;
        let ring = 0;
        while (index < ordered.length && ring < 8) {
          const count = ringSize[Math.min(ring, ringSize.length - 1)] || 28;
          const slice = ordered.slice(index, index + count);
          const radius = ring === 0 ? 0 : (Math.min(width, height) / 2 - 44) * (ring / 3.2);
          slice.forEach((n, i) => {
            const angle = (i / slice.length) * Math.PI * 2 - Math.PI / 2 + ring * 0.4;
            positions.set(n.id, {
              x: cx + Math.cos(angle) * radius,
              y: cy + Math.sin(angle) * radius,
              r: Math.max(4, Math.min(11, 3 + Math.sqrt(n.degree) * 1.6)),
            });
          });
          index += slice.length;
          ring += 1;
        }
        const maxLift = Math.max(...links.map((l) => l.lift), 1);

        return (
          <svg className="chart" width={width} height={height} role="img"
               aria-label="Association rule network">
            {links.map((l, i) => {
              const a = positions.get(l.source);
              const b = positions.get(l.target);
              if (!a || !b) return null;
              return (
                <line
                  key={i}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={seriesColor(2)}
                  strokeWidth={1 + (l.lift / maxLift) * 2.5}
                  opacity={0.28 + (l.lift / maxLift) * 0.4}
                  onMouseMove={(e) =>
                    tip.show(
                      e.nativeEvent.offsetX,
                      e.nativeEvent.offsetY,
                      <>
                        <span className="tooltip__title">
                          {l.source} → {l.target}
                        </span>
                        <TipRow k="Lift" v={num(l.lift, 2)} />
                        {l.confidence !== undefined && (
                          <TipRow k="Confidence" v={pct(l.confidence)} />
                        )}
                      </>,
                    )
                  }
                  onMouseLeave={tip.hide}
                />
              );
            })}
            {ordered.map((n) => {
              const p = positions.get(n.id);
              if (!p) return null;
              return (
                <circle
                  key={n.id}
                  cx={p.x}
                  cy={p.y}
                  r={p.r}
                  fill={seriesColor(0)}
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                  onMouseMove={(e) =>
                    tip.show(
                      e.nativeEvent.offsetX,
                      e.nativeEvent.offsetY,
                      <>
                        <span className="tooltip__title">{n.id}</span>
                        <TipRow k="Rules touching it" v={String(n.degree)} />
                        {n.support !== undefined && <TipRow k="Support" v={pct(n.support, 2)} />}
                      </>,
                    )
                  }
                  onMouseLeave={tip.hide}
                />
              );
            })}
          </svg>
        );
      }}
    </Responsive>
  );
}

/* ============================================================== gauges ==== */
export function ScoreBar({
  value,
  color,
  height = 8,
  max = 1,
}: {
  value: number;
  color?: string;
  height?: number;
  max?: number;
}) {
  const t = Math.max(0, Math.min(1, value / max));
  return (
    <div
      style={{
        background: "var(--surface-3)",
        borderRadius: height / 2,
        height,
        overflow: "hidden",
        width: "100%",
      }}
    >
      <div
        style={{
          width: `${t * 100}%`,
          height: "100%",
          background: color ?? seriesColor(0),
          borderRadius: height / 2,
          transition: "width 0.25s ease",
        }}
      />
    </div>
  );
}

export { ChartFrame, Responsive, TipRow };
