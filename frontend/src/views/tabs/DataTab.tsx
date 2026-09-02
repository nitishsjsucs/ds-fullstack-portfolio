import { useState } from "react";
import { useApi } from "../../lib/api";
import { compact, num, pct, sequential, seriesColor } from "../../lib/format";
import type { CategoricalProfile, Eda, NumericProfile } from "../../lib/types";
import {
  BarChartH,
  ChartFrame,
  CorrelationMatrix,
  Heatmap,
  Histogram,
} from "../../charts/Charts";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  ScoreBarRow,
  Section,
  Stat,
} from "../../components/ui";

export default function DataTab({ slug }: { slug: string }) {
  const eda = useApi<Eda>(`/projects/${slug}/eda`);
  const [selected, setSelected] = useState<string | null>(null);

  if (eda.status === "loading") return <Loading what="the exploratory analysis" />;
  if (eda.status === "error") return <ErrorState error={eda.error} onRetry={eda.reload} />;

  const d = eda.data;
  const numeric = d.numeric ?? [];
  const categorical = d.categorical ?? [];
  const active = numeric.find((n) => n.column === selected) ?? numeric[0];

  return (
    <>
      <Section
        title="Data quality scorecard"
        description={d.quality?.note}
        aside={`grade ${d.quality?.grade}`}
      >
        <div className="grid grid--4" style={{ marginBottom: 16 }}>
          <Stat
            label="Overall quality"
            value={pct(d.quality?.overall_score)}
            tone={d.quality?.overall_score >= 0.95 ? "good" : d.quality?.overall_score >= 0.85 ? "warning" : "serious"}
            note={`Grade ${d.quality?.grade} across six dimensions`}
          />
          <Stat label="Rows" value={compact(d.shape?.rows)} note={`${d.shape?.columns} columns profiled`} />
          <Stat
            label="Missing cells"
            value={pct(d.missingness?.missing_pct, 2)}
            note={`${compact(d.missingness?.missing_cells)} of ${compact(d.missingness?.total_cells)}`}
          />
          <Stat
            label="Complete rows"
            value={pct(d.missingness?.complete_rows_pct)}
            note="No null in any column"
          />
        </div>

        <div className="card">
          {(d.quality?.dimensions ?? []).map((dim) => (
            <ScoreBarRow
              key={dim.dimension}
              label={dim.dimension}
              score={dim.score}
              detail={dim.detail}
            />
          ))}
        </div>
      </Section>

      {Array.isArray((d.exclusions as { rules?: unknown[] } | undefined)?.rules) && (
        <Section
          title="Row exclusions"
          description={(d.exclusions as { policy?: string }).policy}
        >
          <DataTable
            columns={[
              { key: "rule", header: "Rule", render: (r: any) => <code>{r.rule}</code> },
              { key: "expr", header: "Condition", render: (r: any) => <code className="small">{r.expression}</code> },
              { key: "n", header: "Rows removed", numeric: true, render: (r: any) => compact(r.rows_removed) },
              { key: "pct", header: "Share", numeric: true, render: (r: any) => pct(r.pct_removed, 3) },
              { key: "why", header: "Why", render: (r: any) => <span className="small">{r.reason}</span> },
            ]}
            rows={(d.exclusions as { rules: any[] }).rules}
          />
        </Section>
      )}

      {numeric.length > 0 && (
        <Section
          title="Feature distributions"
          description="Histograms are drawn over the 0.5th–99.5th percentile range so a single extreme value cannot compress every real bar into one bucket; the counts clipped at each end are reported below the chart. Dashed markers show the mean and median, and the Tukey fences mark the outlier boundary."
        >
          <div className="chiprow">
            {numeric.map((n) => (
              <button
                key={n.column}
                className={`chip${active?.column === n.column ? " chip--active" : ""}`}
                onClick={() => setSelected(n.column)}
              >
                {n.column}
              </button>
            ))}
          </div>
          {active && <NumericPanel profile={active} />}
        </Section>
      )}

      {d.correlation?.columns?.length > 1 && (
        <Section
          title="Correlation structure"
          description="Pearson measures linear association, Spearman measures monotone association. A pair that is strongly Spearman-correlated but weakly Pearson-correlated is a non-linear relationship — direct evidence that a tree model should beat a linear one."
        >
          <div className="grid grid--2">
            <ChartFrame
              title="Pearson (linear)"
              subtitle="Diverging ramp: blue is positive, red negative, grey near zero."
            >
              <CorrelationMatrix columns={d.correlation.columns} matrix={d.correlation.pearson} />
            </ChartFrame>
            <ChartFrame title="Spearman (monotone)" subtitle="Same pairs, rank-based.">
              <CorrelationMatrix columns={d.correlation.columns} matrix={d.correlation.spearman} />
            </ChartFrame>
          </div>
          {d.correlation.strong_pairs?.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <DataTable
                columns={[
                  { key: "pair", header: "Pair", render: (r) => `${r.a} ↔ ${r.b}` },
                  { key: "p", header: "Pearson", numeric: true, render: (r) => num(r.pearson, 3) },
                  { key: "s", header: "Spearman", numeric: true, render: (r) => num(r.spearman, 3) },
                  {
                    key: "gap",
                    header: "Non-linear gap",
                    numeric: true,
                    render: (r) =>
                      r.nonlinear_gap > 0.05 ? (
                        <Badge tone="warning">{num(r.nonlinear_gap, 3)}</Badge>
                      ) : (
                        num(r.nonlinear_gap, 3)
                      ),
                  },
                ]}
                rows={d.correlation.strong_pairs}
              />
            </div>
          )}
        </Section>
      )}

      {(d.temporal_heatmap || d.demand_heatmap) && (
        <Section
          title="Temporal rhythm"
          description="Day-of-week by hour-of-day. Sequential single-hue ramp: darker means larger."
        >
          <div className="grid grid--2">
            {d.demand_heatmap && (
              <ChartFrame
                title="Volume"
                subtitle={`Peak: ${d.demand_heatmap.peak_cell.day} ${d.demand_heatmap.peak_cell.hour}:00 (${compact(d.demand_heatmap.peak_cell.value)})`}
              >
                <Heatmap
                  rows={d.demand_heatmap.days}
                  cols={d.demand_heatmap.hours}
                  matrix={d.demand_heatmap.matrix}
                  valueFormat={(v) => compact(v)}
                  colorOf={(t) => sequential(t)}
                />
              </ChartFrame>
            )}
            {d.temporal_heatmap && (
              <ChartFrame
                title={`Mean ${d.temporal_heatmap.aggregation === "mean" ? "value" : d.temporal_heatmap.aggregation}`}
                subtitle={`Peak: ${d.temporal_heatmap.peak_cell.day} ${d.temporal_heatmap.peak_cell.hour}:00 (${num(d.temporal_heatmap.peak_cell.value, 1)})`}
              >
                <Heatmap
                  rows={d.temporal_heatmap.days}
                  cols={d.temporal_heatmap.hours}
                  matrix={d.temporal_heatmap.matrix}
                  valueFormat={(v) => num(v, 1)}
                  colorOf={(t) => sequential(t)}
                />
              </ChartFrame>
            )}
          </div>
        </Section>
      )}

      {categorical.length > 0 && (
        <Section
          title="Categorical fields"
          description="Top levels by share, with a flag on any column whose cardinality approaches the row count — that is the signature of an identifier, which must never become a feature."
        >
          <div className="grid grid--2">
            {categorical.slice(0, 8).map((c) => (
              <CategoricalPanel key={c.column} profile={c} />
            ))}
          </div>
        </Section>
      )}

      {d.missingness?.columns?.length > 0 && (
        <Section
          title="Missingness"
          description="Columns that go blank together usually share one upstream cause, which changes the remedy from per-column imputation to a structural fix."
        >
          <div className="grid grid--2">
            <ChartFrame title="Missing by column">
              <BarChartH
                data={d.missingness.columns.map((c) => ({
                  label: c.column,
                  value: c.missing_pct,
                  color: seriesColor(3),
                }))}
                format={(v) => pct(v, 2)}
                valueLabel="Missing"
              />
            </ChartFrame>
            {d.missingness.co_missing_pairs?.length > 0 && (
              <div className="card">
                <div className="chartframe__title" style={{ marginBottom: 8 }}>
                  Jointly missing pairs
                </div>
                <DataTable
                  columns={[
                    { key: "pair", header: "Columns", render: (r) => `${r.a} + ${r.b}` },
                    {
                      key: "pct",
                      header: "Both missing",
                      numeric: true,
                      render: (r) => pct(r.jointly_missing_pct, 3),
                    },
                  ]}
                  rows={d.missingness.co_missing_pairs}
                />
              </div>
            )}
          </div>
        </Section>
      )}

      {typeof d.label_policy === "string" && (
        <Callout tone="good" title="Label quarantine">
          {d.label_policy}
        </Callout>
      )}
    </>
  );
}

function NumericPanel({ profile }: { profile: NumericProfile }) {
  const clipped =
    profile.histogram_range.clipped_below + profile.histogram_range.clipped_above;
  return (
    <div className="grid grid--2">
      <ChartFrame
        title={profile.column}
        subtitle={`${compact(profile.count)} values · ${compact(profile.n_unique)} distinct · skew ${num(profile.skew, 2)}`}
        footnote={
          <>
            Tukey fences at [{num(profile.outlier_fences.lower, 2)},{" "}
            {num(profile.outlier_fences.upper, 2)}] flag {compact(profile.outlier_count)} values (
            {pct(profile.outlier_pct, 2)}).{" "}
            {clipped > 0 && (
              <>
                {compact(clipped)} values fall outside the plotted percentile range and are
                counted rather than drawn.
              </>
            )}
          </>
        }
      >
        <Histogram
          bins={profile.histogram}
          color={seriesColor(0)}
          markers={[
            { value: profile.mean, label: "mean", color: "var(--series-2)" },
            { value: profile.median, label: "median", color: "var(--ink-primary)" },
          ]}
          xFormat={(v) => compact(v)}
        />
      </ChartFrame>

      <div className="card">
        <div className="chartframe__title" style={{ marginBottom: 12 }}>
          Summary statistics
        </div>
        <div className="grid grid--2" style={{ gap: 8 }}>
          <Stat label="Mean" value={num(profile.mean, 2)} />
          <Stat label="Median" value={num(profile.median, 2)} />
          <Stat label="Std dev" value={num(profile.std, 2)} />
          <Stat label="IQR" value={num(profile.iqr, 2)} />
          <Stat label="Min" value={num(profile.min, 2)} />
          <Stat label="Max" value={num(profile.max, 2)} />
        </div>
        <hr className="divider" />
        <div className="kv small">
          {Object.entries(profile.percentiles).map(([k, v]) => (
            <div key={k} style={{ display: "contents" }}>
              <div className="kv__k">{k}</div>
              <div className="kv__v tnum">{num(v, 2)}</div>
            </div>
          ))}
          <div className="kv__k">Zeros</div>
          <div className="kv__v tnum">{compact(profile.zeros)}</div>
          <div className="kv__k">Negatives</div>
          <div className="kv__v tnum">{compact(profile.negatives)}</div>
          <div className="kv__k">Missing</div>
          <div className="kv__v tnum">{pct(profile.missing_pct, 2)}</div>
          <div className="kv__k">Kurtosis</div>
          <div className="kv__v tnum">{num(profile.kurtosis, 2)}</div>
        </div>
      </div>
    </div>
  );
}

function CategoricalPanel({ profile }: { profile: CategoricalProfile }) {
  return (
    <ChartFrame
      title={profile.column}
      subtitle={`${compact(profile.n_unique)} distinct · mode "${profile.mode ?? "—"}"`}
      aside={
        profile.is_high_cardinality ? (
          <Badge tone="critical" icon="⚠">identifier-like</Badge>
        ) : undefined
      }
      footnote={
        profile.tail_share > 0
          ? `The levels below the top ${profile.top_values.length} account for ${pct(profile.tail_share)} of rows.`
          : undefined
      }
    >
      <BarChartH
        data={profile.top_values.map((v) => ({
          label: v.value,
          value: v.share,
          color: seriesColor(2),
        }))}
        format={(v) => pct(v, 1)}
        valueLabel="Share"
        labelWidth={140}
      />
    </ChartFrame>
  );
}
