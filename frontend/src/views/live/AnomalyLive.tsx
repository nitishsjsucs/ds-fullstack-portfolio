import { useState } from "react";
import { useApi } from "../../lib/api";
import { compact, num, pct, seriesColor } from "../../lib/format";
import { ChartFrame, LineChart, ScatterPlot } from "../../charts/Charts";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  Section,
  SliderField,
  Stat,
} from "../../components/ui";

interface Sample {
  values: Record<string, string | number>;
  anomaly_score: number;
  true_label: string;
  family: string;
}

export default function AnomalyLive() {
  const samples = useApi<{ samples: Sample[]; attack_families: Record<string, number>; note: string }>(
    "/projects/anomaly/samples",
  );
  const budget = useApi<{ curve: any[]; per_family_recall: any[]; note: string }>(
    "/projects/anomaly/alert-budget",
  );
  const projection = useApi<any>("/projects/anomaly/projection");
  const [queue, setQueue] = useState(100);

  if (samples.status === "loading") return <Loading what="scored connections" />;
  if (samples.status === "error") return <ErrorState error={samples.error} onRetry={samples.reload} />;

  const rows = samples.data.samples;
  const curve = budget.status === "ready" ? budget.data.curve : [];
  const nearest =
    curve.length > 0
      ? curve.reduce((best, c) =>
          Math.abs(c.alert_budget - queue) < Math.abs(best.alert_budget - queue) ? c : best,
        )
      : null;

  return (
    <>
      <Section
        title="Analyst queue simulator"
        description="A security team reviews the top N alerts a shift can absorb, not 'everything above 0.5'. Drag the queue size and watch the trade-off: more alerts catch more intrusions, at a falling hit rate and a rising hour cost."
      >
        <div className="grid grid--3">
          <div className="card">
            <SliderField
              label="Alerts reviewed per run"
              value={queue}
              min={25}
              max={1600}
              step={25}
              onChange={setQueue}
              format={(v) => compact(v)}
              hint="Roughly six minutes of analyst time per alert."
            />
            {nearest && (
              <div className="grid grid--2" style={{ gap: 10, marginTop: 16 }}>
                <Stat label="Real intrusions caught" value={compact(nearest.true_positives)} tone="good" />
                <Stat label="False alarms" value={compact(nearest.false_positives)} tone="warning" />
                <Stat label="Precision" value={pct(nearest.precision_at_k)} />
                <Stat label="Recall" value={pct(nearest.recall_at_k)} />
                <Stat label="Missed attacks" value={compact(nearest.missed_attacks)} tone="critical" />
                <Stat label="Analyst hours" value={num(nearest.analyst_hours, 1)} />
              </div>
            )}
          </div>

          <div className="card" style={{ gridColumn: "span 2" }}>
            {curve.length > 0 && (
              <ChartFrame
                title="Precision and recall against queue size"
                subtitle="Precision falls as the queue grows because the clearest cases are already at the top. This curve is how you size a team for a target recall."
                legend={[
                  { label: "Precision@k", color: seriesColor(0), shape: "line" },
                  { label: "Recall@k", color: seriesColor(1), shape: "line" },
                ]}
              >
                <LineChart
                  height={260}
                  series={[
                    {
                      name: "Precision@k",
                      points: curve.map((c) => ({ x: c.alert_budget, y: c.precision_at_k })),
                      color: seriesColor(0),
                    },
                    {
                      name: "Recall@k",
                      points: curve.map((c) => ({ x: c.alert_budget, y: c.recall_at_k })),
                      color: seriesColor(1),
                    },
                  ]}
                  markers
                  xLabel="Alerts reviewed"
                  xFormat={(v) => compact(v)}
                  yFormat={(v) => v.toFixed(1)}
                  tipFormat={(v) => pct(v)}
                  yDomain={[0, 1]}
                />
              </ChartFrame>
            )}
          </div>
        </div>
      </Section>

      <Section
        title="Scored connections"
        description={samples.data.note}
      >
        <DataTable
          columns={[
            {
              key: "score",
              header: "Anomaly score",
              numeric: true,
              render: (r: Sample) => (
                <Badge tone={r.anomaly_score > 0.99 ? "critical" : r.anomaly_score > 0.9 ? "warning" : "neutral"}>
                  {num(r.anomaly_score, 4)}
                </Badge>
              ),
            },
            {
              key: "truth",
              header: "Ground truth",
              render: (r: Sample) =>
                r.true_label === "attack" ? (
                  <Badge tone="critical" icon="⚠">
                    {r.family}
                  </Badge>
                ) : (
                  <Badge tone="good" icon="✓">normal</Badge>
                ),
            },
            { key: "proto", header: "Protocol", render: (r: Sample) => String(r.values.protocol_type ?? "—") },
            { key: "svc", header: "Service", render: (r: Sample) => String(r.values.service ?? "—") },
            { key: "flag", header: "Flag", render: (r: Sample) => String(r.values.flag ?? "—") },
            { key: "src", header: "Src bytes", numeric: true, render: (r: Sample) => compact(r.values.src_bytes) },
            { key: "dst", header: "Dst bytes", numeric: true, render: (r: Sample) => compact(r.values.dst_bytes) },
            { key: "cnt", header: "Count", numeric: true, render: (r: Sample) => compact(r.values.count) },
            {
              key: "serr",
              header: "SYN err rate",
              numeric: true,
              render: (r: Sample) => num(r.values.serror_rate as number, 2),
            },
          ]}
          rows={rows}
          rowKey={(_, i) => String(i)}
        />
        <Callout tone="warning" title="An anomaly score is not a verdict">
          These scores measure deviation from the majority of traffic, not maliciousness.
          Where a high-scoring row is labelled normal, the detector is doing its job and
          the connection is genuinely unusual — triage is a human decision.
        </Callout>
      </Section>

      {budget.status === "ready" && budget.data.per_family_recall?.length > 0 && (
        <Section
          title="Which attacks stay invisible"
          description="The most operationally important table in the project: a strong average can coexist with an entire attack class never being surfaced."
        >
          <DataTable
            columns={[
              { key: "f", header: "Attack family", render: (r: any) => r.family },
              { key: "n", header: "Instances", numeric: true, render: (r: any) => compact(r.instances) },
              { key: "c", header: "Caught", numeric: true, render: (r: any) => compact(r.caught) },
              { key: "r", header: "Recall", numeric: true, render: (r: any) => pct(r.recall) },
              {
                key: "v",
                header: "Verdict",
                render: (r: any) => (
                  <Badge
                    tone={
                      r.verdict === "well detected"
                        ? "good"
                        : r.verdict === "partially detected"
                          ? "warning"
                          : "critical"
                    }
                    icon={r.verdict === "INVISIBLE" ? "✕" : undefined}
                  >
                    {r.verdict}
                  </Badge>
                ),
              },
            ]}
            rows={budget.data.per_family_recall}
          />
        </Section>
      )}

      {projection.status === "ready" && projection.data?.points?.length > 0 && (
        <Section
          title="Where detection is impossible"
          description={projection.data.note}
          aside={`PCA explains ${pct((projection.data.explained_variance ?? []).reduce((a: number, b: number) => a + b, 0))}`}
        >
          <ChartFrame
            title="Connections projected onto two principal components"
            legend={[
              { label: "Normal traffic", color: seriesColor(0) },
              { label: "Intrusion", color: seriesColor(7) },
            ]}
            footnote="Where red points sit inside the dense blue mass, no density-based detector can separate them — which is precisely why three of the five detectors are inverted on this data."
          >
            <ScatterPlot
              height={420}
              points={projection.data.points.map((pt: any) => ({
                x: pt.x,
                y: pt.y,
                group: pt.is_attack,
                meta: pt,
              }))}
              xLabel="PC1"
              yLabel="PC2"
              colorOf={(pt) => (pt.group === 1 ? seriesColor(7) : seriesColor(0))}
              tipContent={(pt) => (
                <>
                  <span className="tooltip__title">
                    {(pt.meta as any)?.is_attack ? "Intrusion" : "Normal traffic"}
                  </span>
                  <div className="tooltip__row">
                    <span className="tooltip__key">Anomaly score</span>
                    <span>{num((pt.meta as any)?.score, 3)}</span>
                  </div>
                </>
              )}
            />
          </ChartFrame>
        </Section>
      )}
    </>
  );
}
