import { useEffect, useState } from "react";
import { useApi, useMutation } from "../../lib/api";
import { compact, num, pct, seriesColor, usd } from "../../lib/format";
import { ChartFrame, ScatterPlot } from "../../charts/Charts";
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

interface Profile {
  cluster: number;
  name: string;
  recommended_action: string;
  size: number;
  share: number;
  revenue: number;
  revenue_share: number;
  median: Record<string, number>;
  vs_population: Record<string, number>;
  mean_silhouette: number;
}

interface Prediction {
  cluster: number;
  segment: string;
  recommended_action: string | null;
  confidence: string;
  distance_ratio: number;
  distances: { cluster: number; distance: number; segment: string }[];
  segment_profile: Profile | null;
  caveat: string;
}

export default function SegmentsLive() {
  const profiles = useApi<{ k: number; champion: string; profiles: Profile[]; rationale: string }>(
    "/projects/segments/profiles",
  );
  const scatter = useApi<{ points: any[]; projection: any }>("/projects/segments/scatter");
  const predict = useMutation<Record<string, unknown>, Prediction>("/projects/segments/predict");

  const [recency, setRecency] = useState(30);
  const [frequency, setFrequency] = useState(5);
  const [monetary, setMonetary] = useState(1500);
  const [skus, setSkus] = useState(35);

  useEffect(() => {
    predict.run({
      recency_days: recency,
      frequency,
      monetary,
      n_distinct_skus: skus,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recency, frequency, monetary, skus]);

  if (profiles.status === "loading") return <Loading what="the segment profiles" />;
  if (profiles.status === "error") return <ErrorState error={profiles.error} onRetry={profiles.reload} />;

  const p = predict.data;

  return (
    <>
      <Section
        title="Assign a customer to a segment"
        description="Nearest centroid in log-scaled RFM space. The confidence measure is the ratio between the closest and second-closest centroid — a customer sitting between two segments should get generic treatment, not a wrong label."
      >
        <div className="grid grid--3">
          <div className="card">
            <SliderField
              label="Recency (days since last order)"
              value={recency}
              min={1}
              max={365}
              onChange={setRecency}
              format={(v) => `${v} days`}
            />
            <SliderField
              label="Frequency (distinct orders)"
              value={frequency}
              min={1}
              max={40}
              onChange={setFrequency}
            />
            <SliderField
              label="Monetary (gross spend)"
              value={monetary}
              min={50}
              max={15000}
              step={50}
              onChange={setMonetary}
              format={(v) => usd(v)}
            />
            <SliderField
              label="Distinct products bought"
              value={skus}
              min={1}
              max={250}
              onChange={setSkus}
            />
          </div>

          <div className="card">
            {p && (
              <>
                <Stat label="Assigned segment" value={p.segment} hero tone="accent" />
                <div style={{ margin: "14px 0" }}>
                  <Badge
                    tone={
                      p.confidence === "high" ? "good" : p.confidence === "medium" ? "warning" : "critical"
                    }
                    icon={p.confidence === "borderline" ? "⚠" : "✓"}
                  >
                    {p.confidence} confidence
                  </Badge>
                  <span className="small muted" style={{ marginLeft: 10 }}>
                    distance ratio {num(p.distance_ratio, 3)}
                  </span>
                </div>
                {p.recommended_action && (
                  <Callout tone="info" title="Recommended action">
                    {p.recommended_action}
                  </Callout>
                )}
                <Callout tone={p.confidence === "borderline" ? "warning" : "good"}>
                  {p.caveat}
                </Callout>
              </>
            )}
          </div>

          <div className="card">
            {p && (
              <>
                <div className="chartframe__title" style={{ marginBottom: 10 }}>
                  Distance to every centroid
                </div>
                <DataTable
                  columns={[
                    {
                      key: "s",
                      header: "Segment",
                      render: (r: any, i: number) => (
                        <span style={{ color: i === 0 ? "var(--accent, var(--series-1))" : undefined, fontWeight: i === 0 ? 600 : 400 }}>
                          {i === 0 && "★ "}
                          {r.segment}
                        </span>
                      ),
                    },
                    { key: "d", header: "Distance", numeric: true, render: (r: any) => num(r.distance, 3) },
                  ]}
                  rows={p.distances}
                />
                <div className="small muted" style={{ marginTop: 10, lineHeight: 1.5 }}>
                  A near-tie between the top two rows is what "borderline" means: the
                  customer genuinely sits on a boundary rather than the model being unsure.
                </div>
              </>
            )}
          </div>
        </div>
      </Section>

      <Section
        title="The segments"
        description={profiles.data.rationale}
        aside={`k=${profiles.data.k} · ${profiles.data.champion}`}
      >
        <div className="grid grid--2">
          {profiles.data.profiles.map((prof, i) => (
            <div
              className="card"
              key={prof.cluster}
              style={{
                borderLeft: `3px solid ${seriesColor(i)}`,
                outline: p?.cluster === prof.cluster ? "1px solid var(--accent, var(--series-1))" : undefined,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
                <h3 style={{ fontSize: "var(--text-md)" }}>{prof.name}</h3>
                <Badge tone="accent">{pct(prof.revenue_share)} of revenue</Badge>
              </div>
              <div className="grid grid--4" style={{ gap: 8, margin: "12px 0" }}>
                <Stat label="Customers" value={compact(prof.size)} note={pct(prof.share)} />
                <Stat label="Recency" value={`${num(prof.median.recency_days, 0)}d`} />
                <Stat label="Frequency" value={num(prof.median.frequency, 0)} />
                <Stat label="Monetary" value={usd(prof.median.monetary)} />
              </div>
              <p className="small" style={{ color: "var(--ink-secondary)", margin: 0, lineHeight: 1.55 }}>
                {prof.recommended_action}
              </p>
              <div className="small muted" style={{ marginTop: 8 }}>
                Mean silhouette {num(prof.mean_silhouette, 3)} · spends{" "}
                {num(prof.vs_population.monetary, 1)}× the median customer
              </div>
            </div>
          ))}
        </div>
      </Section>

      {scatter.status === "ready" && scatter.data.points.length > 0 && (
        <Section
          title="Segment map"
          description={scatter.data.projection?.why}
          aside={`PCA explains ${pct(scatter.data.projection?.total_explained)} of variance`}
        >
          <ChartFrame
            title="Customers projected onto the first two principal components"
            subtitle="Colour is the assigned segment. Because this is a linear projection, apparent separation here is real separation in the feature space — which is exactly why PCA was chosen over t-SNE for the primary view."
            legend={profiles.data.profiles.map((prof, i) => ({
              label: prof.name,
              color: seriesColor(i),
            }))}
          >
            <ScatterPlot
              height={420}
              points={scatter.data.points.map((pt) => ({
                x: pt.x,
                y: pt.y,
                group: pt.cluster,
                meta: pt,
              }))}
              xLabel="PC1"
              yLabel="PC2"
              colorOf={(pt) => {
                const idx = profiles.data.profiles.findIndex((pr) => pr.cluster === pt.group);
                return seriesColor(idx >= 0 ? idx : 0);
              }}
              tipContent={(pt) => (
                <>
                  <span className="tooltip__title">
                    {profiles.data.profiles.find((pr) => pr.cluster === pt.group)?.name ??
                      `Cluster ${pt.group}`}
                  </span>
                  <div className="tooltip__row">
                    <span className="tooltip__key">Recency</span>
                    <span>{num((pt.meta as any)?.recency, 0)} days</span>
                  </div>
                  <div className="tooltip__row">
                    <span className="tooltip__key">Frequency</span>
                    <span>{num((pt.meta as any)?.frequency, 0)}</span>
                  </div>
                  <div className="tooltip__row">
                    <span className="tooltip__key">Monetary</span>
                    <span>{usd((pt.meta as any)?.monetary)}</span>
                  </div>
                  <div className="tooltip__row">
                    <span className="tooltip__key">Silhouette</span>
                    <span>{num((pt.meta as any)?.silhouette, 3)}</span>
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
