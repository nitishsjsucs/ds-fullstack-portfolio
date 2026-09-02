import { useState } from "react";
import { useApi } from "../../lib/api";
import { compact, num, seriesColor } from "../../lib/format";
import type { Explain } from "../../lib/types";
import {
  BarChartH,
  ChartFrame,
  Heatmap,
  LineChart,
  ScatterPlot,
  Waterfall,
} from "../../charts/Charts";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  Section,
} from "../../components/ui";

export default function ExplainTab({ slug }: { slug: string }) {
  const ex = useApi<Explain>(`/projects/${slug}/explain`);

  if (ex.status === "loading") return <Loading what="explainability payloads" />;
  if (ex.status === "error") return <ErrorState error={ex.error} onRetry={ex.reload} />;

  const d = ex.data;

  return (
    <>
      {d.permutation && (
        <Section
          title="Permutation importance"
          description={d.permutation.note}
          aside={`${d.permutation.n_repeats} repeats · ${compact(d.permutation.n_samples)} held-out rows`}
        >
          <div className="grid grid--2">
            <ChartFrame
              title="How much does shuffling this feature cost?"
              subtitle="Measured on held-out rows, so it reflects generalisation rather than how often a split was taken. Features whose drop is smaller than twice their own noise band are marked as not distinguishable from irrelevant."
            >
              <BarChartH
                data={d.permutation.features.map((f) => ({
                  label: f.feature,
                  value: f.importance,
                  color: f.significant ? seriesColor(0) : "var(--surface-3)",
                  note: f.significant
                    ? `± ${num(f.std, 4)} across repeats`
                    : `Not significant: drop ${num(f.importance, 4)} vs noise ± ${num(f.std, 4)}`,
                }))}
                format={(v) => num(v, 4)}
                valueLabel={d.permutation.scoring}
                labelWidth={200}
              />
            </ChartFrame>

            {d.impurity ? (
              <ChartFrame
                title="Impurity importance, for contrast"
                subtitle={d.impurity.caveat}
              >
                <BarChartH
                  data={d.impurity.features.map((f) => ({
                    label: f.feature,
                    value: f.importance,
                    color: seriesColor(3),
                  }))}
                  format={(v) => num(v, 4)}
                  valueLabel="Mean impurity decrease"
                  labelWidth={200}
                />
              </ChartFrame>
            ) : (
              <div className="card">
                <Callout tone="info" title="No impurity importance">
                  The deployed estimator does not expose tree impurity importances, so only
                  the model-agnostic permutation measure is available.
                </Callout>
              </div>
            )}
          </div>
          <Callout tone="warning" title="Why both are shown">
            A feature that ranks high on impurity and near-zero on permutation is a
            memorisation smell: the model split on it constantly and gained nothing that
            generalises. Showing only the flattering one is how that goes unnoticed.
          </Callout>
        </Section>
      )}

      {d.shap && <ShapPanels shap={d.shap} />}

      {d.partial_dependence && d.partial_dependence.length > 0 && (
        <Section
          title="Partial dependence"
          description="The shape of the learned relationship, averaged over all other features. The rug marks the deciles of the feature's own distribution — where the ticks are sparse, the curve is extrapolating and should not be trusted."
        >
          <div className="grid grid--2">
            {d.partial_dependence.map((p) => (
              <ChartFrame
                key={p.feature}
                title={p.feature}
                subtitle={`Effect range ${num(p.effect_range, 3)}${p.monotone ? " · monotone" : " · non-monotone"}`}
                aside={p.monotone ? <Badge tone="good">monotone</Badge> : undefined}
              >
                <LineChart
                  height={200}
                  series={[
                    {
                      name: p.feature,
                      points: p.curve.map((c) => ({ x: c.x, y: c.y })),
                      color: seriesColor(0),
                    },
                  ]}
                  xLabel={p.feature}
                  xFormat={(v) => compact(v)}
                  tipFormat={(v) => num(v, 3)}
                />
              </ChartFrame>
            ))}
          </div>
        </Section>
      )}

      {d.architecture !== undefined && <ArchitecturePanel arch={d.architecture as any} />}
      {d.attention !== undefined && <AttentionPanel attention={d.attention as any} />}

      {Array.isArray(d.feature_importance) && (
        <Section
          title="Feature separation"
          description={String(d.note ?? "")}
        >
          <ChartFrame
            title="Eta-squared: between-cluster variance over total variance"
            subtitle="Unsupervised models have no target, so 'importance' means how strongly a feature separates the discovered groups."
          >
            <BarChartH
              data={(d.feature_importance as any[]).map((f) => ({
                label: f.feature,
                value: f.eta_squared,
                color: seriesColor(2),
                note: f.interpretation,
              }))}
              format={(v) => num(v, 3)}
              valueLabel="η²"
              labelWidth={180}
            />
          </ChartFrame>
        </Section>
      )}

      {Array.isArray(d.detector_assumptions) && (
        <Section
          title="What each detector believes an anomaly is"
          description="Five methods, five different definitions. Their disagreement is the most informative output of the project — 'anomalous' is not one concept."
        >
          <div className="grid grid--2">
            {(d.detector_assumptions as any[]).map((a, i) => (
              <div className="card" key={a.detector}>
                <h3 style={{ fontSize: "var(--text-md)", color: seriesColor(i), marginBottom: 8 }}>
                  {a.detector}
                </h3>
                <p className="small" style={{ color: "var(--ink-secondary)", marginBottom: 8 }}>
                  <strong style={{ color: "var(--ink-primary)" }}>Assumes:</strong> {a.assumption}
                </p>
                {a.blind_spot && (
                  <p className="small" style={{ color: "var(--status-serious)", margin: 0 }}>
                    <strong>Blind to:</strong> {a.blind_spot}
                  </p>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {Array.isArray(d.metric_glossary) && (
        <Section
          title="Reading the rule metrics"
          description={String(d.why_lift_over_confidence ?? "")}
        >
          <DataTable
            columns={[
              { key: "m", header: "Metric", render: (r: any) => <strong>{r.metric}</strong> },
              { key: "f", header: "Formula", render: (r: any) => <code className="small">{r.formula}</code> },
              { key: "r", header: "Reads as", render: (r: any) => <span className="small">{r.reads_as}</span> },
              {
                key: "w",
                header: "Watch out",
                render: (r: any) => (
                  <span className="small" style={{ color: "var(--status-warning)" }}>
                    {r.watch_out}
                  </span>
                ),
              },
            ]}
            rows={d.metric_glossary as any[]}
          />
        </Section>
      )}

      {Array.isArray(d.feature_deviation) && (
        <Section
          title="What makes the flagged rows unusual"
          description={String(d.note ?? "")}
        >
          <ChartFrame title="Median of flagged rows vs the population, in IQR units">
            <BarChartH
              data={(d.feature_deviation as any[]).map((f) => ({
                label: f.feature,
                value: f.deviation_iqr,
                color: f.deviation_iqr >= 0 ? seriesColor(0) : seriesColor(7),
                note: `flagged median ${num(f.flagged_median, 3)} vs population ${num(f.population_median, 3)}`,
              }))}
              format={(v) => num(v, 2)}
              valueLabel="Deviation (IQR)"
              labelWidth={210}
            />
          </ChartFrame>
        </Section>
      )}

      {d.distilled_tree ? (
        <Section
          title="The distilled decision tree"
          description="Trained to imitate the ensemble's decisions. Four levels of if-statements a non-specialist can read and contest."
        >
          <pre className="pre">{String(d.distilled_tree)}</pre>
        </Section>
      ) : null}
    </>
  );
}

/* -------------------------------------------------------------- SHAP ----- */
function ShapPanels({ shap }: { shap: NonNullable<Explain["shap"]> }) {
  const waterfallKeys = Object.keys(shap.waterfalls ?? {});
  const [pick, setPick] = useState(waterfallKeys[0] ?? "median");
  const wf = shap.waterfalls?.[pick];

  return (
    <>
      <Section
        title="SHAP attributions"
        description={
          <>
            {shap.method}. {shap.identity}
          </>
        }
        aside={`${compact(shap.n_explained)} rows explained`}
      >
        <div className="grid grid--2">
          <ChartFrame
            title="Global importance"
            subtitle="Mean absolute SHAP value: how much each feature moves predictions on average, regardless of direction."
          >
            <BarChartH
              data={shap.global_importance.map((g) => ({
                label: g.feature,
                value: g.mean_abs_shap,
                color: seriesColor(1),
              }))}
              format={(v) => num(v, 4)}
              valueLabel="mean |SHAP|"
              labelWidth={200}
            />
          </ChartFrame>

          {shap.beeswarm?.length > 0 && (
            <ChartFrame
              title="Direction of effect"
              subtitle="Each point is one row. Horizontal position is its SHAP value; colour is where that row sits in the feature's own distribution — dark blue is a high value, light is low. That pairing is what turns 'important' into 'high values push the prediction up'."
            >
              <BeeswarmPanel beeswarm={shap.beeswarm} />
            </ChartFrame>
          )}
        </div>
      </Section>

      {wf && (
        <Section
          title="Individual predictions, decomposed"
          description="SHAP is additive: the base value plus every feature's contribution reconstructs the prediction exactly. These three instances span the prediction range."
        >
          <div className="chiprow">
            {waterfallKeys.map((k) => (
              <button
                key={k}
                className={`chip${pick === k ? " chip--active" : ""}`}
                onClick={() => setPick(k)}
              >
                {k} prediction
              </button>
            ))}
          </div>
          <ChartFrame
            title={`${pick} instance`}
            subtitle={`Base value ${num(wf.base_value, 4)} → prediction ${num(wf.prediction, 4)}. Features not shown contribute ${num(wf.residual_other_features, 4)} in total.`}
          >
            <Waterfall
              base={wf.base_value}
              contributions={wf.contributions}
              prediction={wf.prediction}
              format={(v) => num(v, 3)}
            />
          </ChartFrame>
        </Section>
      )}
    </>
  );
}

function BeeswarmPanel({ beeswarm }: { beeswarm: { feature: string; points: { shap: number; value_rank: number }[] }[] }) {
  const [feature, setFeature] = useState(beeswarm[0]?.feature ?? "");
  const active = beeswarm.find((b) => b.feature === feature) ?? beeswarm[0];
  const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
  return (
    <>
      <div className="chiprow">
        {beeswarm.map((b) => (
          <button
            key={b.feature}
            className={`chip${active?.feature === b.feature ? " chip--active" : ""}`}
            onClick={() => setFeature(b.feature)}
          >
            {b.feature.length > 18 ? `${b.feature.slice(0, 17)}…` : b.feature}
          </button>
        ))}
      </div>
      {active && (
        <ScatterPlot
          height={220}
          points={active.points.map((p) => ({ x: p.shap, y: p.value_rank, group: p.value_rank }))}
          xLabel={`SHAP value for ${active.feature}`}
          yLabel="Feature value (percentile)"
          colorOf={(p) => SEQ[Math.min(SEQ.length - 1, Math.floor(((p.group as number) ?? 0.5) * SEQ.length))]}
        />
      )}
    </>
  );
}

/* ------------------------------------------------------- transformer ----- */
function ArchitecturePanel({ arch }: { arch: any }) {
  return (
    <Section
      title="Architecture"
      description={`${compact(arch.total_parameters)} parameters across ${arch.config.n_layer} pre-norm blocks. Every component is implemented from primitives rather than imported, because the point of the project is that the mechanism be legible.`}
    >
      <DataTable
        columns={[
          { key: "n", header: "Component", render: (c: any) => <strong>{c.name}</strong> },
          { key: "s", header: "Shape", render: (c: any) => <code className="small">{c.shape}</code> },
          { key: "p", header: "Parameters", numeric: true, render: (c: any) => compact(c.params) },
          { key: "r", header: "Role", render: (c: any) => <span className="small">{c.role}</span> },
          {
            key: "note",
            header: "Note",
            render: (c: any) => (
              <span className="small" style={{ color: "var(--ink-muted)" }}>
                {c.note}
              </span>
            ),
          },
        ]}
        rows={arch.components}
      />
      <div style={{ marginTop: 16 }}>
        <div className="small muted" style={{ marginBottom: 8, fontWeight: 600 }}>
          Design choices
        </div>
        <ul className="prose small">
          {arch.design_notes.map((n: string) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </div>
    </Section>
  );
}

function AttentionPanel({ attention }: { attention: any }) {
  const [head, setHead] = useState(0);
  if (!attention?.heads?.length) return null;
  const active = attention.heads[Math.min(head, attention.heads.length - 1)];

  return (
    <Section
      title="Attention, read from the trained weights"
      description={attention.note}
      aside={`layer ${attention.layer} · ${attention.heads.length} heads`}
    >
      <div className="chiprow">
        {attention.heads.map((h: any, i: number) => (
          <button
            key={i}
            className={`chip${head === i ? " chip--active" : ""}`}
            onClick={() => setHead(i)}
          >
            Head {h.head}
          </button>
        ))}
      </div>
      <ChartFrame
        title={`Head ${active.head}`}
        subtitle="Row i shows where position i looked. Everything above the diagonal is exactly zero — that is the causal mask, visible in the weights rather than asserted in prose."
      >
        <Heatmap
          rows={attention.tokens}
          cols={attention.tokens}
          matrix={active.matrix}
          height={Math.max(260, attention.tokens.length * 13 + 50)}
          valueFormat={(v) => v.toFixed(3)}
        />
      </ChartFrame>
    </Section>
  );
}
