import { useApi } from "../../lib/api";
import { compact, num, pct, seriesColor, signed, usd } from "../../lib/format";
import type { Evaluation } from "../../lib/types";
import {
  BarChartH,
  BarChartV,
  ChartFrame,
  LineChart,
  ScatterPlot,
  SeparationHistogram,
  Histogram,
} from "../../charts/Charts";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  MetricList,
  Section,
  Stat,
} from "../../components/ui";

export default function EvaluationTab({ slug }: { slug: string }) {
  const ev = useApi<Evaluation>(`/projects/${slug}/evaluation`);

  if (ev.status === "loading") return <Loading what="held-out evaluation" />;
  if (ev.status === "error") return <ErrorState error={ev.error} onRetry={ev.reload} />;

  const d = ev.data;
  const m = d.metrics ?? {};
  const isClassification = d.roc_curve !== undefined;
  const isRegression = d.scatter !== undefined;

  return (
    <>
      {d.split && (
        <Section title="Evaluation protocol" description={String(d.split.rationale ?? "")}>
          <div className="grid grid--4">
            <Stat label="Strategy" value={String(d.split.strategy ?? "—")} />
            <Stat label="Training rows" value={compact(d.split.n_train)} />
            <Stat label="Hold-out rows" value={compact(d.split.n_test)} />
            <Stat label="Test fraction" value={pct(d.split.test_fraction as number)} />
          </div>
          {d.split.boundary ? (
            <Callout tone="good" title="Chronological boundary">
              Training covers everything before <code>{String(d.split.boundary)}</code>; the
              hold-out is strictly the future of that point. No shuffling at any stage.
            </Callout>
          ) : null}
        </Section>
      )}

      <Section
        title="Held-out performance"
        description="Scored once, after model selection closed. Nothing on this page fed back into training."
        aside={d.n_test ? `${compact(d.n_test)} test rows` : undefined}
      >
        <div className="grid grid--2">
          <div className="card">
            <MetricList metrics={m} highlight={["pr_auc", "roc_auc", "mae", "mase", "r2"]} />
          </div>
          <div className="card">
            {m.accuracy_is_misleading ? (
              <Callout tone="warning" title="Accuracy is misleading here">
                Prevalence is {pct(m.prevalence as number)}. A model that always predicts the
                majority class scores {pct(m.baseline_accuracy as number)} accuracy while
                catching nothing, which is why PR-AUC is the headline instead. The random
                PR-AUC floor equals prevalence: {num(m.baseline_pr_auc as number, 4)}.
              </Callout>
            ) : null}
            {d.baselines && (
              <>
                <div className="small muted" style={{ marginBottom: 8, fontWeight: 600 }}>
                  Baselines the model has to beat
                </div>
                <DataTable
                  columns={[
                    { key: "n", header: "Baseline", render: (b: any) => b.name },
                    {
                      key: "v",
                      header: "Score",
                      numeric: true,
                      render: (b: any) =>
                        num(
                          (b.mae ?? b.pr_auc ?? b.roc_auc ?? b.value) as number,
                          4,
                        ),
                    },
                    {
                      key: "d",
                      header: "What it does",
                      render: (b: any) => <span className="small">{b.description}</span>,
                    },
                  ]}
                  rows={d.baselines}
                />
              </>
            )}
          </div>
        </div>
      </Section>

      {d.skill_score !== undefined && <SkillPanel skill={d.skill_score as Record<string, unknown>} />}

      {isClassification && <ClassificationPanels d={d} />}
      {isRegression && <RegressionPanels d={d} />}

      {d.interval !== undefined && <IntervalPanel interval={d.interval as Record<string, unknown>} />}
      {d.series !== undefined && <ForecastSeriesPanel d={d} />}
      {d.horizons !== undefined && <HorizonPanel horizons={d.horizons as any} />}
      {d.campaign_simulation !== undefined && (
        <CampaignPanel sim={d.campaign_simulation as any} economics={d.economics as any} />
      )}
      {d.alert_budget !== undefined && (
        <AlertBudgetPanel budget={d.alert_budget as any[]} families={d.per_family_recall as any[]} />
      )}
      {d.distillation !== undefined && <DistillationPanel d={d.distillation as any} />}
      {d.search_overfitting !== undefined && (
        <OverfitPanel data={d.search_overfitting as any} />
      )}
      {d.segment_errors !== undefined && (
        <Section
          title="Where the model fails"
          description="An aggregate error hides which users are served badly. Per-segment error is the diagnostic that finds systematic weakness."
        >
          <DataTable
            columns={[
              { key: "d", header: "Dimension", render: (r: any) => r.dimension },
              { key: "s", header: "Segment", render: (r: any) => r.segment },
              { key: "n", header: "n", numeric: true, render: (r: any) => compact(r.n) },
              { key: "mae", header: "MAE", numeric: true, render: (r: any) => num(r.mae, 3) },
              {
                key: "vs",
                header: "vs overall",
                numeric: true,
                render: (r: any) => (
                  <Badge tone={r.vs_overall > 0.5 ? "warning" : "neutral"}>
                    {signed(r.vs_overall, 3)}
                  </Badge>
                ),
              },
              { key: "b", header: "Bias", numeric: true, render: (r: any) => signed(r.bias, 3) },
            ]}
            rows={d.segment_errors as any[]}
            maxHeight={360}
          />
        </Section>
      )}
    </>
  );
}

/* ------------------------------------------------------- classification --- */
function ClassificationPanels({ d }: { d: Evaluation }) {
  const cm = d.confusion_matrix;
  const threshold = (d.deployed_threshold as number) ?? 0.5;
  return (
    <>
      <Section
        title="Discrimination and calibration"
        description="ROC measures ranking across all thresholds; the precision–recall curve is the one that matters when positives are rare. The calibration plot asks a different question entirely: when the model says 0.7, does it happen 70% of the time?"
      >
        <div className="grid grid--3">
          {d.roc_curve && (
            <ChartFrame
              title="ROC curve"
              subtitle={`AUC ${num(d.metrics?.roc_auc as number, 4)} — the dashed line is random.`}
            >
              <LineChart
                height={230}
                series={[
                  {
                    name: "ROC",
                    points: d.roc_curve.map((p) => ({ x: p.fpr, y: p.tpr })),
                    color: seriesColor(0),
                  },
                  {
                    name: "Random",
                    points: [
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ],
                    color: "var(--ink-muted)",
                    dashed: true,
                  },
                ]}
                xLabel="False positive rate"
                yLabel="True positive rate"
                xFormat={(v) => v.toFixed(1)}
                yFormat={(v) => v.toFixed(1)}
                yDomain={[0, 1]}
              />
            </ChartFrame>
          )}
          {d.pr_curve && (
            <ChartFrame
              title="Precision–recall"
              subtitle={`AP ${num(d.metrics?.pr_auc as number, 4)}; the floor is the prevalence ${num(d.metrics?.prevalence as number, 3)}.`}
            >
              <LineChart
                height={230}
                series={[
                  {
                    name: "PR",
                    points: d.pr_curve.map((p) => ({ x: p.recall, y: p.precision })),
                    color: seriesColor(1),
                  },
                ]}
                xLabel="Recall"
                yLabel="Precision"
                xFormat={(v) => v.toFixed(1)}
                yFormat={(v) => v.toFixed(1)}
                yDomain={[0, 1]}
                reference={{
                  y: (d.metrics?.prevalence as number) ?? 0,
                  label: "random",
                  color: "var(--ink-muted)",
                }}
              />
            </ChartFrame>
          )}
          {d.calibration && d.calibration.length > 0 && (
            <ChartFrame
              title="Calibration"
              subtitle={`Expected calibration error ${num(d.expected_calibration_error as number, 4)}. Perfect calibration is the diagonal.`}
            >
              <LineChart
                height={230}
                series={[
                  {
                    name: "Observed",
                    points: d.calibration.map((p) => ({ x: p.predicted, y: p.observed })),
                    color: seriesColor(2),
                  },
                  {
                    name: "Perfect",
                    points: [
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ],
                    color: "var(--ink-muted)",
                    dashed: true,
                  },
                ]}
                markers
                xLabel="Predicted probability"
                yLabel="Observed frequency"
                xFormat={(v) => v.toFixed(1)}
                yFormat={(v) => v.toFixed(1)}
                yDomain={[0, 1]}
              />
            </ChartFrame>
          )}
        </div>
      </Section>

      {d.calibration_report !== undefined && (
        <CalibrationComparison report={d.calibration_report as Record<string, any>} />
      )}

      <Section title="The decision, not the score">
        <div className="grid grid--2">
          {d.score_distribution && (
            <ChartFrame
              title="Score separation"
              subtitle="Predicted probability split by true class. Overlap is the model's irreducible uncertainty; the threshold decides how it is resolved."
              legend={[
                { label: "Negatives", color: seriesColor(0) },
                { label: "Positives", color: seriesColor(1) },
              ]}
            >
              <SeparationHistogram bins={d.score_distribution} threshold={threshold} />
            </ChartFrame>
          )}
          {cm && (
            <div className="card">
              <div className="chartframe__title">Confusion matrix</div>
              <div className="chartframe__sub" style={{ marginBottom: 14 }}>
                At the deployed threshold of {num(threshold, 2)}.
              </div>
              <div className="grid grid--2" style={{ gap: 8 }}>
                <Stat label="True positives" value={compact(cm.tp)} tone="good" />
                <Stat label="False positives" value={compact(cm.fp)} tone="warning" />
                <Stat label="False negatives" value={compact(cm.fn)} tone="critical" />
                <Stat label="True negatives" value={compact(cm.tn)} />
              </div>
              {d.at_deployed_threshold ? (
                <>
                  <hr className="divider" />
                  <MetricList
                    metrics={d.at_deployed_threshold as Record<string, unknown>}
                    highlight={["precision", "recall", "f1"]}
                  />
                </>
              ) : null}
            </div>
          )}
        </div>
      </Section>

      {d.threshold_sweep && (
        <Section
          title="Threshold economics"
          description={d.threshold_sweep.note}
        >
          {d.economics ? (
            <Callout tone="info" title="The cost model">
              {String((d.economics as any).cost_model ?? (d.economics as any).interpretation ?? "")}
            </Callout>
          ) : null}
          <ChartFrame
            title="Precision, recall and expected cost across the threshold grid"
            subtitle={`F1 peaks at ${num(d.threshold_sweep.best_f1_threshold, 2)}; the cost-optimal cut is ${num(d.threshold_sweep.best_cost_threshold, 2)}. They differ because F1 assumes both error types cost the same.`}
            legend={[
              { label: "Precision", color: seriesColor(0), shape: "line" },
              { label: "Recall", color: seriesColor(1), shape: "line" },
              { label: "F1", color: seriesColor(2), shape: "line" },
            ]}
          >
            <LineChart
              height={260}
              series={[
                { name: "Precision", points: d.threshold_sweep.grid.map((g) => ({ x: g.threshold, y: g.precision })), color: seriesColor(0) },
                { name: "Recall", points: d.threshold_sweep.grid.map((g) => ({ x: g.threshold, y: g.recall })), color: seriesColor(1) },
                { name: "F1", points: d.threshold_sweep.grid.map((g) => ({ x: g.threshold, y: g.f1 })), color: seriesColor(2) },
              ]}
              xLabel="Decision threshold"
              xFormat={(v) => v.toFixed(2)}
              yFormat={(v) => v.toFixed(1)}
              yDomain={[0, 1]}
              tipFormat={(v) => pct(v)}
            />
          </ChartFrame>
        </Section>
      )}

      {d.gain_chart && (
        <Section
          title="Gain and lift"
          description="If you can only act on part of the population, how much of the positive mass do you capture? This is the curve that turns a ranking into a targeting plan."
        >
          <ChartFrame title="Cumulative positives captured by decile">
            <BarChartV
              data={d.gain_chart.map((g) => ({
                label: `${g.decile * 10}%`,
                value: g.captured_positives,
                color: seriesColor(0),
                note: `Lift ${num(g.lift, 2)}×`,
              }))}
              format={(v) => pct(v)}
              valueLabel="Positives captured"
              yLabel="Share captured"
            />
          </ChartFrame>
        </Section>
      )}
    </>
  );
}

function CalibrationComparison({ report }: { report: Record<string, any> }) {
  const methods = ["raw", "isotonic", "sigmoid"].filter(
    (k) => report[k] && typeof report[k].ece === "number",
  );
  return (
    <Section
      title="Calibration comparison"
      description={report.rationale}
      aside={`deployed: ${report.selected}`}
    >
      <div className="grid grid--2">
        <ChartFrame
          title="Reliability curves"
          subtitle="Calibration is a monotone transform, so it barely moves ranking metrics while substantially changing whether the probability can be multiplied by a dollar amount."
          legend={methods.map((k, i) => ({ label: k, color: seriesColor(i), shape: "line" }))}
        >
          <LineChart
            height={250}
            series={[
              ...methods.map((k, i) => ({
                name: k,
                points: (report[k].curve ?? []).map((p: any) => ({ x: p.predicted, y: p.observed })),
                color: seriesColor(i),
              })),
              {
                name: "Perfect",
                points: [
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ],
                color: "var(--ink-muted)",
                dashed: true,
              },
            ]}
            markers
            xLabel="Predicted"
            yLabel="Observed"
            xFormat={(v) => v.toFixed(1)}
            yFormat={(v) => v.toFixed(1)}
            yDomain={[0, 1]}
          />
        </ChartFrame>
        <div className="card">
          <div className="chartframe__title" style={{ marginBottom: 12 }}>
            Calibration error by method
          </div>
          <DataTable
            columns={[
              {
                key: "m",
                header: "Method",
                render: (r: any) => (
                  <>
                    {r.method}
                    {r.method === report.selected && (
                      <Badge tone="good" icon="✓">deployed</Badge>
                    )}
                  </>
                ),
              },
              { key: "ece", header: "ECE", numeric: true, render: (r: any) => num(r.ece, 4) },
              { key: "b", header: "Brier", numeric: true, render: (r: any) => num(r.brier, 4) },
              { key: "pr", header: "PR-AUC", numeric: true, render: (r: any) => num(r.pr_auc, 4) },
            ]}
            rows={methods.map((k) => report[k])}
          />
          <div className="small muted" style={{ marginTop: 12, lineHeight: 1.5 }}>
            Notice how little PR-AUC moves across the three rows while ECE changes
            substantially. That is exactly what theory predicts: calibration reorders
            nothing, it only rescales.
          </div>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------ regression -- */
function RegressionPanels({ d }: { d: Evaluation }) {
  return (
    <Section
      title="Residual analysis"
      description="A scatter against the identity line shows where predictions break down; residuals against fitted values expose heteroscedasticity — error that grows with the prediction."
    >
      <div className="grid grid--3">
        {d.scatter && (
          <ChartFrame
            title="Predicted vs actual"
            subtitle="Perfect prediction lies on the dashed diagonal."
          >
            <ScatterPlot
              points={d.scatter.map((p) => ({ x: p.actual, y: p.predicted }))}
              height={250}
              xLabel="Actual"
              yLabel="Predicted"
              identity
            />
          </ChartFrame>
        )}
        {d.residuals_vs_fitted && (
          <ChartFrame
            title="Residuals vs fitted"
            subtitle="A horizontal band means constant variance; a fan means the model is less certain at larger values."
          >
            <ScatterPlot
              points={d.residuals_vs_fitted.map((p) => ({ x: p.fitted, y: p.residual }))}
              height={250}
              xLabel="Fitted"
              yLabel="Residual"
            />
          </ChartFrame>
        )}
        {d.residual_histogram && (
          <ChartFrame
            title="Residual distribution"
            subtitle={`Bias ${signed(d.metrics?.bias as number, 3)} — a centred distribution means no systematic over- or under-prediction.`}
          >
            <Histogram bins={d.residual_histogram} height={250} color={seriesColor(2)} />
          </ChartFrame>
        )}
      </div>
    </Section>
  );
}

/* -------------------------------------------------------------- forecast -- */
function SkillPanel({ skill }: { skill: Record<string, unknown> }) {
  const beats = skill.beats_seasonal_naive as boolean;
  return (
    <Section title="Skill against the naive baseline" description={String(skill.definition ?? "")}>
      <div className="grid grid--4">
        <Stat label="Model MAE" value={num(skill.model_mae as number, 2)} unit=" rides" tone="accent" />
        <Stat label="Seasonal naive MAE" value={num(skill.seasonal_naive_mae as number, 2)} unit=" rides" />
        <Stat
          label="Skill score"
          value={pct(skill.skill as number)}
          tone={beats ? "good" : "critical"}
          note="1 − MAE_model / MAE_naive on the same hold-out"
        />
        <Stat
          label="Beats naive"
          value={beats ? "Yes" : "No"}
          tone={beats ? "good" : "critical"}
        />
      </div>
      <Callout tone="warning" title="Why MASE alone would not settle this">
        MASE divides by <em>in-sample</em> naive error, so on a test period that is easier
        than the training span every method — including the naive baseline — scores below
        1. Here the model's MASE is {num(skill.model_mase as number, 3)} and seasonal
        naive's is {num(skill.seasonal_naive_mase as number, 3)}. The head-to-head skill
        score is the claim that actually holds up.
      </Callout>
    </Section>
  );
}

function IntervalPanel({ interval }: { interval: Record<string, unknown> }) {
  const before = interval.before_calibration as Record<string, number> | undefined;
  const conformal = interval.conformal as Record<string, unknown> | undefined;
  const gap = interval.coverage_gap as number;
  return (
    <Section
      title="Prediction interval coverage"
      description="An 80% band that contains 70% of outcomes is not conservative — it is wrong, and downstream planning will treat it as a bound. Coverage is therefore measured rather than asserted."
    >
      <div className="grid grid--4" style={{ marginBottom: 16 }}>
        <Stat label="Nominal" value={pct(interval.nominal_coverage as number, 0)} />
        <Stat
          label="Empirical"
          value={pct(interval.empirical_coverage as number)}
          tone={Math.abs(gap) <= 0.05 ? "good" : "warning"}
        />
        <Stat label="Gap" value={signed(gap, 3)} />
        <Stat label="Mean width" value={num(interval.mean_interval_width as number, 1)} />
      </div>
      <Callout tone={Math.abs(gap) <= 0.05 ? "good" : "warning"} title={String(interval.verdict)}>
        {String(interval.calibration_effect ?? "")}
      </Callout>
      {conformal && before && (
        <div className="grid grid--2">
          <div className="card">
            <div className="chartframe__title" style={{ marginBottom: 10 }}>
              Before and after conformal calibration
            </div>
            <DataTable
              columns={[
                { key: "s", header: "Stage", render: (r: any) => r.stage },
                { key: "c", header: "Coverage", numeric: true, render: (r: any) => pct(r.coverage) },
                { key: "w", header: "Mean width", numeric: true, render: (r: any) => num(r.width, 1) },
              ]}
              rows={[
                { stage: "Raw quantile band", coverage: before.empirical_coverage, width: before.mean_interval_width },
                {
                  stage: "Conformally calibrated",
                  coverage: interval.empirical_coverage,
                  width: interval.mean_interval_width,
                },
              ]}
            />
          </div>
          <div className="card">
            <div className="chartframe__title" style={{ marginBottom: 10 }}>
              {String(conformal.method)}
            </div>
            <div className="kv small">
              <div className="kv__k">Adjustment</div>
              <div className="kv__v tnum">±{num(conformal.adjustment as number, 2)}</div>
              <div className="kv__k">Calibration rows</div>
              <div className="kv__v tnum">{compact(conformal.n_calibration)}</div>
              <div className="kv__k">Direction</div>
              <div className="kv__v">{String(conformal.direction)}</div>
            </div>
            <div className="small muted" style={{ marginTop: 10, lineHeight: 1.5 }}>
              {String(conformal.guarantee)}
            </div>
            <div className="small" style={{ marginTop: 8, color: "var(--status-warning)", lineHeight: 1.5 }}>
              {String(conformal.caveat)}
            </div>
          </div>
        </div>
      )}
    </Section>
  );
}

function ForecastSeriesPanel({ d }: { d: Evaluation }) {
  const series = (d.series ?? []) as {
    t: string;
    actual: number;
    predicted: number;
    lower: number;
    upper: number;
    inside: boolean;
  }[];
  if (!series.length) return null;
  const inside = series.filter((p) => p.inside).length;
  return (
    <Section
      title="Forecast against actuals"
      description="The hold-out period, never seen during training or tuning. The band is the conformally calibrated 80% interval."
      aside={`${pct(inside / series.length)} of shown points inside the band`}
    >
      <ChartFrame
        title="Hourly demand: actual, forecast and 80% band"
        legend={[
          { label: "Actual", color: seriesColor(0), shape: "line" },
          { label: "Forecast", color: seriesColor(1), shape: "line" },
        ]}
      >
        <LineChart
          height={300}
          series={[
            { name: "Actual", points: series.map((p, i) => ({ x: i, y: p.actual })), color: seriesColor(0) },
            { name: "Forecast", points: series.map((p, i) => ({ x: i, y: p.predicted })), color: seriesColor(1) },
          ]}
          band={{ points: series.map((p, i) => ({ x: i, lo: p.lower, hi: p.upper })), color: seriesColor(1) }}
          xLabel="Hours into the hold-out period"
          yLabel="Rides per hour"
          xFormat={(v) => String(Math.round(v))}
          tipFormat={(v) => num(v, 0)}
        />
      </ChartFrame>
    </Section>
  );
}

function HorizonPanel({ horizons }: { horizons: { results: any[]; method: string; interpretation: string } }) {
  return (
    <Section title="Accuracy by forecast horizon" description={horizons.method}>
      <div className="grid grid--2">
        <ChartFrame title="Error grows as recent history falls away">
          <BarChartV
            data={horizons.results.map((r, i) => ({
              label: `${r.horizon_hours}h`,
              value: r.mae,
              color: seriesColor(i),
            }))}
            format={(v) => num(v, 2)}
            valueLabel="MAE"
            yLabel="MAE (rides)"
          />
        </ChartFrame>
        <div className="card">
          <DataTable
            columns={[
              { key: "h", header: "Horizon", render: (r: any) => `${r.horizon_hours}h ahead` },
              { key: "f", header: "Features", numeric: true, render: (r: any) => String(r.features_available) },
              { key: "mae", header: "MAE", numeric: true, render: (r: any) => num(r.mae, 2) },
              { key: "mase", header: "MASE", numeric: true, render: (r: any) => num(r.mase, 3) },
              { key: "r2", header: "R²", numeric: true, render: (r: any) => num(r.r2, 4) },
            ]}
            rows={horizons.results}
          />
          <Callout tone="info" title="Interpretation">
            {horizons.interpretation}
          </Callout>
        </div>
      </div>
    </Section>
  );
}

/* -------------------------------------------------------------- business -- */
function CampaignPanel({ sim, economics }: { sim: any; economics: any }) {
  return (
    <Section
      title="Campaign value simulation"
      description={sim.interpretation}
      aside={`optimum: top ${sim.best.targeted_pct}%`}
    >
      <div className="grid grid--4" style={{ marginBottom: 16 }}>
        <Stat label="Contact depth" value={`${sim.best.targeted_pct}%`} tone="accent" />
        <Stat label="Churners reached" value={pct(sim.best.recall)} />
        <Stat label="Net value" value={usd(sim.best.net_value)} tone="good" />
        <Stat label="ROI" value={pct(sim.best.roi)} />
      </div>
      <ChartFrame
        title="Net value by targeting depth"
        subtitle={`Contacting everyone destroys value: most incentives go to customers who were never going to leave. Economics: ${usd(economics?.retention_offer_cost)} offer, ${pct(economics?.offer_acceptance_rate)} acceptance.`}
      >
        <LineChart
          height={250}
          series={[
            {
              name: "Net value",
              points: sim.curve.map((r: any) => ({ x: r.targeted_pct, y: r.net_value })),
              color: seriesColor(0),
            },
          ]}
          markers
          xLabel="Percent of base contacted"
          yLabel="Net value (USD)"
          xFormat={(v) => `${Math.round(v)}%`}
          yFormat={(v) => compact(v)}
          tipFormat={(v) => usd(v)}
          reference={{ y: 0, label: "break-even", color: "var(--status-critical)" }}
        />
      </ChartFrame>
    </Section>
  );
}

function AlertBudgetPanel({ budget, families }: { budget: any[]; families: any[] }) {
  return (
    <>
      <Section
        title="Analyst alert budget"
        description="Nobody reviews 'everything above 0.5'. A team reviews the top N alerts a shift can absorb, so precision@k is the operational metric and this table prices the recall you buy per analyst hour."
      >
        <div className="grid grid--2">
          <ChartFrame
            title="Precision and recall against queue size"
            legend={[
              { label: "Precision@k", color: seriesColor(0), shape: "line" },
              { label: "Recall@k", color: seriesColor(1), shape: "line" },
            ]}
          >
            <LineChart
              height={250}
              series={[
                { name: "Precision@k", points: budget.map((b) => ({ x: b.alert_budget, y: b.precision_at_k })), color: seriesColor(0) },
                { name: "Recall@k", points: budget.map((b) => ({ x: b.alert_budget, y: b.recall_at_k })), color: seriesColor(1) },
              ]}
              markers
              xLabel="Alerts reviewed"
              xFormat={(v) => compact(v)}
              yFormat={(v) => v.toFixed(1)}
              tipFormat={(v) => pct(v)}
              yDomain={[0, 1]}
            />
          </ChartFrame>
          <div className="card">
            <DataTable
              columns={[
                { key: "k", header: "Queue", numeric: true, render: (b: any) => compact(b.alert_budget) },
                { key: "tp", header: "Real", numeric: true, render: (b: any) => compact(b.true_positives) },
                { key: "p", header: "Precision", numeric: true, render: (b: any) => pct(b.precision_at_k) },
                { key: "r", header: "Recall", numeric: true, render: (b: any) => pct(b.recall_at_k) },
                { key: "h", header: "Analyst hrs", numeric: true, render: (b: any) => num(b.analyst_hours, 1) },
              ]}
              rows={budget}
            />
          </div>
        </div>
      </Section>

      {families?.length > 0 && (
        <Section
          title="Recall by attack family"
          description="An aggregate score can be strong while an entire attack class is never surfaced. This is the table a security team actually needs."
        >
          <DataTable
            columns={[
              { key: "f", header: "Family", render: (r: any) => r.family },
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
                  >
                    {r.verdict}
                  </Badge>
                ),
              },
            ]}
            rows={families}
          />
        </Section>
      )}
    </>
  );
}

function DistillationPanel({ d }: { d: any }) {
  return (
    <Section
      title="Was the complexity necessary?"
      description={`A ${d.student} was fitted to imitate the ensemble's decisions — trained on ${d.trained_on}. If four levels of if-statements recover most of the performance, the interpretable model is the defensible choice.`}
    >
      <div className="grid grid--4" style={{ marginBottom: 16 }}>
        <Stat label="Teacher AUC" value={num(d.teacher_auc, 4)} />
        <Stat label="Student AUC" value={num(d.student_auc, 4)} />
        <Stat label="AUC retained" value={pct(d.auc_retained)} tone="accent" />
        <Stat label="Decision fidelity" value={pct(d.fidelity)} />
      </div>
      <Callout tone="info" title="Verdict">
        {d.verdict}
      </Callout>
      {d.tree_text && (
        <details>
          <summary className="small muted" style={{ cursor: "pointer", marginBottom: 8 }}>
            The distilled tree, in full
          </summary>
          <pre className="pre">{d.tree_text}</pre>
        </details>
      )}
    </Section>
  );
}

function OverfitPanel({ data }: { data: any }) {
  return (
    <Section
      title="Did the search overfit?"
      description={data.interpretation}
      aside={`${data.n_flagged} flagged above ${data.threshold}`}
    >
      <div className="grid grid--2">
        <ChartFrame
          title="Cross-validated vs held-out score"
          subtitle="The gap between the two is the cost of searching. Quoting the CV number alone overstates deployed performance by exactly this margin."
          legend={[
            { label: "Cross-validated", color: seriesColor(0) },
            { label: "Hold-out", color: seriesColor(1) },
          ]}
        >
          <BarChartH
            data={data.rows.flatMap((r: any) => [
              { label: `${r.model} · CV`, value: r.cv_auc, color: seriesColor(0) },
              { label: `${r.model} · hold-out`, value: r.holdout_auc, color: seriesColor(1) },
            ])}
            format={(v) => num(v, 4)}
            valueLabel="ROC-AUC"
            labelWidth={210}
            domain={[
              Math.min(...data.rows.map((r: any) => r.holdout_auc)) - 0.02,
              Math.max(...data.rows.map((r: any) => r.cv_auc)) + 0.005,
            ]}
          />
        </ChartFrame>
        <div className="card">
          <DataTable
            columns={[
              { key: "m", header: "Model", render: (r: any) => r.model },
              { key: "cv", header: "CV", numeric: true, render: (r: any) => num(r.cv_auc, 4) },
              { key: "h", header: "Hold-out", numeric: true, render: (r: any) => num(r.holdout_auc, 4) },
              {
                key: "g",
                header: "Gap",
                numeric: true,
                render: (r: any) => (
                  <Badge tone={r.flagged ? "warning" : "neutral"}>{signed(r.gap, 4)}</Badge>
                ),
              },
              { key: "t", header: "Trials", numeric: true, render: (r: any) => String(r.trials) },
            ]}
            rows={data.rows}
          />
          {data.expressiveness_pattern && (
            <Callout tone="info" title="The gap tracks model capacity">
              {data.expressiveness_pattern}
            </Callout>
          )}
        </div>
      </div>
    </Section>
  );
}
