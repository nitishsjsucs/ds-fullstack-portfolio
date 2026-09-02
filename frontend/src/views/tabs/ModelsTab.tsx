import { useMemo, useState } from "react";
import { useApi } from "../../lib/api";
import { compact, num, pct, seriesColor, signed } from "../../lib/format";
import type { Leaderboard, LeaderboardRow, Trial } from "../../lib/types";
import { BarChartH, ChartFrame, LineChart } from "../../charts/Charts";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  Section,
  Stat,
} from "../../components/ui";

export default function ModelsTab({ slug }: { slug: string }) {
  const lb = useApi<Leaderboard>(`/projects/${slug}/leaderboard`);

  if (lb.status === "loading") return <Loading what="the model tournament" />;
  if (lb.status === "error") return <ErrorState error={lb.error} onRetry={lb.reload} />;

  const d = lb.data;
  const rows = d.leaderboard ?? [];
  const scoreKey = pickScoreKey(rows);
  const protocolText = typeof d.protocol === "string" ? d.protocol : undefined;

  return (
    <>
      <Section
        title="Leaderboard"
        description={
          <>
            Scored by <strong>{d.scoring}</strong>. {protocolText}
          </>
        }
        aside={
          d.total_trials
            ? `${compact(d.total_trials)} trials · ${num(d.total_seconds, 0)}s`
            : undefined
        }
      >
        {rows.length > 0 && scoreKey && (
          <div style={{ marginBottom: 16 }}>
            <ChartFrame
              title="Ranked performance"
              subtitle={`Bars show ${scoreKey.replace(/_/g, " ")}. Longer is ${scoreKey.includes("mase") || scoreKey.includes("davies") ? "worse" : "better"} unless the metric is an error.`}
            >
              <BarChartH
                data={rows
                  .filter((r) => typeof r[scoreKey] === "number")
                  .map((r, i) => ({
                    label: rowName(r, i),
                    value: r[scoreKey] as number,
                    color: seriesColor(i),
                    note: r.hypothesis ?? (r.assumption as string | undefined),
                  }))}
                format={(v) => num(v, 4)}
                valueLabel={scoreKey.replace(/_/g, " ")}
                labelWidth={190}
              />
            </ChartFrame>
          </div>
        )}
        <LeaderboardTable rows={rows} />
      </Section>

      {d.ensemble && Object.keys(d.ensemble).length > 0 && (
        <EnsemblePanel ensemble={d.ensemble as Record<string, unknown>} />
      )}

      {d.stacking !== undefined && <StackingPanel stacking={d.stacking as Record<string, unknown>} />}

      {Array.isArray(d.trajectory) && d.trajectory.length > 0 && (
        <TrajectoryPanel trials={d.trajectory} scoring={d.scoring} />
      )}

      {Array.isArray(d.agreement) && (
        <Section
          title="Do the algorithms agree?"
          description="Adjusted Rand index between every pair of clusterings. High agreement between methods with different assumptions is the strongest available evidence that the structure is real rather than imposed."
        >
          <DataTable
            columns={[
              { key: "pair", header: "Pair", render: (r: any) => `${r.a} ↔ ${r.b}` },
              {
                key: "ari",
                header: "Adjusted Rand",
                numeric: true,
                render: (r: any) => (
                  <Badge tone={r.adjusted_rand > 0.7 ? "good" : r.adjusted_rand > 0.4 ? "warning" : "serious"}>
                    {num(r.adjusted_rand, 3)}
                  </Badge>
                ),
              },
            ]}
            rows={d.agreement as any[]}
          />
        </Section>
      )}

      {Array.isArray(d.k_sweep) && <KSweepPanel sweep={d.k_sweep as any[]} />}

      {Array.isArray(d.results) && d.identical_output !== undefined && (
        <AlgorithmComparison data={d as unknown as AlgoComparison} />
      )}
    </>
  );
}

/* --------------------------------------------------------------- helpers -- */
/** Projects name their leaderboard rows differently -- 'model' for the supervised
 *  tournaments, 'algorithm' for clustering, 'detector' for anomaly. Resolving that
 *  in one place stops a row rendering as "#1" or a callout titled "undefined". */
function rowName(r: LeaderboardRow, i = 0): string {
  return String(r.model ?? r.algorithm ?? r.detector ?? `#${i + 1}`);
}

function pickScoreKey(rows: LeaderboardRow[]): string | null {
  if (!rows.length) return null;
  for (const key of [
    "holdout_score",
    "holdout_mase",
    "pr_auc",
    "silhouette",
    "cv_score",
    "val_bpc",
    "cv_mase_mean",
  ]) {
    if (rows.some((r) => typeof r[key] === "number")) return key;
  }
  return null;
}

function LeaderboardTable({ rows }: { rows: LeaderboardRow[] }) {
  if (!rows.length) return <div className="state">No leaderboard in the artifacts.</div>;
  const has = (k: string) => rows.some((r) => r[k] !== undefined && r[k] !== null);

  const columns: any[] = [
    {
      key: "model",
      header: "Model",
      render: (r: LeaderboardRow, i: number) => (
        <div>
          <span className="strong" style={{ color: "var(--ink-primary)", fontWeight: 600 }}>
            {i === 0 && "★ "}
            {rowName(r, i)}
          </span>
          {Boolean(r.hypothesis || r.assumption) && (
            <div className="small muted" style={{ maxWidth: "48ch", marginTop: 2 }}>
              {String(r.hypothesis ?? r.assumption)}
            </div>
          )}
          {r.ineligible_reason ? (
            <div className="small" style={{ color: "var(--status-serious)", marginTop: 3 }}>
              excluded: {String(r.ineligible_reason)}
            </div>
          ) : null}
          {r.error ? (
            <div className="small" style={{ color: "var(--status-critical)", marginTop: 3 }}>
              unusable: {String(r.error)}
            </div>
          ) : null}
        </div>
      ),
    },
  ];

  if (has("cv_score")) {
    columns.push({
      key: "cv",
      header: "CV",
      numeric: true,
      render: (r: LeaderboardRow) => num(r.cv_score, 4),
    });
  }
  if (has("cv_mase_mean")) {
    columns.push({
      key: "cvm",
      header: "CV MASE",
      numeric: true,
      render: (r: LeaderboardRow) => (
        <>
          {num(r.cv_mase_mean as number, 4)}
          <div className="small muted">± {num(r.cv_mase_std as number, 4)}</div>
        </>
      ),
    });
  }
  if (has("holdout_score")) {
    columns.push({
      key: "hold",
      header: "Hold-out",
      numeric: true,
      render: (r: LeaderboardRow) => (
        <span style={{ color: "var(--ink-primary)", fontWeight: 600 }}>
          {num(r.holdout_score, 4)}
        </span>
      ),
    });
  }
  if (has("holdout_mase")) {
    columns.push({
      key: "hm",
      header: "Hold-out MASE",
      numeric: true,
      render: (r: LeaderboardRow) => (
        <span style={{ color: "var(--ink-primary)", fontWeight: 600 }}>
          {num(r.holdout_mase as number, 4)}
        </span>
      ),
    });
  }
  if (has("generalisation_gap")) {
    columns.push({
      key: "gap",
      header: "CV − hold-out",
      numeric: true,
      render: (r: LeaderboardRow) =>
        r.generalisation_gap === null || r.generalisation_gap === undefined ? (
          "—"
        ) : (
          <Badge tone={r.overfit_flag ? "warning" : "neutral"}>
            {signed(r.generalisation_gap, 4)}
          </Badge>
        ),
    });
  }
  if (has("silhouette")) {
    columns.push({
      key: "sil",
      header: "Silhouette",
      numeric: true,
      render: (r: any) => (r.silhouette === null ? "—" : num(r.silhouette, 4)),
    });
    columns.push({
      key: "cov",
      header: "Coverage",
      numeric: true,
      render: (r: any) => (r.coverage === undefined ? "—" : pct(r.coverage)),
    });
    columns.push({
      key: "ari",
      header: "Bootstrap ARI",
      numeric: true,
      render: (r: any) => (r.stability_ari === null ? "—" : num(r.stability_ari, 3)),
    });
  }
  if (has("pr_auc")) {
    columns.push({
      key: "pr",
      header: "PR-AUC",
      numeric: true,
      render: (r: any) => (
        <span style={{ color: "var(--ink-primary)", fontWeight: 600 }}>{num(r.pr_auc, 4)}</span>
      ),
    });
    columns.push({
      key: "roc",
      header: "ROC-AUC",
      numeric: true,
      render: (r: any) =>
        r.inverted ? (
          <Badge tone="critical" icon="⚠">
            {num(r.roc_auc, 4)}
          </Badge>
        ) : (
          num(r.roc_auc, 4)
        ),
    });
    columns.push({
      key: "p100",
      header: "P@100",
      numeric: true,
      render: (r: any) => pct(r.precision_at_100),
    });
  }
  if (has("val_bpc")) {
    columns.push({
      key: "bpc",
      header: "Bits / char",
      numeric: true,
      render: (r: any) => (
        <span style={{ color: "var(--ink-primary)", fontWeight: 600 }}>{num(r.val_bpc, 4)}</span>
      ),
    });
    columns.push({
      key: "params",
      header: "Parameters",
      numeric: true,
      render: (r: any) => compact(r.parameters),
    });
  }
  if (has("itemsets")) {
    columns.push({
      key: "itemsets",
      header: "Itemsets found",
      numeric: true,
      render: (r: any) => compact(r.itemsets),
    });
  }
  if (has("seconds")) {
    columns.push({
      key: "secs",
      header: "Seconds",
      numeric: true,
      render: (r: any) => (
        <span style={{ color: "var(--ink-primary)", fontWeight: 600 }}>{num(r.seconds, 2)}</span>
      ),
    });
  }
  if (has("n_trials")) {
    columns.push({
      key: "trials",
      header: "Trials",
      numeric: true,
      render: (r: LeaderboardRow) => String(r.n_trials ?? "—"),
    });
  }

  return (
    <>
      <DataTable columns={columns} rows={rows} rowKey={(r, i) => `${rowName(r, i)}-${i}`} />
      {rows.some((r) => r.diagnosis) && (
        <div style={{ marginTop: 14, display: "grid", gap: 8 }}>
          {rows
            .filter((r) => r.diagnosis)
            .map((r, i) => (
              <Callout
                key={i}
                tone={r.inverted ? "critical" : "info"}
                title={rowName(r, i)}
              >
                {String(r.diagnosis)}
                {r.blind_spot ? (
                  <div style={{ marginTop: 6, color: "var(--ink-muted)" }}>
                    <strong>Blind spot:</strong> {String(r.blind_spot)}
                  </div>
                ) : null}
              </Callout>
            ))}
        </div>
      )}
      {rows.some((r) => r.best_params && Object.keys(r.best_params).length) && (
        <details style={{ marginTop: 14 }}>
          <summary className="small muted" style={{ cursor: "pointer" }}>
            Winning hyperparameters per family
          </summary>
          <div className="grid grid--2" style={{ marginTop: 10 }}>
            {rows
              .filter((r) => r.best_params && Object.keys(r.best_params).length)
              .map((r, i) => (
                <div className="card" key={i} style={{ padding: 14 }}>
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>{rowName(r, i)}</div>
                  <pre className="pre">{JSON.stringify(r.best_params, null, 2)}</pre>
                </div>
              ))}
          </div>
        </details>
      )}
    </>
  );
}

function EnsemblePanel({ ensemble }: { ensemble: Record<string, unknown> }) {
  const weights = (ensemble.weights ?? {}) as Record<string, number>;
  const trajectory = (ensemble.trajectory ?? []) as { round: number; score: number; added: string }[];
  const helps = ensemble.holdout_vs_best_single;

  return (
    <Section
      title="Greedy ensemble selection"
      description={String(ensemble.method ?? "Caruana forward selection with replacement over out-of-fold predictions.")}
    >
      <div className="grid grid--2">
        <ChartFrame
          title="Ensemble weights"
          subtitle="Selection with replacement: a model picked repeatedly earns a larger weight without any optimisation that could overfit."
        >
          <BarChartH
            data={Object.entries(weights).map(([name, w], i) => ({
              label: name,
              value: w,
              color: seriesColor(i),
            }))}
            format={(v) => pct(v, 1)}
            valueLabel="Weight"
            labelWidth={180}
          />
        </ChartFrame>

        <div className="card">
          <div className="chartframe__title" style={{ marginBottom: 12 }}>
            Did it help?
          </div>
          <div className="grid grid--2" style={{ gap: 10, marginBottom: 14 }}>
            <Stat label="Best single" value={num(ensemble.best_single_score as number, 4)} />
            <Stat label="Ensemble (OOF)" value={num(ensemble.score as number, 4)} />
            {ensemble.holdout_score !== undefined && (
              <Stat label="Ensemble (hold-out)" value={num(ensemble.holdout_score as number, 4)} />
            )}
            {helps !== undefined && (
              <Stat
                label="vs best single"
                value={signed(helps as number, 5)}
                tone={(helps as number) > 0 ? "good" : "serious"}
              />
            )}
          </div>
          {typeof helps === "number" && helps <= 0 && (
            <Callout tone="warning" title="An honest negative result">
              Ensembling did not beat the best single model on the untouched hold-out. That
              is reported rather than tuned away — when base learners agree closely there
              is nothing left for a blend to exploit.
            </Callout>
          )}
          {trajectory.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <LineChart
                height={170}
                series={[
                  {
                    name: "OOF score",
                    points: trajectory.map((t) => ({ x: t.round, y: t.score })),
                    color: seriesColor(0),
                  },
                ]}
                markers
                xLabel="Selection round"
                xFormat={(v) => String(Math.round(v))}
                tipFormat={(v) => num(v, 5)}
              />
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}

function StackingPanel({ stacking }: { stacking: Record<string, unknown> }) {
  const corr = (stacking.base_model_correlation ?? []) as { a: string; b: string; correlation: number }[];
  const gain = (stacking.gain_over_best_single as number) ?? 0;
  // A gain smaller than this is within the noise of a single hold-out split, so it
  // is not styled as a success. Colouring +0.0001 green would be the chart-level
  // version of the overclaiming this portfolio is meant to avoid.
  const material = gain > 0.002;
  return (
    <Section
      title="Stacked generalisation"
      description={String(stacking.leakage_note ?? "")}
    >
      <div className="grid grid--3" style={{ marginBottom: 16 }}>
        <Stat label="Best single (hold-out)" value={num(stacking.best_single_holdout_auc as number, 4)} />
        <Stat label="Greedy ensemble" value={num(stacking.greedy_holdout_auc as number, 4)} />
        <Stat label="Logistic meta-learner" value={num(stacking.holdout_auc as number, 4)} />
      </div>
      <Callout
        tone={material ? "good" : "warning"}
        title={
          material
            ? `Ensembling gained ${signed(gain, 5)}`
            : `Ensembling changed hold-out score by ${signed(gain, 5)} — noise`
        }
      >
        {!material && (
          <>
            A difference in the fourth decimal place is indistinguishable from sampling
            noise on a hold-out this size. Reported at face value rather than presented
            as a win.{" "}
          </>
        )}
        Base learners correlate at{" "}
        {corr.length ? num(corr[corr.length - 1].correlation, 3) : "—"} at the top end and{" "}
        {corr.length ? num(corr[0].correlation, 3) : "—"} at the bottom. Ensembling can only
        exploit disagreement; near-identical members leave nothing to combine.
      </Callout>
      {corr.length > 0 && (
        <DataTable
          columns={[
            { key: "pair", header: "Base model pair", render: (r: any) => `${r.a} ↔ ${r.b}` },
            {
              key: "c",
              header: "OOF correlation",
              numeric: true,
              render: (r: any) => (
                <Badge tone={r.correlation > 0.95 ? "critical" : r.correlation > 0.85 ? "warning" : "good"}>
                  {num(r.correlation, 4)}
                </Badge>
              ),
            },
          ]}
          rows={corr}
        />
      )}
    </Section>
  );
}

function TrajectoryPanel({ trials, scoring }: { trials: Trial[]; scoring: string }) {
  const families = useMemo(
    () => Array.from(new Set(trials.map((t) => t.family))),
    [trials],
  );
  const [family, setFamily] = useState<string | "all">("all");
  const filtered = family === "all" ? trials : trials.filter((t) => t.family === family);
  const accepted = filtered.filter((t) => t.accepted);

  return (
    <Section
      title="Hill-climbing trajectory"
      description="Every cross-validated fit the search performed, in order. Rejected trials are shown alongside accepted ones — a search that reports only its successes hides how much of the improvement was luck."
      aside={`${filtered.length} trials · ${accepted.length} accepted`}
    >
      <div className="chiprow">
        <button
          className={`chip${family === "all" ? " chip--active" : ""}`}
          onClick={() => setFamily("all")}
        >
          All families
        </button>
        {families.map((f) => (
          <button
            key={f}
            className={`chip${family === f ? " chip--active" : ""}`}
            onClick={() => setFamily(f)}
          >
            {f}
          </button>
        ))}
      </div>

      <ChartFrame
        title={`Score by trial — ${scoring}`}
        subtitle="Filled markers are accepted moves; the line traces the running best."
        legend={[
          { label: "Trial score", color: seriesColor(0) },
          { label: "Running best", color: seriesColor(1), shape: "line" },
        ]}
      >
        <LineChart
          height={250}
          series={[
            {
              name: "Trial score",
              points: filtered.map((t, i) => ({ x: i + 1, y: t.score })),
              color: seriesColor(0),
            },
            {
              name: "Running best",
              points: runningBest(filtered),
              color: seriesColor(1),
            },
          ]}
          xLabel="Trial"
          xFormat={(v) => String(Math.round(v))}
          tipFormat={(v) => num(v, 5)}
        />
      </ChartFrame>

      <div style={{ marginTop: 16 }}>
        <DataTable
          maxHeight={340}
          columns={[
            { key: "step", header: "#", numeric: true, render: (t: Trial) => String(t.step) },
            { key: "family", header: "Family", render: (t: Trial) => t.family },
            { key: "note", header: "Change", render: (t: Trial) => <span className="mono small">{t.note}</span> },
            {
              key: "score",
              header: scoring,
              numeric: true,
              render: (t: Trial) => (t.score === null ? "—" : num(t.score, 5)),
            },
            {
              key: "acc",
              header: "Outcome",
              render: (t: Trial) =>
                t.accepted ? (
                  <Badge tone="good" icon="✓">accepted</Badge>
                ) : (
                  <span className="muted small">rejected</span>
                ),
            },
          ]}
          rows={filtered}
          rowKey={(t) => String(t.step)}
        />
      </div>
    </Section>
  );
}

function runningBest(trials: Trial[]): { x: number; y: number | null }[] {
  let best: number | null = null;
  return trials.map((t, i) => {
    if (t.score !== null && (best === null || t.score > best)) best = t.score;
    return { x: i + 1, y: best };
  });
}

function KSweepPanel({ sweep }: { sweep: any[] }) {
  return (
    <Section
      title="Choosing k"
      description="Three internal indices plus the inertia elbow. They routinely disagree, which is normal — the project states which index decided and why rather than presenting a single curve as settled."
    >
      <div className="grid grid--2">
        <ChartFrame title="Silhouette (higher is better)">
          <LineChart
            height={200}
            series={[{ name: "Silhouette", points: sweep.map((s) => ({ x: s.k, y: s.silhouette })), color: seriesColor(0) }]}
            markers
            xLabel="k"
            xFormat={(v) => String(Math.round(v))}
            tipFormat={(v) => num(v, 4)}
          />
        </ChartFrame>
        <ChartFrame title="Inertia (the elbow)">
          <LineChart
            height={200}
            series={[{ name: "Inertia", points: sweep.map((s) => ({ x: s.k, y: s.inertia })), color: seriesColor(1) }]}
            markers
            xLabel="k"
            xFormat={(v) => String(Math.round(v))}
            tipFormat={(v) => compact(v)}
          />
        </ChartFrame>
        <ChartFrame title="Davies–Bouldin (lower is better)">
          <LineChart
            height={200}
            series={[{ name: "Davies–Bouldin", points: sweep.map((s) => ({ x: s.k, y: s.davies_bouldin })), color: seriesColor(2) }]}
            markers
            xLabel="k"
            xFormat={(v) => String(Math.round(v))}
            tipFormat={(v) => num(v, 4)}
          />
        </ChartFrame>
        <ChartFrame title="Calinski–Harabasz (higher is better)">
          <LineChart
            height={200}
            series={[{ name: "Calinski–Harabasz", points: sweep.map((s) => ({ x: s.k, y: s.calinski_harabasz })), color: seriesColor(3) }]}
            markers
            xLabel="k"
            xFormat={(v) => String(Math.round(v))}
            tipFormat={(v) => compact(v)}
          />
        </ChartFrame>
      </div>
    </Section>
  );
}

interface AlgoComparison {
  results: { algorithm: string; itemsets: number; seconds: number; strategy: string; wins_when: string; cost: string }[];
  identical_output: boolean;
  faster_algorithm: string;
  speedup: number;
  conclusion: string;
  min_support: number;
}

function AlgorithmComparison({ data }: { data: AlgoComparison }) {
  return (
    <Section
      title="Apriori vs FP-Growth"
      description="Both algorithms are exact, so they must return identical itemsets. Only the search strategy differs — which makes this the cleanest possible demonstration that the algorithm choice is a performance decision and never a correctness one."
    >
      <div className="grid grid--3" style={{ marginBottom: 16 }}>
        <Stat
          label="Identical output"
          value={data.identical_output ? "Yes" : "No"}
          tone={data.identical_output ? "good" : "critical"}
          note="Both are exact algorithms over the same support floor"
        />
        <Stat label="Faster here" value={data.faster_algorithm} tone="accent" />
        <Stat label="Speed-up" value={`${num(data.speedup, 2)}×`} />
      </div>
      <Callout tone="info" title="Measured, not assumed">
        {data.conclusion}
      </Callout>
      <div className="grid grid--2">
        {data.results.map((r, i) => (
          <div className="card" key={r.algorithm}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <h3 style={{ fontSize: "var(--text-md)", color: seriesColor(i) }}>{r.algorithm}</h3>
              <span className="mono tnum">{num(r.seconds, 2)}s</span>
            </div>
            <p className="small" style={{ color: "var(--ink-secondary)", marginTop: 8 }}>
              {r.strategy}
            </p>
            <div className="small" style={{ color: "var(--status-good)", marginTop: 8 }}>
              <strong>Wins when:</strong> {r.wins_when}
            </div>
            <div className="small" style={{ color: "var(--ink-muted)", marginTop: 4 }}>
              <strong>Cost:</strong> {r.cost}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}
