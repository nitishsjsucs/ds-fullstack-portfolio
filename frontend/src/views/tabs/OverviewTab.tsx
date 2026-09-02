import { useApi } from "../../lib/api";
import { compact, humanKey, num, pct, shortDate } from "../../lib/format";
import type { CrispDm, Overview, ProjectMeta } from "../../lib/types";
import { Badge, Callout, ErrorState, Loading, Section, Stat, TagList } from "../../components/ui";
import { href } from "../../lib/router";

/** Values in a headline block that are naturally percentages rather than raw ratios. */
const PERCENTISH = /coverage|rate|share|fidelity|precision|recall|_pct|impact/i;
const INTEGERISH = /parameters|rows|trials|families|count|baskets|items|itemsets|rules|segments|layers|heads|context|steps|detectors|k$/i;

export default function OverviewTab({ slug, meta }: { slug: string; meta: ProjectMeta }) {
  const overview = useApi<Overview>(`/projects/${slug}/overview`);
  const crisp = useApi<CrispDm>(`/projects/${slug}/crispdm`);

  if (overview.status === "loading") return <Loading what="the project overview" />;
  if (overview.status === "error") return <ErrorState error={overview.error} onRetry={overview.reload} />;

  const o = overview.data;
  const headline = Object.entries(o.headline ?? {});

  return (
    <>
      <Section
        title="Headline results"
        description={`Every figure below was produced by the training run stamped ${
          o._provenance?.trained_at ? shortDate(o._provenance.trained_at) : "in the artifacts"
        } and is served from disk rather than recomputed, so the report and the model cannot disagree.`}
        aside={o.train_seconds ? `${Math.round(o.train_seconds)}s to train` : undefined}
      >
        <div className="grid grid--4">
          {headline.map(([k, v]) => (
            <Stat
              key={k}
              label={humanKey(k)}
              value={renderHeadline(k, v)}
              tone={toneFor(k, v)}
            />
          ))}
        </div>
      </Section>

      <div className="grid grid--2">
        <Section title="What this system does">
          <div className="card">
            <p className="prose">{meta.tagline}</p>
            <hr className="divider" />
            <div className="kv">
              <div className="kv__k">Task</div>
              <div className="kv__v">{meta.task}</div>
              <div className="kv__k">Domain</div>
              <div className="kv__v">{meta.domain}</div>
              <div className="kv__k">Dataset</div>
              <div className="kv__v">
                {meta.dataset_title}
                <div className="small muted">
                  {compact(meta.dataset_rows)} rows · <code>{meta.dataset}</code>
                </div>
              </div>
              <div className="kv__k">Champion</div>
              <div className="kv__v">{o.champion ?? "—"}</div>
              {o.rows_modelled !== undefined && (
                <>
                  <div className="kv__k">Rows modelled</div>
                  <div className="kv__v tnum">
                    {compact(o.rows_modelled)}
                    {o.rows_raw ? (
                      <span className="small muted"> of {compact(o.rows_raw)} after filtering</span>
                    ) : null}
                  </div>
                </>
              )}
            </div>
          </div>
        </Section>

        <Section title="Why this metric">
          <div className="card">
            <div style={{ marginBottom: 12 }}>
              <Badge tone="accent">{meta.primary_metric}</Badge>
            </div>
            <p className="prose">{meta.metric_reason}</p>
            <hr className="divider" />
            <div className="small muted" style={{ marginBottom: 6 }}>
              Techniques applied
            </div>
            <TagList items={meta.techniques} />
          </div>
        </Section>
      </div>

      {crisp.status === "ready" && (
        <Section
          title="Business question"
          description="Fixed before any modelling, together with the criteria that decide whether the result is good enough."
        >
          <div className="card">
            <p style={{ fontSize: "var(--text-md)", color: "var(--ink-primary)", marginBottom: 16 }}>
              {crisp.data.business_question}
            </p>
            <div className="small muted" style={{ marginBottom: 8 }}>
              Success criteria
            </div>
            <ul className="prose" style={{ marginBottom: 16 }}>
              {crisp.data.success_criteria.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
            <Callout tone={crisp.data.all_gates_passed ? "good" : "warning"}>
              {crisp.data.gates_passed} of {crisp.data.gates_total} CRISP-DM phase gates
              passed.{" "}
              <a href={href({ page: "project", slug, tab: "crispdm" })}>
                Read the full six-phase record →
              </a>
            </Callout>
          </div>
        </Section>
      )}

      <Section title="What you can do here" description="The highlights worth opening on each tab.">
        <div className="grid grid--3">
          {meta.highlights.map((h, i) => (
            <div className="card" key={i}>
              <p className="small" style={{ margin: 0, color: "var(--ink-secondary)", lineHeight: 1.55 }}>
                {h}
              </p>
            </div>
          ))}
        </div>
      </Section>
    </>
  );
}

function renderHeadline(key: string, v: unknown) {
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (typeof v !== "number") return String(v);
  if (PERCENTISH.test(key) && Math.abs(v) <= 1) return pct(v);
  if (INTEGERISH.test(key)) return compact(v);
  if (Math.abs(v) >= 1000) return compact(v);
  return num(v, Math.abs(v) < 1 ? 4 : 3);
}

function toneFor(key: string, v: unknown) {
  if (typeof v !== "boolean") return undefined;
  if (/beats|supports|passed/i.test(key)) return v ? ("good" as const) : ("critical" as const);
  return undefined;
}
