import { href } from "../lib/router";
import type { ProjectIndex, ProjectMeta } from "../lib/types";
import { compact } from "../lib/format";
import { Badge, Callout, Section, TagList } from "../components/ui";

const ACCENT: Record<string, string> = {
  blue: "var(--series-1)",
  amber: "var(--series-4)",
  rose: "var(--series-5)",
  violet: "var(--series-7)",
  emerald: "var(--series-3)",
  red: "var(--series-8)",
  sky: "var(--series-1)",
  cyan: "var(--series-3)",
  fuchsia: "var(--series-5)",
};

export default function Home({ index }: { index: ProjectIndex }) {
  const totalRows = index.projects.reduce((a, p) => a + p.dataset_rows, 0);
  const trained = index.projects.filter((p) => p.trained).length;

  return (
    <>
      <header className="pagehead">
        <div className="pagehead__inner">
          <div className="pagehead__eyebrow">
            <span>Portfolio</span>
            <span>·</span>
            <span>CRISP-DM</span>
            <span>·</span>
            <span>{trained} of {index.projects.length} trained</span>
          </div>
          <h1 className="pagehead__title">
            Eight end-to-end data science systems, on eight real datasets
          </h1>
          <p className="pagehead__tagline">
            Every number on this site comes from a model trained on genuine public data —{" "}
            {compact(totalRows)} rows of NYC taxi trips, telecom subscribers, UK retail
            invoices, network intrusions, bike-share demand, census records and
            Shakespeare. Each project follows CRISP-DM with phase gates, publishes its own
            leakage audit, and states plainly what it cannot do.
          </p>
          <div style={{ height: 20 }} />
        </div>
      </header>

      <div className="content">
        <Callout tone="info" title="What makes this different from a notebook dump">
          <p>
            Three commitments run through every project. <strong>Real data</strong> — the
            ingestion script downloads from the original sources and records a SHA-256 for
            every curated file. <strong>Leakage discipline</strong> — every fitted
            transform lives inside a pipeline that cross-validation clones, time series are
            never shuffled, and each project ships a static audit of its own training path.{" "}
            <strong>Honest reporting</strong> — baselines are published alongside results,
            negative findings are kept rather than deleted, and the metric is chosen to
            suit the problem instead of to flatter the model.
          </p>
        </Callout>

        <Section
          title="The systems"
          description="Each card opens a full workspace: exploratory analysis, the six CRISP-DM phases with their exit gates, the model tournament, held-out evaluation, explainability, live inference and the audit."
        >
          <div className="grid grid--auto">
            {index.projects.map((p) => (
              <ProjectCard key={p.slug} project={p} />
            ))}
          </div>
        </Section>

        <Section
          title="Where to start"
          description="If you only have a few minutes, these are the three most interesting results in the portfolio."
        >
          <div className="grid grid--3">
            <Highlight
              slug="anomaly"
              tab="models"
              title="Three detectors are inverted"
              body="On network telemetry, LOF, one-class SVM and PCA reconstruction all score below 0.5 ROC-AUC — they rank intrusions as more normal than benign traffic, because the dominant flood attacks are the densest region of the feature space. A fourth returned NaN outright."
            />
            <Highlight
              slug="automl"
              tab="models"
              title="Ensembling bought nothing"
              body="After a full hill-climbing tournament, greedy ensembling reaches 0.9314 against 0.9313 for the best single model, and the logistic meta-learner lands below it. A fourth-decimal difference is noise — which is what a 0.92 correlation between base learners predicts."
            />
            <Highlight
              slug="forecast"
              tab="evaluation"
              title="An 80% interval that covered 67%"
              body="Raw quantile regression produced bands that were confidently wrong. Conformal calibration on a held-out block moved empirical coverage from 67.1% to 79.2% — at the cost of a band 17% wider."
            />
          </div>
        </Section>
      </div>
    </>
  );
}

function ProjectCard({ project }: { project: ProjectMeta }) {
  const accent = ACCENT[project.accent] ?? "var(--series-1)";
  return (
    <a
      className="card"
      href={href({ page: "project", slug: project.slug })}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        borderTop: `2px solid ${accent}`,
        color: "inherit",
        textDecoration: "none",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <div className="small muted mono">{String(project.number).padStart(2, "0")}</div>
          <h3 style={{ fontSize: "var(--text-md)", marginTop: 2 }}>{project.title}</h3>
        </div>
        {project.trained ? (
          <Badge tone="good" icon="✓">trained</Badge>
        ) : (
          <Badge tone="warning" icon="⚠">untrained</Badge>
        )}
      </div>

      <p className="small" style={{ color: "var(--ink-secondary)", margin: 0, lineHeight: 1.5 }}>
        {project.tagline}
      </p>

      <div className="kv small" style={{ gridTemplateColumns: "auto 1fr", gap: "3px 10px" }}>
        <div className="kv__k">Task</div>
        <div className="kv__v" style={{ color: "var(--ink-secondary)" }}>{project.task}</div>
        <div className="kv__k">Data</div>
        <div className="kv__v" style={{ color: "var(--ink-secondary)" }}>
          {compact(project.dataset_rows)} rows · {project.dataset_title.split("(")[0].trim()}
        </div>
        <div className="kv__k">Metric</div>
        <div className="kv__v" style={{ color: accent, fontWeight: 600 }}>{project.primary_metric}</div>
      </div>

      <div style={{ marginTop: "auto", paddingTop: 4 }}>
        <TagList items={project.skills.slice(0, 3)} />
      </div>
    </a>
  );
}

function Highlight({
  slug,
  tab,
  title,
  body,
}: {
  slug: string;
  tab: string;
  title: string;
  body: string;
}) {
  return (
    <a className="card" href={href({ page: "project", slug, tab })} style={{ color: "inherit", textDecoration: "none" }}>
      <h3 style={{ fontSize: "var(--text-base)", marginBottom: 6 }}>{title}</h3>
      <p className="small" style={{ color: "var(--ink-secondary)", margin: 0, lineHeight: 1.55 }}>
        {body}
      </p>
      <div className="small" style={{ marginTop: 10, color: "var(--accent, var(--series-1))", fontWeight: 600 }}>
        Open {slug} →
      </div>
    </a>
  );
}
