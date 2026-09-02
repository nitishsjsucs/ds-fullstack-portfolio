import { useApi } from "../../lib/api";
import { compact, humanKey, num, shortDate } from "../../lib/format";
import type { Audit, ModelCard } from "../../lib/types";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  KeyValues,
  Section,
  Stat,
} from "../../components/ui";

export default function AuditTab({ slug }: { slug: string }) {
  const audit = useApi<Audit>(`/projects/${slug}/audit`);
  const card = useApi<ModelCard>(`/projects/${slug}/model_card`);

  if (audit.status === "loading") return <Loading what="the audit" />;
  if (audit.status === "error") return <ErrorState error={audit.error} onRetry={audit.reload} />;

  const a = audit.data;
  const failures = a.checks.filter((c) => c.status !== "pass");

  return (
    <>
      <Section
        title="Leakage & methodology audit"
        description={a.scope}
        aside={`grade ${a.grade}`}
      >
        <div className="grid grid--4" style={{ marginBottom: 16 }}>
          <Stat
            label="Checks passed"
            value={`${a.passed}/${a.total}`}
            tone={a.passed === a.total ? "good" : "warning"}
          />
          <Stat label="Grade" value={a.grade} tone={a.grade === "A" ? "good" : "warning"} />
          <Stat
            label="Warnings"
            value={String(failures.length)}
            tone={failures.length ? "warning" : "good"}
          />
          <Stat
            label="Audited"
            value={a._provenance?.trained_at ? shortDate(a._provenance.trained_at) : "—"}
            note={a._provenance?.git_commit ? `commit ${a._provenance.git_commit}` : undefined}
          />
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          {a.checks.map((c) => (
            <div
              key={c.check}
              className="card"
              style={{
                padding: "14px 16px",
                borderLeft: `3px solid var(--status-${c.status === "pass" ? "good" : c.status === "warn" ? "warning" : "critical"})`,
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 12,
                  alignItems: "flex-start",
                }}
              >
                <div style={{ fontWeight: 600 }}>{c.check}</div>
                <Badge
                  tone={c.status === "pass" ? "good" : c.status === "warn" ? "warning" : "critical"}
                  icon={c.status === "pass" ? "✓" : c.status === "warn" ? "⚠" : "✕"}
                >
                  {c.status}
                </Badge>
              </div>
              <div className="small" style={{ color: "var(--ink-secondary)", marginTop: 6, lineHeight: 1.55 }}>
                {c.evidence}
              </div>
            </div>
          ))}
        </div>
      </Section>

      {card.status === "ready" && <ModelCardPanel card={card.data} />}
    </>
  );
}

function ModelCardPanel({ card }: { card: ModelCard }) {
  return (
    <>
      <Section
        title="Model card"
        description="What this model is for, what it is not for, and what it gets wrong. The out-of-scope and limitations sections are the load-bearing parts."
        aside={`v${card.version}`}
      >
        <div className="grid grid--2">
          <div className="card">
            <h3 style={{ fontSize: "var(--text-md)", marginBottom: 10 }}>{card.model}</h3>
            <p className="prose">{card.intended_use}</p>
            <hr className="divider" />
            <KeyValues
              entries={[
                ["Task", card.task],
                [
                  "Training data",
                  <>
                    {String(card.training_data.source ?? "—")}
                    <div className="small muted">
                      {compact(card.training_data.rows ?? card.training_data.baskets ?? card.training_data.characters)}{" "}
                      rows
                      {card.training_data.period ? ` · ${card.training_data.period}` : ""}
                      {card.training_data.licence ? ` · ${card.training_data.licence}` : ""}
                    </div>
                  </>,
                ],
                ...(card.evaluation_data
                  ? ([
                      [
                        "Evaluation data",
                        <>
                          {compact(card.evaluation_data.rows)} rows
                          <div className="small muted">{String(card.evaluation_data.protocol ?? "")}</div>
                        </>,
                      ],
                    ] as [string, React.ReactNode][])
                  : []),
              ]}
            />
          </div>

          <div className="card">
            <div className="chartframe__title" style={{ marginBottom: 10 }}>
              Reported metrics
            </div>
            <div className="kv small">
              {Object.entries(card.metrics).map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <div className="kv__k">{humanKey(k)}</div>
                  <div className="kv__v tnum">
                    {typeof v === "number" ? num(v, 4) : typeof v === "boolean" ? (v ? "yes" : "no") : String(v ?? "—")}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      <div className="grid grid--2">
        <Section title="Out of scope">
          <div className="card">
            <ul className="prose">
              {card.out_of_scope.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        </Section>

        <Section title="Limitations">
          <div className="card">
            <ul className="prose">
              {card.limitations.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        </Section>
      </div>

      {card.features ? (
        <Section title="Features">
          <div className="card">
            <KeyValues
              entries={[
                ["Count", String(card.features.count ?? "—")],
                [
                  "Excluded by design",
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {(card.features.excluded_by_design as string[] | undefined)?.map((f) => (
                      <Badge key={f} tone="critical">
                        {f}
                      </Badge>
                    )) ?? "—"}
                  </div>,
                ],
                ...(card.features.exclusion_reason
                  ? ([["Why excluded", String(card.features.exclusion_reason)]] as [string, React.ReactNode][])
                  : []),
              ]}
            />
            {card.features.exclusion_reasons ? (
              <div style={{ marginTop: 14 }}>
                <DataTable
                  columns={[
                    { key: "c", header: "Column", render: ([k]: [string, string]) => <code>{k}</code> },
                    {
                      key: "r",
                      header: "Reason for exclusion",
                      render: ([, v]: [string, string]) => <span className="small">{v}</span>,
                    },
                  ]}
                  rows={Object.entries(card.features.exclusion_reasons as Record<string, string>)}
                  rowKey={([k]) => k}
                />
              </div>
            ) : null}
          </div>
        </Section>
      ) : null}

      {card.fairness ? <FairnessSummary fairness={card.fairness as Record<string, unknown>} /> : null}

      <Section title="Ethical considerations">
        <div className="card">
          {card.ethical_considerations.map((e, i) => (
            <Callout key={i} tone="warning">
              {e}
            </Callout>
          ))}
        </div>
      </Section>

      <Section title="Maintenance">
        <div className="card">
          <KeyValues
            entries={[
              ["Retrain trigger", card.maintenance.retrain_trigger],
              [
                "Monitored signals",
                card.maintenance.monitored_signals.length ? (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {card.maintenance.monitored_signals.map((s) => (
                      <Badge key={s}>{s}</Badge>
                    ))}
                  </div>
                ) : (
                  "—"
                ),
              ],
            ]}
          />
        </div>
      </Section>
    </>
  );
}

function FairnessSummary({ fairness }: { fairness: Record<string, unknown> }) {
  const failures = (fairness.four_fifths_failures ?? []) as string[];
  return (
    <Section
      title="Fairness summary"
      description={String(fairness.note ?? "")}
      aside={`prioritises ${String(fairness.prioritised_criterion ?? "—")}`}
    >
      <div className="grid grid--3" style={{ marginBottom: 12 }}>
        <Stat label="Attributes audited" value={String(fairness.attributes_audited ?? "—")} />
        <Stat
          label="Worst disparate impact"
          value={num(fairness.worst_disparate_impact as number, 3)}
          tone={(fairness.worst_disparate_impact as number) >= 0.8 ? "good" : "critical"}
          note="Four-fifths rule threshold is 0.80"
        />
        <Stat
          label="Four-fifths failures"
          value={failures.length ? failures.join(", ") : "None"}
          tone={failures.length ? "critical" : "good"}
        />
      </div>
      {fairness.key_finding ? (
        <Callout tone="critical" title="Key finding">
          {String(fairness.key_finding)}
        </Callout>
      ) : null}
    </Section>
  );
}
