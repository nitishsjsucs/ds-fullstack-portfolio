import { useState } from "react";
import { useApi } from "../../lib/api";
import { humanKey, num } from "../../lib/format";
import type { CrispDm, Finding, Phase } from "../../lib/types";
import { Badge, Callout, ErrorState, Loading, PassFail, Section } from "../../components/ui";

const SEVERITY_TONE = {
  info: "good",
  watch: "warning",
  critical: "critical",
} as const;

const SEVERITY_ICON = { info: "ℹ", watch: "⚠", critical: "✕" } as const;

export default function CrispDmTab({ slug }: { slug: string }) {
  const crisp = useApi<CrispDm>(`/projects/${slug}/crispdm`);
  const [open, setOpen] = useState<string | null>(null);

  if (crisp.status === "loading") return <Loading what="the CRISP-DM record" />;
  if (crisp.status === "error") return <ErrorState error={crisp.error} onRetry={crisp.reload} />;

  const d = crisp.data;
  const expanded = open ?? d.phases[0]?.key;

  return (
    <>
      <Section
        title="The business question"
        description={d.methodology}
        aside={`${d.gates_passed}/${d.gates_total} gates passed`}
      >
        <div className="card">
          <p style={{ fontSize: "var(--text-md)", color: "var(--ink-primary)", marginBottom: 18 }}>
            {d.business_question}
          </p>
          <div className="small muted" style={{ marginBottom: 8 }}>
            Success criteria, fixed before modelling
          </div>
          <ul className="prose">
            {d.success_criteria.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      </Section>

      <Section
        title="Phase timeline"
        description="Each phase carries its own exit gate. A gate is evaluated against the training run's actual numbers, not asserted — so a phase can fail."
      >
        <div style={{ display: "flex", gap: 2, marginBottom: 20, flexWrap: "wrap" }}>
          {d.phases.map((p) => (
            <button
              key={p.key}
              onClick={() => setOpen(p.key)}
              className="chip"
              style={{
                flex: "1 1 130px",
                borderColor: expanded === p.key ? "var(--accent, var(--series-1))" : undefined,
                borderTopWidth: 3,
                borderTopColor: p.gate.passed ? "var(--status-good)" : "var(--status-critical)",
                padding: "10px 12px",
              }}
            >
              <div
                style={{
                  fontWeight: 620,
                  color: expanded === p.key ? "var(--accent, var(--series-1))" : "var(--ink-primary)",
                  fontSize: "var(--text-sm)",
                }}
              >
                {p.title}
              </div>
              <div className="small muted" style={{ marginTop: 3 }}>
                {p.gate.passed ? "gate passed" : "gate failed"} · {p.findings.length} findings
              </div>
            </button>
          ))}
        </div>

        {d.phases
          .filter((p) => p.key === expanded)
          .map((p) => (
            <PhasePanel key={p.key} phase={p} />
          ))}
      </Section>

      {d.iteration_notes.length > 0 && (
        <Section
          title="Where this project looped back"
          description="CRISP-DM is iterative, not linear. These are the points where a later phase sent the work back to an earlier one — usually the most instructive part of a project and the part normally deleted before publication."
        >
          {d.iteration_notes.map((n, i) => (
            <Callout key={i} tone="warning" title={`Loop ${i + 1}`}>
              {n}
            </Callout>
          ))}
        </Section>
      )}
    </>
  );
}

function PhasePanel({ phase }: { phase: Phase }) {
  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "flex-start" }}>
        <div>
          <h3 style={{ fontSize: "var(--text-lg)" }}>{phase.title}</h3>
          <p className="small muted" style={{ marginTop: 4, maxWidth: "70ch" }}>
            {phase.purpose}
          </p>
        </div>
        <PassFail passed={phase.gate.passed} labels={["Gate passed", "Gate failed"]} />
      </div>

      <p className="prose" style={{ marginTop: 16, fontSize: "var(--text-base)" }}>
        {phase.summary}
      </p>

      {Object.keys(phase.metrics).length > 0 && (
        <div className="grid grid--4" style={{ marginTop: 18, gap: 10 }}>
          {Object.entries(phase.metrics).map(([k, v]) => (
            <div key={k} className="stat" style={{ padding: "10px 12px" }}>
              <div className="stat__label">{humanKey(k)}</div>
              <div style={{ fontSize: "var(--text-md)", fontWeight: 620 }}>
                {typeof v === "number" ? num(v, Math.abs(v) < 1 ? 4 : 2) : String(v)}
              </div>
            </div>
          ))}
        </div>
      )}

      <hr className="divider" />

      <div className="grid grid--2">
        <div>
          <div className="small muted" style={{ marginBottom: 8, fontWeight: 600 }}>
            Activities
          </div>
          <ul className="prose small">
            {phase.activities.map((a) => (
              <li key={a}>{a}</li>
            ))}
          </ul>
        </div>

        <div>
          <div className="small muted" style={{ marginBottom: 8, fontWeight: 600 }}>
            Exit gate
          </div>
          <ul className="prose small">
            {phase.gate.criteria.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          {phase.gate.notes && (
            <div className="small mono" style={{ color: "var(--ink-primary)", marginTop: 8 }}>
              {phase.gate.notes}
            </div>
          )}
        </div>
      </div>

      {phase.findings.length > 0 && (
        <>
          <hr className="divider" />
          <div className="small muted" style={{ marginBottom: 10, fontWeight: 600 }}>
            Findings — each backed by a measured value
          </div>
          <div style={{ display: "grid", gap: 10 }}>
            {phase.findings.map((f, i) => (
              <FindingCard key={i} finding={f} />
            ))}
          </div>
        </>
      )}

      {phase.decisions.length > 0 && (
        <>
          <hr className="divider" />
          <div className="small muted" style={{ marginBottom: 10, fontWeight: 600 }}>
            Decisions — and the alternative each one rejected
          </div>
          <div style={{ display: "grid", gap: 12 }}>
            {phase.decisions.map((d, i) => (
              <div
                key={i}
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-sm)",
                  padding: "12px 14px",
                  background: "var(--surface-2)",
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.decision}</div>
                <p className="small" style={{ color: "var(--ink-secondary)", marginBottom: 8 }}>
                  {d.rationale}
                </p>
                {d.alternative_rejected && (
                  <div
                    className="small"
                    style={{
                      borderTop: "1px solid var(--border)",
                      paddingTop: 8,
                      color: "var(--ink-muted)",
                    }}
                  >
                    <strong style={{ color: "var(--status-serious)" }}>
                      Rejected: {d.alternative_rejected}
                    </strong>
                    {d.rejection_reason && <> — {d.rejection_reason}</>}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function FindingCard({ finding }: { finding: Finding }) {
  const tone = SEVERITY_TONE[finding.severity] ?? "good";
  return (
    <div
      style={{
        borderLeft: `3px solid var(--status-${tone})`,
        background: "var(--surface-2)",
        borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
        padding: "10px 14px",
      }}
    >
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <span aria-hidden="true" style={{ color: `var(--status-${tone})` }}>
          {SEVERITY_ICON[finding.severity]}
        </span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 580 }}>{finding.statement}</div>
          <div className="small mono" style={{ color: "var(--ink-secondary)", marginTop: 3 }}>
            {finding.evidence}
          </div>
          {finding.implication && (
            <div className="small" style={{ color: "var(--ink-muted)", marginTop: 5 }}>
              → {finding.implication}
            </div>
          )}
        </div>
        <Badge tone={tone}>{finding.severity}</Badge>
      </div>
    </div>
  );
}
