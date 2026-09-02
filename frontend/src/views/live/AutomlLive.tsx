import { useEffect, useState } from "react";
import { useApi, useMutation } from "../../lib/api";
import { num, pct } from "../../lib/format";
import { ChartFrame, Waterfall } from "../../charts/Charts";
import {
  Badge,
  Callout,
  ErrorState,
  Loading,
  Section,
  SelectField,
  SliderField,
  Stat,
} from "../../components/ui";

interface Schema {
  domains: Record<string, string[]> & { _numeric_ranges?: Record<string, number[]> };
  profiles: { label: string; note: string; payload: Record<string, unknown> }[];
  dropped_columns: Record<string, string>;
}

interface Prediction {
  probability_above_50k: number;
  prediction: string;
  context: Record<string, unknown>;
  distilled_tree?: {
    probability_above_50k: number;
    prediction: string;
    agrees_with_champion: boolean;
    note: string;
  };
  attribution: {
    base_value: number;
    prediction: number;
    contributions: { feature: string; value: unknown; shap: number }[];
  } | null;
  disclaimer: string;
}

export default function AutomlLive() {
  const schema = useApi<Schema>("/projects/automl/form-schema");
  const predict = useMutation<Record<string, unknown>, Prediction>("/projects/automl/predict");
  const [record, setRecord] = useState<Record<string, unknown>>({
    age: 38,
    education_num: 13,
    hours_per_week: 45,
    occupation: "Prof-specialty",
    marital_status: "Never-married",
    relationship: "Not-in-family",
    workclass: "Private",
    capital_gain: 0,
  });

  useEffect(() => {
    predict.run(record);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record]);

  if (schema.status === "loading") return <Loading what="the record form" />;
  if (schema.status === "error") return <ErrorState error={schema.error} onRetry={schema.reload} />;

  const domains = schema.data.domains;
  const ranges = domains._numeric_ranges ?? {};
  const p = predict.data;
  const set = (k: string, v: unknown) => setRecord((r) => ({ ...r, [k]: v }));

  return (
    <>
      <Callout tone="critical" title="This is a benchmark, not a service">
        Trained on a 1994 US census extract. It must not inform any decision about any real
        person — lending, hiring, housing, insurance or eligibility of any kind. What it
        demonstrates is the methodology of automated model selection.
      </Callout>

      <Section
        title="Score a record with both models"
        description="The champion ensemble and the depth-4 distilled tree run side by side. Where they agree, the ensemble's answer is explainable in four questions — which is the whole argument for shipping the interpretable model in a regulated setting."
      >
        <div className="grid grid--3">
          <div className="card">
            <div className="small muted" style={{ marginBottom: 10, fontWeight: 600 }}>
              Example records
            </div>
            <div className="chiprow" style={{ flexDirection: "column", marginBottom: 16 }}>
              {schema.data.profiles.map((prof) => (
                <button key={prof.label} className="chip" onClick={() => setRecord(prof.payload)}>
                  <div style={{ fontWeight: 600 }}>{prof.label}</div>
                  <div className="small muted" style={{ marginTop: 2 }}>
                    {prof.note}
                  </div>
                </button>
              ))}
            </div>

            <SliderField label="Age" value={Number(record.age ?? 38)} min={17} max={90} onChange={(v) => set("age", v)} />
            <SliderField
              label="Education (ordinal)"
              value={Number(record.education_num ?? 13)}
              min={1}
              max={16}
              onChange={(v) => set("education_num", v)}
              hint="1 = pre-school, 9 = high-school grad, 13 = bachelor's, 16 = doctorate."
            />
            <SliderField
              label="Hours per week"
              value={Number(record.hours_per_week ?? 40)}
              min={1}
              max={99}
              onChange={(v) => set("hours_per_week", v)}
            />
            <SliderField
              label="Capital gain"
              value={Number(record.capital_gain ?? 0)}
              min={0}
              max={Math.min(30000, ranges.capital_gain?.[1] ?? 30000)}
              step={500}
              onChange={(v) => set("capital_gain", v)}
              hint="Sparse but enormously predictive — the single most decisive feature in the model."
            />
            {["occupation", "marital_status", "relationship", "workclass"].map((f) =>
              domains[f] ? (
                <SelectField
                  key={f}
                  label={f.replace(/_/g, " ")}
                  value={String(record[f] ?? domains[f][0])}
                  options={domains[f].map((v) => ({ value: v, label: v }))}
                  onChange={(v) => set(f, v)}
                />
              ) : null,
            )}
          </div>

          <div className="card">
            {p && (
              <>
                <Stat
                  label="Champion ensemble"
                  value={pct(p.probability_above_50k)}
                  hero
                  tone="accent"
                  note={`predicts ${p.prediction}`}
                />
                {p.distilled_tree && (
                  <>
                    <hr className="divider" />
                    <Stat
                      label="Depth-4 distilled tree"
                      value={pct(p.distilled_tree.probability_above_50k)}
                      note={`predicts ${p.distilled_tree.prediction}`}
                    />
                    <div style={{ marginTop: 10 }}>
                      {p.distilled_tree.agrees_with_champion ? (
                        <Badge tone="good" icon="✓">models agree</Badge>
                      ) : (
                        <Badge tone="warning" icon="⚠">models disagree</Badge>
                      )}
                    </div>
                    <div className="small muted" style={{ marginTop: 10, lineHeight: 1.5 }}>
                      {p.distilled_tree.note}
                    </div>
                  </>
                )}
              </>
            )}
          </div>

          <div className="card">
            {p?.attribution && (
              <ChartFrame
                title="Contribution breakdown"
                subtitle="TreeSHAP on the champion. The bars sum exactly from the population base rate to this record's score."
              >
                <Waterfall
                  base={p.attribution.base_value}
                  contributions={p.attribution.contributions}
                  prediction={p.attribution.prediction}
                  format={(v) => num(v, 3)}
                />
              </ChartFrame>
            )}
          </div>
        </div>
      </Section>

      <Section
        title="Columns excluded by design"
        description="Two of these are the traps in this dataset. Dropping them costs measurable AUC — which is the point: that performance was never available in deployment."
      >
        <div className="grid grid--2">
          {Object.entries(schema.data.dropped_columns).map(([col, why]) => (
            <div className="card" key={col}>
              <div style={{ marginBottom: 6 }}>
                <Badge tone="critical">{col}</Badge>
              </div>
              <p className="small" style={{ color: "var(--ink-secondary)", margin: 0, lineHeight: 1.55 }}>
                {why}
              </p>
            </div>
          ))}
        </div>
      </Section>

      <FairnessSection />
    </>
  );
}

function FairnessSection() {
  const fair = useApi<any>("/projects/automl/fairness");
  if (fair.status !== "ready") return null;
  const d = fair.data;

  return (
    <Section
      title="Fairness audit"
      description={d.measurement_only_note}
      aside={`prioritises ${d.prioritised_criterion}`}
    >
      <Callout tone="critical" title="Exclusion did not remove the disparity">
        Sex and race are absent from the feature set, yet the worst disparate-impact ratio
        is {num(d.worst_disparate_impact, 3)} and{" "}
        {d.four_fifths_failures.length
          ? `${d.four_fifths_failures.join(", ")} fail the four-fifths rule`
          : "no attribute fails the four-fifths rule"}
        . Correlated proxies — relationship, occupation, hours — carry the same
        information. This is why "fairness through unawareness" is not a strategy.
      </Callout>

      <div className="grid grid--3">
        {d.attributes.map((attr: any) => (
          <div className="card" key={attr.attribute}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
              <h3 style={{ fontSize: "var(--text-md)" }}>{attr.attribute}</h3>
              {attr.passes_four_fifths === undefined ? null : attr.passes_four_fifths ? (
                <Badge tone="good" icon="✓">passes 4/5</Badge>
              ) : (
                <Badge tone="critical" icon="✕">fails 4/5</Badge>
              )}
            </div>
            {attr.disparate_impact && (
              <div style={{ marginTop: 12 }}>
                {Object.entries(attr.disparate_impact as Record<string, number>).map(([g, v]) => (
                  <div key={g} style={{ marginBottom: 8 }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "var(--text-sm)",
                        marginBottom: 3,
                      }}
                    >
                      <span>{g}</span>
                      <span className="tnum mono" style={{ color: v >= 0.8 ? "var(--status-good)" : "var(--status-critical)" }}>
                        {num(v, 3)}
                      </span>
                    </div>
                    <div style={{ background: "var(--surface-3)", height: 5, borderRadius: 3 }}>
                      <div
                        style={{
                          width: `${Math.min(1, v) * 100}%`,
                          height: "100%",
                          borderRadius: 3,
                          background: v >= 0.8 ? "var(--status-good)" : "var(--status-critical)",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="kv small" style={{ marginTop: 12 }}>
              <div className="kv__k">Equal opportunity gap</div>
              <div className="kv__v tnum">{num(attr.equal_opportunity_difference, 4)}</div>
              <div className="kv__k">Equalised odds gap</div>
              <div className="kv__v tnum">{num(attr.equalised_odds_difference, 4)}</div>
              <div className="kv__k">Base rate spread</div>
              <div className="kv__v tnum">{num(attr.base_rate_difference, 4)}</div>
            </div>
          </div>
        ))}
      </div>

      <Callout tone="warning" title="Why all four criteria are shown">
        {d.impossibility_note}
      </Callout>
    </Section>
  );
}
