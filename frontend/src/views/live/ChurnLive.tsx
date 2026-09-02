import { useEffect, useState } from "react";
import { useApi, useMutation } from "../../lib/api";
import { num, pct, seriesColor, usd } from "../../lib/format";
import { ChartFrame, LineChart, Waterfall } from "../../charts/Charts";
import {
  Badge,
  Callout,
  ErrorState,
  Loading,
  SelectField,
  SliderField,
  Section,
  Stat,
} from "../../components/ui";

interface Schema {
  domains: Record<string, string[] | { tenure: number[]; MonthlyCharges: number[] }>;
  personas: { label: string; note: string; payload: Record<string, unknown> }[];
  economics: Record<string, number | string>;
  deployed_threshold: number;
}

interface Prediction {
  churn_probability: number;
  risk_band: string;
  decision_threshold: number;
  flagged_for_outreach: boolean;
  economics: {
    customer_lifetime_margin: number;
    expected_loss_if_no_action: number;
    expected_value_of_offer: number;
    breakeven_probability: number;
    recommended_action: string;
  };
  context: Record<string, unknown>;
  attribution: {
    base_value: number;
    prediction: number;
    contributions: { feature: string; value: unknown; shap: number }[];
  } | null;
  note: string;
}

export default function ChurnLive() {
  const schema = useApi<Schema>("/projects/churn/form-schema");
  const predict = useMutation<Record<string, unknown>, Prediction>("/projects/churn/predict");

  const [customer, setCustomer] = useState<Record<string, unknown>>({
    tenure: 2,
    MonthlyCharges: 89.9,
    Contract: "Month-to-month",
    PaymentMethod: "Electronic check",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    TechSupport: "No",
  });
  const [offerCost, setOfferCost] = useState(45);
  const [acceptance, setAcceptance] = useState(0.35);
  const [whatIf, setWhatIf] = useState<any>(null);
  const [field, setField] = useState("tenure");

  useEffect(() => {
    predict.run({
      ...customer,
      economics: { retention_offer_cost: offerCost, offer_acceptance_rate: acceptance },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customer, offerCost, acceptance]);

  useEffect(() => {
    fetch("/api/projects/churn/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ field, customer }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then(setWhatIf)
      .catch(() => setWhatIf(null));
  }, [field, customer]);

  if (schema.status === "loading") return <Loading what="the customer form" />;
  if (schema.status === "error") return <ErrorState error={schema.error} onRetry={schema.reload} />;

  const domains = schema.data.domains as Record<string, string[]>;
  const ranges = (schema.data.domains as any)._numeric_ranges ?? {};
  const p = predict.data;

  const set = (k: string, v: unknown) => setCustomer((c) => ({ ...c, [k]: v }));

  return (
    <>
      <Section
        title="Retention scorer"
        description="Not just a probability — the expected value of making an offer. Drag the economics and watch the recommendation change: that is the whole point of a calibrated model."
      >
        <div className="grid grid--3">
          <div className="card">
            <div className="small muted" style={{ marginBottom: 10, fontWeight: 600 }}>
              Customer profiles
            </div>
            <div className="chiprow" style={{ flexDirection: "column", marginBottom: 16 }}>
              {schema.data.personas.map((persona) => (
                <button
                  key={persona.label}
                  className="chip"
                  onClick={() => setCustomer(persona.payload)}
                >
                  <div style={{ fontWeight: 600 }}>{persona.label}</div>
                  <div className="small muted" style={{ marginTop: 2 }}>
                    {persona.note}
                  </div>
                </button>
              ))}
            </div>

            <SliderField
              label="Tenure (months)"
              value={Number(customer.tenure ?? 12)}
              min={0}
              max={ranges.tenure?.[1] ?? 72}
              onChange={(v) => set("tenure", v)}
            />
            <SliderField
              label="Monthly charges"
              value={Number(customer.MonthlyCharges ?? 70)}
              min={Math.floor(ranges.MonthlyCharges?.[0] ?? 18)}
              max={Math.ceil(ranges.MonthlyCharges?.[1] ?? 120)}
              step={0.5}
              onChange={(v) => set("MonthlyCharges", v)}
              format={(v) => usd(v, 2)}
            />
            {["Contract", "PaymentMethod", "InternetService", "OnlineSecurity", "TechSupport"].map(
              (f) =>
                domains[f] ? (
                  <SelectField
                    key={f}
                    label={f.replace(/([A-Z])/g, " $1").trim()}
                    value={String(customer[f] ?? domains[f][0])}
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
                  label="Churn probability"
                  value={pct(p.churn_probability)}
                  hero
                  tone={p.risk_band === "high" ? "critical" : p.risk_band === "medium" ? "warning" : "good"}
                  note={`${p.risk_band} risk`}
                />
                <div style={{ margin: "16px 0" }}>
                  {p.flagged_for_outreach ? (
                    <Badge tone="critical" icon="⚠">
                      flagged for outreach
                    </Badge>
                  ) : (
                    <Badge tone="good" icon="✓">
                      no action needed
                    </Badge>
                  )}
                </div>
                <div className="small muted" style={{ lineHeight: 1.5, marginBottom: 14 }}>
                  {p.note}
                </div>

                <hr className="divider" />
                <div className="small muted" style={{ marginBottom: 8, fontWeight: 600 }}>
                  Retention economics — drag these
                </div>
                <SliderField
                  label="Retention offer cost"
                  value={offerCost}
                  min={5}
                  max={200}
                  step={5}
                  onChange={setOfferCost}
                  format={(v) => usd(v)}
                />
                <SliderField
                  label="Offer acceptance rate"
                  value={acceptance}
                  min={0.05}
                  max={0.9}
                  step={0.05}
                  onChange={setAcceptance}
                  format={(v) => pct(v, 0)}
                  hint="What fraction of customers who receive an offer actually stay because of it."
                />
              </>
            )}
          </div>

          <div className="card">
            {p && (
              <>
                <div className="chartframe__title" style={{ marginBottom: 12 }}>
                  Is an offer worth making?
                </div>
                <div className="grid grid--2" style={{ gap: 10, marginBottom: 14 }}>
                  <Stat label="Lifetime margin" value={usd(p.economics.customer_lifetime_margin)} />
                  <Stat
                    label="Expected loss if ignored"
                    value={usd(p.economics.expected_loss_if_no_action)}
                    tone="warning"
                  />
                  <Stat
                    label="Value of offering"
                    value={usd(p.economics.expected_value_of_offer)}
                    tone={p.economics.expected_value_of_offer > 0 ? "good" : "critical"}
                  />
                  <Stat
                    label="Break-even probability"
                    value={pct(p.economics.breakeven_probability)}
                    note="Below this, the offer destroys value"
                  />
                </div>
                <Callout
                  tone={p.economics.expected_value_of_offer > 0 ? "good" : "warning"}
                  title="Recommendation"
                >
                  {p.economics.recommended_action}
                </Callout>

                {p.attribution && (
                  <ChartFrame
                    title="What drives this score"
                    subtitle="TreeSHAP on the uncalibrated model — the calibrated wrapper is a monotone rescale, so the ordering of contributions is unchanged."
                  >
                    <Waterfall
                      base={p.attribution.base_value}
                      contributions={p.attribution.contributions}
                      prediction={p.attribution.prediction}
                      format={(v) => num(v, 3)}
                    />
                  </ChartFrame>
                )}
              </>
            )}
          </div>
        </div>
      </Section>

      {whatIf && (
        <Section
          title="Counterfactual sweep"
          description={whatIf.note}
          aside={`swing ${pct(whatIf.swing)}`}
        >
          <div className="chiprow">
            {["tenure", "MonthlyCharges", "Contract", "PaymentMethod", "InternetService"].map((f) => (
              <button
                key={f}
                className={`chip${field === f ? " chip--active" : ""}`}
                onClick={() => setField(f)}
              >
                {f}
              </button>
            ))}
          </div>
          <ChartFrame
            title={`Churn probability as ${whatIf.field} varies`}
            subtitle={`Lowest risk at ${String(whatIf.best_value)}, highest at ${String(whatIf.worst_value)}. Everything else is held fixed.`}
          >
            {typeof whatIf.curve[0]?.value === "number" ? (
              <LineChart
                height={230}
                series={[
                  {
                    name: "Churn probability",
                    points: whatIf.curve.map((c: any) => ({ x: c.value, y: c.churn_probability })),
                    color: seriesColor(5),
                  },
                ]}
                markers
                xLabel={whatIf.field}
                yLabel="P(churn)"
                yFormat={(v) => v.toFixed(2)}
                tipFormat={(v) => pct(v)}
                reference={{
                  y: whatIf.threshold,
                  label: "deployed threshold",
                  color: "var(--status-warning)",
                }}
              />
            ) : (
              <div className="tablewrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>{whatIf.field}</th>
                      <th className="num">P(churn)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {whatIf.curve.map((c: any) => (
                      <tr key={String(c.value)}>
                        <td>{String(c.value)}</td>
                        <td className="num">{pct(c.churn_probability)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </ChartFrame>
        </Section>
      )}
    </>
  );
}
