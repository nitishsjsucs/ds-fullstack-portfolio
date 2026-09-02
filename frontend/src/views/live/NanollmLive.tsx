import { useState } from "react";
import { useApi, useMutation } from "../../lib/api";
import { compact, num, seriesColor } from "../../lib/format";
import { ChartFrame, Heatmap, LineChart } from "../../charts/Charts";
import {
  Badge,
  Callout,
  ErrorState,
  Loading,
  Section,
  SliderField,
  Stat,
} from "../../components/ui";

interface Generation {
  prompt: string;
  generated: string;
  full_text: string;
  settings: { temperature: number; top_k: number | null; max_new_tokens: number; seed: number };
  characters_dropped_from_prompt: number;
  prompt_note: string | null;
  caveat: string;
}

export default function NanollmLive() {
  const extras = useApi<{ prompts: string[]; temperature_guide: { temperature: number; character: string }[] }>(
    "/projects/nanollm/extras",
  );
  const curve = useApi<any>("/projects/nanollm/training-curve");
  const attention = useApi<any>("/projects/nanollm/attention");
  const generate = useMutation<Record<string, unknown>, Generation>("/projects/nanollm/generate");

  const [prompt, setPrompt] = useState("ROMEO:\n");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [tokens, setTokens] = useState(300);
  const [head, setHead] = useState(0);

  if (extras.status === "loading") return <Loading what="the model" />;
  if (extras.status === "error") return <ErrorState error={extras.error} onRetry={extras.reload} />;

  const g = generate.data;
  const guide = extras.data.temperature_guide.reduce((best, t) =>
    Math.abs(t.temperature - temperature) < Math.abs(best.temperature - temperature) ? t : best,
  );

  return (
    <>
      <Section
        title="Generation studio"
        description="Sampling runs on the trained weights in the browser's request cycle. Temperature and top-k are exposed because experiencing the trade-off directly is the most instructive thing this project offers."
      >
        <div className="grid grid--3">
          <div className="card">
            <div className="field">
              <label className="field__label">
                <span>Prompt</span>
              </label>
              <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} />
              <div className="field__hint">
                65-character vocabulary. Anything outside it is dropped rather than mapped to
                an unknown token — with a vocabulary this small there is no meaningful
                fallback symbol.
              </div>
            </div>
            <div className="chiprow">
              {extras.data.prompts.map((pr) => (
                <button key={pr} className="chip" onClick={() => setPrompt(pr)}>
                  {pr.replace(/\n/g, "⏎")}
                </button>
              ))}
            </div>

            <SliderField
              label="Temperature"
              value={temperature}
              min={0}
              max={1.6}
              step={0.1}
              onChange={setTemperature}
              format={(v) => v.toFixed(1)}
              hint={guide?.character}
            />
            <SliderField
              label="Top-k"
              value={topK}
              min={1}
              max={65}
              onChange={setTopK}
              hint="Restricts sampling to the k most likely next characters. k=1 is greedy; k=65 is unrestricted."
            />
            <SliderField
              label="Characters to generate"
              value={tokens}
              min={60}
              max={800}
              step={20}
              onChange={setTokens}
            />
            <button
              className="btn"
              style={{ width: "100%" }}
              disabled={generate.pending}
              onClick={() =>
                generate.run({
                  prompt,
                  temperature,
                  top_k: topK,
                  max_new_tokens: tokens,
                  seed: 42,
                })
              }
            >
              {generate.pending ? "Generating…" : "Generate"}
            </button>
          </div>

          <div className="card" style={{ gridColumn: "span 2" }}>
            {generate.error && <ErrorState error={generate.error} />}
            {g ? (
              <>
                <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
                  <Badge tone="accent">T = {g.settings.temperature}</Badge>
                  <Badge>top-k {g.settings.top_k}</Badge>
                  <Badge>seed {g.settings.seed}</Badge>
                </div>
                <pre
                  className="pre"
                  style={{ whiteSpace: "pre-wrap", minHeight: 300, fontSize: "var(--text-sm)", lineHeight: 1.65 }}
                >
                  <span style={{ color: "var(--accent, var(--series-1))", fontWeight: 600 }}>
                    {g.prompt}
                  </span>
                  {g.generated}
                </pre>
                {g.prompt_note && (
                  <Callout tone="warning" title="Prompt adjusted">
                    {g.prompt_note}
                  </Callout>
                )}
                <Callout tone="critical" title="What this model is">
                  {g.caveat}
                </Callout>
              </>
            ) : (
              <div className="state">
                <div className="state__title">Press Generate</div>
                Try temperature 0.2 first, then 1.4 — the difference between looping and
                spelling breakdown is the clearest demonstration of what temperature does.
              </div>
            )}
          </div>
        </div>
      </Section>

      {curve.status === "ready" && curve.data.history?.length > 0 && (
        <Section
          title="Training curve"
          description={curve.data.metric_note}
          aside={`perplexity ${num(curve.data.perplexity, 2)}`}
        >
          <div className="grid grid--4" style={{ marginBottom: 16 }}>
            <Stat
              label="Validation"
              value={num(curve.data.best?.val_bpc, 3)}
              unit=" bits/char"
              tone="accent"
            />
            <Stat
              label="Unigram baseline"
              value={num(curve.data.baselines?.unigram_frequency?.bits_per_char, 3)}
              unit=" bits/char"
              note="Marginal character distribution, no context"
            />
            <Stat
              label="Uniform baseline"
              value={num(curve.data.baselines?.uniform_random?.bits_per_char, 3)}
              unit=" bits/char"
              note="Guess among 65 characters"
            />
            <Stat
              label="Improvement"
              value={num(curve.data.improvement_over_unigram, 3)}
              unit=" bits"
              tone="good"
              note="Against the context-free floor"
            />
          </div>

          <ChartFrame
            title="Train and validation loss together"
            subtitle="Both curves are published in full. The step where they separate is the practical capacity limit of a model this size — cropping the chart there is how overfitting gets hidden."
            legend={[
              { label: "Train", color: seriesColor(0), shape: "line" },
              { label: "Validation", color: seriesColor(1), shape: "line" },
            ]}
            footnote={curve.data.overfitting?.verdict}
          >
            <LineChart
              height={280}
              series={[
                {
                  name: "Train",
                  points: curve.data.history.map((h: any) => ({ x: h.step, y: h.train_bpc })),
                  color: seriesColor(0),
                },
                {
                  name: "Validation",
                  points: curve.data.history.map((h: any) => ({ x: h.step, y: h.val_bpc })),
                  color: seriesColor(1),
                },
              ]}
              markers
              xLabel="Training step"
              yLabel="Bits per character"
              xFormat={(v) => compact(v)}
              tipFormat={(v) => `${num(v, 4)} bits`}
              reference={{
                y: curve.data.baselines?.unigram_frequency?.bits_per_char ?? 4.8,
                label: "unigram baseline",
                color: "var(--status-warning)",
              }}
            />
          </ChartFrame>
        </Section>
      )}

      {attention.status === "ready" && attention.data?.heads?.length > 0 && (
        <Section
          title="Attention, from the trained weights"
          description={attention.data.note}
          aside={`layer ${attention.data.layer}`}
        >
          <div className="chiprow">
            {attention.data.heads.map((h: any, i: number) => (
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
            title={`Head ${attention.data.heads[head]?.head} on "${attention.data.text}"`}
            subtitle="Row i shows where position i looked. The entire upper triangle is exactly zero — that is the causal mask, verifiable in the weights rather than asserted."
          >
            <Heatmap
              rows={attention.data.tokens}
              cols={attention.data.tokens}
              matrix={attention.data.heads[head]?.matrix ?? []}
              height={Math.max(300, attention.data.tokens.length * 13 + 60)}
              valueFormat={(v) => v.toFixed(3)}
            />
          </ChartFrame>
        </Section>
      )}
    </>
  );
}
