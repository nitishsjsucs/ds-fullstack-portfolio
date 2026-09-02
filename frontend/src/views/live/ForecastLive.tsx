import { useEffect, useState } from "react";
import { apiPost, useApi } from "../../lib/api";
import { compact, num, pct, seriesColor, signed } from "../../lib/format";
import { ChartFrame, LineChart } from "../../charts/Charts";
import {
  Callout,
  Loading,
  Section,
  SelectField,
  SliderField,
  Stat,
} from "../../components/ui";

const WEATHER = [
  { value: 1, label: "Clear or partly cloudy" },
  { value: 2, label: "Mist / cloudy" },
  { value: 3, label: "Light rain or snow" },
];

interface ScenarioResult {
  date: string;
  scenarios: Record<string, { hour: number; predicted_rides: number; lower: number | null; upper: number | null }[]>;
  daily_totals: Record<string, number>;
  weather_impact?: { scenario: string; rides_lost: number; pct_change: number };
}

export default function ForecastLive() {
  const decomposition = useApi<any>("/projects/forecast/decomposition");
  const [weather, setWeather] = useState(3);
  const [temp, setTemp] = useState(0.55);
  const [working, setWorking] = useState(1);
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPending(true);
    apiPost<ScenarioResult>("/projects/forecast/scenario", {
      date: "2012-10-17",
      weathersit: weather,
      temp,
      atemp: temp,
      workingday: working,
    })
      .then((r) => {
        setResult(r);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Scenario failed"))
      .finally(() => setPending(false));
  }, [weather, temp, working]);

  const names = result ? Object.keys(result.scenarios) : [];

  return (
    <>
      <Section
        title="Weather scenario planner"
        description="Forecast a full day under different conditions. The comparison against clear weather is the actionable output — it prices the demand impact of rain for a rebalancing team."
      >
        <div className="grid grid--3">
          <div className="card">
            <SelectField
              label="Weather"
              value={weather}
              options={WEATHER}
              onChange={(v) => setWeather(Number(v))}
            />
            <SliderField
              label="Temperature (normalised)"
              value={temp}
              min={0.05}
              max={0.95}
              step={0.05}
              onChange={setTemp}
              format={(v) => `${(v * 41).toFixed(0)}°C scale`}
              hint="The archive publishes temperature min-max normalised; 0.5 is roughly 20°C."
            />
            <SelectField
              label="Day type"
              value={working}
              options={[
                { value: 1, label: "Working day" },
                { value: 0, label: "Weekend or holiday" },
              ]}
              onChange={(v) => setWorking(Number(v))}
            />

            {result?.weather_impact && (
              <>
                <hr className="divider" />
                <Stat
                  label={`Impact of ${result.weather_impact.scenario}`}
                  value={signed(result.weather_impact.rides_lost, 0)}
                  unit=" rides"
                  tone={result.weather_impact.rides_lost > 0 ? "critical" : "good"}
                  note={`${pct(result.weather_impact.pct_change)} against a clear day`}
                />
              </>
            )}
            {result && (
              <div style={{ marginTop: 14 }}>
                {Object.entries(result.daily_totals).map(([k, v]) => (
                  <div key={k} className="kv small" style={{ marginBottom: 4 }}>
                    <div className="kv__k">{k}</div>
                    <div className="kv__v tnum">{compact(v)} rides</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card" style={{ gridColumn: "span 2" }}>
            {error && <Callout tone="critical" title="Scenario failed">{error}</Callout>}
            {pending && !result && <Loading what="the forecast" />}
            {result && (
              <ChartFrame
                title="Hourly demand across the day"
                subtitle="Conformally calibrated 80% band on the primary scenario. The twin commuting peaks are the model reading the working-day rhythm out of the lag features."
                legend={names.map((n, i) => ({ label: n, color: seriesColor(i), shape: "line" }))}
              >
                <LineChart
                  height={300}
                  series={names.map((n, i) => ({
                    name: n,
                    points: result.scenarios[n].map((p) => ({ x: p.hour, y: p.predicted_rides })),
                    color: seriesColor(i),
                  }))}
                  band={
                    result.scenarios[names[names.length - 1]]?.[0]?.lower != null
                      ? {
                          points: result.scenarios[names[names.length - 1]].map((p) => ({
                            x: p.hour,
                            lo: p.lower as number,
                            hi: p.upper as number,
                          })),
                          color: seriesColor(names.length - 1),
                        }
                      : undefined
                  }
                  markers
                  xLabel="Hour of day"
                  yLabel="Rides per hour"
                  xFormat={(v) => `${String(Math.round(v)).padStart(2, "0")}:00`}
                  tipFormat={(v) => `${num(v, 0)} rides`}
                />
              </ChartFrame>
            )}
          </div>
        </div>
      </Section>

      {decomposition.status === "ready" && decomposition.data?.points?.length > 0 && (
        <Section
          title="What the series is made of"
          description={decomposition.data.note}
          aside={`seasonal share ${pct(decomposition.data.variance_share?.seasonal)}`}
        >
          <div className="grid grid--2" style={{ marginBottom: 16 }}>
            <Stat
              label="Seasonal variance"
              value={pct(decomposition.data.variance_share?.seasonal)}
              tone="accent"
              note="Pure daily rhythm — available from knowing the hour alone"
            />
            <Stat
              label="Residual variance"
              value={pct(decomposition.data.variance_share?.residual)}
              note="What a model has to earn beyond the calendar"
            />
          </div>
          <ChartFrame
            title="STL decomposition"
            subtitle="Observed demand split into trend, a 24-hour seasonal component and the residual. The seasonal share is the ceiling on how much a calendar-only model could achieve."
            legend={[
              { label: "Observed", color: seriesColor(0), shape: "line" },
              { label: "Trend", color: seriesColor(1), shape: "line" },
              { label: "Seasonal", color: seriesColor(2), shape: "line" },
              { label: "Residual", color: seriesColor(3), shape: "line" },
            ]}
          >
            <LineChart
              height={300}
              series={[
                { name: "Observed", points: decomposition.data.points.map((p: any) => ({ x: p.i, y: p.observed })), color: seriesColor(0) },
                { name: "Trend", points: decomposition.data.points.map((p: any) => ({ x: p.i, y: p.trend })), color: seriesColor(1) },
                { name: "Seasonal", points: decomposition.data.points.map((p: any) => ({ x: p.i, y: p.seasonal })), color: seriesColor(2) },
                { name: "Residual", points: decomposition.data.points.map((p: any) => ({ x: p.i, y: p.residual })), color: seriesColor(3) },
              ]}
              xLabel="Hour index"
              yLabel="Rides"
              xFormat={(v) => compact(v)}
              tipFormat={(v) => num(v, 1)}
            />
          </ChartFrame>

          {decomposition.data.stationarity && (
            <Callout tone="info" title="Stationarity">
              {decomposition.data.stationarity.interpretation} ADF p-value{" "}
              {num(decomposition.data.stationarity.adf?.p_value, 4)}, KPSS p-value{" "}
              {num(decomposition.data.stationarity.kpss?.p_value, 4)}.
            </Callout>
          )}
        </Section>
      )}

      <CorrelogramSection />
    </>
  );
}

function CorrelogramSection() {
  const corr = useApi<any>("/projects/forecast/correlogram");
  if (corr.status !== "ready") return null;
  const acf = corr.data.acf ?? [];
  const pacf = corr.data.pacf ?? [];

  return (
    <Section
      title="Correlogram"
      description={corr.data.note}
      aside={`95% band ±${num(corr.data.confidence_band, 3)}`}
    >
      <div className="grid grid--2">
        <ChartFrame
          title="Autocorrelation (ACF)"
          subtitle="Peaks at lag 24 and 168 are yesterday's same hour and last week's same hour."
        >
          <LineChart
            height={230}
            series={[{ name: "ACF", points: acf.map((a: any) => ({ x: a.lag, y: a.value })), color: seriesColor(0) }]}
            xLabel="Lag (hours)"
            xFormat={(v) => String(Math.round(v))}
            tipFormat={(v) => num(v, 3)}
            reference={{ y: corr.data.confidence_band, label: "95% band", color: "var(--ink-muted)" }}
          />
        </ChartFrame>
        <ChartFrame
          title="Partial autocorrelation (PACF)"
          subtitle="Correlation with each lag after removing the effect of the shorter lags — this is what separates a genuine weekly signal from an echo of the daily one."
        >
          <LineChart
            height={230}
            series={[{ name: "PACF", points: pacf.map((a: any) => ({ x: a.lag, y: a.value })), color: seriesColor(1) }]}
            xLabel="Lag (hours)"
            xFormat={(v) => String(Math.round(v))}
            tipFormat={(v) => num(v, 3)}
            reference={{ y: corr.data.confidence_band, label: "95% band", color: "var(--ink-muted)" }}
          />
        </ChartFrame>
      </div>
    </Section>
  );
}
