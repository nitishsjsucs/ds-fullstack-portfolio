import { useEffect, useMemo, useState } from "react";
import { apiGet, useApi, useMutation } from "../../lib/api";
import { num, seriesColor, usd } from "../../lib/format";
import { ChartFrame, LineChart, Waterfall } from "../../charts/Charts";
import {
  Callout,
  ErrorState,
  Loading,
  SelectField,
  SliderField,
  Section,
  Stat,
} from "../../components/ui";

interface Zone {
  location_id: number;
  zone: string;
  borough: string;
}

interface Extras {
  zones: Zone[];
  presets: {
    label: string;
    pu_location_id: number;
    do_location_id: number;
    pickup_datetime: string;
    passenger_count: number;
    note: string;
  }[];
}

interface Prediction {
  duration_minutes: number;
  duration_interval_80: { lower: number; upper: number } | null;
  fare_usd: number | null;
  implied_speed_mph: number | null;
  context: Record<string, unknown>;
  attribution: {
    base_value: number;
    prediction: number;
    contributions: { feature: string; value: unknown; shap: number }[];
  } | null;
  latency_ms?: number;
  disclaimer: string;
}

interface Sensitivity {
  route: { from: string; to: string; distance_mi: number };
  curve: { hour: number; duration_minutes: number; lower?: number; upper?: number }[];
  peak: { hour: number; duration_minutes: number };
  trough: { hour: number; duration_minutes: number };
  peak_penalty_minutes: number;
  note: string;
}

export default function TaxiLive() {
  const extras = useApi<Extras>("/projects/taxi/extras");
  const predict = useMutation<Record<string, unknown>, Prediction>("/projects/taxi/predict");

  const [pu, setPu] = useState(161);
  const [dep, setDo] = useState(132);
  const [hour, setHour] = useState(17);
  const [dow, setDow] = useState(4);
  const [passengers, setPassengers] = useState(2);
  const [sensitivity, setSensitivity] = useState<Sensitivity | null>(null);

  // 2024-01-22 is a Monday, so adding the weekday index lands on the right day.
  const datetime = useMemo(
    () => `2024-01-${String(22 + dow).padStart(2, "0")}T${String(hour).padStart(2, "0")}:30:00`,
    [hour, dow],
  );

  useEffect(() => {
    predict.run({
      pu_location_id: pu,
      do_location_id: dep,
      pickup_datetime: datetime,
      passenger_count: passengers,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pu, dep, datetime, passengers]);

  useEffect(() => {
    apiGet<Sensitivity>(
      `/projects/taxi/sensitivity?pu_location_id=${pu}&do_location_id=${dep}&passenger_count=${passengers}`,
      { fresh: true },
    )
      .then(setSensitivity)
      .catch(() => setSensitivity(null));
  }, [pu, dep, passengers]);

  if (extras.status === "loading") return <Loading what="the zone dictionary" />;
  if (extras.status === "error") return <ErrorState error={extras.error} onRetry={extras.reload} />;

  const zones = extras.data.zones;
  const options = zones.map((z) => ({
    value: z.location_id,
    label: `${z.borough} · ${z.zone}`,
  }));
  const p = predict.data;

  return (
    <>
      <Section
        title="Trip estimator"
        description="A live call to the deployed pipeline on every change. The zone list is the TLC's own 265-zone dictionary, so these are real New York pickup and drop-off points."
        aside={p?.latency_ms ? `${num(p.latency_ms, 1)} ms` : undefined}
      >
        <div className="grid grid--3">
          <div className="card">
            <div className="small muted" style={{ marginBottom: 10, fontWeight: 600 }}>
              Route presets
            </div>
            <div className="chiprow" style={{ flexDirection: "column", marginBottom: 16 }}>
              {extras.data.presets.map((preset) => (
                <button
                  key={preset.label}
                  className="chip"
                  onClick={() => {
                    setPu(preset.pu_location_id);
                    setDo(preset.do_location_id);
                    const d = new Date(preset.pickup_datetime);
                    setHour(d.getHours());
                    setDow((d.getDay() + 6) % 7);
                    setPassengers(preset.passenger_count);
                  }}
                >
                  <div style={{ fontWeight: 600 }}>{preset.label}</div>
                  <div className="small muted" style={{ marginTop: 2 }}>
                    {preset.note}
                  </div>
                </button>
              ))}
            </div>

            <SelectField
              label="Pick-up zone"
              value={pu}
              options={options}
              onChange={(v) => setPu(Number(v))}
            />
            <SelectField
              label="Drop-off zone"
              value={dep}
              options={options}
              onChange={(v) => setDo(Number(v))}
            />
            <SliderField
              label="Departure hour"
              value={hour}
              min={0}
              max={23}
              onChange={setHour}
              format={(v) => `${String(v).padStart(2, "0")}:30`}
            />
            <SelectField
              label="Day of week"
              value={dow}
              options={["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map(
                (d, i) => ({ value: i, label: d }),
              )}
              onChange={(v) => setDow(Number(v))}
            />
            <SliderField
              label="Passengers"
              value={passengers}
              min={1}
              max={6}
              onChange={setPassengers}
            />
          </div>

          <div className="card" style={{ gridColumn: "span 2" }}>
            {predict.error && <ErrorState error={predict.error} />}
            {p && (
              <>
                <div className="grid grid--3" style={{ marginBottom: 16 }}>
                  <Stat
                    label="Duration"
                    value={num(p.duration_minutes, 1)}
                    unit="min"
                    hero
                    tone="accent"
                  />
                  <Stat
                    label="80% interval"
                    value={
                      p.duration_interval_80
                        ? `${num(p.duration_interval_80.lower, 0)}–${num(p.duration_interval_80.upper, 0)}`
                        : "—"
                    }
                    unit="min"
                    note="Paired quantile models, coverage-tested at 77.6%"
                  />
                  <Stat
                    label="Fare"
                    value={p.fare_usd !== null ? usd(p.fare_usd, 2) : "—"}
                    note="Excludes tips, tolls and surcharges"
                  />
                </div>

                <div className="kv small" style={{ marginBottom: 14 }}>
                  <div className="kv__k">Route</div>
                  <div className="kv__v">
                    {String(p.context.pickup_zone)} → {String(p.context.dropoff_zone)}
                  </div>
                  <div className="kv__k">Distance</div>
                  <div className="kv__v tnum">
                    {num(p.context.trip_distance_mi as number, 1)} mi
                    {p.context.distance_was_estimated ? (
                      <span className="small muted"> (estimated from the borough pair)</span>
                    ) : null}
                  </div>
                  <div className="kv__k">Implied speed</div>
                  <div className="kv__v tnum">{num(p.implied_speed_mph, 1)} mph</div>
                </div>

                {p.attribution && (
                  <ChartFrame
                    title="Why this number"
                    subtitle="TreeSHAP decomposition of this single prediction. The bars sum exactly from the base value to the final estimate."
                  >
                    <Waterfall
                      base={p.attribution.base_value}
                      contributions={p.attribution.contributions}
                      prediction={p.attribution.prediction}
                      format={(v) => num(v, 2)}
                      unit=" min"
                    />
                  </ChartFrame>
                )}

                <Callout tone="warning" title="Model limits">
                  {p.disclaimer}
                </Callout>
              </>
            )}
            {!p && predict.pending && <Loading what="the estimate" />}
          </div>
        </div>
      </Section>

      {sensitivity && (
        <Section
          title="The same trip, every hour of the day"
          description={sensitivity.note}
          aside={`${sensitivity.route.from} → ${sensitivity.route.to}`}
        >
          <ChartFrame
            title="Congestion cost across a Wednesday"
            subtitle={`Peak at ${sensitivity.peak.hour}:00 (${num(sensitivity.peak.duration_minutes, 1)} min) against a trough at ${sensitivity.trough.hour}:00 (${num(sensitivity.trough.duration_minutes, 1)} min) — a ${num(sensitivity.peak_penalty_minutes, 1)}-minute spread the model learned from the data alone.`}
          >
            <LineChart
              height={260}
              series={[
                {
                  name: "Estimated duration",
                  points: sensitivity.curve.map((c) => ({ x: c.hour, y: c.duration_minutes })),
                  color: seriesColor(0),
                },
              ]}
              band={
                sensitivity.curve[0]?.lower !== undefined
                  ? {
                      points: sensitivity.curve.map((c) => ({
                        x: c.hour,
                        lo: c.lower as number,
                        hi: c.upper as number,
                      })),
                      color: seriesColor(0),
                    }
                  : undefined
              }
              markers
              xLabel="Departure hour"
              yLabel="Minutes"
              xFormat={(v) => `${String(Math.round(v)).padStart(2, "0")}:00`}
              tipFormat={(v) => `${num(v, 1)} min`}
            />
          </ChartFrame>
        </Section>
      )}
    </>
  );
}
