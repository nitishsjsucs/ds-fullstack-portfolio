import { useEffect, useState } from "react";
import { apiGet, useApi, useMutation } from "../../lib/api";
import { num, pct } from "../../lib/format";
import { ChartFrame, NetworkGraph } from "../../charts/Charts";
import {
  Badge,
  Callout,
  DataTable,
  ErrorState,
  Loading,
  Section,
  SliderField,
} from "../../components/ui";

interface Item {
  item: string;
  baskets: number;
  support: number;
}

interface Recommendation {
  item: string;
  lift: number;
  confidence: number;
  support: number;
  because_of: string[];
  interpretation: string;
  holdout_lift?: number;
  lift_retained?: number;
}

interface Prediction {
  basket: string[];
  recommendations: Recommendation[];
  rules_matched: number;
  served_from: string;
  note: string;
}

export default function BasketLive() {
  const items = useApi<{ count: number; items: Item[] }>("/projects/basket/items?limit=400");
  const predict = useMutation<Record<string, unknown>, Prediction>("/projects/basket/predict");

  const [basket, setBasket] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [minLift, setMinLift] = useState(2);
  const [network, setNetwork] = useState<any>(null);

  useEffect(() => {
    if (basket.length) predict.run({ basket, top_k: 8 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [basket]);

  useEffect(() => {
    apiGet<any>(`/projects/basket/network?min_lift=${minLift}`, { fresh: true })
      .then(setNetwork)
      .catch(() => setNetwork(null));
  }, [minLift]);

  if (items.status === "loading") return <Loading what="the item catalogue" />;
  if (items.status === "error") return <ErrorState error={items.error} onRetry={items.reload} />;

  const filtered = query
    ? items.data.items.filter((i) => i.item.includes(query.toUpperCase())).slice(0, 40)
    : items.data.items.slice(0, 40);
  const p = predict.data;

  return (
    <>
      <Section
        title="Basket completion"
        description="Add real products from the UCI Online Retail catalogue and the recommender fires only rules that survived the temporal hold-out — patterns that still held on invoices the miner never saw."
      >
        <div className="grid grid--2">
          <div className="card">
            <div className="field">
              <label className="field__label">
                <span>Search the catalogue</span>
                <span className="field__value">{items.data.count} items</span>
              </label>
              <input
                type="text"
                placeholder="e.g. HEART, BAG, CHRISTMAS…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>

            <div style={{ maxHeight: 260, overflowY: "auto", marginBottom: 14 }}>
              {filtered.map((i) => (
                <button
                  key={i.item}
                  className="chip"
                  style={{ display: "block", width: "100%", marginBottom: 4 }}
                  onClick={() =>
                    setBasket((b) => (b.includes(i.item) ? b : [...b, i.item]))
                  }
                  disabled={basket.includes(i.item)}
                >
                  <span style={{ fontWeight: 500 }}>{i.item}</span>
                  <span className="small muted" style={{ float: "right" }}>
                    {pct(i.support, 1)}
                  </span>
                </button>
              ))}
            </div>

            <div className="small muted" style={{ marginBottom: 8, fontWeight: 600 }}>
              Current basket ({basket.length})
            </div>
            {basket.length === 0 ? (
              <div className="small muted">Add an item to see what goes with it.</div>
            ) : (
              <div className="chiprow">
                {basket.map((b) => (
                  <button key={b} className="chip chip--active" onClick={() => setBasket((x) => x.filter((y) => y !== b))}>
                    {b} ✕
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="card">
            <div className="chartframe__title" style={{ marginBottom: 4 }}>
              Recommendations
            </div>
            {p ? (
              <>
                <div className="chartframe__sub" style={{ marginBottom: 12 }}>
                  {p.rules_matched} rules fired · served from {p.served_from}
                </div>
                {p.recommendations.length === 0 ? (
                  <Callout tone="warning" title="No confirmed rule fires for this basket">
                    {p.note}
                  </Callout>
                ) : (
                  <DataTable
                    columns={[
                      {
                        key: "i",
                        header: "Suggested item",
                        render: (r: Recommendation) => (
                          <>
                            <div style={{ fontWeight: 560 }}>{r.item}</div>
                            <div className="small muted">because of {r.because_of.join(" + ")}</div>
                          </>
                        ),
                      },
                      {
                        key: "l",
                        header: "Lift",
                        numeric: true,
                        render: (r: Recommendation) => (
                          <Badge tone={r.lift >= 3 ? "good" : r.lift >= 1.5 ? "warning" : "neutral"}>
                            {num(r.lift, 1)}×
                          </Badge>
                        ),
                      },
                      {
                        key: "c",
                        header: "Confidence",
                        numeric: true,
                        render: (r: Recommendation) => pct(r.confidence),
                      },
                      {
                        key: "h",
                        header: "Out-of-sample lift",
                        numeric: true,
                        render: (r: Recommendation) =>
                          r.holdout_lift ? num(r.holdout_lift, 1) : "—",
                      },
                    ]}
                    rows={p.recommendations}
                    rowKey={(r) => r.item}
                  />
                )}
                <div className="small muted" style={{ marginTop: 12, lineHeight: 1.5 }}>
                  {p.note}
                </div>
              </>
            ) : (
              <div className="state">Add items to the basket to get recommendations.</div>
            )}
          </div>
        </div>
      </Section>

      {network && (
        <Section
          title="Rule network"
          description="Every single-item rule above the lift threshold. Node size is how many rules touch an item; edge thickness is lift. Raising the threshold prunes the weak dependencies and leaves the genuine complements."
          aside={`${network.links.length} rules · ${network.nodes.length} items`}
        >
          <div style={{ maxWidth: 420, marginBottom: 12 }}>
            <SliderField
              label="Minimum lift"
              value={minLift}
              min={1.2}
              max={12}
              step={0.2}
              onChange={setMinLift}
              format={(v) => `${v.toFixed(1)}×`}
              hint="Lift 1.0 means the two items are independent — the rule would just be restating base rates."
            />
          </div>
          <ChartFrame title={`Co-occurrence above ${minLift.toFixed(1)}× lift`}>
            <NetworkGraph nodes={network.nodes} links={network.links} height={460} />
          </ChartFrame>
        </Section>
      )}
    </>
  );
}
