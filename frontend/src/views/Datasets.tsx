import { useApi } from "../lib/api";
import { compact } from "../lib/format";
import type { DatasetManifest } from "../lib/types";
import { Badge, Callout, ErrorState, Loading, Section } from "../components/ui";

export default function Datasets() {
  const manifest = useApi<DatasetManifest>("/datasets");

  return (
    <>
      <header className="pagehead">
        <div className="pagehead__inner">
          <div className="pagehead__eyebrow">
            <span>Reference</span>
          </div>
          <h1 className="pagehead__title">Data provenance</h1>
          <p className="pagehead__tagline">
            Every dataset behind this portfolio, with its original source, licence, exact
            curation rule and a SHA-256 of the committed file. Nothing here is synthetic.
          </p>
          <div style={{ height: 20 }} />
        </div>
      </header>

      <div className="content">
        {manifest.status === "loading" && <Loading what="the data manifest" />}
        {manifest.status === "error" && (
          <ErrorState error={manifest.error} onRetry={manifest.reload} />
        )}
        {manifest.status === "ready" && (
          <>
            <Callout tone="good" title="The curation contract">
              {manifest.data.contract}
            </Callout>

            <Section
              title={`${manifest.data.datasets.length} curated datasets`}
              description={`All subsampling is a pure function of seed ${manifest.data.seed}, so two clean checkouts produce byte-identical files. The checksums below are how you verify that.`}
              aside={`${compact(
                manifest.data.datasets.reduce((a, d) => a + d.rows, 0),
              )} rows total`}
            >
              <div style={{ display: "grid", gap: 16 }}>
                {manifest.data.datasets.map((d) => (
                  <div className="card" key={d.name}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 16,
                        alignItems: "flex-start",
                        flexWrap: "wrap",
                      }}
                    >
                      <div style={{ minWidth: 0 }}>
                        <h3 style={{ fontSize: "var(--text-md)" }}>{d.source.title}</h3>
                        <code className="small muted">{d.file}</code>
                      </div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        <Badge tone="accent">{compact(d.rows)} rows</Badge>
                        <Badge>{d.columns.length} columns</Badge>
                        <Badge>{d.size_mb.toFixed(2)} MB</Badge>
                        <Badge tone="good">{d.source.licence}</Badge>
                      </div>
                    </div>

                    <div className="kv small" style={{ marginTop: 16 }}>
                      <div className="kv__k">Source</div>
                      <div className="kv__v">
                        <a href={d.source.url} target="_blank" rel="noreferrer">
                          {d.source.url}
                        </a>
                      </div>
                      <div className="kv__k">Citation</div>
                      <div className="kv__v">{d.source.citation}</div>
                      <div className="kv__k">Curation rule</div>
                      <div className="kv__v">{d.curation_rule}</div>
                      <div className="kv__k">SHA-256</div>
                      <div className="kv__v mono small" style={{ wordBreak: "break-all" }}>
                        {d.sha256}
                      </div>
                    </div>

                    {d.notes.length > 0 && (
                      <ul className="prose small" style={{ marginTop: 12, marginBottom: 0 }}>
                        {d.notes.map((n) => (
                          <li key={n}>{n}</li>
                        ))}
                      </ul>
                    )}

                    <details style={{ marginTop: 12 }}>
                      <summary className="small muted" style={{ cursor: "pointer" }}>
                        Columns ({d.columns.length})
                      </summary>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
                        {d.columns.map((c) => (
                          <code
                            key={c}
                            className="small"
                            style={{
                              background: "var(--surface-2)",
                              padding: "2px 7px",
                              borderRadius: 3,
                              color: "var(--ink-secondary)",
                            }}
                          >
                            {c}
                          </code>
                        ))}
                      </div>
                    </details>
                  </div>
                ))}
              </div>
            </Section>
          </>
        )}
      </div>
    </>
  );
}
