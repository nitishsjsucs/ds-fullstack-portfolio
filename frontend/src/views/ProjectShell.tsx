/** Per-project workspace: header, tab bar and tab content. */

import { href } from "../lib/router";
import type { ProjectMeta } from "../lib/types";
import { shortDate } from "../lib/format";
import { Badge } from "../components/ui";
import OverviewTab from "./tabs/OverviewTab";
import DataTab from "./tabs/DataTab";
import CrispDmTab from "./tabs/CrispDmTab";
import ModelsTab from "./tabs/ModelsTab";
import EvaluationTab from "./tabs/EvaluationTab";
import ExplainTab from "./tabs/ExplainTab";
import AuditTab from "./tabs/AuditTab";
import LiveTab from "./tabs/LiveTab";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "data", label: "Data" },
  { id: "crispdm", label: "CRISP-DM" },
  { id: "models", label: "Models" },
  { id: "evaluation", label: "Evaluation" },
  { id: "explain", label: "Explain" },
  { id: "live", label: "Live inference" },
  { id: "audit", label: "Audit & card" },
] as const;

export default function ProjectShell({ meta, tab }: { meta: ProjectMeta; tab: string }) {
  const active = TABS.some((t) => t.id === tab) ? tab : "overview";
  const prov = meta.provenance;

  return (
    <>
      <header className="pagehead">
        <div className="pagehead__inner">
          <div className="pagehead__eyebrow">
            <span>{String(meta.number).padStart(2, "0")}</span>
            <span>·</span>
            <span>{meta.task}</span>
            <span>·</span>
            <span>{meta.domain}</span>
          </div>
          <h1 className="pagehead__title">{meta.title}</h1>
          <p className="pagehead__tagline">{meta.tagline}</p>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
            <Badge tone="accent">{meta.primary_metric}</Badge>
            {meta.trained ? (
              <Badge tone="good" icon="✓">
                trained {prov?.trained_at ? shortDate(prov.trained_at) : ""}
              </Badge>
            ) : (
              <Badge tone="warning" icon="⚠">not trained</Badge>
            )}
            {prov?.quick_mode ? (
              <Badge tone="warning" icon="⚠">
                quick mode — reduced fidelity
              </Badge>
            ) : null}
            {prov?.git_commit ? <Badge>commit {prov.git_commit}</Badge> : null}
            {prov?.seed !== undefined ? <Badge>seed {String(prov.seed)}</Badge> : null}
          </div>

          <nav className="tabs" aria-label="Project sections">
            {TABS.map((t) => (
              <a
                key={t.id}
                className={`tab${active === t.id ? " tab--active" : ""}`}
                href={href({ page: "project", slug: meta.slug, tab: t.id })}
              >
                {t.label}
              </a>
            ))}
          </nav>
        </div>
      </header>

      <div className="content">
        <TabContent slug={meta.slug} tab={active} meta={meta} />
      </div>
    </>
  );
}

function TabContent({ slug, tab, meta }: { slug: string; tab: string; meta: ProjectMeta }) {
  switch (tab) {
    case "data":
      return <DataTab slug={slug} />;
    case "crispdm":
      return <CrispDmTab slug={slug} />;
    case "models":
      return <ModelsTab slug={slug} />;
    case "evaluation":
      return <EvaluationTab slug={slug} />;
    case "explain":
      return <ExplainTab slug={slug} />;
    case "live":
      return <LiveTab slug={slug} meta={meta} />;
    case "audit":
      return <AuditTab slug={slug} />;
    default:
      return <OverviewTab slug={slug} meta={meta} />;
  }
}
