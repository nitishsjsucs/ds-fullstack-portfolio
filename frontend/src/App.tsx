/** Application shell: sidebar navigation, project routing and tab layout. */

import { useApi } from "./lib/api";
import { href, useRoute, type Route } from "./lib/router";
import type { ProjectIndex, ProjectMeta } from "./lib/types";
import { ErrorState, Loading } from "./components/ui";
import Home from "./views/Home";
import Methods from "./views/Methods";
import Datasets from "./views/Datasets";
import ProjectShell from "./views/ProjectShell";

const ACCENT_DOT: Record<string, string> = {
  blue: "var(--series-1)",
  amber: "var(--series-4)",
  rose: "var(--series-5)",
  violet: "var(--series-7)",
  emerald: "var(--series-3)",
  red: "var(--series-8)",
  sky: "var(--series-1)",
  cyan: "var(--series-3)",
  fuchsia: "var(--series-5)",
  indigo: "var(--series-7)",
};

export default function App() {
  const [route] = useRoute();
  const index = useApi<ProjectIndex>("/projects");

  const active = index.data?.projects.find((p) => p.slug === route.slug);
  const accent = active?.accent ?? "blue";

  return (
    <div className="shell" data-accent={accent}>
      <Sidebar route={route} projects={index.data?.projects ?? []} />
      <main className="main">
        {index.status === "loading" && <Loading what="the portfolio index" />}
        {index.status === "error" && (
          <div className="content">
            <ErrorState error={index.error} onRetry={index.reload} />
          </div>
        )}
        {index.status === "ready" && <Router route={route} index={index.data} />}
      </main>
    </div>
  );
}

function Router({ route, index }: { route: Route; index: ProjectIndex }) {
  if (route.page === "methods") return <Methods projects={index.projects} />;
  if (route.page === "datasets") return <Datasets />;
  if (route.page === "project") {
    const meta = index.projects.find((p) => p.slug === route.slug);
    if (!meta) {
      return (
        <div className="content">
          <div className="state">
            <div className="state__title">No such project</div>
            <p>
              <code>{route.slug}</code> is not in the registry.{" "}
              <a href={href({ page: "home" })}>Back to the portfolio</a>.
            </p>
          </div>
        </div>
      );
    }
    return <ProjectShell meta={meta} tab={route.tab ?? "overview"} />;
  }
  return <Home index={index} />;
}

function Sidebar({ route, projects }: { route: Route; projects: ProjectMeta[] }) {
  const trained = projects.filter((p) => p.trained).length;
  return (
    <nav className="sidebar" aria-label="Portfolio navigation">
      <a className="sidebar__brand" href={href({ page: "home" })}>
        <div className="sidebar__title">Data Science Full Stack</div>
        <div className="sidebar__subtitle">
          Eight systems · eight real datasets
        </div>
      </a>

      <div className="sidebar__label">Portfolio</div>
      <a
        className={`navitem${route.page === "home" ? " navitem--active" : ""}`}
        href={href({ page: "home" })}
      >
        <span className="navitem__num">◆</span>
        <span className="navitem__body">
          <span className="navitem__name">Overview</span>
        </span>
      </a>
      <a
        className={`navitem${route.page === "datasets" ? " navitem--active" : ""}`}
        href={href({ page: "datasets" })}
      >
        <span className="navitem__num">◈</span>
        <span className="navitem__body">
          <span className="navitem__name">Data provenance</span>
        </span>
      </a>
      <a
        className={`navitem${route.page === "methods" ? " navitem--active" : ""}`}
        href={href({ page: "methods" })}
      >
        <span className="navitem__num">◇</span>
        <span className="navitem__body">
          <span className="navitem__name">Method &amp; guarantees</span>
        </span>
      </a>

      <div className="sidebar__label">
        Projects{" "}
        <span style={{ float: "right", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
          {trained}/{projects.length} trained
        </span>
      </div>
      {projects.map((p) => (
        <a
          key={p.slug}
          className={`navitem${route.slug === p.slug ? " navitem--active" : ""}`}
          href={href({ page: "project", slug: p.slug, tab: route.slug === p.slug ? route.tab : "overview" })}
          style={{ "--dot": ACCENT_DOT[p.accent] ?? "var(--series-1)" } as React.CSSProperties}
        >
          <span className="navitem__num">{String(p.number).padStart(2, "0")}</span>
          <span className="navitem__body">
            <span className="navitem__name">{p.title}</span>
            <span className="navitem__meta">
              {p.trained ? p.task.split(" ")[0] : "not trained"}
            </span>
          </span>
          <span className="navitem__dot" style={{ background: p.trained ? undefined : "var(--surface-3)" }} />
        </a>
      ))}

      <div className="sidebar__label">Reference</div>
      <a className="navitem" href="/api/docs" target="_blank" rel="noreferrer">
        <span className="navitem__num">↗</span>
        <span className="navitem__body">
          <span className="navitem__name">API documentation</span>
        </span>
      </a>
    </nav>
  );
}
