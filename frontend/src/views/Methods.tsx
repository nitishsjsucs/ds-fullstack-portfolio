import type { ProjectMeta } from "../lib/types";
import { Callout, Section } from "../components/ui";
import { href } from "../lib/router";

export default function Methods({ projects }: { projects: ProjectMeta[] }) {
  return (
    <>
      <header className="pagehead">
        <div className="pagehead__inner">
          <div className="pagehead__eyebrow">
            <span>Reference</span>
          </div>
          <h1 className="pagehead__title">Method &amp; guarantees</h1>
          <p className="pagehead__tagline">
            What every project in this portfolio promises, how those promises are enforced
            in code rather than in prose, and where the honest limits are.
          </p>
          <div style={{ height: 20 }} />
        </div>
      </header>

      <div className="content">
        <Section
          title="The four guarantees"
          description="These hold for all eight projects. Each is enforced by a shared module, so a change lands everywhere at once and a violation fails the test suite rather than quietly inflating a score."
        >
          <div className="grid grid--2">
            <Guarantee
              n="01"
              title="Every fitted transform lives inside the pipeline"
              body="Imputers, encoders, scalers and frequency tables are all steps of a scikit-learn Pipeline, which cross-validation clones per fold. That is what makes 'no preprocessing leakage' structural rather than a claim. Computing a median over the full frame before splitting is the single most common way a portfolio project silently cheats."
              module="backend/app/core/model_zoo.py"
            />
            <Guarantee
              n="02"
              title="Temporal data is never shuffled"
              body="Time-ordered projects split chronologically and assert it: assert_chronological() raises if any training timestamp exceeds the earliest test timestamp. Cross-validation uses expanding windows with a purge equal to the longest feature look-back, plus an embargo — without that purge, a trailing rolling window reaches across the fold boundary."
              module="backend/app/core/splitting.py"
            />
            <Guarantee
              n="03"
              title="The hold-out is scored once"
              body="Hyperparameter search optimises a cross-validated score only. Each model family is refitted and scored on the untouched hold-out exactly once, after the search closes. The gap between the two numbers is published per family. Where it is positive, that gap is the cost of searching and quoting the CV score alone would overstate deployed performance by exactly that margin; on the census dataset it comes back negative, which is the measurement rather than the expectation."
              module="backend/app/core/autoresearch.py"
            />
            <Guarantee
              n="04"
              title="The metric suits the problem, not the model"
              body="At 3.4% prevalence a detector that flags nothing is 96.6% accurate, so accuracy appears nowhere as a headline. Imbalanced problems report PR-AUC and precision@k; forecasting reports skill against a seasonal-naive baseline; regression reports MAE with the baselines it must beat. Where accuracy is computed at all, it is flagged as misleading."
              module="backend/app/core/evaluation.py"
            />
          </div>
        </Section>

        <Section
          title="What each project is really about"
          description="Beyond the algorithm, each system is designed to demonstrate one methodological idea properly."
        >
          <div className="tablewrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Project</th>
                  <th>The idea it demonstrates</th>
                  <th>The honest finding it reports</th>
                </tr>
              </thead>
              <tbody>
                {LESSONS.map((l) => {
                  const project = projects.find((p) => p.slug === l.slug);
                  return (
                    <tr key={l.slug}>
                      <td className="strong">
                        <a href={href({ page: "project", slug: l.slug })}>
                          {project?.title ?? l.slug}
                        </a>
                      </td>
                      <td>{l.idea}</td>
                      <td style={{ color: "var(--ink-muted)" }}>{l.finding}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>

        <Section
          title="What this portfolio does not claim"
          description="The limits worth stating plainly, because a project that lists none is not being careful."
        >
          <Callout tone="warning" title="These are demonstrations, not products">
            Every model card names its out-of-scope uses explicitly. The census model must
            not touch a real decision about a real person; the intrusion detector runs on a
            1999 simulation and says so; the language model has no knowledge and cannot
            answer a question. Sample sizes here are chosen so the whole portfolio retrains
            in under an hour on a laptop, which bounds what any single result can claim.
          </Callout>
          <Callout tone="warning" title="Some datasets are curated samples">
            The NYC taxi project uses a deterministic 150,000-trip sample of the 2.96M the
            TLC published for January 2024, and the intrusion project samples 80,000 of
            100,655 connections. Both sampling rules are recorded in the manifest with a
            seed and a SHA-256, so they reproduce exactly — but they are samples, and
            confidence intervals on segment-level results are correspondingly wider.
          </Callout>
          <Callout tone="warning" title="Correlation is not causation, anywhere">
            The market-basket rules describe co-occurrence in a historical archive. The
            churn what-if sweep shows the model's learned sensitivity to a field, which is
            not the effect of changing that field in reality — moving a customer to a
            one-year contract also changes what kind of customer they are. Every such view
            says so on the page.
          </Callout>
        </Section>

        <Section
          title="Reproducing this"
          description="Three commands from a clean checkout. Everything is seeded, so two runs on the same data produce the same leaderboard."
        >
          <pre className="pre">{`git clone <repo> && cd ds-fullstack-portfolio

# 1. Download the eight real datasets and write data/MANIFEST.json
#    with a SHA-256 for every curated file.
python scripts/fetch_data.py

# 2. Train every project and persist its artifacts.
#    Add --quick for a low-fidelity run in a couple of minutes.
python scripts/train_all.py

# 3. Serve the API and the console.
uvicorn app.main:app --app-dir backend --port 8000
cd frontend && npm install && npm run dev`}</pre>
        </Section>
      </div>
    </>
  );
}

const LESSONS = [
  {
    slug: "taxi",
    idea: "Uncertainty is the deliverable. A point estimate implies a precision the model does not have, so the project ships an interval and coverage-tests it.",
    finding: "Airport trips carry roughly double the error of city trips; the per-segment table makes that visible rather than hiding it in an average.",
  },
  {
    slug: "churn",
    idea: "A probability that gets multiplied by a dollar amount has to be literally true, so calibration is selected on expected calibration error rather than AUC.",
    finding: "Senior-citizen and partner status both fail the four-fifths rule at the deployed threshold, despite neither being a model feature.",
  },
  {
    slug: "segments",
    idea: "Unsupervised work has no test set, so confidence comes from convergent evidence: four algorithms, a null-reference gap statistic and bootstrap stability.",
    finding: "DBSCAN scores the highest silhouette but only by discarding 13% of customers as noise; it is excluded by an explicit coverage rule. The evaluation gate then fails on cohesion — silhouette 0.239 against a 0.25 bar — and stays failed.",
  },
  {
    slug: "basket",
    idea: "Two exact algorithms must return identical output, so their difference is purely computational — the cleanest possible demonstration of that distinction.",
    finding: "Apriori is 3.2× faster than FP-Growth at this support floor (7.2s vs 23.2s on identical output), which is the opposite of the textbook ordering, and the report says so.",
  },
  {
    slug: "anomaly",
    idea: "'Anomalous' and 'malicious' are different properties. Five detectors with genuinely different assumptions expose how much the definition matters.",
    finding: "Three detectors are inverted — they rank intrusions as more normal than benign traffic, because flood attacks are the densest region of the space. A fourth returns NaN: its covariance is singular on one-hot columns.",
  },
  {
    slug: "forecast",
    idea: "Time-series leakage is invisible: nothing errors, the score just improves. Purged cross-validation and shift-before-roll are the two defences.",
    finding: "The raw 80% interval covered 67.1% of hours. Conformal calibration moved it to 79.2%, at the cost of a band 17% wider.",
  },
  {
    slug: "automl",
    idea: "Whether automated search overfits its own folds is an empirical question, so the CV-to-hold-out gap is measured per family rather than assumed.",
    finding: "Ensembling moved hold-out AUC by +0.0001 — noise. And the search did not overfit at this data size: every gap came back negative, which is the measurement, not the expectation.",
  },
  {
    slug: "nanollm",
    idea: "A page of plausible text is not a measurement. Held-out cross-entropy against explicit baselines is, and samples are labelled as illustration.",
    finding: "Train and validation loss separate at a locatable step; both curves are published in full rather than cropped at the flattering point.",
  },
];

function Guarantee({
  n,
  title,
  body,
  module,
}: {
  n: string;
  title: string;
  body: string;
  module: string;
}) {
  return (
    <div className="card">
      <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
        <div
          className="mono"
          style={{
            fontSize: "var(--text-lg)",
            color: "var(--accent, var(--series-1))",
            fontWeight: 700,
            lineHeight: 1,
          }}
        >
          {n}
        </div>
        <div style={{ minWidth: 0 }}>
          <h3 style={{ fontSize: "var(--text-md)", marginBottom: 8 }}>{title}</h3>
          <p className="small" style={{ color: "var(--ink-secondary)", lineHeight: 1.6, marginBottom: 10 }}>
            {body}
          </p>
          <code className="small muted">{module}</code>
        </div>
      </div>
    </div>
  );
}
