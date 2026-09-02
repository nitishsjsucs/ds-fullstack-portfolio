"""The project registry -- the single source of truth for what this portfolio contains.

Adding a project means adding one :class:`ProjectMeta` entry here and a package
under ``app/projects/<slug>/``. The API, the navigation, the training runner, the
audit and the documentation generator all read from this list, so a new project
appears everywhere at once and cannot be half-registered.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from types import ModuleType


@dataclass(frozen=True)
class ProjectMeta:
    slug: str
    number: int
    title: str
    tagline: str
    task: str                      # the modelling task family
    domain: str                    # the business domain
    dataset: str                   # curated dataset stem
    dataset_title: str             # human-readable source
    dataset_rows: int
    primary_metric: str            # the number that decides "better"
    metric_reason: str             # why that metric and not accuracy
    techniques: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    accent: str = "indigo"         # UI accent token
    icon: str = "chart"
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


PROJECTS: list[ProjectMeta] = [
    ProjectMeta(
        slug="taxi", number=1,
        title="NYC Taxi Fare & Duration",
        tagline="Spatio-temporal gradient boosting on 150,000 real January-2024 yellow-cab trips.",
        task="Supervised regression (dual target)",
        domain="Urban mobility / pricing",
        dataset="nyc_taxi_trips",
        dataset_title="NYC TLC Yellow Taxi Trip Records, January 2024",
        dataset_rows=150_000,
        primary_metric="MAE (minutes)",
        metric_reason=(
            "Riders experience absolute lateness, not squared error, and MAE is "
            "robust to the genuine 4-hour outlier trips the TLC publishes."
        ),
        techniques=["Gradient boosting (LightGBM/XGBoost)", "Haversine & zone features",
                    "Cyclical time encoding", "TreeSHAP", "Partial dependence",
                    "Quantile regression intervals"],
        skills=["Feature engineering", "Geospatial analysis", "Outlier diagnostics",
                "Model explainability", "Prediction intervals"],
        accent="amber", icon="map",
        highlights=["Interactive borough-to-borough estimator with a live SHAP waterfall",
                    "80% prediction intervals from paired quantile models, coverage-tested",
                    "Data-quality scorecard over the TLC's real anomalies"],
    ),
    ProjectMeta(
        slug="churn", number=2,
        title="Telco Churn & Retention Economics",
        tagline="Cost-sensitive classification with calibration and a fairness audit on 7,043 subscribers.",
        task="Supervised binary classification (imbalanced)",
        domain="Subscription retention",
        dataset="telco_churn",
        dataset_title="IBM Telco Customer Churn",
        dataset_rows=7_043,
        primary_metric="PR-AUC (average precision)",
        metric_reason=(
            "Churn prevalence is 26.5%. Accuracy rewards predicting 'stays' for "
            "everyone (73.5%); PR-AUC only rewards ranking real churners highly."
        ),
        techniques=["Regularised logistic regression", "Gradient boosting",
                    "Isotonic & Platt calibration", "Cost-matrix threshold search",
                    "Demographic parity / equalised odds"],
        skills=["Imbalanced learning", "Probability calibration", "Cost-sensitive decisions",
                "Fairness auditing", "Business value modelling"],
        accent="rose", icon="users",
        highlights=["Drag the retention-offer economics and watch the optimal threshold move",
                    "Calibration curve with expected calibration error before/after isotonic fit",
                    "Disparate-impact audit across gender, senior-citizen and partner status"],
    ),
    ProjectMeta(
        slug="segments", number=3,
        title="RFM Customer Segmentation",
        tagline="Unsupervised segmentation of 4,339 real UK e-commerce customers.",
        task="Unsupervised clustering",
        domain="Customer intelligence",
        dataset="online_retail_rfm",
        dataset_title="UCI Online Retail (customer-level RFM aggregation)",
        dataset_rows=4_339,
        primary_metric="Silhouette coefficient",
        metric_reason=(
            "There is no ground-truth label, so cluster quality must be judged "
            "internally; silhouette balances cohesion against separation."
        ),
        techniques=["K-Means", "Gaussian mixture models", "Hierarchical (Ward)", "DBSCAN",
                    "PCA projection", "Log/Yeo-Johnson transforms", "Gap statistic"],
        skills=["Unsupervised learning", "Cluster validation", "Dimensionality reduction",
                "Segment profiling", "RFM analysis"],
        accent="violet", icon="scatter",
        highlights=["Four algorithms compared on the same features with agreement analysis",
                    "Interactive PCA scatter that recolours as you change k",
                    "Plain-English persona cards derived from cluster centroids"],
    ),
    ProjectMeta(
        slug="basket", number=4,
        title="Market Basket Association Mining",
        tagline="Apriori and FP-Growth over 20,136 real invoices and 3,925 SKUs.",
        task="Association rule mining",
        domain="Retail merchandising",
        dataset="online_retail_transactions",
        dataset_title="UCI Online Retail (invoice line items)",
        dataset_rows=530_693,
        primary_metric="Lift (with conviction & leverage)",
        metric_reason=(
            "Confidence alone promotes rules that merely restate a popular item's "
            "base rate; lift and conviction measure genuine dependence."
        ),
        techniques=["Apriori", "FP-Growth", "Lift / leverage / conviction / Zhang's metric",
                    "Frequent itemset lattice", "Rule network graph"],
        skills=["Pattern mining", "Support-confidence framework", "Graph visualisation",
                "Recommendation heuristics"],
        accent="emerald", icon="network",
        highlights=["Side-by-side Apriori vs FP-Growth runtime at identical support",
                    "Force-directed rule network you can filter by lift in real time",
                    "Live 'customers who bought this' recommender over the mined rules"],
    ),
    ProjectMeta(
        slug="anomaly", number=5,
        title="Network Intrusion Anomaly Detection",
        tagline="Unsupervised outlier detection on 80,000 KDD-99 connections with a 3.4% attack base rate.",
        task="Anomaly / outlier detection",
        domain="Security telemetry",
        dataset="kdd99_intrusion",
        dataset_title="KDD Cup 1999 network intrusion (SA subset)",
        dataset_rows=80_000,
        primary_metric="PR-AUC + precision@k",
        metric_reason=(
            "At 3.4% prevalence a detector that flags nothing is 96.6% accurate. "
            "Analysts triage a fixed queue, so precision@k is the operational metric."
        ),
        techniques=["Isolation Forest", "Local Outlier Factor", "One-Class SVM",
                    "Elliptic Envelope", "PCA reconstruction error", "Score fusion"],
        skills=["Unsupervised anomaly detection", "Extreme class imbalance",
                "Threshold-free evaluation", "Alert triage economics"],
        accent="red", icon="shield",
        highlights=["Five detectors scored on identical folds; labels used only post hoc",
                    "Analyst-queue simulator: set a daily alert budget, see caught vs missed",
                    "Per-attack-family recall showing which intrusions are invisible"],
    ),
    ProjectMeta(
        slug="forecast", number=6,
        title="Bike-Share Demand Forecasting",
        tagline="Multi-horizon hourly forecasting over 17,379 real Capital Bikeshare observations.",
        task="Time series forecasting",
        domain="Demand planning",
        dataset="bike_sharing_hourly",
        dataset_title="UCI Bike Sharing (hourly, 2011-2012)",
        dataset_rows=17_379,
        primary_metric="MASE",
        metric_reason=(
            "MASE is scale-free and divides by the seasonal-naive error, so a value "
            "below 1 proves the model beats 'same hour last week' -- the only "
            "baseline that matters."
        ),
        techniques=["Seasonal naive baseline", "SARIMAX", "Gradient boosting on lags",
                    "STL decomposition", "ACF/PACF", "Purged & embargoed CV",
                    "Quantile prediction intervals"],
        skills=["Temporal validation", "Seasonality decomposition", "Lag feature engineering",
                "Interval calibration", "Backtesting"],
        accent="sky", icon="trend",
        highlights=["Rolling-origin backtest across 5 purged folds, never shuffled",
                    "80% intervals with empirical coverage reported against nominal",
                    "STL trend/season/residual decomposition with a 48-lag correlogram"],
    ),
    ProjectMeta(
        slug="automl", number=7,
        title="AutoML Stacking Tournament",
        tagline="Hill-climbing search and Caruana ensembling over 48,842 census records.",
        task="AutoML / model selection",
        domain="Income prediction & fairness",
        dataset="adult_census_income",
        dataset_title="UCI Adult / Census Income (1994 CPS)",
        dataset_rows=48_842,
        primary_metric="ROC-AUC (CV) with hold-out gap",
        metric_reason=(
            "Model selection needs a ranking metric stable across folds; the "
            "CV-to-hold-out gap is reported alongside to expose search overfitting."
        ),
        techniques=["Coordinate-ascent hill climbing", "Stacked generalisation",
                    "Caruana greedy ensemble selection", "Out-of-fold prediction",
                    "Fairness constraints", "Model distillation"],
        skills=["Hyperparameter search", "Ensembling", "Leakage-free stacking",
                "Search-overfitting diagnosis", "Responsible AI"],
        accent="cyan", icon="tournament",
        highlights=["Every hill-climbing trial replayable as an animated trajectory",
                    "Greedy ensemble weights derived purely from out-of-fold predictions",
                    "Disparate-impact ratio before and after threshold adjustment"],
    ),
    ProjectMeta(
        slug="nanollm", number=8,
        title="NanoGPT: A Transformer From Scratch",
        tagline="A decoder-only language model built layer by layer and trained on 1.1M characters of Shakespeare.",
        task="Self-supervised sequence modelling",
        domain="Deep learning / NLP",
        dataset="tiny_shakespeare",
        dataset_title="Tiny Shakespeare character corpus",
        dataset_rows=1_115_394,
        primary_metric="Validation cross-entropy (bits/char)",
        metric_reason=(
            "Held-out perplexity is the only honest measure for a generative model; "
            "sampled text is a demo, not an evaluation."
        ),
        techniques=["Multi-head causal self-attention", "Rotary position embeddings",
                    "RMSNorm & SwiGLU", "AdamW with cosine schedule", "Temperature/top-k sampling"],
        skills=["Attention mechanics", "Autoregressive training", "Tokenisation",
                "Optimisation scheduling", "Generative sampling"],
        accent="fuchsia", icon="sparkles",
        highlights=["Attention heat-maps rendered from the real trained weights",
                    "Streaming generation with live temperature and top-k controls",
                    "Loss curves showing the exact epoch where over-fitting begins"],
    ),
]

BY_SLUG: dict[str, ProjectMeta] = {p.slug: p for p in PROJECTS}
SLUGS: list[str] = [p.slug for p in PROJECTS]


def get(slug: str) -> ProjectMeta:
    if slug not in BY_SLUG:
        raise KeyError(f"Unknown project {slug!r}. Known: {', '.join(SLUGS)}")
    return BY_SLUG[slug]


def module(slug: str) -> ModuleType:
    """Import a project package lazily (training pulls heavy deps)."""
    get(slug)
    return importlib.import_module(f"app.projects.{slug}")


def index_payload() -> dict:
    """The navigation payload the frontend boots from."""
    from .core.artifacts import ArtifactStore

    items = []
    for p in PROJECTS:
        store = ArtifactStore(p.slug)
        items.append({
            **p.to_dict(),
            "trained": store.is_trained,
            "provenance": store.provenance(),
        })
    return {
        "portfolio": "Data Science Full Stack",
        "count": len(items),
        "projects": items,
        "datasets_are_real": True,
        "note": (
            "Every project trains on genuine public data curated by "
            "scripts/fetch_data.py; no metric on this site comes from synthetic numbers."
        ),
    }
