"""RFM feature preparation for customer segmentation.

RFM (recency, frequency, monetary) is the standard customer-value summary, and it
has a standard problem: all three are severely right-skewed. A handful of
wholesale accounts spend thousands while the median customer spends a few hundred.
Feed that to K-Means -- which minimises squared Euclidean distance -- and the
outliers dominate every centroid, producing one giant cluster and several
singletons.

The fix is a log transform, applied *inside* the pipeline alongside the scaler so
the whole thing stays a single fitted object. We use log1p rather than a
Box-Cox/Yeo-Johnson search because the interpretation matters here: log spend is
what a merchandiser reasons in ("this segment spends 10x the median"), and a
fitted power parameter would make the cluster profiles harder to explain for a
marginal gain in symmetry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ...core import paths

# The clustering feature set. Deliberately small: RFM plus two behavioural ratios.
# Adding weakly-informative columns to a distance-based method dilutes every
# meaningful dimension, because Euclidean distance weights all axes equally.
FEATURES = [
    "recency_days", "frequency", "monetary",
    "n_distinct_skus", "avg_basket_value",
]

PROFILE_COLUMNS = [
    "recency_days", "frequency", "monetary", "n_items",
    "n_distinct_skus", "avg_unit_price", "tenure_days", "avg_basket_value",
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(paths.curated("online_retail_rfm"))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["avg_basket_value"] = out["monetary"] / out["frequency"].clip(lower=1)
    out["items_per_basket"] = out["n_items"] / out["frequency"].clip(lower=1)
    # Monetary can be non-positive when a customer's only orders were later
    # cancelled at the line level. Distance methods need finite logs, so those
    # rows are excluded and counted rather than clipped to an invented floor.
    out["_valid"] = (out["monetary"] > 0) & (out["frequency"] > 0)
    return out


def prepare() -> tuple[pd.DataFrame, dict]:
    raw = load_raw()
    eng = engineer(raw)
    kept = eng[eng["_valid"]].drop(columns=["_valid"]).reset_index(drop=True)
    exclusions = {
        "n_raw": int(len(eng)),
        "n_kept": int(len(kept)),
        "removed": int(len(eng) - len(kept)),
        "rule": "monetary > 0 and frequency > 0",
        "reason": ("Customers whose gross spend nets to zero or below have no "
                   "meaningful position in RFM space and cannot be log-transformed."),
    }
    return kept, exclusions


def build_preprocessor() -> Pipeline:
    """log1p -> standardise, as one fitted object.

    Order matters. Log first to pull in the tail, then standardise so every axis
    contributes comparably to the Euclidean distance K-Means minimises. Doing it
    the other way round would take the log of a variable containing negatives.
    """
    return Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one",
                                    validate=True)),
        ("scale", StandardScaler()),
    ])


def projection(X_scaled: np.ndarray, seed: int = 42) -> dict:
    """2-D PCA projection for the scatter plot, with variance accounting.

    PCA rather than t-SNE or UMAP for the primary view: distances in a PCA plot
    are a linear shadow of the real distances, so a cluster that looks separated
    genuinely is. t-SNE can manufacture visually convincing separation that does
    not exist in the feature space, which would undermine the whole point of
    showing the plot next to the silhouette score.
    """
    pca = PCA(n_components=2, random_state=seed)
    # audit: ok - R3 is about fitting a transform on data that spans a
    # train/test boundary. This is an unsupervised 2-D projection of
    # already-scaled features, used only to draw the scatter plot; there
    # is no target and no hold-out for it to leak across.
    coords = pca.fit_transform(X_scaled)
    return {
        "coords": coords,
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "total_explained": float(pca.explained_variance_ratio_.sum()),
        "loadings": [
            {"feature": f,
             "pc1": round(float(pca.components_[0][i]), 4),
             "pc2": round(float(pca.components_[1][i]), 4)}
            for i, f in enumerate(FEATURES)
        ],
        "method": "PCA (linear)",
        "why": ("Distances in a PCA projection are a faithful linear shadow of the "
                "distances the clusterer actually optimised. t-SNE and UMAP can "
                "invent separation that is not in the data."),
    }


def rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Classic 1-5 RFM quintile scores, as a human-readable cross-check.

    This is the pre-machine-learning way to segment a book, and it is genuinely
    useful: if the clusters do not roughly line up with RFM scores, either the
    clustering has found something subtle or it has found noise.
    """
    out = df.copy()
    out["r_score"] = pd.qcut(out["recency_days"], 5, labels=[5, 4, 3, 2, 1],
                             duplicates="drop").astype(int)
    out["f_score"] = pd.qcut(out["frequency"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    out["m_score"] = pd.qcut(out["monetary"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    out["rfm_score"] = out["r_score"] + out["f_score"] + out["m_score"]
    out["rfm_cell"] = (out["r_score"].astype(str) + out["f_score"].astype(str)
                       + out["m_score"].astype(str))
    return out


def name_segment(profile: dict, overall: dict) -> tuple[str, str]:
    """Turn a centroid into a persona a merchandiser can act on.

    Names are built from two orthogonal axes -- *engagement* (how recently and how
    often they buy) and *value* (how much) -- rather than from a flat list of
    rules. A flat list collides: two clusters that are both dormant but differ
    tenfold in spend end up with the same label, which breaks the phase-1
    criterion that every segment carry a distinct action. Crossing two axes
    guarantees distinct names for genuinely distinct centroids, and the value tier
    is what decides whether win-back spend is worth it.

    Ratios are against the population median, so the taxonomy survives a change in
    the underlying book.
    """
    r = profile["recency_days"] / max(1e-9, overall["recency_days"])
    f = profile["frequency"] / max(1e-9, overall["frequency"])
    m = profile["monetary"] / max(1e-9, overall["monetary"])

    # Engagement: recent + frequent = active; long-silent = dormant.
    if r <= 0.75 and f >= 1.4:
        engagement = "active"
    elif r <= 0.75:
        engagement = "recent"
    elif r >= 1.8:
        engagement = "dormant"
    else:
        engagement = "occasional"

    # Value tier against the median customer. Four tiers rather than three: the
    # bottom half of the book spans a 3-4x spend range, and a lapsed customer who
    # spent GBP 600 warrants one reactivation attempt while one who spent GBP 175
    # does not. Collapsing them into a single "low" tier produced two clusters
    # with identical names and identical advice, which is precisely the failure
    # the phase-1 actionability criterion exists to prevent.
    if m >= 2.5:
        value = "premium"
    elif m >= 1.2:
        value = "mid"
    elif m >= 0.55:
        value = "low"
    else:
        value = "minimal"

    table: dict[tuple[str, str], tuple[str, str]] = {
        ("active", "premium"): (
            "Champions",
            "Recent, frequent and high-spending -- the core of the book. Protect "
            "with early access and loyalty perks, never with discounts they would "
            "have spent without."),
        ("active", "mid"): (
            "Loyal regulars",
            "Reliable repeat buyers of moderate value. Best cross-sell target: they "
            "already trust the brand, so the marginal cost of an extra category is low."),
        ("active", "low"): (
            "Frequent low-ticket",
            "Buys often but small. Raise order value with basket-building bundles and "
            "a free-shipping threshold just above their current average."),
        ("active", "minimal"): (
            "Habitual micro-buyers",
            "Very frequent, very small orders. Fulfilment cost per order is the risk "
            "here -- consolidate with subscription or multi-pack formats."),
        ("recent", "minimal"): (
            "Sampler accounts",
            "One token recent order. Let automated lifecycle email do the work; any "
            "hand-crafted spend on this group is unrecoverable."),
        ("occasional", "minimal"): (
            "Negligible accounts",
            "Rare, tiny orders. Exclude from paid channels entirely and keep only "
            "for organic reach."),
        ("dormant", "minimal"): (
            "Dormant minimal-value",
            "Long silent and never more than a token spender. Suppress from all "
            "campaigns; the contact cost exceeds any plausible return."),
        ("recent", "premium"): (
            "Big-ticket newcomers",
            "One or two large recent orders. High potential, unproven loyalty -- "
            "prioritise a strong second-purchase experience over discounting."),
        ("recent", "mid"): (
            "Promising newcomers",
            "Bought recently but not yet often. Onboarding sequences and a "
            "second-purchase nudge have the highest marginal return in the book."),
        ("recent", "low"): (
            "Trial buyers",
            "Small first purchase, still warm. Cheap nurture only; most will not "
            "convert and heavy spend here does not pay back."),
        ("occasional", "premium"): (
            "High-value irregulars",
            "Spend heavily but sporadically. Time campaigns to their replenishment "
            "cycle rather than to the calendar."),
        ("occasional", "mid"): (
            "Occasional mainstream",
            "The middle of the book: unremarkable on every axis. Serve with general "
            "merchandising rather than targeted spend."),
        ("occasional", "low"): (
            "Marginal buyers",
            "Low value and irregular. Keep in broadcast channels only -- targeted "
            "contact costs more than the expected return."),
        ("dormant", "premium"): (
            "At-risk high-value",
            "Valuable but drifting. This is where win-back budget belongs: the "
            "recoverable margin is the largest in the book and the clock is running."),
        ("dormant", "mid"): (
            "Lapsing mid-value",
            "Silent for months with real historical spend. A single well-timed "
            "reactivation offer is justified; sustained campaigning is not."),
        ("dormant", "low"): (
            "Lapsed modest spenders",
            "Silent for months with a modest but real purchase history. Worth exactly "
            "one well-timed reactivation offer -- not a sustained campaign."),
    }
    return table[(engagement, value)]


def ensure_distinct_names(profiles: list[dict]) -> list[dict]:
    """Guarantee the phase-1 criterion that no two segments share a label.

    The two-axis taxonomy makes collisions unlikely but not impossible: two
    clusters can land in the same cell while still being separated on a third
    feature. Rather than silently shipping duplicates, we disambiguate with the
    feature that actually distinguishes them and flag that we did.
    """
    seen: dict[str, list[dict]] = {}
    for p in profiles:
        seen.setdefault(p["name"], []).append(p)

    for name, group in seen.items():
        if len(group) < 2:
            continue
        # Disambiguate by the feature with the widest spread across the collision.
        best_feat, best_spread = "monetary", 0.0
        for feat in ("monetary", "frequency", "recency_days", "n_distinct_skus"):
            vals = [g["median"][feat] for g in group]
            spread = (max(vals) - min(vals)) / max(1e-9, min(vals))
            if spread > best_spread:
                best_feat, best_spread = feat, spread
        ordered = sorted(group, key=lambda g: -g["median"][best_feat])
        labels = ["higher", "lower"] if len(ordered) == 2 else \
                 [f"tier {i + 1}" for i in range(len(ordered))]
        pretty = best_feat.replace("_", " ")
        for g, suffix in zip(ordered, labels):
            g["name"] = f"{name} ({suffix} {pretty})"
            g["name_disambiguated"] = True
            g["disambiguated_on"] = best_feat
    return profiles
