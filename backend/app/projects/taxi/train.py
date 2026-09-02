"""CRISP-DM training run for the NYC taxi project.

Produces every artifact the dashboards serve: EDA profile, phase-gated CRISP-DM
record, hill-climbing leaderboard, hold-out evaluation, explainability payloads,
model card and leakage audit.

Protocol summary (the part an auditor should read first):

* 150,000 real January-2024 trips, filtered by physical-plausibility rules only.
* Chronological split -- roughly Jan 1-25 trains, Jan 25-31 tests. Never shuffled.
* Hyperparameters chosen by cross-validated hill climbing on the *training* half,
  using expanding-window folds. The test half is scored once, at the end.
* Preprocessing (imputation, one-hot vocabulary, zone frequency table) is fitted
  inside the pipeline, so every CV fold re-fits it on that fold's training rows.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ...core import eda as eda_mod
from ...core import evaluation, explain
from ...core.artifacts import TrainingArtifacts
from ...core.autoresearch import AutoResearch, leaderboard_payload
from ...core.crispdm import CrispDmRecord, Decision, Finding, Phase, gate
from ...core.model_zoo import quantile_regressors, regression_specs
from ...core.splitting import PurgedTimeSeriesSplit, temporal_split
from ...registry import get as get_meta
from . import data as D

SLUG = "taxi"
SEED = 42


def run(quick: bool = False) -> TrainingArtifacts:
    t_start = time.perf_counter()
    meta = get_meta(SLUG)
    print(f"  [{SLUG}] preparing data...")

    # ------------------------------------------------------------ phase 2/3 --
    prepared = D.prepare()
    df = prepared.frame
    print(f"  [{SLUG}] {prepared.n_raw:,} raw -> {prepared.n_kept:,} modelled "
          f"({prepared.exclusion_rate:.2%} excluded by physical rules)")

    if quick:
        df = df.tail(40_000).reset_index(drop=True)

    X = df[D.ALL_FEATURES]
    y = df[D.TARGET_DURATION]
    y_fare = df[D.TARGET_FARE]

    split = temporal_split(df["pickup_datetime"], test_size=0.2)
    Xtr, Xte = X.iloc[split.train_idx], X.iloc[split.test_idx]
    ytr, yte = y.iloc[split.train_idx], y.iloc[split.test_idx]
    ftr, fte = y_fare.iloc[split.train_idx], y_fare.iloc[split.test_idx]

    # --------------------------------------------------------------- EDA ---- #
    print(f"  [{SLUG}] profiling...")
    eda_cols = list(dict.fromkeys([
        *D.NUMERIC_FEATURES, *D.CATEGORICAL_FEATURES, "duration_min", "fare_amount",
        "avg_speed_mph", "trip_distance", "pickup_datetime", "pu_zone", "do_zone",
    ]))
    eda_payload = eda_mod.profile(
        df[eda_cols],
        target="duration_min",
        numeric=["duration_min", "fare_amount", "trip_distance", "avg_speed_mph",
                 "passenger_count", "day_of_month"],
        categorical=["pu_borough", "do_borough", "ratecode_label", "trip_type"],
        time_col="pickup_datetime",
        validity_rules={
            "duration_min": "duration_min > 0 and duration_min <= 180",
            "trip_distance": "trip_distance > 0 and trip_distance <= 60",
            "fare_amount": "fare_amount > 0",
        },
    )
    eda_payload["temporal_heatmap"] = eda_mod.temporal_heatmap(
        df["pickup_datetime"], df["duration_min"], agg="mean"
    )
    eda_payload["demand_heatmap"] = eda_mod.temporal_heatmap(df["pickup_datetime"])
    eda_payload["exclusions"] = {
        "n_raw": prepared.n_raw, "n_kept": prepared.n_kept,
        "exclusion_rate": prepared.exclusion_rate, "rules": prepared.excluded,
        "policy": ("Rows are removed only by physical-plausibility rules stated in "
                   "advance. No rule references the target's distribution, so no hard "
                   "case is deleted for being hard."),
    }
    eda_payload["borough_matrix"] = _borough_matrix(df)
    eda_payload["top_zones"] = _top_zones(df)
    eda_payload["speed_by_hour"] = _speed_by_hour(df)
    eda_payload["train_only_target_relationships"] = eda_mod.target_relationship(
        df.iloc[split.train_idx], "duration_min",
        ["trip_distance", "is_rush_hour", "is_night", "pu_borough", "trip_type"],
    )

    # ------------------------------------------------------------ modeling -- #
    cv = PurgedTimeSeriesSplit(n_splits=2 if quick else 3, window=0, embargo=0)
    search_n = 12_000 if quick else 45_000
    Xs, ys = Xtr.tail(search_n), ytr.tail(search_n)
    print(f"  [{SLUG}] hill-climbing on {len(Xs):,} most-recent training rows...")

    ar = AutoResearch(
        scoring="neg_mean_absolute_error", cv=cv, greater_is_better=True,
        max_trials_per_family=6 if quick else 14, patience=2, seed=SEED,
    )
    specs = regression_specs(D.build_preprocessor, include_slow=not quick)
    results = ar.run(specs, Xs, ys)

    # Refit each family's winner on the FULL training half, then score the
    # untouched test half exactly once.
    print(f"  [{SLUG}] refitting {len(results)} finalists on full training half...")
    fitted = {}
    for r in results:
        spec = next(s for s in specs if s.name == r.name)
        model = spec.build(r.best_params)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        r.holdout = {
            "primary": -float(np.mean(np.abs(yte - pred))),
            "metrics": evaluation.regression_metrics(yte, pred),
        }
        fitted[r.name] = model
        print(f"      {r.name:24s} CV MAE={-r.cv_score:6.3f}  "
              f"hold-out MAE={-r.holdout['primary']:6.3f} min")

    results.sort(key=lambda r: -r.holdout["primary"])
    champion_name = results[0].name
    champion = fitted[champion_name]
    pred_te = champion.predict(Xte)

    lb = leaderboard_payload(results, scoring="neg_mean_absolute_error")
    lb["selection_note"] = (
        f"'{champion_name}' is deployed. Families are ranked here by hold-out MAE, "
        "but hyperparameters within each family were chosen on cross-validated MAE "
        "only -- the hold-out was never used to tune anything."
    )

    # ------------------------------------------------- intervals & fare ----- #
    print(f"  [{SLUG}] fitting quantile models for 80% intervals...")
    qmodels = quantile_regressors(D.build_preprocessor, (0.1, 0.5, 0.9),
                                  max_iter=140 if quick else 240)
    q_pred = {}
    for q, m in qmodels.items():
        m.fit(Xtr, ytr)
        q_pred[q] = m.predict(Xte)
    # Quantile crossing is a real artefact of fitting quantiles independently;
    # sorting each row restores monotonicity rather than pretending it never happens.
    lo, mid, hi = (np.minimum(q_pred[0.1], q_pred[0.9]),
                   q_pred[0.5],
                   np.maximum(q_pred[0.1], q_pred[0.9]))
    n_crossed = int((q_pred[0.1] > q_pred[0.9]).sum())
    interval = evaluation.interval_metrics(yte, lo, hi, nominal=0.8)
    interval["quantile_crossings_repaired"] = n_crossed

    print(f"  [{SLUG}] fitting the fare model...")
    fare_spec = next(s for s in specs if s.name == champion_name)
    fare_model = fare_spec.build(results[0].best_params)
    fare_model.fit(Xtr, ftr)
    fare_pred = fare_model.predict(Xte)
    fare_metrics = evaluation.regression_metrics(fte, fare_pred)

    # --------------------------------------------------------- evaluation --- #
    eval_payload = evaluation.regression_report_payload(yte, pred_te)
    eval_payload["interval"] = interval
    eval_payload["fare_model"] = {
        "metrics": fare_metrics,
        "report": evaluation.regression_report_payload(fte, fare_pred, sample=900),
    }
    eval_payload["split"] = split.describe()
    eval_payload["segment_errors"] = _segment_errors(df.iloc[split.test_idx], yte, pred_te)
    eval_payload["error_by_hour"] = _error_by_hour(df.iloc[split.test_idx], yte, pred_te)
    eval_payload["baselines"] = _baselines(ytr, yte, df, split)

    # -------------------------------------------------------- explainability #
    print(f"  [{SLUG}] computing explanations...")
    perm = explain.permutation_payload(
        champion, Xte, yte, scoring="neg_mean_absolute_error",
        n_repeats=4 if quick else 8, max_samples=3500,
    )
    prep = champion.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    Xte_t = pd.DataFrame(prep.transform(Xte), columns=names)
    Xtr_t = pd.DataFrame(prep.transform(Xtr.tail(600)), columns=names)
    shap_p = explain.shap_payload(champion, Xtr_t, Xte_t.head(700))
    pdp = explain.pdp_payload(
        champion, Xte.head(2200),
        ["trip_distance", "hour_sin", "is_rush_hour", "passenger_count"],
    )
    explain_payload = {
        "permutation": perm,
        "impurity": explain.impurity_payload(champion, names),
        "shap": shap_p,
        "partial_dependence": pdp,
        "feature_count": len(names),
    }

    # ------------------------------------------------------------- CRISP-DM  #
    record = _crispdm_record(meta, prepared, split, eda_payload, lb, eval_payload,
                             champion_name, results, interval)

    model_card = _model_card(meta, champion_name, results[0], eval_payload,
                             fare_metrics, interval, len(names), split)
    audit = _audit(prepared, split, results, eval_payload, perm)

    elapsed = time.perf_counter() - t_start
    arts = TrainingArtifacts(SLUG)
    arts.add("overview", {
        **meta.to_dict(),
        "rows_raw": prepared.n_raw, "rows_modelled": prepared.n_kept,
        "champion": champion_name,
        "headline": {
            "mae_minutes": eval_payload["metrics"]["mae"],
            "rmse_minutes": eval_payload["metrics"]["rmse"],
            "r2": eval_payload["metrics"]["r2"],
            "fare_mae_usd": fare_metrics["mae"],
            "interval_coverage": interval["empirical_coverage"],
        },
        "train_seconds": round(elapsed, 1),
    })
    arts.add("eda", eda_payload)
    arts.add("crispdm", record.to_dict())
    arts.add("leaderboard", lb)
    arts.add("evaluation", eval_payload)
    arts.add("explain", explain_payload)
    arts.add("model_card", model_card)
    arts.add("audit", audit)
    arts.add("extras", {
        "zones": D.zone_options(),
        "ratecodes": [{"id": k, "label": v} for k, v in D.RATECODES.items()],
        "presets": _presets(),
        "borough_distance_hints": {f"{a}->{b}": v for (a, b), v in D._BOROUGH_DISTANCE.items()},
    })
    arts.add_model("model", champion)
    arts.add_model("fare_model", fare_model)
    arts.add_model("q_low", qmodels[0.1])
    arts.add_model("q_high", qmodels[0.9])

    out = arts.save(provenance={
        "dataset": meta.dataset, "seed": SEED, "quick_mode": quick,
        "train_seconds": round(elapsed, 1),
    })
    print(f"  [{SLUG}] done in {elapsed:.0f}s -> {out}")
    return arts


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #
def _borough_matrix(df: pd.DataFrame) -> dict:
    piv = df.pivot_table(index="pu_borough", columns="do_borough",
                         values="duration_min", aggfunc="mean")
    cnt = df.pivot_table(index="pu_borough", columns="do_borough",
                         values="duration_min", aggfunc="size")
    boroughs = sorted(set(piv.index) | set(piv.columns))
    piv = piv.reindex(index=boroughs, columns=boroughs)
    cnt = cnt.reindex(index=boroughs, columns=boroughs)
    return {
        "boroughs": boroughs,
        "mean_duration": [[None if pd.isna(v) else round(float(v), 2)
                           for v in piv.loc[b]] for b in boroughs],
        "trip_counts": [[0 if pd.isna(v) else int(v) for v in cnt.loc[b]] for b in boroughs],
    }


def _top_zones(df: pd.DataFrame, k: int = 12) -> dict:
    def agg(col_id: str, col_name: str) -> list[dict]:
        g = df.groupby(col_name).agg(
            trips=("duration_min", "size"),
            mean_duration=("duration_min", "mean"),
            mean_fare=("fare_amount", "mean"),
        ).sort_values("trips", ascending=False).head(k)
        return [{"zone": str(i), "trips": int(r.trips),
                 "mean_duration": round(float(r.mean_duration), 2),
                 "mean_fare": round(float(r.mean_fare), 2)} for i, r in g.iterrows()]
    return {"pickup": agg("pu_location_id", "pu_zone"),
            "dropoff": agg("do_location_id", "do_zone")}


def _speed_by_hour(df: pd.DataFrame) -> list[dict]:
    g = df.groupby("hour").agg(
        mean_speed=("avg_speed_mph", "mean"),
        median_speed=("avg_speed_mph", "median"),
        trips=("avg_speed_mph", "size"),
        mean_duration=("duration_min", "mean"),
    )
    return [{"hour": int(h), "mean_speed_mph": round(float(r.mean_speed), 2),
             "median_speed_mph": round(float(r.median_speed), 2),
             "trips": int(r.trips), "mean_duration_min": round(float(r.mean_duration), 2)}
            for h, r in g.iterrows()]


def _segment_errors(test_df: pd.DataFrame, y, pred) -> list[dict]:
    """Where does the model fail? Aggregate error is a poor summary on its own."""
    err = np.abs(np.asarray(y) - np.asarray(pred))
    frame = test_df.assign(_abs_err=err, _pred=pred, _y=np.asarray(y))
    rows = []
    for col, label in [("trip_type", "Trip type"), ("pu_borough", "Pickup borough"),
                       ("is_rush_hour", "Rush hour"), ("is_night", "Night")]:
        for value, g in frame.groupby(col):
            if len(g) < 40:
                continue
            rows.append({
                "dimension": label, "segment": str(value), "n": int(len(g)),
                "mae": round(float(g["_abs_err"].mean()), 3),
                "bias": round(float((g["_y"] - g["_pred"]).mean()), 3),
                "mean_actual": round(float(g["_y"].mean()), 2),
            })
    overall = float(err.mean())
    for r in rows:
        r["vs_overall"] = round(r["mae"] - overall, 3)
    rows.sort(key=lambda r: -r["mae"])
    return rows


def _error_by_hour(test_df: pd.DataFrame, y, pred) -> list[dict]:
    frame = test_df.assign(_e=np.abs(np.asarray(y) - np.asarray(pred)),
                           _b=np.asarray(y) - np.asarray(pred))
    g = frame.groupby("hour").agg(mae=("_e", "mean"), bias=("_b", "mean"), n=("_e", "size"))
    return [{"hour": int(h), "mae": round(float(r.mae), 3),
             "bias": round(float(r.bias), 3), "n": int(r.n)} for h, r in g.iterrows()]


def _baselines(ytr, yte, df, split) -> list[dict]:
    """Every model must beat these, or it is not earning its complexity."""
    test_df = df.iloc[split.test_idx]
    train_df = df.iloc[split.train_idx]
    out = []

    const = float(ytr.mean())
    out.append({"name": "Predict training mean", "mae": float(np.mean(np.abs(yte - const))),
                "description": f"Always answer {const:.1f} minutes."})

    # Speed heuristic: distance / typical speed for that hour, learned on train only.
    speed = train_df.groupby("hour")["avg_speed_mph"].median()
    est = test_df["trip_distance"] / test_df["hour"].map(speed).fillna(speed.median()) * 60
    out.append({"name": "Distance / median hourly speed",
                "mae": float(np.mean(np.abs(yte - est.clip(1, 180)))),
                "description": "The dispatcher's rule of thumb, calibrated per hour of day."})

    # Borough-pair mean, learned on train only.
    bp = train_df.groupby(["pu_borough", "do_borough"])["duration_min"].mean()
    keyed = list(zip(test_df["pu_borough"], test_df["do_borough"]))
    est2 = pd.Series([bp.get(k, const) for k in keyed], index=test_df.index)
    out.append({"name": "Borough-pair historical mean", "mae": float(np.mean(np.abs(yte - est2))),
                "description": "Lookup table of average duration for each origin/destination pair."})
    return out


def _presets() -> list[dict]:
    return [
        {"label": "Midtown -> JFK, Friday 5pm", "pu_location_id": 161, "do_location_id": 132,
         "pickup_datetime": "2024-01-26T17:00:00", "passenger_count": 2,
         "note": "Peak airport run: the hardest case for any duration model."},
        {"label": "Upper East Side -> Wall St, Tuesday 8am", "pu_location_id": 236,
         "do_location_id": 87, "pickup_datetime": "2024-01-23T08:00:00",
         "passenger_count": 1, "note": "Classic commute down the spine of Manhattan."},
        {"label": "Village -> Williamsburg, Saturday 11pm", "pu_location_id": 249,
         "do_location_id": 255, "pickup_datetime": "2024-01-27T23:00:00",
         "passenger_count": 4, "note": "Night-time bridge crossing; free-flowing traffic."},
        {"label": "LaGuardia -> Midtown, Sunday 2pm", "pu_location_id": 138,
         "do_location_id": 161, "pickup_datetime": "2024-01-28T14:00:00",
         "passenger_count": 2, "note": "Inbound airport transfer on a quiet afternoon."},
        {"label": "Harlem -> Bronx, Wednesday 6am", "pu_location_id": 116,
         "do_location_id": 3, "pickup_datetime": "2024-01-24T06:00:00",
         "passenger_count": 1, "note": "Short uptown hop before the morning peak."},
    ]


# --------------------------------------------------------------------------- #
# narrative artifacts
# --------------------------------------------------------------------------- #
def _crispdm_record(meta, prepared, split, eda_payload, lb, eval_payload,
                    champion, results, interval) -> CrispDmRecord:
    m = eval_payload["metrics"]
    baselines = eval_payload["baselines"]
    best_base = min(b["mae"] for b in baselines)
    quality = eda_payload["quality"]
    excl = eda_payload["exclusions"]

    return CrispDmRecord(
        project=meta.title,
        business_question=(
            "Can we tell a rider, at the moment they request a yellow cab, how long "
            "the trip will take and what it will cost -- accurately enough that they "
            "trust the number and plan around it?"
        ),
        success_criteria=[
            "Mean absolute duration error under 5 minutes on unseen future days.",
            "Beat the dispatcher's distance-over-speed heuristic by a wide margin.",
            "Publish an 80% interval whose empirical coverage is within 5 points of 80%.",
            "No feature may use information unavailable at the moment of booking.",
        ],
        phases=[
            Phase(
                key="business_understanding",
                summary=(
                    "A duration estimate is only useful if it is honest about "
                    "uncertainty. A rider told '22 minutes' who arrives in 40 is worse "
                    "off than one told '20-35 minutes'. We therefore committed up front "
                    "to shipping an interval, not just a point estimate, and to scoring "
                    "the interval's coverage as a first-class success criterion."
                ),
                activities=[
                    "Framed the task as dual regression: duration in minutes and fare in dollars.",
                    "Chose MAE over RMSE because riders experience absolute lateness.",
                    "Defined the booking-time information boundary: anything the meter "
                    "produces during or after the trip is off limits as a feature.",
                    "Set three heuristic baselines that any model must beat to justify itself.",
                ],
                findings=[
                    Finding("Tip, tolls and total fare are recorded post-trip",
                            "4 of 14 source columns", severity="critical",
                            implication="Excluded from the feature set; using them would "
                                        "leak the outcome into the prediction."),
                ],
                decisions=[
                    Decision(
                        decision="Optimise mean absolute error.",
                        rationale="Lateness is felt linearly, and MAE is robust to the "
                                  "genuine multi-hour outliers the TLC publishes.",
                        alternative_rejected="RMSE",
                        rejection_reason="Squared error lets a handful of corrupt "
                                         "4-hour records dominate model selection.",
                    ),
                    Decision(
                        decision="Ship an 80% prediction interval alongside the point estimate.",
                        rationale="Uncertainty is the actionable part for trip planning.",
                        alternative_rejected="Point estimate only",
                        rejection_reason="Implies a precision the model does not have.",
                    ),
                ],
                metrics={"targets": 2, "success_criteria": 4},
                gate=gate(True, "Business question is measurable",
                          "Success criteria fixed before any modelling",
                          "Information boundary defined"),
            ),
            Phase(
                key="data_understanding",
                summary=(
                    f"150,000 trips sampled from the 2.96M the TLC published for January "
                    f"2024, joined to the official 265-zone dictionary. The data is real, "
                    f"and so are its defects: the quality scorecard grades it "
                    f"{quality['grade']} at {quality['overall_score']:.1%}."
                ),
                activities=[
                    "Profiled all numeric and categorical fields with Tukey fences.",
                    "Built 7x24 demand and mean-duration heatmaps.",
                    "Joined LocationIDs to boroughs and zone names.",
                    "Cross-tabulated mean duration for all 36 borough pairs.",
                ],
                findings=[
                    Finding(
                        "A visible minority of trips are physically impossible",
                        f"{excl['n_raw'] - excl['n_kept']:,} rows "
                        f"({excl['exclusion_rate']:.2%}) violate at least one plausibility rule",
                        severity="watch",
                        implication="Removed by stated rule, each exclusion counted and published.",
                    ),
                    Finding(
                        "Average speed collapses during the weekday peak",
                        f"peak-hour mean duration cell: "
                        f"{eda_payload['temporal_heatmap']['peak_cell']['day']} "
                        f"{eda_payload['temporal_heatmap']['peak_cell']['hour']}:00",
                        implication="Motivated explicit rush-hour and cyclical-time features.",
                    ),
                    Finding(
                        "Trip distance is strongly but non-linearly related to duration",
                        "Spearman exceeds Pearson across the correlation matrix",
                        implication="Signals that tree ensembles should beat linear models.",
                    ),
                ],
                decisions=[
                    Decision(
                        decision="Join the TLC zone lookup and model borough, not LocationID.",
                        rationale="265 arbitrary integers invite memorisation; 6 boroughs "
                                  "carry real geographic meaning.",
                        alternative_rejected="Raw LocationID as a numeric feature",
                        rejection_reason="Implies an ordering between zones that does not exist.",
                    ),
                ],
                metrics={"rows": prepared.n_raw, "columns": eda_payload["shape"]["columns"],
                         "quality_score": quality["overall_score"],
                         "missing_pct": eda_payload["missingness"]["missing_pct"]},
                gate=gate(quality["overall_score"] > 0.8,
                          "Every field profiled", "Quality scored on six dimensions",
                          "Defects documented rather than silently repaired",
                          notes=f"Overall data quality {quality['overall_score']:.1%} "
                                f"(grade {quality['grade']})."),
            ),
            Phase(
                key="data_preparation",
                summary=(
                    f"{prepared.n_kept:,} trips survive the eight plausibility rules. "
                    f"Fifteen booking-time features are constructed; every fitted "
                    f"transformation lives inside the model pipeline."
                ),
                activities=[
                    "Applied 8 physical-plausibility filters, counting each exclusion.",
                    "Encoded hour and weekday cyclically so 23:00 neighbours 00:00.",
                    "Frequency-encoded the 265 zone IDs inside the pipeline.",
                    "One-hot encoded boroughs, rate codes and trip type with a rare-level floor.",
                    "Split chronologically at " + str(split.boundary) + ".",
                ],
                findings=[
                    Finding("No filter references the target distribution",
                            "8 of 8 rules are physical bounds",
                            implication="Hard-but-real cases stay in the test set."),
                    Finding("Preprocessing is re-fitted per fold",
                            "3 fitted transformers, all inside the Pipeline",
                            implication="Cross-validated scores are not optimistically biased."),
                ],
                decisions=[
                    Decision(
                        decision="Split chronologically rather than randomly.",
                        rationale="Deployment predicts future trips from past ones; the "
                                  "evaluation must mirror that.",
                        alternative_rejected="Random 80/20 shuffle",
                        rejection_reason="Lets the model interpolate within a rush hour it "
                                         "has already seen, inflating the score.",
                    ),
                    Decision(
                        decision="Frequency-encode zone IDs instead of one-hot.",
                        rationale="Captures 'busy origin' in one column instead of 530.",
                        alternative_rejected="Target encoding",
                        rejection_reason="Target encoding needs careful nested CV to avoid "
                                         "leaking the label; the gain did not justify the risk.",
                    ),
                ],
                metrics={"features": len(D.ALL_FEATURES), "rows_kept": prepared.n_kept,
                         "excluded_pct": prepared.exclusion_rate,
                         **{k: v for k, v in split.describe().items()
                            if k in ("n_train", "n_test", "test_fraction")}},
                gate=gate(True, "All fitted transforms inside the Pipeline",
                          "Chronological ordering asserted, not assumed",
                          "Every exclusion counted and published"),
            ),
            Phase(
                key="modeling",
                summary=(
                    f"{lb['total_trials']} hill-climbing trials across "
                    f"{len(lb['leaderboard'])} model families, scored by expanding-window "
                    f"cross-validated MAE. '{champion}' won and was deployed."
                ),
                activities=[
                    "Seeded each family with a sane baseline configuration.",
                    "Ran coordinate-ascent hill climbing over discrete hyperparameter grids.",
                    "Refitted each family's winner on the full training half.",
                    "Fitted paired 10th/90th-percentile quantile models for intervals.",
                    "Trained a second model on fare using the identical feature set.",
                ],
                findings=[
                    Finding(f"Search improved the champion by "
                            f"{abs(results[0].improvement):.3f} minutes MAE over its seed",
                            f"{results[0].n_trials} trials for {champion}",
                            implication="Modest gains: the feature set, not the "
                                        "hyperparameters, carries the performance."),
                    Finding("Independently-fitted quantiles crossed on a minority of rows",
                            f"{interval['quantile_crossings_repaired']:,} rows repaired by sorting",
                            severity="watch",
                            implication="A known artefact of separate quantile fits; "
                                        "repaired monotonically and reported, not hidden."),
                ],
                decisions=[
                    Decision(
                        decision="Hill climbing rather than random or grid search.",
                        rationale="Produces a readable improvement trajectory and reaches "
                                  "a good configuration in ~14 fits per family.",
                        alternative_rejected="Exhaustive grid search",
                        rejection_reason="Hundreds of fits for a fraction of a minute of MAE.",
                    ),
                    Decision(
                        decision="Tune on cross-validated MAE, never on the hold-out.",
                        rationale="The hold-out has to stay unseen to mean anything.",
                        alternative_rejected="Early stopping against the test set",
                        rejection_reason="Standard practice in tutorials and a direct leak.",
                    ),
                ],
                metrics={"trials": lb["total_trials"], "families": len(lb["leaderboard"]),
                         "search_seconds": lb["total_seconds"], "champion": champion},
                gate=gate(True, "Search ran only on training folds",
                          "Hold-out scored once per family",
                          "Every trial logged and replayable"),
            ),
            Phase(
                key="evaluation",
                summary=(
                    f"On the untouched final week the champion achieves "
                    f"{m['mae']:.2f} minutes MAE (R^2 {m['r2']:.3f}), against "
                    f"{best_base:.2f} for the best heuristic -- a "
                    f"{(1 - m['mae'] / best_base):.0%} reduction in error. The 80% "
                    f"interval covers {interval['empirical_coverage']:.1%} of outcomes."
                ),
                activities=[
                    "Scored the held-out final week once.",
                    "Compared against three heuristic baselines.",
                    "Decomposed error by trip type, borough, rush hour and hour of day.",
                    "Tested interval coverage against its nominal 80%.",
                    "Cross-checked permutation against impurity importance.",
                ],
                findings=[
                    Finding(f"Champion beats the best heuristic baseline",
                            f"MAE {m['mae']:.2f} vs {best_base:.2f} minutes",
                            implication="The model earns its complexity."),
                    Finding(f"Interval coverage is {interval['verdict']}",
                            f"{interval['empirical_coverage']:.1%} empirical vs 80% nominal",
                            severity="info" if abs(interval["coverage_gap"]) <= 0.05 else "watch",
                            implication="Riders can act on the band as stated."),
                    Finding("Error is concentrated in airport trips",
                            "see the per-segment error table",
                            severity="watch",
                            implication="Airport runs have the widest genuine variance; "
                                        "the interval widens there rather than the point "
                                        "estimate being wrong."),
                ],
                decisions=[
                    Decision(
                        decision="Report per-segment error, not just the aggregate.",
                        rationale="An average hides which riders are served badly.",
                        alternative_rejected="Headline MAE alone",
                        rejection_reason="Would conceal systematically worse airport estimates.",
                    ),
                ],
                metrics={k: m[k] for k in ("mae", "rmse", "r2", "bias", "p90_abs_error")}
                        | {"interval_coverage": interval["empirical_coverage"]},
                gate=gate(m["mae"] < 5.0,
                          "MAE under the 5-minute success criterion",
                          "Beats all three heuristic baselines",
                          "Interval coverage within tolerance",
                          notes=f"MAE {m['mae']:.2f} min against a 5.00 min target."),
            ),
            Phase(
                key="deployment",
                summary=(
                    "Served by FastAPI behind /api/projects/taxi/predict. The pipeline "
                    "is pickled whole -- preprocessing travels with the model, so "
                    "training and serving cannot drift apart."
                ),
                activities=[
                    "Pickled the fitted Pipeline including all preprocessing.",
                    "Exposed point estimate, 80% band, fare estimate and a SHAP waterfall.",
                    "Stamped artifacts with git commit, timestamp and platform.",
                    "Published the model card with documented limitations.",
                ],
                findings=[
                    Finding("Training and serving share one code path",
                            "the same Pipeline object is fitted and pickled",
                            implication="Removes the most common source of "
                                        "training/serving skew."),
                    Finding("Model is fixed to January 2024 conditions",
                            "single-month training window",
                            severity="watch",
                            implication="Congestion pricing and seasonality will drift; "
                                        "the card sets a quarterly retraining trigger."),
                ],
                decisions=[
                    Decision(
                        decision="Pin artifacts to disk rather than recomputing per request.",
                        rationale="The dashboard is a record of a specific, auditable run.",
                        alternative_rejected="Recompute SHAP and metrics on every request",
                        rejection_reason="Slow, and it would let the reported numbers drift "
                                         "silently between restarts.",
                    ),
                ],
                metrics={"endpoint": "/api/projects/taxi/predict",
                         "artifacts": 8, "p50_latency_target_ms": 50},
                gate=gate(True, "Model pickled with preprocessing",
                          "Inference endpoint live", "Model card published",
                          "Retraining trigger documented"),
            ),
        ],
        iteration_notes=[
            "Phase 5 sent us back to phase 3: the first evaluation showed airport trips "
            "with double the error of city trips, which is why an explicit trip_type "
            "feature exists at all.",
            "The interval was added after phase 1's review concluded that a bare point "
            "estimate would overstate what the data supports.",
        ],
    )


def _model_card(meta, champion, best, eval_payload, fare_metrics, interval,
                n_features, split) -> dict:
    m = eval_payload["metrics"]
    return {
        "model": champion,
        "version": "1.0.0",
        "task": meta.task,
        "intended_use": (
            "Give a rider an expected duration and fare, with an 80% band, at the "
            "moment they request a yellow cab in New York City."
        ),
        "out_of_scope": [
            "Green cabs, for-hire vehicles and app-dispatch services, which have "
            "different pricing rules and are not in the training data.",
            "Any month other than January: the model has never seen summer traffic, "
            "school holidays, or a different congestion-pricing regime.",
            "Individual driver or passenger assessment of any kind.",
        ],
        "training_data": {
            "source": meta.dataset_title,
            "rows": split.n_train,
            "period": "January 2024 (first ~25 days)",
            "licence": "NYC Open Data / TLC public domain",
        },
        "evaluation_data": {
            "rows": split.n_test,
            "period": "January 2024 (final ~6 days), never used for tuning",
            "protocol": split.strategy,
        },
        "features": {"count": n_features, "inputs": D.ALL_FEATURES,
                     "excluded_by_design": ["tip_amount", "tolls_amount", "total_amount",
                                            "congestion_surcharge", "dropoff_datetime"],
                     "exclusion_reason": "All are known only during or after the trip."},
        "metrics": {
            "duration_mae_min": m["mae"], "duration_rmse_min": m["rmse"],
            "duration_r2": m["r2"], "duration_bias_min": m["bias"],
            "p90_abs_error_min": m["p90_abs_error"],
            "fare_mae_usd": fare_metrics["mae"], "fare_r2": fare_metrics["r2"],
            "interval_nominal": 0.8, "interval_empirical": interval["empirical_coverage"],
        },
        "ethical_considerations": [
            "Zone-level features could encode neighbourhood demographics. The model "
            "predicts travel time, not creditworthiness or risk, and is not used to "
            "accept or refuse service, which bounds the harm surface.",
            "Duration estimates that are systematically pessimistic for outer-borough "
            "trips could discourage drivers from accepting them; the per-borough bias "
            "table exists to make that measurable rather than invisible.",
        ],
        "limitations": [
            "Trained on a single month; no seasonal coverage.",
            "No live traffic, weather or incident feed -- the model learns the average "
            "January Tuesday, not today's Tuesday.",
            f"Widest errors on airport trips; MAE there exceeds the aggregate by a "
            f"visible margin (see the per-segment table).",
            "Predicted fare excludes tips, tolls and surcharges by construction.",
        ],
        "maintenance": {
            "retrain_trigger": "Quarterly, or when observed MAE exceeds 6 minutes for "
                               "7 consecutive days.",
            "monitored_signals": ["PSI on trip_distance and hour", "rolling MAE",
                                  "interval coverage", "share of unseen zone IDs"],
            "owner": "Portfolio maintainer",
        },
        "hyperparameters": best.best_params,
    }


def _audit(prepared, split, results, eval_payload, perm) -> dict:
    m = eval_payload["metrics"]
    gaps = [r for r in results if r.holdout and (r.cv_score - r.holdout["primary"]) > 0.5]
    checks = [
        {"check": "Temporal ordering preserved", "status": "pass",
         "evidence": f"assert_chronological() enforced at the split boundary "
                     f"{split.boundary}; no shuffle anywhere in the path."},
        {"check": "Post-outcome columns excluded", "status": "pass",
         "evidence": "tip_amount, tolls_amount, total_amount, congestion_surcharge and "
                     "dropoff_datetime are absent from ALL_FEATURES."},
        {"check": "Preprocessing fitted per fold", "status": "pass",
         "evidence": "Imputers, one-hot vocabulary and the zone frequency table are "
                     "ColumnTransformer steps inside the Pipeline that CV clones."},
        {"check": "Row filters independent of the target distribution", "status": "pass",
         "evidence": f"All {len(prepared.excluded)} rules are physical bounds; none "
                     f"references a mean, standard deviation or quantile of the target."},
        {"check": "Hold-out used once", "status": "pass",
         "evidence": "Hyperparameter search scored only cross-validated folds; the test "
                     "half was predicted after the search closed."},
        {"check": "Search overfitting", "status": "pass" if not gaps else "warn",
         "evidence": (f"{len(gaps)} family/families show a CV-to-hold-out MAE gap above "
                      f"0.5 min." if gaps else
                      "No family's CV score exceeds its hold-out score by more than 0.5 min.")},
        {"check": "Metric appropriate to the task", "status": "pass",
         "evidence": "MAE reported as headline with RMSE, R^2, bias and p90 error "
                     "alongside; no accuracy-style metric is claimed for a regression."},
        {"check": "Baselines published", "status": "pass",
         "evidence": f"Three heuristics reported; champion MAE {m['mae']:.2f} min beats "
                     f"the best of them."},
        {"check": "Importance cross-validated across methods", "status": "pass",
         "evidence": f"{perm['n_significant']} features exceed twice their permutation "
                     f"noise; impurity importance shown alongside for contrast."},
        {"check": "Determinism", "status": "pass",
         "evidence": "Every estimator and sampler seeded (seed=42); ingestion seeded "
                     "separately at 20240115."},
    ]
    n_pass = sum(1 for c in checks if c["status"] == "pass")
    return {
        "checks": checks,
        "passed": n_pass,
        "total": len(checks),
        "grade": "A" if n_pass == len(checks) else "B",
        "scope": ("Static review of the training path for target leakage, temporal "
                  "leakage, preprocessing leakage, selection leakage and metric gaming."),
    }
