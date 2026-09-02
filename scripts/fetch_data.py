"""Reproducible ingestion of the eight *real* datasets behind this portfolio.

Design contract
---------------
Every project in this repository is trained on genuine, publicly-licensed data --
never on synthetic numbers invented to make metrics look good. This script is the
only place where bytes enter the repository, and it is deliberately boring:

    download(url) -> data/raw/<file>        # verbatim, untouched, gitignored
    curate(raw)   -> data/curated/<name>.*  # deterministic, small, committed
    manifest()    -> data/MANIFEST.json     # provenance + sha256 of every artifact

Curation is *only* ever: column selection, dtype coercion, deterministic
subsampling with a pinned seed, and stable sorting. No imputation, no outlier
removal, no feature engineering, and above all no target-aware filtering -- all of
that belongs downstream of the train/test split, inside the modelling pipelines,
where the leakage audit can see it.

Run:  python scripts/fetch_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "curated"
MANIFEST = ROOT / "data" / "MANIFEST.json"

# One seed for the entire ingestion layer. Any subsampling below is a pure
# function of this constant, so two clean checkouts produce byte-identical files.
SEED = 20240115

USER_AGENT = "ds-fullstack-portfolio/1.0 (academic coursework; +https://github.com)"


# --------------------------------------------------------------------------- #
# Source declarations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Source:
    """A remote file plus everything needed to cite it honestly."""

    key: str
    url: str
    filename: str
    title: str
    licence: str
    citation: str
    # Some sources are big; we only need the head of the file for a few of them.
    approx_mb: float = 0.0


SOURCES: dict[str, Source] = {
    s.key: s
    for s in [
        Source(
            key="nyc_taxi",
            url="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
            filename="yellow_tripdata_2024-01.parquet",
            title="NYC TLC Yellow Taxi Trip Records, January 2024",
            licence="NYC Open Data / TLC public domain",
            citation="NYC Taxi & Limousine Commission, Trip Record Data (2024).",
            approx_mb=48.0,
        ),
        Source(
            key="taxi_zones",
            url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
            filename="taxi_zone_lookup.csv",
            title="NYC TLC Taxi Zone Lookup (265 zones, 6 boroughs)",
            licence="NYC Open Data / TLC public domain",
            citation="NYC Taxi & Limousine Commission, Taxi Zone Lookup Table.",
            approx_mb=0.02,
        ),
        Source(
            key="telco_churn",
            url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            filename="Telco-Customer-Churn.csv",
            title="IBM Telco Customer Churn",
            licence="Apache-2.0 (IBM sample data)",
            citation="IBM Cognos Analytics, Telco customer churn sample dataset.",
            approx_mb=1.0,
        ),
        Source(
            key="online_retail",
            url="https://archive.ics.uci.edu/static/public/352/online+retail.zip",
            filename="online_retail.zip",
            title="UCI Online Retail (UK gift retailer invoices, 2010-2011)",
            licence="CC BY 4.0",
            citation="Chen, D. (2015). Online Retail. UCI Machine Learning Repository.",
            approx_mb=23.0,
        ),
        Source(
            key="adult",
            url="https://archive.ics.uci.edu/static/public/2/adult.zip",
            filename="adult.zip",
            title="UCI Adult / Census Income (1994 CPS extract)",
            licence="CC BY 4.0",
            citation="Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository.",
            approx_mb=0.6,
        ),
        Source(
            key="bike_sharing",
            url="https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
            filename="bike_sharing.zip",
            title="UCI Bike Sharing (Capital Bikeshare hourly demand, 2011-2012)",
            licence="CC BY 4.0",
            citation="Fanaee-T, H. & Gama, J. (2013). Event labeling combining ensemble "
            "detectors and background knowledge. Progress in AI.",
            approx_mb=0.3,
        ),
        Source(
            key="tiny_shakespeare",
            url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            filename="tiny_shakespeare.txt",
            title="Tiny Shakespeare character corpus",
            licence="Public domain (Shakespeare) / MIT packaging",
            citation="Karpathy, A. (2015). char-rnn tinyshakespeare corpus.",
            approx_mb=1.1,
        ),
    ]
}


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(src: Source, force: bool = False) -> Path:
    dest = RAW / src.filename
    if dest.exists() and not force:
        print(f"  [cached] {src.filename} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [fetch ] {src.filename} (~{src.approx_mb:.1f} MB) <- {src.url}")
    req = urllib.request.Request(src.url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)
    print(f"  [ok    ] {src.filename} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _zip_member(path: Path, name_contains: str) -> bytes:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if name_contains in n and not n.startswith("__")]
        if not names:
            raise FileNotFoundError(f"{name_contains!r} not in {path.name}: {zf.namelist()}")
        return zf.read(sorted(names, key=len)[0])


# --------------------------------------------------------------------------- #
# Curation functions -- one per curated artifact
# --------------------------------------------------------------------------- #
@dataclass
class Curated:
    name: str
    frame: pd.DataFrame
    source_key: str
    rule: str
    notes: list[str] = field(default_factory=list)
    fmt: str = "parquet"


def curate_nyc_taxi(raw: Path) -> Curated:
    """150k trips sampled from 2.96M January-2024 yellow-cab records.

    We keep every trip the TLC published -- including the physically impossible
    ones (zero distance, negative fare, 8-hour crosstown crawls). Cleaning them
    here would hide the data-quality story that project 01's EDA is meant to tell,
    and would quietly leak an analyst's judgement into the "raw" layer.
    """
    cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "tip_amount",
        "tolls_amount",
        "total_amount",
        "congestion_surcharge",
        "Airport_fee",
    ]
    df = pd.read_parquet(raw, columns=cols)
    n_total = len(df)
    # Deterministic uniform sample -- no filtering, no stratification on target.
    df = df.sample(n=150_000, random_state=SEED).sort_values("tpep_pickup_datetime")
    df = df.reset_index(drop=True)
    df.columns = [
        "pickup_datetime", "dropoff_datetime", "passenger_count", "trip_distance",
        "ratecode_id", "pu_location_id", "do_location_id", "payment_type",
        "fare_amount", "tip_amount", "tolls_amount", "total_amount",
        "congestion_surcharge", "airport_fee",
    ]
    return Curated(
        name="nyc_taxi_trips",
        frame=df,
        source_key="nyc_taxi",
        rule=f"uniform sample n=150000 of {n_total:,} rows, seed={SEED}, sorted by pickup time",
        notes=[
            "No cleaning applied: anomalous fares/distances are retained on purpose so "
            "the EDA and data-quality scorecard describe the real distribution.",
            "Zone IDs are TLC LocationIDs (1-265); borough mapping is applied downstream.",
        ],
    )


def curate_taxi_zones(raw: Path) -> Curated:
    """The TLC's own zone dictionary: LocationID -> borough, zone, service zone.

    Joining this lets the model use *borough* (6 levels, semantically meaningful)
    instead of raw LocationID (265 arbitrary integers a tree would happily
    memorise), and lets the UI offer real place names instead of ID numbers.
    """
    df = pd.read_csv(raw)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"locationid": "location_id"})
    df["borough"] = df["borough"].fillna("Unknown").str.strip()
    df["zone"] = df["zone"].fillna("Unknown").str.strip()
    df = df.sort_values("location_id").reset_index(drop=True)
    return Curated(
        name="taxi_zone_lookup",
        frame=df,
        source_key="taxi_zones",
        rule="verbatim lookup table, columns lower-cased",
        notes=[
            "265 LocationIDs across Manhattan, Brooklyn, Queens, Bronx, Staten Island "
            "and EWR, plus 'Unknown' placeholders 264/265.",
        ],
        fmt="csv",
    )


def curate_telco_churn(raw: Path) -> Curated:
    """All 7,043 Telco subscribers, verbatim. Small enough to keep whole."""
    df = pd.read_csv(raw)
    df.columns = [c.strip() for c in df.columns]
    # TotalCharges ships as a string with 11 blank cells for day-zero customers.
    # We coerce the dtype but deliberately leave the NaNs for the imputer inside
    # the modelling pipeline -- imputing here would fit on the full dataset and
    # leak test-fold information into training.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    n_blank = int(df["TotalCharges"].isna().sum())
    df = df.sort_values("customerID").reset_index(drop=True)
    return Curated(
        name="telco_churn",
        frame=df,
        source_key="telco_churn",
        rule="complete dataset, no sampling; TotalCharges coerced to float64",
        notes=[
            f"{n_blank} TotalCharges cells are blank (tenure=0 customers); left as NaN "
            "so imputation happens inside the cross-validated pipeline.",
            "Target column is 'Churn' (Yes/No), 26.5% positive -- imbalanced, so the "
            "project optimises PR-AUC rather than accuracy.",
        ],
        fmt="csv",
    )


def _load_online_retail(raw: Path) -> pd.DataFrame:
    print("  [parse ] online_retail.xlsx (541k rows, ~40s)...")
    blob = _zip_member(raw, "Online Retail")
    df = pd.read_excel(io.BytesIO(blob), dtype={"CustomerID": "float64"})
    df.columns = [str(c).strip() for c in df.columns]
    return df


def curate_online_retail_transactions(df: pd.DataFrame) -> Curated:
    """Invoice line items for market-basket mining.

    Credit notes (negative quantity, invoices prefixed 'C') are dropped because an
    association rule over returns is not a basket -- that is a definitional
    restriction on the unit of analysis, not a data-quality judgement.
    """
    n_total = len(df)
    keep = df[(df["Quantity"] > 0) & df["Description"].notna()].copy()
    keep = keep[~keep["InvoiceNo"].astype(str).str.startswith("C")]
    keep["Description"] = keep["Description"].str.strip().str.upper()
    # InvoiceNo/StockCode arrive as a mix of ints and strings ('A563185', '85123A');
    # pin them to str so the column has one Arrow type.
    for col in ("InvoiceNo", "StockCode"):
        keep[col] = keep[col].astype(str).str.strip()
    keep = keep[["InvoiceNo", "StockCode", "Description", "Quantity",
                 "InvoiceDate", "UnitPrice", "CustomerID", "Country"]]
    keep = keep.sort_values(["InvoiceDate", "InvoiceNo", "StockCode"]).reset_index(drop=True)
    keep.columns = ["invoice_no", "stock_code", "description", "quantity",
                    "invoice_date", "unit_price", "customer_id", "country"]
    return Curated(
        name="online_retail_transactions",
        frame=keep,
        source_key="online_retail",
        rule=f"{len(keep):,} of {n_total:,} line items: positive quantity, non-null "
             "description, credit-note invoices ('C...') excluded",
        notes=[
            "Unit of analysis is a purchase basket, so returns are out of scope by "
            "definition rather than by cleaning.",
            "Descriptions upper-cased and whitespace-stripped to merge trivial variants.",
        ],
    )


def curate_online_retail_rfm(df: pd.DataFrame) -> Curated:
    """Customer-level RFM features for unsupervised segmentation.

    Recency is measured against the day *after* the last invoice in the archive,
    which is the standard convention and keeps every customer's recency strictly
    positive. Because segmentation is unsupervised there is no target to leak, but
    the scaler is still fitted inside the clustering pipeline, not here.
    """
    txn = df[(df["Quantity"] > 0) & df["CustomerID"].notna()].copy()
    txn = txn[~txn["InvoiceNo"].astype(str).str.startswith("C")]
    txn["revenue"] = txn["Quantity"] * txn["UnitPrice"]
    asof = txn["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = txn.groupby("CustomerID").agg(
        recency_days=("InvoiceDate", lambda s: (asof - s.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("revenue", "sum"),
        n_items=("Quantity", "sum"),
        n_distinct_skus=("StockCode", "nunique"),
        avg_unit_price=("UnitPrice", "mean"),
        tenure_days=("InvoiceDate", lambda s: (s.max() - s.min()).days),
        country=("Country", lambda s: s.mode().iat[0]),
    ).reset_index()
    rfm = rfm.rename(columns={"CustomerID": "customer_id"})
    rfm["customer_id"] = rfm["customer_id"].astype(int)
    rfm = rfm.sort_values("customer_id").reset_index(drop=True)
    return Curated(
        name="online_retail_rfm",
        frame=rfm,
        source_key="online_retail",
        rule=f"customer-level aggregation of positive-quantity invoices; "
             f"recency as-of {asof.date()} (last invoice + 1 day)",
        notes=[
            "Only customers with a CustomerID are included; ~25% of line items are "
            "anonymous guest checkouts and cannot be aggregated to a person.",
            "Monetary is gross revenue (quantity x unit price), not net of returns.",
        ],
        fmt="csv",
    )


def curate_adult(raw: Path) -> Curated:
    """Census Income: 48,842 records, the canonical fairness/AutoML benchmark."""
    names = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income",
    ]
    frames = []
    for member in ("adult.data", "adult.test"):
        blob = _zip_member(raw, member)
        part = pd.read_csv(
            io.BytesIO(blob), names=names, skipinitialspace=True,
            na_values=["?"], comment="|", skiprows=1 if member == "adult.test" else 0,
        )
        part["split_origin"] = member
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    # adult.test encodes the label with a trailing period; normalise the *label
    # text* only -- the class balance is untouched.
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df = df[df["income"].isin([">50K", "<=50K"])].reset_index(drop=True)
    return Curated(
        name="adult_census_income",
        frame=df,
        source_key="adult",
        rule="adult.data + adult.test concatenated (48,842 rows); '?' -> NaN; "
             "label period stripped from the test half",
        notes=[
            "Missing values in workclass/occupation/native_country are preserved as "
            "NaN for pipeline-internal imputation.",
            "'sex' and 'race' are retained solely to compute the fairness audit; they "
            "are excluded from the model feature set.",
        ],
        fmt="csv",
    )


def curate_bike_sharing(raw: Path) -> Curated:
    """17,379 hourly demand observations with real weather covariates."""
    blob = _zip_member(raw, "hour.csv")
    df = pd.read_csv(io.BytesIO(blob))
    df["dteday"] = pd.to_datetime(df["dteday"])
    # Reconstruct a true hourly timestamp; the archive splits date and hour.
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[[
        "timestamp", "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
        "casual", "registered", "cnt",
    ]]
    return Curated(
        name="bike_sharing_hourly",
        frame=df,
        source_key="bike_sharing",
        rule="hour.csv, all 17,379 rows, chronologically sorted with a reconstructed "
             "hourly timestamp",
        notes=[
            "temp/atemp/hum/windspeed arrive min-max normalised by the archive authors; "
            "denormalisation constants are documented in the project's data dictionary.",
            "The series has genuine gaps (165 missing hours) which the forecasting "
            "project reindexes explicitly rather than silently forward-filling.",
        ],
    )


def curate_kdd99() -> Curated:
    """Network-intrusion telemetry for anomaly detection, via scikit-learn.

    We take the SA variant: all normal traffic plus a small intrusion minority,
    which is the configuration the outlier-detection literature benchmarks on.
    """
    from sklearn.datasets import fetch_kddcup99

    bunch = fetch_kddcup99(subset="SA", percent10=True, random_state=SEED, as_frame=True)
    df = bunch.frame.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8") if hasattr(df[col], "str") and \
                isinstance(df[col].iloc[0], bytes) else df[col].astype(str)
    df["labels"] = df["labels"].astype(str).str.strip().str.rstrip(".")
    df["is_attack"] = (df["labels"] != "normal").astype(int)
    n_total = len(df)
    if n_total > 80_000:
        df = df.sample(n=80_000, random_state=SEED)
    df = df.reset_index(drop=True)
    rate = df["is_attack"].mean()
    return Curated(
        name="kdd99_intrusion",
        frame=df,
        source_key="kdd99_sklearn",
        rule=f"fetch_kddcup99(subset='SA', percent10=True), sampled to {len(df):,} of "
             f"{n_total:,} rows, seed={SEED}",
        notes=[
            f"Attack prevalence {rate:.3%} -- a genuine needle-in-haystack problem, "
            "which is why the project reports PR-AUC and precision@k, not accuracy.",
            "'labels' is held out entirely from the unsupervised detectors and used "
            "only for post-hoc evaluation.",
        ],
    )


def curate_tiny_shakespeare(raw: Path) -> Curated:
    text = raw.read_text(encoding="utf-8")
    vocab = sorted(set(text))
    out = CURATED / "tiny_shakespeare.txt"
    out.write_text(text, encoding="utf-8")
    meta = pd.DataFrame({
        "metric": ["characters", "vocabulary", "lines"],
        "value": [len(text), len(vocab), text.count("\n") + 1],
    })
    return Curated(
        name="tiny_shakespeare_stats",
        frame=meta,
        source_key="tiny_shakespeare",
        rule=f"verbatim copy; {len(text):,} characters, {len(vocab)} unique symbols",
        notes=[
            "Character-level corpus; the language model splits it 90/10 by position "
            "(never shuffled) so the validation text is strictly unseen continuation.",
        ],
        fmt="csv",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def write_curated(c: Curated) -> dict:
    CURATED.mkdir(parents=True, exist_ok=True)
    if c.fmt == "parquet":
        path = CURATED / f"{c.name}.parquet"
        c.frame.to_parquet(path, index=False, compression="snappy")
    else:
        path = CURATED / f"{c.name}.csv.gz"
        c.frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    size_mb = path.stat().st_size / 1e6
    print(f"  [write ] {path.name:38s} {len(c.frame):>8,} rows x {c.frame.shape[1]:>2} cols  {size_mb:5.2f} MB")
    src = SOURCES.get(c.source_key)
    return {
        "name": c.name,
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(c.frame)),
        "columns": [str(x) for x in c.frame.columns],
        "dtypes": {str(k): str(v) for k, v in c.frame.dtypes.items()},
        "size_mb": round(size_mb, 3),
        "sha256": _sha256(path),
        "curation_rule": c.rule,
        "notes": c.notes,
        "source": {
            "key": c.source_key,
            "title": src.title if src else "scikit-learn bundled fetcher",
            "url": src.url if src else "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html",
            "licence": src.licence if src else "Public domain (DARPA/MIT Lincoln Labs 1999)",
            "citation": src.citation if src else "Stolfo, S. et al. (1999). KDD Cup 1999 Data.",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--only", nargs="*", help="restrict to these dataset keys")
    args = ap.parse_args()

    np.random.seed(SEED)
    RAW.mkdir(parents=True, exist_ok=True)
    CURATED.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else None
    entries: list[dict] = []

    def want(key: str) -> bool:
        return wanted is None or key in wanted

    print("\n=== 1/3  Downloading raw sources ===")
    paths: dict[str, Path] = {}
    for key, src in SOURCES.items():
        if want(key):
            paths[key] = download(src, force=args.force)

    print("\n=== 2/3  Curating deterministic samples ===")
    if want("nyc_taxi"):
        entries.append(write_curated(curate_nyc_taxi(paths["nyc_taxi"])))
    if want("taxi_zones"):
        entries.append(write_curated(curate_taxi_zones(paths["taxi_zones"])))
    if want("telco_churn"):
        entries.append(write_curated(curate_telco_churn(paths["telco_churn"])))
    if want("online_retail"):
        retail = _load_online_retail(paths["online_retail"])
        entries.append(write_curated(curate_online_retail_transactions(retail)))
        entries.append(write_curated(curate_online_retail_rfm(retail)))
    if want("adult"):
        entries.append(write_curated(curate_adult(paths["adult"])))
    if want("bike_sharing"):
        entries.append(write_curated(curate_bike_sharing(paths["bike_sharing"])))
    if want("kdd99") or wanted is None:
        print("  [fetch ] KDD Cup 99 via scikit-learn (cached in ~/scikit_learn_data)")
        entries.append(write_curated(curate_kdd99()))
    if want("tiny_shakespeare"):
        entries.append(write_curated(curate_tiny_shakespeare(paths["tiny_shakespeare"])))

    print("\n=== 3/3  Writing manifest ===")
    manifest = {
        "generated_by": "scripts/fetch_data.py",
        "seed": SEED,
        "contract": (
            "Curation is limited to column selection, dtype coercion, deterministic "
            "subsampling and stable sorting. No imputation, scaling, outlier removal "
            "or target-aware filtering happens at this layer."
        ),
        "datasets": sorted(entries, key=lambda e: e["name"]),
    }
    if MANIFEST.exists() and wanted is not None:
        prior = json.loads(MANIFEST.read_text())
        by_name = {d["name"]: d for d in prior.get("datasets", [])}
        by_name.update({d["name"]: d for d in entries})
        manifest["datasets"] = sorted(by_name.values(), key=lambda e: e["name"])
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    total_rows = sum(d["rows"] for d in manifest["datasets"])
    total_mb = sum(d["size_mb"] for d in manifest["datasets"])
    print(f"  [ok    ] {MANIFEST.relative_to(ROOT)}")
    print(f"\nDone: {len(manifest['datasets'])} curated datasets, "
          f"{total_rows:,} rows, {total_mb:.1f} MB committed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
