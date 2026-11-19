#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dan Vy Le | EM1421 Social Media and Web Analytics
==========================
Olist E-commerce Review Analysis


Analysis of delivery performance, review timing, customer ratings,
and delivery-related expressions in Portuguese reviews.

Outputs:
- Three figures for the research paper
- Supporting CSV files
- Supplementary logistic regression results
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
SEED = 42                      # only used for the seeded example sample of text matches
ALPHA = 0.05                   # 95% confidence intervals
NEG_MAX_SCORE = 2              # negative rating = review_score <= 2
SMALL_N = 100                  # groups below this size are flagged as unstable
N_EXAMPLES_PER_CATEGORY = 15   # exploratory text-match examples exported per category

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Olist datasets"
ROOT_OUT = SCRIPT_DIR / "olist_results"
OUT_DIR = ROOT_OUT / "main_analysis"  # existing functions write main CSVs here
FIG_DIR = ROOT_OUT / "figures"
REG_DIR = ROOT_OUT / "supplementary_regression"

REQUIRED_FILES = {
    "orders": (
        "olist_orders_dataset.csv",
        ["order_id", "customer_id", "order_status", "order_purchase_timestamp",
         "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date"],
    ),
    "reviews": (
        "olist_order_reviews_dataset.csv",
        ["review_id", "order_id", "review_score", "review_comment_title",
         "review_comment_message", "review_creation_date", "review_answer_timestamp"],
    ),
    "items": (
        "olist_order_items_dataset.csv",
        ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
    ),
    "products": (
        "olist_products_dataset.csv",
        ["product_id", "product_category_name", "product_weight_g"],
    ),
}
DATE_COLUMNS = {
    "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
               "order_delivered_customer_date", "order_estimated_delivery_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "items": [],
    "products": [],
}

# Category orders (ASCII hyphens keep regression term names simple)
DELIVERY_LEVELS = ["On time", "Delayed"]
TIMING_LEVELS = ["Before delivery", "On/after delivery", "Unknown"]
SEVERITY4_LEVELS = ["On time", "1-3 days late", "4-7 days late", "8+ days late"]
SEVERITY_FINE_LEVELS = ["On time", "1-3 days late", "4-7 days late", "8-14 days late",
                        "15-30 days late", "31+ days late"]
GROUP4_LEVELS = [f"{d} | {t}" for d in DELIVERY_LEVELS for t in ["On/after delivery", "Before delivery"]]
# Display labels for figures (two lines, en dash for ranges)
GROUP4_DISPLAY = {
    "On time | On/after delivery": "On time,\nreviewed on/after delivery",
    "On time | Before delivery": "On time,\nreviewed before delivery",
    "Delayed | On/after delivery": "Delayed,\nreviewed on/after delivery",
    "Delayed | Before delivery": "Delayed,\nreviewed before delivery",
}
SEVERITY_DISPLAY = {
    "On time": "On time", "1-3 days late": "1\u20133 days", "4-7 days late": "4\u20137 days",
    "8+ days late": "8+ days", "8-14 days late": "8\u201314 days",
    "15-30 days late": "15\u201330 days", "31+ days late": "31+ days",
}

# Accessible diverging palette (colour-blind safe, RdBu-based) for 1..5 stars
RATING_COLORS = {1: "#b2182b", 2: "#ef8a62", 3: "#d9d9d9", 4: "#67a9cf", 5: "#2166ac"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


# ----------------------------------------------------------------------------
# 1. SMALL STATISTICAL HELPERS
# ----------------------------------------------------------------------------
def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=ALPHA, method="wilson")
    return float(lo), float(hi)


def mean_ci(x: pd.Series) -> tuple[float, float]:
    """t-based confidence interval for a mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - ALPHA / 2, n - 1)
    return float(x.mean() - t * se), float(x.mean() + t * se)


def present_levels(series: pd.Series, order: list[str]) -> list[str]:
    """Levels of `order` that actually occur in `series` (empty groups are never plotted)."""
    seen = set(series.dropna().unique())
    return [lvl for lvl in order if lvl in seen]


def chi_square_association(table: pd.DataFrame) -> dict:
    """Chi-square test of independence + Cramer's V, with an explicit suitability check.

    Suitability rule: expected counts should be >= 5 in at least 80% of cells and
    >= 1 everywhere. For 2x2 tables with small expected counts Fisher's exact test is added.
    The Yates correction is not applied (n is large); Cramer's V uses the uncorrected chi-square.
    """
    obs = table.values.astype(float)
    n = obs.sum()
    chi2, p, dof, expected = stats.chi2_contingency(obs, correction=False)
    k = min(obs.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else np.nan
    share_ok = float((expected >= 5).mean())
    suitable = bool(share_ok >= 0.8 and expected.min() >= 1)
    out = {"n": int(n), "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
           "cramers_v": cramers_v, "min_expected_count": float(expected.min()),
           "share_cells_expected_ge5": share_ok, "chi2_suitable": suitable,
           "fisher_p_value": np.nan, "odds_ratio": np.nan, "or_ci_low": np.nan, "or_ci_high": np.nan}
    if obs.shape == (2, 2):
        a, b, c, d = obs.ravel()
        if min(a, b, c, d) > 0:
            or_ = (a * d) / (b * c)
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            out.update(odds_ratio=float(or_), or_ci_low=float(np.exp(np.log(or_) - 1.96 * se)),
                       or_ci_high=float(np.exp(np.log(or_) + 1.96 * se)))
        if not suitable:
            out["fisher_p_value"] = float(stats.fisher_exact(obs)[1])
    return out


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = 2*AUC - 1 from the Mann-Whitney U statistic (ordinal effect size)."""
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset '{name}' is missing required columns: {missing}. "
                       f"Found columns: {list(df.columns)}")


def save_csv(df: pd.DataFrame, filename: str, folder: Path | None = None) -> Path:
    """Write a traceable CSV either to main_analysis/ or to a specified supplementary folder."""
    path = (folder if folder is not None else OUT_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ----------------------------------------------------------------------------
# 2. LOADING AND RAW-DATA AUDIT
# ----------------------------------------------------------------------------
def load_raw() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Read the four raw CSVs, check required columns, parse dates (conversion failures are counted)."""
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_DIR}\n"
            "Expected layout: <script folder>/Olist datasets/<csv files>.")
    raw: dict[str, pd.DataFrame] = {}
    conversion_log: list[dict] = []
    for name, (filename, cols) in REQUIRED_FILES.items():
        path = DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
        df = pd.read_csv(path)
        require_columns(df, cols, name)
        for col in DATE_COLUMNS[name]:
            before_missing = int(df[col].isna().sum())
            df[col] = pd.to_datetime(df[col], errors="coerce")
            failed = int(df[col].isna().sum()) - before_missing
            conversion_log.append({"dataset": name, "column": col, "values_failed_to_parse": failed})
        raw[name] = df
    return raw, conversion_log


def audit_raw(raw: dict[str, pd.DataFrame], conversion_log: list[dict]) -> pd.DataFrame:
    """Section-4 audit: sizes, identifiers, missingness, duplicates, statuses, timestamp consistency."""
    rows: list[dict] = []

    def add(dataset: str, category: str, metric: str, value, note: str = "") -> None:
        rows.append({"dataset": dataset, "category": category, "metric": metric,
                     "value": value, "note": note})

    orders, reviews, items, products = (raw[k] for k in ["orders", "reviews", "items", "products"])

    # Size and identifiers
    add("orders", "size", "rows", len(orders))
    add("orders", "identifiers", "unique order_id", orders["order_id"].nunique())
    add("orders", "identifiers", "unique customer_id", orders["customer_id"].nunique())
    add("reviews", "size", "rows", len(reviews))
    add("reviews", "identifiers", "unique review_id", reviews["review_id"].nunique())
    add("reviews", "identifiers", "unique order_id", reviews["order_id"].nunique())
    add("reviews", "identifiers", "unique (review_id, order_id) pairs",
        len(reviews.drop_duplicates(["review_id", "order_id"])))
    add("items", "size", "rows", len(items))
    add("items", "identifiers", "unique order_id", items["order_id"].nunique())
    add("items", "identifiers", "unique product_id", items["product_id"].nunique())
    add("items", "identifiers", "unique (order_id, order_item_id)",
        len(items.drop_duplicates(["order_id", "order_item_id"])))
    add("products", "size", "rows", len(products))
    add("products", "identifiers", "unique product_id", products["product_id"].nunique())

    # Missing values and exact duplicates (all columns, all datasets)
    for name, df in raw.items():
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                add(name, "missing values", f"missing: {col}", n_missing,
                    f"{100 * n_missing / len(df):.2f}% of rows")
        add(name, "duplicates", "exact duplicate rows", int(df.duplicated().sum()))
    add("reviews", "duplicates", "duplicate (review_id, order_id) pairs",
        int(reviews.duplicated(["review_id", "order_id"]).sum()))
    add("reviews", "duplicates", "review_id appearing under more than one order_id",
        int((reviews.groupby("review_id")["order_id"].nunique() > 1).sum()),
        "same review text/score linked to several orders (multi-order purchases)")
    add("reviews", "duplicates", "orders with more than one review record",
        int((reviews.groupby("order_id").size() > 1).sum()),
        "repeated reviews per order: statistical independence is assessed later")
    add("items", "duplicates", "duplicate (order_id, order_item_id)", int(items.duplicated(["order_id", "order_item_id"]).sum()))
    add("items", "structure", "orders with more than one item row",
        int((items.groupby("order_id").size() > 1).sum()),
        "joining items to reviews directly would duplicate reviews")

    # Referential integrity
    add("reviews", "integrity", "review rows whose order_id is not in orders",
        int((~reviews["order_id"].isin(orders["order_id"])).sum()))
    add("items", "integrity", "item rows whose product_id is not in products",
        int((~items["product_id"].isin(products["product_id"])).sum()))
    add("orders", "integrity", "orders without any item row",
        int((~orders["order_id"].isin(items["order_id"])).sum()))

    # Order status distribution
    for status, cnt in orders["order_status"].value_counts().items():
        add("orders", "order_status", f"status = {status}", int(cnt),
            f"{100 * cnt / len(orders):.2f}% of orders")

    # Timestamp conversion and format
    for rec in conversion_log:
        add(rec["dataset"], "timestamp parsing", f"unparseable values: {rec['column']}", rec["values_failed_to_parse"])
    est = orders["order_estimated_delivery_date"].dropna()
    add("orders", "timestamp format", "estimated delivery dates with a non-midnight time component",
        int((est != est.dt.normalize()).sum()),
        "0 means the estimate is a calendar-day deadline (date only)")
    deliv = orders["order_delivered_customer_date"].dropna()
    add("orders", "timestamp format", "actual delivery timestamps with a non-midnight time component",
        int((deliv != deliv.dt.normalize()).sum()), "actual delivery carries a time of day")
    created = reviews["review_creation_date"].dropna()
    add("reviews", "timestamp format", "review_creation_date with a non-midnight time component",
        int((created != created.dt.normalize()).sum()))

    # Order-level timestamp / status consistency
    delivered_status = orders["order_status"] == "delivered"
    has_delivery_date = orders["order_delivered_customer_date"].notna()
    add("orders", "inconsistency", "status delivered but delivery timestamp missing",
        int((delivered_status & ~has_delivery_date).sum()))
    add("orders", "inconsistency", "status not delivered but delivery timestamp present",
        int((~delivered_status & has_delivery_date).sum()))
    add("orders", "inconsistency", "delivery timestamp earlier than purchase timestamp",
        int((orders["order_delivered_customer_date"] < orders["order_purchase_timestamp"]).sum()))
    add("orders", "inconsistency", "delivery to customer earlier than hand-over to carrier",
        int((orders["order_delivered_customer_date"] < orders["order_delivered_carrier_date"]).sum()))
    add("orders", "inconsistency", "estimated delivery earlier than purchase",
        int((orders["order_estimated_delivery_date"] < orders["order_purchase_timestamp"].dt.normalize()).sum()))
    valid_ts = (delivered_status & has_delivery_date & orders["order_estimated_delivery_date"].notna())
    add("orders", "eligibility", "delivered orders with valid actual AND estimated delivery timestamps",
        int(valid_ts.sum()))

    # Review-level consistency
    add("reviews", "inconsistency", "review_answer_timestamp earlier than review_creation_date",
        int((reviews["review_answer_timestamp"] < reviews["review_creation_date"]).sum()))
    merged = reviews.merge(orders[["order_id", "order_purchase_timestamp", "order_delivered_customer_date"]],
                           on="order_id", how="left", validate="many_to_one")
    add("reviews", "inconsistency", "review_answer_timestamp earlier than order purchase",
        int((merged["review_answer_timestamp"] < merged["order_purchase_timestamp"]).sum()))
    add("reviews", "inconsistency", "review_score outside 1-5", int((~reviews["review_score"].between(1, 5)).sum()))
    msg = reviews["review_comment_message"].fillna("").astype(str).str.strip()
    ttl = reviews["review_comment_title"].fillna("").astype(str).str.strip()
    add("reviews", "text", "reviews with non-empty comment message", int((msg != "").sum()))
    add("reviews", "text", "reviews with title but no message", int(((msg == "") & (ttl != "")).sum()))
    for score, cnt in reviews["review_score"].value_counts().sort_index().items():
        add("reviews", "review_score", f"score = {score}", int(cnt), f"{100 * cnt / len(reviews):.2f}% of review records")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. SAMPLE CONSTRUCTION AND ANALYTICAL VARIABLES
# ----------------------------------------------------------------------------
def aggregate_order_covariates(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item/product data to ONE ROW PER ORDER before any merge with reviews."""
    prod = products[["product_id", "product_weight_g"]].copy()
    if prod["product_id"].duplicated().any():
        raise ValueError("products contains duplicate product_id values; cannot merge safely.")
    it = items.merge(prod, on="product_id", how="left", validate="many_to_one")
    it["weight_missing"] = it["product_weight_g"].isna().astype(int)
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        n_items_missing_weight=("weight_missing", "sum"),
    ).reset_index()
    # weight is only trusted when every item in the order has a weight
    agg.loc[agg["n_items_missing_weight"] > 0, "total_weight_g"] = np.nan
    return agg.drop(columns="n_items_missing_weight")


def severity_labels(days_late: pd.Series, fine: bool) -> pd.Categorical:
    """Delay-severity category from calendar days late (<=0 is on time)."""
    if fine:
        bins = [-np.inf, 0, 3, 7, 14, 30, np.inf]
        labels = SEVERITY_FINE_LEVELS
    else:
        bins = [-np.inf, 0, 3, 7, np.inf]
        labels = SEVERITY4_LEVELS
    return pd.Categorical(pd.cut(days_late, bins=bins, labels=labels), categories=labels, ordered=True)


def construct_samples(raw: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply eligibility rules, log every step, and build the review-level analytical table.

    Analytical unit: one REVIEW RECORD (review_id, order_id) linked to one eligible delivered order.
    Repeated reviews per order are retained and handled by cluster-robust errors and an
    order-level sensitivity sample.
    """
    orders, reviews = raw["orders"].copy(), raw["reviews"].copy()
    steps: list[dict] = []

    def log_step(population, step, description, n_orders=None, n_reviews=None, n_review_orders=None):
        prev = [s for s in steps if s["population"] == population]
        key = "n_orders" if population == "Orders" else "n_review_records"
        value = n_orders if population == "Orders" else n_reviews
        removed = (prev[-1][key] - value) if prev else 0
        steps.append({"population": population, "step": step, "description": description,
                      "n_orders": n_orders, "n_review_records": n_reviews,
                      "n_unique_orders_with_review": n_review_orders, "removed_vs_previous_step": removed})

    # --- orders ---
    log_step("Orders", "O1", "All orders in olist_orders_dataset", n_orders=len(orders))
    o = orders[orders["order_status"] == "delivered"]
    log_step("Orders", "O2", "order_status == 'delivered'", n_orders=len(o))
    o = o.dropna(subset=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    log_step("Orders", "O3", "...and purchase, actual-delivery and estimated-delivery timestamps all present", n_orders=len(o))
    o = o[o["order_delivered_customer_date"] >= o["order_purchase_timestamp"]]
    log_step("Orders", "O4", "...and delivery not earlier than purchase (chronology check)", n_orders=len(o))
    eligible = o.copy()

    # Delay variables. The estimate is a date-only deadline (verified in the audit), so the actual
    # delivery is reduced to its calendar day before comparing. Primary convention (calendar-day):
    # late <=> delivery date > estimated date. Same-day deliveries are on time.
    eligible["delivered_date"] = eligible["order_delivered_customer_date"].dt.normalize()
    eligible["estimated_date"] = eligible["order_estimated_delivery_date"].dt.normalize()
    eligible["days_vs_estimate"] = (eligible["delivered_date"] - eligible["estimated_date"]).dt.days
    eligible["delayed"] = eligible["days_vs_estimate"] > 0
    eligible["delayed_strict_timestamp"] = eligible["order_delivered_customer_date"] > eligible["order_estimated_delivery_date"]
    eligible["delivery_status"] = pd.Categorical(np.where(eligible["delayed"], "Delayed", "On time"),
                                                 categories=DELIVERY_LEVELS, ordered=True)
    eligible["severity4"] = severity_labels(eligible["days_vs_estimate"], fine=False)
    eligible["severity_fine"] = severity_labels(eligible["days_vs_estimate"], fine=True)
    eligible["promised_days"] = (eligible["estimated_date"] - eligible["order_purchase_timestamp"].dt.normalize()).dt.days
    eligible["purchased_2018"] = (eligible["order_purchase_timestamp"].dt.year == 2018).astype(int)

    # Order-level covariates (items/products aggregated BEFORE the merge with reviews)
    covariates = aggregate_order_covariates(raw["items"], raw["products"])
    eligible = eligible.merge(covariates, on="order_id", how="left", validate="one_to_one")

    # --- reviews ---
    n_all = len(reviews)
    log_step("Reviews", "R1", "All review records in olist_order_reviews_dataset", n_reviews=n_all,
             n_review_orders=reviews["order_id"].nunique())
    rv = reviews.drop_duplicates()
    log_step("Reviews", "R2", "After removing exact duplicate rows", n_reviews=len(rv), n_review_orders=rv["order_id"].nunique())
    rv = rv[rv["order_id"].isin(orders["order_id"])]
    log_step("Reviews", "R3", "...linked to an order present in the orders table", n_reviews=len(rv), n_review_orders=rv["order_id"].nunique())
    rv = rv[rv["order_id"].isin(eligible["order_id"])]
    log_step("Reviews", "R4", "...linked to an ELIGIBLE delivered order (steps O1-O4)", n_reviews=len(rv), n_review_orders=rv["order_id"].nunique())
    rv = rv[rv["review_score"].between(1, 5)]
    log_step("Reviews", "R5", "...with a valid 1-5 review_score (all-scored-reviews sample)", n_reviews=len(rv), n_review_orders=rv["order_id"].nunique())
    n_unknown = int(rv["review_answer_timestamp"].isna().sum())
    log_step("Reviews", "R5a", "   of which review_answer_timestamp missing (kept as timing 'Unknown'; not dropped)",
             n_reviews=n_unknown, n_review_orders=rv.loc[rv["review_answer_timestamp"].isna(), "order_id"].nunique())
    msg = rv["review_comment_message"].fillna("").astype(str).str.strip()
    ttl = rv["review_comment_title"].fillna("").astype(str).str.strip()
    log_step("Reviews", "R5b", "   of which title only, no message (excluded from written sample)",
             n_reviews=int(((msg == "") & (ttl != "")).sum()), n_review_orders=rv.loc[(msg == "") & (ttl != ""), "order_id"].nunique())
    written_mask = msg != ""
    log_step("Reviews", "R6", "...with a non-empty written comment message (written-review sample)",
             n_reviews=int(written_mask.sum()), n_review_orders=rv.loc[written_mask, "order_id"].nunique())
    # R5a/R5b are informational sub-counts; do not chain them in the 'removed' column
    for s in steps:
        if s["step"] in ("R5a", "R5b", "R5c"):
            s["removed_vs_previous_step"] = np.nan
        if s["step"] == "R6":
            s["removed_vs_previous_step"] = int(len(rv) - written_mask.sum())

    rv = rv.copy()
    rv["has_text"] = written_mask.values
    rv["text"] = np.where(written_mask, np.where(ttl != "", ttl + " . " + msg, msg), "")

    # Merge reviews to eligible orders (many reviews -> one order; validated)
    keep_order_cols = ["order_id", "order_purchase_timestamp", "order_delivered_customer_date", "estimated_date",
                       "delivered_date", "days_vs_estimate", "delayed", "delayed_strict_timestamp",
                       "delivery_status", "severity4", "severity_fine", "promised_days", "purchased_2018",
                       "n_items", "n_sellers", "total_price", "total_freight", "total_weight_g"]
    df = rv.merge(eligible[keep_order_cols], on="order_id", how="left", validate="many_to_one")

    # Review timing from review_answer_timestamp vs actual delivery TIMESTAMP.
    # A tie (answer == delivery to the second) is classified as 'On/after delivery'.
    invalid_answer = (df["review_answer_timestamp"].isna()
                      | (df["review_answer_timestamp"] < df["order_purchase_timestamp"])
                      | (df["review_answer_timestamp"] < df["review_creation_date"]))
    timing = np.where(invalid_answer, "Unknown",
                      np.where(df["review_answer_timestamp"] < df["order_delivered_customer_date"],
                               "Before delivery", "On/after delivery"))
    df["timing"] = pd.Categorical(timing, categories=TIMING_LEVELS)
    df["group4"] = pd.Categorical(df["delivery_status"].astype(str) + " | " + df["timing"].astype(str),
                                  categories=GROUP4_LEVELS + [f"{d} | Unknown" for d in DELIVERY_LEVELS])
    df["negative"] = (df["review_score"] <= NEG_MAX_SCORE).astype(int)
    df["hours_answer_vs_delivery"] = (df["review_answer_timestamp"] - df["order_delivered_customer_date"]).dt.total_seconds() / 3600
    df["survey_created_before_delivery_day"] = df["review_creation_date"] < df["delivered_date"]
    df["survey_created_vs_estimate_days"] = (df["review_creation_date"] - df["estimated_date"]).dt.days
    df["n_reviews_in_order"] = df.groupby("order_id")["review_id"].transform("size")
    df["word_count"] = df["text"].str.split().str.len().fillna(0).astype(int)

    n_anomalous = int((df["review_answer_timestamp"] < df["order_purchase_timestamp"]).sum())
    r6_pos = next(i for i, st in enumerate(steps) if st["step"] == "R6")
    steps.insert(r6_pos, {"population": "Reviews", "step": "R5c",
                  "description": "   of which review_answer_timestamp earlier than order purchase (anomalous; retained, timing classified Unknown)",
                  "n_orders": None, "n_review_records": n_anomalous,
                  "n_unique_orders_with_review": df.loc[df["review_answer_timestamp"] < df["order_purchase_timestamp"], "order_id"].nunique(),
                  "removed_vs_previous_step": np.nan})
    sample_log = pd.DataFrame(steps)
    return eligible, df.reset_index(drop=True), sample_log


def one_review_per_order(df: pd.DataFrame) -> pd.DataFrame:
    """Order-level sensitivity sample: keep the most recently answered review of each order."""
    return (df.sort_values(["order_id", "review_answer_timestamp", "review_id"])
              .drop_duplicates("order_id", keep="last").reset_index(drop=True))


# ----------------------------------------------------------------------------
# 4. PORTUGUESE TEXT CLASSIFICATION (transparent, rule-based, no manual validation)
# ----------------------------------------------------------------------------
# Design principles
#  * Text is lower-cased, accent-stripped and punctuation-normalised ("nao" for "nao/n/nn/naum").
#  * Clauses are separated by punctuation and by contrast words (mas, porem, ...), so negation
#    does not leak across clauses ("nao gostei, atrasou" -> 'atrasou' is not negated).
#  * Rules are PROBLEM phrases ("nao recebi", "atrasou", "veio quebrado"), not topic words.
#    Topic-only mentions ("entrega", "prazo", "frete") are tracked separately as mentions.
#  * Positive-form problem words are skipped when negated within the 3 preceding tokens of the
#    clause ("sem atraso", "nao demorou", "nao veio quebrado").
#  * A review can carry several categories (multi-label).
NEGATORS = {"nao", "nunca", "jamais", "nem", "sem", "nenhum", "nenhuma", "ninguem", "nada"}
NEGATION_STOP = {"porque", "pois", "que", "e", "ja", "quando", "se", "como"}
HYPOTHETICAL = {"se", "caso", "quando", "qualquer", "necessario", "necessidade", "precisar", "podera", "possa", "for"}
CLAUSE_PUNCT = re.compile(r"[.,;:!?]")

W1, W2, W3 = r"(?:\w+ )?", r"(?:\w+ ){0,2}", r"(?:\w+ ){0,3}"
CARRIER = (r"(?:transportadora|transportador|correios|correio|sedex|jadlog|loggi|fedex|total express|"
           r"mercado envios|entregador|motoboy|carteiro|courier)")
TRACK = r"(?:rastreio|rastreamento|rastrear|rastreavel|codigo de rastreio|tracking)"
NOT_PAYLOAD = (r"(?! (?:o |a |os |as |um |uma )?(?:nenhum[a]? |qualquer )?(?:resposta|retorno|contato|email|e mail|"
               r"mensagem|nota|nf|nfe|codigo|rastreio|rastreamento|aviso|notificacao|ligacao|reembolso|estorno|"
               r"dinheiro|valor|satisfacao|explicacao|justificativa|atualizacao|comprovante|boleto|fatura|"
               r"completo|inteiro|todos|todas|tudo)\b)")
NOT_ARRIVAL_QUALIFIER = (r"(?! (?:no|na|dentro|ate|a tempo|completo|inteir\w+|correto|certo|conforme|corretamente|"
                         r"direito|certinho|com|nenhum)\b)")
PK_BAD = (r"(?:ruim|pessima|danificad\w+|amassad\w+|rasgad\w+|aberta|aberto|violad\w+|fraca|inadequad\w+|precaria|"
          r"estragad\w+|furad\w+|molhad\w+|suja|sujo|horrivel|quebrad\w+|deficiente|insuficiente|vazia|destruid\w+|"
          r"esmagad\w+|rompid\w+|mal feita|mal feito)")
DAMAGE = (r"(?:defeito\w*|defeituos[oa]s?|quebrad[oa]s?|quebrou|quebraram|trincad[oa]s?|trincou|rachad[oa]s?|"
          r"rachou|riscad[oa]s?|arranhad[oa]s?|manchad[oa]s?|estragad[oa]s?|estragou|danificad[oa]s?|avariad[oa]s?|"
          r"avaria|rasgad[oa]s?|rasgou|mofo|mofad[oa]|enferrujad[oa]|descascou|descascando|vazando|vazou|queimou|"
          r"queimad[oa]|pifou|falsific\w+|pirata|despregou|descolou|descolad[oa]|desbotou|desbotad[oa])")
NOT_AS_DESCRIBED = r"(?! d[oa]s? (?:anunc\w+|fotos?|descri\w+|propaganda|site|imagens?|mostrad\w+|exibid\w+))"
SUPPORT_OBJECT = (r"(?:retorno|resposta|satisfacao|atencao|suporte|atendimento|solucao|contato|explicacao|"
                  r"justificativa|posicionamento|esclarecimento)")

CATEGORIES = ["delay", "non_receipt", "tracking_carrier", "packaging", "product_condition",
              "wrong_missing", "customer_support", "return_refund"]
CATEGORY_LABELS = {
    "delay": "Delivery delay language",
    "non_receipt": "Non-receipt / lost in transit language",
    "delivery_delay_nonreceipt": "Delivery delay / non-receipt (delay OR non-receipt)",
    "tracking_carrier": "Tracking / carrier problems",
    "packaging": "Packaging problems",
    "product_condition": "Product condition / quality / not as described",
    "wrong_missing": "Wrong / missing / incomplete items",
    "customer_support": "Customer support problems",
    "return_refund": "Returns / refunds / exchange / cancellation language",
}


@dataclass(frozen=True)
class Rule:
    category: str
    rule_id: str
    pattern: re.Pattern
    negation: str = "none"      # 'prefix': skip when a negator precedes the match in the same clause
                                # 'inner' : skip when a negator occurs inside the matched span
                                # 'both'  : either of the above
    hypothetical: bool = False  # skip when preceded by a conditional word ("caso precise trocar")


def _r(category: str, rule_id: str, regex: str, negation: str = "none", hypothetical: bool = False) -> Rule:
    return Rule(category, rule_id, re.compile(regex), negation, hypothetical)


RULES: list[Rule] = [
    # ---- delivery delay ----
    _r("delay", "atraso", r"\batras(?:o|os|ou|ado|ada|ados|adas|aram|ar|ei|aria|ando|am)\b", "prefix"),
    _r("delay", "demora", r"\bdemor(?:a|as|ou|aram|ar|ado|ada|ados|adas|ando|aria|ei|ava|avam|am)\b", "prefix"),
    _r("delay", "fora_do_prazo", r"\b(?:fora d[oa]|alem d[oa]|apos o|depois d[oa]|passou d[oa]|passaram d[oa]|estourou o|"
       r"estouraram o|ultrapassou o|ultrapassaram o|extrapolou o|descumpriu o|desrespeitou o) prazo\b", "prefix"),
    _r("delay", "prazo_estourado", r"\bprazo (?:estourado|vencido|expirado|ultrapassado|estourou|venceu|expirou)\b", "prefix"),
    _r("delay", "nao_cumpriu_prazo", r"\bnao (?:cumpr\w+|respeit\w+|entreg\w+|chegou|chegaram|veio|vieram|recebi|recebemos|"
       r"foi entregue) " + W3 + r"prazo\b"),
    _r("delay", "chegou_tarde", r"\b(?:chegou|chegaram|entregue|entregues|entregaram|entregou|veio|vieram) "
       r"(?:muito |bem |um pouco |meio )?tarde\b"),
    _r("delay", "tempo_excessivo", r"\bmuito tempo (?:para|pra|pro|ate|a) (?:chegar|entregar|receber|enviar|postar|despachar|chegada)\b"),
    _r("delay", "espera_longa", r"\b(?:esper\w+|aguard\w+|levou|leva) " + W3 + r"(?:(?:[1-9]\d|\d{3}) dias|semanas|mes|meses)\b", "both"),
    _r("delay", "prazo_longo", r"\bprazo (?:de entrega )?" + W1 + r"(?:longo|grande|absurdo|exagerado|excessivo)\b", "prefix"),
    _r("delay", "entrega_lenta", r"\b(?:entrega " + W1 + r"(?:lenta|tardia)|lent[ao] (?:na|para|pra) (?:entrega|chegar|chegada))\b", "prefix"),
    # ---- non-receipt / lost in transit ----
    _r("non_receipt", "nao_recebi", r"\b(?:nao|nunca) (?:foi |fui |fora )?(?:recebi|recebemos|recebido|recebida|recebeu)\b" + NOT_PAYLOAD),
    _r("non_receipt", "nao_chegou", r"\b(?:nao|nunca) (?:chegou|chegaram|chegam|apareceu|apareceram|foi entregue|foram entregues|"
       r"foi entregado|foram entregados|entregaram|entregou|entregue|entregues|foi enviado|foi postado|postaram|postou|"
       r"enviaram|enviou|mandaram|mandou)\b" + NOT_ARRIVAL_QUALIFIER),
    _r("non_receipt", "nao_veio", r"\b(?:nao|nunca) (?:veio|vieram)\b(?! (?:no|na|dentro|ate|a tempo|completo|inteir\w+|o|a|os|as|"
       r"um|uma|com|nem|nenhum|nenhuma|todos|todas|manual|cabo|carregador|embalad\w+|certo|certinho|conforme|igual|como)\b)"),
    _r("non_receipt", "ate_agora_nada", r"\bate (?:agora|hoje|o momento|a presente data) (?:ainda )?(?:nada\b|nao (?:\w+ )?"
       r"(?:recebi|recebemos|chegou|chegaram|veio|vieram|foi entregue|foram entregues|entregaram|entregou|apareceu)\b)"),
    _r("non_receipt", "aguardando", r"\b(?:estou|estamos|continuo|continuamos|fico|ficamos) " + W1 + r"(?:aguardando|esperando)\b"
       r"(?! (?:a |o |um |uma |por |pela |pelo )?(?:resposta|retorno|contato|devolutiva|reembolso|estorno|troca|solucao|"
       r"posicionamento|ligacao|email|devolucao|ressarcimento|dinheiro|autorizacao|coleta|ser ressarcid\w+))"),
    _r("non_receipt", "extravio", r"\bextravi\w+\b", "prefix"),
    _r("non_receipt", "pedido_perdido", r"\b(?:(?:foi|ficou|esta) perdid[oa]s?|(?:pedido|produto|pacote|encomenda|mercadoria) "
       + W1 + r"(?:perdeu|sumiu))\b"),
    _r("non_receipt", "sem_previsao", r"\b(?:sem previsao|sem noticias?|nenhuma noticia|nao (?:ha|tem|tenho|tinha|existe) previsao)\b"),
    _r("non_receipt", "nao_houve_entrega", r"\bnao (?:houve|tive|tivemos|ocorreu|fizeram|fez|realizaram|realizou|efetuaram|efetuou) "
       r"(?:a |uma |nenhuma )?(?:tentativa de )?entrega\b"),
    _r("non_receipt", "sem_entrega", r"\bsem (?:a )?entrega\b"),
    # ---- tracking / carrier ----
    _r("tracking_carrier", "rastreio_falho", r"\b" + TRACK + " " + W3 + r"(?:nao (?:atualiz\w+|funcion\w+|mostr\w+|aparec\w+|"
       r"consta\w*|exist\w+|encontr\w+|rastre\w+|mud\w+)|parado|travado|errado|incorreto|falho|indisponivel|bugado|"
       r"invalido|inexistente|sem (?:atualizacao|movimentacao|informacao|informacoes))\b"),
    _r("tracking_carrier", "sem_rastreio", r"\b(?:nao|nunca|sem) (?:(?!problema|dificuldade|erro|falha|atraso)\w+ ){0,2}"
       r"(?:" + TRACK + r"|atualizacao de entrega|atualizacao do pedido|informacao de entrega|informacoes de entrega|aviso de entrega)\b"),
    _r("tracking_carrier", "problema_com_transportadora", r"\b(?:problema\w*|erro|falh\w+|atras\w+|demor\w+|descaso|culpa|extravi\w+|"
       r"greve|negligen\w+|incompeten\w+) " + W3 + r"(?:" + CARRIER + "|" + TRACK + r")\b", "both"),
    _r("tracking_carrier", "transportadora_falha", r"\b" + CARRIER + " " + W3 + r"(?:extravi\w+|perdeu|perderam|devolveu|"
       r"devolveram|errou|erraram|atras\w+|demor\w+|abandon\w+|retornou|greve|culpa|problema\w*|falh\w+|negligen\w+|descaso|"
       r"incompeten\w+|pessim\w+|horrivel|ruim|deplora\w+|lament\w+|terrivel|vergonh\w+|mal educad\w+|grosseir\w+|rude|"
       r"deixou|deixaram|largou|largaram|jogou|jogaram|furt\w+|roub\w+)\b", "inner"),
    _r("tracking_carrier", "transportadora_nao_fez", r"\b" + CARRIER + " " + W2 + r"nao (?:entreg\w+|avis\w+|tent\w+|encontr\w+|"
       r"cumpr\w+|deix\w+|respeit\w+|localiz\w+|ligou|ligaram|atend\w+|quis|quiseram)\b"),
    _r("tracking_carrier", "endereco_errado", r"\b(?:entreg\w+|deixou|deixaram|largou|largaram) " + W3 + r"(?:endereco|local|casa|lugar|"
       r"predio|apartamento|vizinho|porteiro|portaria) " + W1 + r"(?:errado|errada|diferente|incorreto)\b"),
    _r("tracking_carrier", "entrega_outra_pessoa", r"\bentreg\w+ (?:no|em|para|pra|a) (?:outro|outra|um outro) (?:endereco|local|lugar|casa|pessoa|vizinho)\b"),
    _r("tracking_carrier", "tentativa_frustrada", r"\b(?:tentativas? de entrega " + W1 + r"(?:sem sucesso|frustrada\w*|falha\w*|nao realizada)|"
       r"(?:cliente|destinatario|ninguem) (?:estava )?ausente)\b"),
    # ---- packaging ----
    _r("packaging", "mal_embalado", r"\b(?:mal|pessimamente|porcamente) (?:embalad[oa]s?|embalagem|acondicionad[oa]s?|empacotad[oa]s?)\b", "prefix"),
    _r("packaging", "embalagem_ruim", r"\bembalagem " + W2 + PK_BAD + r"\b", "both"),
    _r("packaging", "caixa_danificada", r"\b(?:caixa|caixinha|pacote|envelope|sacola|saco|papelao|lacre) " + W2 + PK_BAD + r"\b", "both"),
    _r("packaging", "sem_embalagem", r"\b(?:sem (?:a |nenhuma )?(?:embalagem|caixa|lacre|protecao|plastico bolha)|veio solto|vieram soltos?|"
       r"nao (?:veio|estava|vinha) " + W1 + r"embalad\w+|sem embalar)\b"),
    # ---- product condition / quality / not as described ----
    _r("product_condition", "dano_defeito", r"\b" + DAMAGE + r"\b", "prefix"),
    _r("product_condition", "nao_funciona", r"\bnao (?:funciona\w*|funcionou|liga|ligou|ligava|carrega\w*|carregou|presta\w*|prestou|"
       r"acende\w*|acendeu|conecta\w*|conectou|encaixa\w*|encaixou|abre|abriu|esquenta|esquentou|sincroniza\w*|"
       r"reconhece\w*|reconheceu|imprime|imprimiu)\b"),
    _r("product_condition", "parou_de_funcionar", r"\b(?:parou|pararam|deixou|deixaram|comecou a dar|comecaram a dar) (?:de |a )?"
       r"(?:funcionar|carregar|ligar|ler|tocar|dar defeito|dar problema|falhar)\b"),
    _r("product_condition", "qualidade_ruim", r"\b(?:qualidade " + W2 + r"(?:ruim|pessima|horrivel|baixa|inferior|duvidosa|fraca|precaria|"
       r"deplora\w+|pobre|lamentavel)|(?:ruim|pessima|horrivel|baixa|pouca|fraca|pessimo|baixissima) qualidade|material "
       + W1 + r"(?:ruim|fraco|pessimo|barato|inferior|vagabundo)|(?:sem|falta de|nao (?:tem|possui)) qualidade|"
       r"produto " + W1 + r"(?:ruim|pessimo|horrivel|fraco|inferior|vagabundo|lixo))\b", "both"),
    _r("product_condition", "diferente_do_anuncio", r"\b(?:diferente d[oa]s? " + W1 + r"(?:anunc\w+|fotos?|descri\w+|propaganda|site|imagens?|"
       r"mostrad\w+|exibid\w+|especifica\w+)|nao (?:e|eh|era|corresponde|condiz|bate|esta|estava) " + W2 +
       r"(?:anunc\w+|fotos?|descri\w+|propaganda|site|imagens?|mostrad\w+|exibid\w+|original|especifica\w+)|"
       r"propaganda enganosa|anuncio enganoso|enganos[oa])\b"),
    _r("product_condition", "produto_amassado", r"\b(?:produto|item|aparelho|peca|pecas) " + W1 + r"(?:amassad[oa]s?|empenad[oa]s?)\b", "prefix"),
    _r("product_condition", "produto_usado", r"\b(?:produto|item|aparelho|veio|chegou|parece|aparenta|vieram|parecia) " + W1 + r"usad[oa]s?\b", "prefix"),
    # ---- wrong / missing / incomplete items ----
    _r("wrong_missing", "item_errado", r"\b(?:produto|item|itens|modelo|cor|tamanho|numero|voltagem|sabor|quantidade|pedido|material|"
       r"marca|medida|versao|capa|peca|pecas|cheiro|estampa)s? " + W2 + r"(?:errad[oa]s?|trocad[oa]s?|incorret[oa]s?|diferente)\b"
       + NOT_AS_DESCRIBED, "both"),
    _r("wrong_missing", "recebi_outro", r"\b(?:recebi|veio|vieram|chegou|chegaram|entregaram|enviaram|mandaram|mandou|enviou) " + W2 +
       r"(?:outro|outra|outros|outras|diferente|errad[oa]s?|trocad[oa]s?)\b" + NOT_AS_DESCRIBED),
    _r("wrong_missing", "errado_item", r"\b(?:errad[oa]s?|incorret[oa]s?) (?:produto|item|modelo|cor|tamanho|pedido|material|sabor|voltagem|numero)\b", "prefix"),
    _r("wrong_missing", "enviaram_errado", r"\b(?:enviaram|mandaram|entregaram|enviou|mandou|vieram|veio|chegou) errad[oa]s?\b"),
    _r("wrong_missing", "faltou", r"\b(?:faltou|faltaram|faltando|incomplet[oa]s?)\b", "prefix"),
    _r("wrong_missing", "falta_item", r"\bfalta(?:m)? (?:o|a|os|as|um|uma|1|2|3|peca|pecas|item|itens|parte|acessorios?|cabo|carregador|manual)\b", "prefix"),
    _r("wrong_missing", "veio_sem", r"\b(?:veio|vieram|chegou|chegaram) sem\b(?! (?:nenhum |nenhuma )?(?:defeito\w*|problema\w*|arranh\w+|"
       r"avaria\w*|amassad\w+|marca\w*|dano\w*|embalagem|caixa|lacre|protecao|nota|riscos?|mancha\w*|trinca\w*|"
       r"fissura\w*|atraso\w*)\b)"),
    _r("wrong_missing", "nao_veio_item", r"\bnao (?:veio|vieram) (?:nem )?(?:o|a|os|as|um|uma|com|nenhum|nenhuma|todos|todas|manual|"
       r"cabo|carregador|pecas?|itens?|acessorios?|brinde)\b"),
    _r("wrong_missing", "recebi_parcial", r"\b(?:(?:so|soh|apenas|somente) (?:chegou|veio|vieram|chegaram|recebi|recebemos)|"
       r"(?:recebi|recebemos|veio|vieram|chegou|chegaram) (?:apenas|somente|so|soh) (?:\d+|um|uma|dois|duas|metade|parte)|"
       r"nao (?:recebi|recebemos) (?:o pedido |o produto )?(?:completo|inteiro|todos|todas|tudo))\b"),
    # ---- customer support ----
    _r("customer_support", "nao_responde", r"\b(?:(?:nao|nem|ninguem) (?:me |nos )?(?:responde\w*|respondeu|responderam|respondem|retorna\w*|"
       r"retornou|retornaram|retornam|atenderam|atendeu o telefone|esclareceram|resolveram|solucionaram)|"
       r"ninguem (?:me |nos )?(?:resolve\w*|resolveu|atende\w*|atendeu)|nao fui atendid\w+|nao (?:me )?atendem)\b"),
    _r("customer_support", "nao_tive_retorno", r"\b(?:nao|nunca) " + W1 + r"(?:tive|obtive|houve|ha|tem|tenho|recebi|recebemos|deu|"
       r"deram|dao|da|dei|teve|tivemos|consegui|conseguimos) " + W1 + SUPPORT_OBJECT + r"\b"),
    _r("customer_support", "sem_resposta", r"\b(?:sem|nenhum[a]?) (?:nenhum[a]? )?(?:retorno|resposta|suporte|atendimento|satisfacao|"
       r"solucao|explicacao|esclarecimento|posicionamento)\b"),
    _r("customer_support", "falta_de", r"\bfalta de (?:respeito|retorno|atencao|comunicacao|atendimento|suporte|compromisso|"
       r"profissionalismo|educacao|consideracao|interesse|resposta|cuidado|organizacao)\b"),
    _r("customer_support", "descaso", r"\b(?:descaso|desrespeit\w+|enrol\w+|ignor\w+|mal atendid\w+|no vacuo|"
       r"empurrando com a barriga|desinteresse|desprezo)\b", "prefix"),
    _r("customer_support", "atendimento_ruim", r"\b(?:(?:pessim\w+|horrivel|ruim|precario|deplora\w+|lamentavel|terrivel|vergonhos\w+|mau|"
       r"mal|fraco|pouco) (?:servico de |o )?(?:atendimento|suporte|sac)|(?:atendimento|suporte|sac|vendedor|vendedora|lojista) "
       + W1 + r"(?:pessim\w+|horrivel|horriveis|ruim|ruins|precario|precaria|deplora\w+|lamentavel|terrivel|vergonhos\w+|"
       r"lixo|fraco|desrespeit\w+|despreparad\w+|incompeten\w+|grosseir\w+|mal educad\w+|deixa a desejar|deixou a desejar))\b", "prefix"),
    _r("customer_support", "tentei_contato", r"\b(?:tentei|tentamos|tentou) " + W1 + r"(?:contato|contatar|entrar em contato|falar|"
       r"ligar|mandar mensagem|enviar mensagem|ligamos)\b"),
    _r("customer_support", "reclame_aqui", r"\b(?:reclame aqui|reclameaqui|procon|ouvidoria|vou reclamar|reclamei|reclamacao formal)\b", "prefix"),
    # ---- returns / refunds / exchange / cancellation (request, dispute or event language) ----
    _r("return_refund", "devolucao_troca", r"\b(?:devolu\w+|devolv\w+|reembols\w+|estorn\w+|cancel\w+|troca|trocas|trocar|trocaram|trocou|troquei|troque|trocamos|ressarc\w+|"
       r"dinheiro de volta|meu dinheiro|arrepend\w+|desist\w+)\b", "prefix", hypothetical=True),
]

DELIVERY_MENTION = re.compile(r"\b(?:entreg\w+|prazo|chegou|chegaram|chegada|receb\w+|frete|envio|enviad\w+|postad\w+|logistic\w+|dias uteis)\b")


def normalize_text(raw: str) -> str:
    """Lower-case, strip accents, keep clause punctuation, standardise 'nao', squeeze letter runs."""
    s = unicodedata.normalize("NFKD", str(raw).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\r\n]+", " . ", s)
    s = re.sub(r"\b(?:mas|porem|entretanto|contudo|todavia)\b", " . ", s)   # contrast words close a clause
    s = re.sub(r"[^a-z0-9.,;:!?\s]", " ", s)
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)                                  # "atrasouuu" -> "atrasou"
    s = re.sub(r"\batraz(?=\w)", "atras", s)                                # common misspelling
    s = re.sub(r"\b(?:n|nn|naum|nao)\b", "nao", s)
    return re.sub(r"\s+", " ", s).strip()


def _preceding_tokens(text: str, start: int) -> list[str]:
    return CLAUSE_PUNCT.split(text[:start])[-1].split()


def _is_negated(text: str, start: int, window: int = 3) -> bool:
    for tok in reversed(_preceding_tokens(text, start)[-window:]):
        if tok in NEGATION_STOP:
            return False
        if tok in NEGATORS:
            return True
    return False


def detect_categories(norm: str) -> dict[str, list[str]]:
    """Return {category: [rule_ids that fired]} for one normalised review."""
    hits: dict[str, list[str]] = {}
    for rule in RULES:
        for m in rule.pattern.finditer(norm):
            if rule.negation in ("prefix", "both") and _is_negated(norm, m.start()):
                continue
            if rule.negation in ("inner", "both") and any(tok in NEGATORS for tok in m.group(0).split()[1:-1]):
                continue
            if rule.hypothetical and any(t in HYPOTHETICAL for t in _preceding_tokens(norm, m.start())[-4:]):
                continue
            hits.setdefault(rule.category, []).append(rule.rule_id)
            break  # one hit per rule is enough
    # "demora" is ambiguous: a product/app can be slow to respond, not the carrier.
    # Retain this phrase only if the same clause also mentions a logistics cue.
    if "demora" in hits.get("delay", []):
        delivery_context = re.compile(
            r"\b(entreg\w*|pedido|encomend\w*|pacote|produto.*(?:cheg|receb|entreg)|"
            r"transportador\w*|correios|frete|envio|enviad\w*|prazo|"
            r"receb\w*|cheg\w*|despach\w*|postag\w*)\b")
        delay_matches = list(re.finditer(r"\bdemor\w*\b", norm))
        has_logistic_context = False
        for m in delay_matches:
            clause = norm[max(0, m.start()-45):min(len(norm), m.end()+65)]
            if delivery_context.search(clause):
                has_logistic_context = True
                break
        if not has_logistic_context:
            hits["delay"] = [r for r in hits["delay"] if r != "demora"]
            if not hits["delay"]:
                del hits["delay"]
    return hits


# Unit tests on constructed Portuguese examples.
# These tests do not measure performance on the Olist reviews.
NLP_UNIT_TESTS = [
    ("Produto chegou antes do prazo, sem atraso nenhum", set(), {"delay"}),
    ("O produto atrasou muito", {"delay"}, set()),
    ("não atrasou nada, recomendo", set(), {"delay"}),
    ("Demorou um pouco mas chegou", {"delay"}, set()),
    ("O tour do celular e lento. Demora ao responder", set(), {"delay"}),
    ("Entrega fora do prazo", {"delay"}, set()),
    ("Nunca chegou o produto", {"non_receipt"}, set()),
    ("Ainda não recebi meu pedido", {"non_receipt"}, set()),
    ("Não recebi resposta do vendedor", {"customer_support"}, {"non_receipt"}),
    ("Estou aguardando até hoje", {"non_receipt"}, set()),
    ("Entrega rápida, tudo certo, recomendo", set(), set(CATEGORIES)),
    ("O produto veio quebrado", {"product_condition"}, set()),
    ("Não veio quebrado, tudo certo", set(), {"product_condition"}),
    ("Veio sem defeito nenhum", set(), {"product_condition", "wrong_missing"}),
    ("Produto não funciona", {"product_condition"}, set()),
    ("Embalagem mal feita e amassada", {"packaging"}, set()),
    ("Bem embalado, chegou rápido", set(), {"packaging", "delay"}),
    ("Recebi o produto errado", {"wrong_missing"}, set()),
    ("Faltou o carregador", {"wrong_missing"}, set()),
    ("Não tive problema com os correios", set(), {"tracking_carrier"}),
    ("Os correios atrasaram a entrega", {"tracking_carrier", "delay"}, set()),
    ("Sem código de rastreio", {"tracking_carrier"}, set()),
    ("Quero meu dinheiro de volta", {"return_refund"}, set()),
    ("Se precisar trocar eu aviso", set(), {"return_refund"}),
    ("Sem necessidade de troca", set(), {"return_refund"}),
    ("O vendedor não respondeu", {"customer_support"}, set()),
    ("Atendimento excelente", set(), {"customer_support"}),
    ("Não gostei porque atrasou", {"delay"}, set()),
]


def run_nlp_unit_tests() -> tuple[int, int, list[str]]:
    passed, failures = 0, []
    for text, must, must_not in NLP_UNIT_TESTS:
        found = set(detect_categories(normalize_text(text)))
        if must <= found and not (found & must_not):
            passed += 1
        else:
            failures.append(f"'{text}' -> found {sorted(found)} (expected {sorted(must)}; forbidden {sorted(must_not & found)})")
    return passed, len(NLP_UNIT_TESTS), failures


def classify_reviews(written: pd.DataFrame) -> pd.DataFrame:
    """Apply the rules to every written review; add multi-label flags and mention flags."""
    out = written.copy()
    norm = out["text"].map(normalize_text)
    out["text_normalized"] = norm
    detected = norm.map(detect_categories)
    for cat in CATEGORIES:
        out[f"flag_{cat}"] = detected.map(lambda d, c=cat: c in d).astype(int)
    out["flag_delivery_delay_nonreceipt"] = ((out["flag_delay"] == 1) | (out["flag_non_receipt"] == 1)).astype(int)
    out["n_categories_flagged"] = out[[f"flag_{c}" for c in CATEGORIES]].sum(axis=1)
    out["any_service_concern_flag"] = (out["n_categories_flagged"] > 0).astype(int)
    out["matched_rules"] = detected.map(lambda d: "; ".join(f"{c}:{'+'.join(r)}" for c, r in d.items()))
    out["delivery_topic_mention"] = norm.map(lambda s: int(bool(DELIVERY_MENTION.search(s))))
    out["delivery_mention_without_complaint"] = ((out["delivery_topic_mention"] == 1) &
                                                 (out["flag_delivery_delay_nonreceipt"] == 0) &
                                                 (out["flag_tracking_carrier"] == 0)).astype(int)
    return out


# ----------------------------------------------------------------------------
# 5. DESCRIPTIVE ANALYSES A, B, C
# ----------------------------------------------------------------------------
def score_summary(sub: pd.DataFrame, sample: str, grouping: str, group: str) -> dict:
    """Sample size, mean/median score, % of each rating and negative prevalence with CIs."""
    n = len(sub)
    row = {"sample": sample, "grouping": grouping, "group": group, "n_reviews": n,
           "n_orders": int(sub["order_id"].nunique()), "small_group_flag": bool(n < SMALL_N)}
    m_lo, m_hi = mean_ci(sub["review_score"])
    row.update(mean_score=sub["review_score"].mean() if n else np.nan, mean_ci_low=m_lo, mean_ci_high=m_hi,
               median_score=sub["review_score"].median() if n else np.nan)
    for s in range(1, 6):
        k = int((sub["review_score"] == s).sum())
        lo, hi = wilson_ci(k, n)
        row.update({f"n_{s}star": k, f"pct_{s}star": 100 * k / n if n else np.nan,
                    f"pct_{s}star_ci_low": 100 * lo, f"pct_{s}star_ci_high": 100 * hi})
    k_neg = int(sub["negative"].sum())
    lo, hi = wilson_ci(k_neg, n)
    row.update(n_negative=k_neg, negative_pct=100 * k_neg / n if n else np.nan,
               negative_ci_low=100 * lo, negative_ci_high=100 * hi)
    return row


def analysis_a_rating_distribution(rv_all: pd.DataFrame, rv_written: pd.DataFrame) -> pd.DataFrame:
    """Rating distribution by delivery performance and review timing; all scored vs written-only vs no-comment."""
    samples = [("All scored reviews", rv_all), ("Written reviews only", rv_written),
               ("Reviews without written comment", rv_all[~rv_all["has_text"]])]
    rows = []
    for label, df in samples:
        rows.append(score_summary(df, label, "Overall", "All eligible reviews"))
        for grouping, col, order in [("Delivery performance", "delivery_status", DELIVERY_LEVELS),
                                     ("Review timing", "timing", TIMING_LEVELS),
                                     ("Delivery x review timing", "group4", GROUP4_LEVELS + [f"{d} | Unknown" for d in DELIVERY_LEVELS])]:
            for g in present_levels(df[col], order):
                rows.append(score_summary(df[df[col] == g], label, grouping, g))
    return pd.DataFrame(rows)


def analysis_b_review_timing(rv: pd.DataFrame) -> pd.DataFrame:
    """Review timing by delivery performance: group summaries plus before-minus-after differences."""
    rows = []
    for ds_label, ds_df in [("On time", rv[rv["delivery_status"] == "On time"]),
                            ("Delayed", rv[rv["delivery_status"] == "Delayed"]),
                            ("All eligible reviews", rv)]:
        total = len(ds_df)
        for t in present_levels(ds_df["timing"], TIMING_LEVELS):
            sub = ds_df[ds_df["timing"] == t]
            n, k = len(sub), int(sub["negative"].sum())
            lo, hi = wilson_ci(k, n)
            m_lo, m_hi = mean_ci(sub["review_score"])
            rows.append({
                "row_type": "group_summary", "delivery_group": ds_label, "review_timing": t, "n_reviews": n,
                "share_of_delivery_group_pct": 100 * n / total, "small_group_flag": bool(n < SMALL_N),
                "mean_score": sub["review_score"].mean(), "mean_ci_low": m_lo, "mean_ci_high": m_hi,
                "median_score": sub["review_score"].median(), "n_negative": k, "negative_pct": 100 * k / n,
                "negative_ci_low": 100 * lo, "negative_ci_high": 100 * hi,
                "median_days_vs_estimate": sub["days_vs_estimate"].median(),
                "median_hours_answer_minus_delivery": sub["hours_answer_vs_delivery"].median(),
                "pct_survey_created_before_delivery_day": 100 * sub["survey_created_before_delivery_day"].mean(),
                "median_survey_created_minus_estimate_days": sub["survey_created_vs_estimate_days"].median(),
                "pct_with_written_comment": 100 * sub["has_text"].mean()})
        before = ds_df[ds_df["timing"] == "Before delivery"]
        after = ds_df[ds_df["timing"] == "On/after delivery"]
        if len(before) and len(after):
            kb, ka = int(before["negative"].sum()), int(after["negative"].sum())
            d_lo, d_hi = confint_proportions_2indep(kb, len(before), ka, len(after), method="newcomb", compare="diff")
            welch = stats.ttest_ind(before["review_score"], after["review_score"], equal_var=False)
            w_ci = welch.confidence_interval(confidence_level=1 - ALPHA)
            mw = stats.mannwhitneyu(before["review_score"], after["review_score"], alternative="two-sided")
            rows.append({
                "row_type": "difference_before_minus_after", "delivery_group": ds_label,
                "review_timing": "Before minus on/after delivery", "n_reviews": len(before) + len(after),
                "n_before": len(before), "n_after": len(after),
                "small_group_flag": bool(min(len(before), len(after)) < SMALL_N),
                "diff_negative_pct_points": 100 * (kb / len(before) - ka / len(after)),
                "diff_negative_ci_low": 100 * d_lo, "diff_negative_ci_high": 100 * d_hi,
                "diff_mean_score": before["review_score"].mean() - after["review_score"].mean(),
                "diff_mean_ci_low": float(w_ci.low), "diff_mean_ci_high": float(w_ci.high),
                "mann_whitney_p_value": float(mw.pvalue),
                "cliffs_delta": cliffs_delta(before["review_score"].values, after["review_score"].values)})
    return pd.DataFrame(rows)


def analysis_c_delay_severity(rv: pd.DataFrame, rv_written: pd.DataFrame) -> pd.DataFrame:
    """Negative-rating prevalence by delay severity, for several review samples."""
    samples = [("All scored reviews", rv),
               ("Reviewed on/after delivery only", rv[rv["timing"] == "On/after delivery"]),
               ("Reviewed before delivery only", rv[rv["timing"] == "Before delivery"]),
               ("Written reviews only", rv_written)]
    rows = []
    for scheme, col, levels in [("4 categories (Figure 2)", "severity4", SEVERITY4_LEVELS),
                                ("6 categories (fine detail)", "severity_fine", SEVERITY_FINE_LEVELS)]:
        for label, df in samples:
            for lvl in levels:
                sub = df[df[col] == lvl]
                n, k = len(sub), int(sub["negative"].sum())
                lo, hi = wilson_ci(k, n)
                m_lo, m_hi = mean_ci(sub["review_score"])
                rows.append({
                    "scheme": scheme, "sample": label, "severity_category": lvl, "n_reviews": n,
                    "n_orders": int(sub["order_id"].nunique()), "small_group_flag": bool(n < SMALL_N),
                    "n_negative": k, "negative_pct": 100 * k / n if n else np.nan,
                    "negative_ci_low": 100 * lo, "negative_ci_high": 100 * hi,
                    "mean_score": sub["review_score"].mean() if n else np.nan, "mean_ci_low": m_lo, "mean_ci_high": m_hi,
                    "median_days_vs_estimate": sub["days_vs_estimate"].median() if n else np.nan,
                    "pct_reviewed_before_delivery": 100 * (sub["timing"] == "Before delivery").mean() if n else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 6. TEXT-BASED PREVALENCE (Analysis D)
# ----------------------------------------------------------------------------
def complaint_prevalence(clf: pd.DataFrame) -> pd.DataFrame:
    """Share of written reviews with each detected phrase category, by group (Wilson 95% CIs)."""
    flags = ["flag_delivery_delay_nonreceipt", "flag_delay", "flag_non_receipt"] + \
            [f"flag_{c}" for c in CATEGORIES if c not in ("delay", "non_receipt")] + \
            ["any_service_concern_flag", "delivery_topic_mention", "delivery_mention_without_complaint"]
    groupings = [("Overall", None, None),
                 ("Delivery performance", "delivery_status", DELIVERY_LEVELS),
                 ("Review timing", "timing", TIMING_LEVELS),
                 ("Delivery x review timing", "group4", GROUP4_LEVELS),
                 ("Delay severity (4 categories)", "severity4", SEVERITY4_LEVELS),
                 ("Review score", "review_score", [1, 2, 3, 4, 5])]
    rows = []
    for grouping, col, order in groupings:
        subsets = [("All written reviews", clf)] if col is None else \
                  [(str(g), clf[clf[col] == g]) for g in (present_levels(clf[col], order) if col != "review_score" else order)]
        for gname, sub in subsets:
            n = len(sub)
            for flag in flags:
                k = int(sub[flag].sum())
                lo, hi = wilson_ci(k, n)
                label = flag.replace("flag_", "")
                rows.append({"grouping": grouping, "group": gname, "category": label,
                             "category_label": CATEGORY_LABELS.get(label, label),
                             "n_written_reviews": n, "n_flagged": k, "pct": 100 * k / n if n else np.nan,
                             "ci_low": 100 * lo, "ci_high": 100 * hi, "small_group_flag": bool(n < SMALL_N)})
    out = pd.DataFrame(rows)
    # Among reviews that mention delivery topics, how many carry a delay/non-receipt/tracking phrase?
    mention = clf[clf["delivery_topic_mention"] == 1]
    k = int(((mention["flag_delivery_delay_nonreceipt"] == 1) | (mention["flag_tracking_carrier"] == 1)).sum())
    lo, hi = wilson_ci(k, len(mention))
    extra = pd.DataFrame([{"grouping": "Mention vs complaint", "group": "Reviews mentioning delivery topics",
                           "category": "mention_with_delivery_phrase", "category_label":
                           "Share of delivery-topic mentions that contain a delay / non-receipt / tracking phrase",
                           "n_written_reviews": len(mention), "n_flagged": k, "pct": 100 * k / len(mention),
                           "ci_low": 100 * lo, "ci_high": 100 * hi, "small_group_flag": False}])
    return pd.concat([out, extra], ignore_index=True)


def nlp_rule_frequency(clf: pd.DataFrame) -> pd.DataFrame:
    """Counts of fired rules, not an accuracy validation or independent labels."""
    from collections import Counter
    counter = Counter()
    for field in clf['matched_rules'].fillna(''):
        for component in field.split('; '):
            if ':' not in component:
                continue
            category, rule_list = component.split(':', 1)
            for rule_name in rule_list.split('+'):
                if rule_name:
                    counter[(category, rule_name)] += 1
    return pd.DataFrame([
        {'category': category, 'rule_id': rule_id, 'n_reviews_triggered': n,
         'sample_n_written': len(clf), 'share_of_written_reviews_pct': 100*n/len(clf),
         'note': 'Exploratory rule triggers, not validated complaints'}
        for (category, rule_id), n in sorted(counter.items())
    ])


def export_example_matches(clf: pd.DataFrame) -> pd.DataFrame:
    """Seeded random examples per category, for exploratory inspection of rule weaknesses (not validation)."""
    rng = np.random.default_rng(SEED)
    parts = []
    for cat in CATEGORIES:
        pool = clf[clf[f"flag_{cat}"] == 1]
        take = min(N_EXAMPLES_PER_CATEGORY, len(pool))
        idx = rng.choice(pool.index.values, size=take, replace=False) if take else []
        ex = clf.loc[idx, ["review_id", "order_id", "review_score", "delivery_status", "timing", "matched_rules", "text"]].copy()
        ex.insert(0, "category", cat)
        parts.append(ex)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------
# 7. ASSOCIATIONS AND REGRESSION (Analysis E)
# ----------------------------------------------------------------------------
def association_tests(samples: dict[str, pd.DataFrame], written_samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Chi-square / Cramer's V for categorical pairs, Mann-Whitney / Kruskal / Spearman for ordinal scores."""
    rows: list[dict] = []
    counter = {"i": 0}

    def base(sample, analysis, note=""):
        counter["i"] += 1
        return {"test_id": f"T{counter['i']:02d}", "sample": sample, "analysis": analysis, "note": note}

    def add_chi(sample, df, var1, var2, analysis, note=""):
        tbl = pd.crosstab(df[var1], df[var2])
        tbl = tbl.loc[tbl.sum(axis=1) > 0, tbl.sum(axis=0) > 0]
        if tbl.shape[0] < 2 or tbl.shape[1] < 2:
            return
        res = chi_square_association(tbl)
        row = base(sample, analysis, note)
        if tbl.shape == (2, 2):
            row["or_definition"] = (f"odds of '{tbl.columns[1]}' in '{tbl.index[1]}' relative to '{tbl.index[0]}' "
                                    f"(columns: '{tbl.columns[0]}' vs '{tbl.columns[1]}')")
        row.update(test="Chi-square test of independence", n=res["n"], statistic=res["chi2"], dof=res["dof"],
                   p_value=res["p_value"], effect_size_name="Cramer's V", effect_size=res["cramers_v"],
                   odds_ratio=res["odds_ratio"], or_ci_low=res["or_ci_low"], or_ci_high=res["or_ci_high"],
                   min_expected_count=res["min_expected_count"], share_cells_expected_ge5=res["share_cells_expected_ge5"],
                   test_suitable=res["chi2_suitable"], fisher_p_value=res["fisher_p_value"])
        rows.append(row)

    def add_mw(sample, a, b, analysis, note=""):
        if len(a) == 0 or len(b) == 0:
            return
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        row = base(sample, analysis, note)
        row.update(test="Mann-Whitney U (review_score)", n=len(a) + len(b), statistic=float(mw.statistic),
                   p_value=float(mw.pvalue), effect_size_name="Cliff's delta (first minus second group)",
                   effect_size=cliffs_delta(a.values, b.values), test_suitable=True)
        rows.append(row)

    for sample, df in samples.items():
        add_chi(sample, df, "delivery_status", "negative", "Delivery status x negative rating (1-2 stars)")
        add_chi(sample, df, "timing", "negative", "Review timing x negative rating")
        add_chi(sample, df, "delivery_status", "timing", "Delivery status x review timing (collinearity check)",
                "strong association => delivery status and timing cannot be separated cleanly")
        add_chi(sample, df, "severity4", "timing", "Delay severity x review timing (collinearity check)")
        add_chi(sample, df, "group4", "review_score", "Delivery x timing group x 1-5 star rating")
        add_chi(sample, df, "severity4", "negative", "Delay severity (4 cat.) x negative rating")
        add_chi(sample, df[df["timing"] == "On/after delivery"], "severity4", "negative",
                "Delay severity x negative rating, reviewed on/after delivery only")
        for ds in DELIVERY_LEVELS:
            add_chi(sample, df[df["delivery_status"] == ds], "timing", "negative",
                    f"Review timing x negative rating within {ds.lower()} orders")
        add_chi(sample, df, "delivery_status", "has_text", "Delivery status x has written comment (selection check)")
        add_chi(sample, df, "timing", "has_text", "Review timing x has written comment (selection check)")
        add_chi(sample, df, "review_score", "has_text", "Review score x has written comment (selection check)")
        add_mw(sample, df.loc[df["delivery_status"] == "On time", "review_score"],
               df.loc[df["delivery_status"] == "Delayed", "review_score"], "Review score: on time vs delayed")
        delayed = df[df["delivery_status"] == "Delayed"]
        add_mw(sample, delayed.loc[delayed["timing"] == "Before delivery", "review_score"],
               delayed.loc[delayed["timing"] == "On/after delivery", "review_score"],
               "Review score within delayed orders: before vs on/after delivery")
        groups = [g["review_score"].values for _, g in df.groupby("group4", observed=True) if len(g)]
        kw = stats.kruskal(*groups)
        row = base(sample, "Review score across delivery x timing groups")
        row.update(test="Kruskal-Wallis H (review_score)", n=sum(len(g) for g in groups), statistic=float(kw.statistic),
                   dof=len(groups) - 1, p_value=float(kw.pvalue), test_suitable=True)
        rows.append(row)
        for label, sub in [("all delayed", delayed), ("delayed, reviewed on/after delivery", delayed[delayed["timing"] == "On/after delivery"]),
                           ("delayed, reviewed before delivery", delayed[delayed["timing"] == "Before delivery"])]:
            if len(sub) > 2:
                rho = stats.spearmanr(sub["days_vs_estimate"], sub["review_score"])
                row = base(sample, f"Days late vs review score, {label}")
                row.update(test="Spearman rank correlation", n=len(sub), statistic=float(rho.statistic),
                           p_value=float(rho.pvalue), effect_size_name="Spearman rho", effect_size=float(rho.statistic),
                           test_suitable=True)
                rows.append(row)
    for sample, df in written_samples.items():
        add_chi(sample, df, "delivery_status", "flag_delivery_delay_nonreceipt", "Delivery status x delay/non-receipt phrase detected")
        add_chi(sample, df, "timing", "flag_delivery_delay_nonreceipt", "Review timing x delay/non-receipt phrase detected")
        add_chi(sample, df, "group4", "flag_delivery_delay_nonreceipt", "Delivery x timing group x delay/non-receipt phrase")
        add_chi(sample, df, "severity4", "flag_delivery_delay_nonreceipt", "Delay severity x delay/non-receipt phrase")
        add_chi(sample, df, "negative", "flag_delivery_delay_nonreceipt", "Negative rating x delay/non-receipt phrase",
                "rating and text are jointly determined by the same experience")
        add_chi(sample, df, "delivery_status", "flag_tracking_carrier", "Delivery status x tracking/carrier phrase")
        add_chi(sample, df[df["delivery_status"] == "Delayed"], "timing", "flag_delivery_delay_nonreceipt",
                "Review timing x delay/non-receipt phrase within delayed orders")
    return pd.DataFrame(rows)


COVARIATE_FORMULA = "multi_item + multi_seller + purchased_2018 + log_price_z + freight_share_z + log_weight_z + promised_days_z"
COVARIATE_LABELS = {
    "multi_item": "Order has 2+ items (vs 1 item)", "multi_seller": "Order has 2+ sellers (vs 1)",
    "purchased_2018": "Purchased in 2018 (vs 2016-2017)", "log_price_z": "Order item value, log (per 1 SD)",
    "freight_share_z": "Freight share of order value (per 1 SD)", "log_weight_z": "Order weight, log (per 1 SD)",
    "promised_days_z": "Promised delivery time (per 1 SD)", "log_words_z": "Review length, log words (per 1 SD)",
    "review_score": "Review score (per +1 star)", "days_late_per7": "Days late (per +7 days)",
}
TERM_PATTERN = re.compile(r'C\((\w+), Treatment\(reference="(.+?)"\)\)\[T\.(.+)\]')


def describe_term(term: str) -> tuple[str, str, str]:
    """Return (group heading, readable label, reference category) for a patsy term name."""
    m = TERM_PATTERN.match(term)
    if m:
        var, ref, level = m.groups()
        if var == "delivery_status":
            return "Delivery performance", f"{level} delivery", ref
        if var == "severity4":
            return "Delay severity", SEVERITY_DISPLAY.get(level, level), ref
        if var == "timing":
            return "Review timing", f"Reviewed {level.lower()}", ref
    group = "Delay (continuous)" if term == "days_late_per7" else "Review text / score" \
        if term in ("log_words_z", "review_score") else "Order covariates"
    return group, COVARIATE_LABELS.get(term, term), ""


def add_covariate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Raw covariates used in the models (NaN is preserved, never silently turned into 0)."""
    d = df.copy()
    d["multi_item"] = np.where(d["n_items"].isna(), np.nan, (d["n_items"] > 1).astype(float))
    d["multi_seller"] = np.where(d["n_sellers"].isna(), np.nan, (d["n_sellers"] > 1).astype(float))
    d["log_price"] = np.log1p(d["total_price"])
    value = d["total_price"] + d["total_freight"]
    d["freight_share"] = np.where(value > 0, d["total_freight"] / value, np.nan)
    d["log_weight"] = np.log1p(d["total_weight_g"])
    d["log_words"] = np.log1p(d["word_count"])
    d["days_late_per7"] = d["days_vs_estimate"] / 7.0
    return d


def prepare_model_frame(df: pd.DataFrame, extra_needed: list[str], z_columns: list[str],
                        require_order_covariates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Remove invalid/unknown timing, apply transparent complete-case exclusions and z-scale."""
    d = add_covariate_columns(df)
    n_start = len(d)
    d = d[d["timing"] != "Unknown"].copy()
    n_unknown = n_start - len(d)
    needed = (['multi_item', 'multi_seller', 'purchased_2018', 'log_price',
               'freight_share', 'log_weight', 'promised_days'] if require_order_covariates else [])
    needed += list(extra_needed)
    d = d.dropna(subset=needed).copy()
    for col in ['delivery_status', 'timing', 'severity4']:
        d[col] = d[col].astype(str).astype(object)
    info = {'n_start': n_start, 'n_excluded_unknown_timing': n_unknown,
            'n_excluded_missing_covariates': n_start - n_unknown - len(d)}
    for col in z_columns:
        sigma = d[col].std()
        if not np.isfinite(sigma) or sigma == 0:
            raise ValueError(f"Covariate {col} has zero/nonfinite standard deviation")
        d[f'{col}_z'] = (d[col] - d[col].mean()) / sigma
    return d, info


def fit_logit(model_id: str, label: str, formula: str, data: pd.DataFrame, outcome: str, sample_label: str,
              info: dict, notes: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Logistic regression with cluster-robust SEs (cluster = order_id); returns OR table and VIF table."""
    groups = pd.factorize(data["order_id"])[0]
    result = smf.logit(formula, data=data).fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    if not result.mle_retvals.get("converged", True):
        print(f"WARNING: model {model_id} did not converge.", file=sys.stderr)
    conf = np.exp(result.conf_int())
    rows = []
    for term in result.params.index:
        if term == "Intercept":
            continue
        heading, term_label, ref = describe_term(term)
        rows.append({"model_id": model_id, "model_label": label, "outcome": outcome, "sample": sample_label,
                     "term": term, "term_group": heading, "term_label": term_label, "reference_category": ref,
                     "odds_ratio": float(np.exp(result.params[term])), "ci_low": float(conf.loc[term, 0]),
                     "ci_high": float(conf.loc[term, 1]), "p_value": float(result.pvalues[term]),
                     "coef_log_odds": float(result.params[term]), "se_cluster_robust": float(result.bse[term]),
                     "n_obs": int(result.nobs), "n_outcome_events": int(data[outcome].sum()),
                     "n_clusters_orders": int(len(np.unique(groups))), "n_excluded_unknown_timing": info["n_excluded_unknown_timing"],
                     "n_excluded_missing_covariates": info["n_excluded_missing_covariates"],
                     "pseudo_r2_mcfadden": float(result.prsquared), "covariance": "cluster-robust by order_id",
                     "formula": formula, "notes": notes})
    exog, names = result.model.exog, result.model.exog_names
    vif_rows = [{"model_id": model_id, "variable": names[i], "vif": float(variance_inflation_factor(exog, i))}
                for i in range(len(names)) if names[i] != "Intercept"]
    return pd.DataFrame(rows), pd.DataFrame(vif_rows)


def run_regressions(rv: pd.DataFrame, rv_one: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rating-outcome models only. N1 = core, N2 = adjusted; N3/N4 sensitivity.

    No unvalidated NLP label is ever the dependent variable in the main regression.
    """
    severity = 'C(severity4, Treatment(reference="On time"))'
    timing = 'C(timing, Treatment(reference="On/after delivery"))'
    specs = [
      ('N1', 'Core: delay severity + review timing',
       f'negative ~ {severity} + {timing}', rv, 'All scored reviews', False),
      ('N2', 'Adjusted: delay severity + review timing + order covariates',
       f'negative ~ {severity} + {timing} + {COVARIATE_FORMULA}', rv, 'All scored reviews', True),
      ('N3', 'Sensitivity: adjusted model, reviews on/after delivery',
       f'negative ~ {severity} + {COVARIATE_FORMULA}',
       rv[rv['timing'] == 'On/after delivery'], 'After delivery scored reviews', True),
      ('N4', 'Sensitivity: adjusted model, one review per order',
       f'negative ~ {severity} + {timing} + {COVARIATE_FORMULA}',
       rv_one, 'One scored review per order', True),
    ]
    models, vifs, diagnostics = [], [], []
    for mid, title, formula, sample, sample_title, adjusted in specs:
        z = ['log_price', 'freight_share', 'log_weight', 'promised_days'] if adjusted else []
        d, info = prepare_model_frame(sample, [], z, require_order_covariates=adjusted)
        try:
            m, vif = fit_logit(mid, title, formula, d, 'negative', sample_title, info,
                               notes='Associations, not causal effects; timing groups are selected.')
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise RuntimeError(f'Model {mid} failed: {exc}') from exc
        models.append(m)
        vifs.append(vif)
        diagnostics.append({'model_id': mid, 'formula': formula, 'n_start': info['n_start'],
                            'n_excluded_unknown_timing': info['n_excluded_unknown_timing'],
                            'n_excluded_missing_covariates': info['n_excluded_missing_covariates'],
                            'n_model': len(d), 'n_negative': int(d['negative'].sum()),
                            'n_orders': d['order_id'].nunique(),
                            'highest_vif': float(vif['vif'].max()) if len(vif) else np.nan})
    return (pd.concat(models, ignore_index=True), pd.concat(vifs, ignore_index=True),
            pd.DataFrame(diagnostics))


# ----------------------------------------------------------------------------
# 8. SENSITIVITY ANALYSES
# ----------------------------------------------------------------------------
def sensitivity_analyses(eligible: pd.DataFrame, rv: pd.DataFrame, rv_written: pd.DataFrame, rv_one: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def neg_row(analysis, spec, group, sub, note=""):
        n, k = len(sub), int(sub["negative"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"analysis": analysis, "specification": spec, "group": group, "n_reviews": n, "n_negative": k,
                     "negative_pct": 100 * k / n if n else np.nan, "ci_low": 100 * lo, "ci_high": 100 * hi,
                     "mean_score": sub["review_score"].mean() if n else np.nan, "note": note})

    # S1: how "delayed" is defined
    conventions = {
        "Calendar-day (primary): delivery date > estimated date": eligible["days_vs_estimate"] > 0,
        "Strict timestamp: delivery timestamp > estimated date at 00:00": eligible["delayed_strict_timestamp"],
        "Calendar-day with 1-day tolerance: delivery date > estimate + 1 day": eligible["days_vs_estimate"] > 1,
    }
    review_flags = {
        "Calendar-day (primary): delivery date > estimated date": rv["days_vs_estimate"] > 0,
        "Strict timestamp: delivery timestamp > estimated date at 00:00": rv["delayed_strict_timestamp"],
        "Calendar-day with 1-day tolerance: delivery date > estimate + 1 day": rv["days_vs_estimate"] > 1,
    }
    for spec, flag in conventions.items():
        rows.append({"analysis": "S1 Delay definition (orders)", "specification": spec, "group": "Delayed orders",
                     "n_reviews": np.nan, "n_negative": np.nan, "negative_pct": 100 * flag.mean(), "ci_low": np.nan,
                     "ci_high": np.nan, "mean_score": np.nan,
                     "note": f"{int(flag.sum())} of {len(eligible)} eligible orders; negative_pct column = % of orders delayed"})
        rflag = review_flags[spec]
        neg_row("S1 Delay definition (reviews)", spec, "Delayed", rv[rflag])
        neg_row("S1 Delay definition (reviews)", spec, "On time", rv[~rflag])
    # S2: how "before delivery" is defined (timestamp vs calendar day) and tie handling
    ans_day = rv["review_answer_timestamp"].dt.normalize()
    before_day = ans_day < rv["delivered_date"]
    for dlabel, dmask in [("Delayed", rv["delivery_status"] == "Delayed"), ("On time", rv["delivery_status"] == "On time")]:
        neg_row("S2 Review-timing definition", "Timestamp (primary): answer timestamp < delivery timestamp",
                f"{dlabel}, before delivery", rv[dmask & (rv["timing"] == "Before delivery")])
        neg_row("S2 Review-timing definition", "Timestamp (primary): answer timestamp < delivery timestamp",
                f"{dlabel}, on/after delivery", rv[dmask & (rv["timing"] == "On/after delivery")])
        neg_row("S2 Review-timing definition", "Calendar day: answer date < delivery date",
                f"{dlabel}, before delivery", rv[dmask & before_day & (rv["timing"] != "Unknown")])
        neg_row("S2 Review-timing definition", "Calendar day: answer date < delivery date",
                f"{dlabel}, on/after delivery", rv[dmask & ~before_day & (rv["timing"] != "Unknown")])
    ties = int((rv["review_answer_timestamp"] == rv["order_delivered_customer_date"]).sum())
    rows.append({"analysis": "S2 Review-timing definition", "specification": "Exact ties (answer == delivery timestamp)",
                 "group": "All reviews", "n_reviews": ties, "n_negative": np.nan, "negative_pct": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "mean_score": np.nan,
                 "note": "ties are classified as 'On/after delivery'; count is the number of tied records"})
    # S3: unit of analysis (all review records vs one review per order) and S4: sample (all vs written)
    for g in GROUP4_LEVELS:
        neg_row("S3 Unit of analysis", "All review records (primary)", g, rv[rv["group4"] == g])
        neg_row("S3 Unit of analysis", "One review per order (latest answered)", g, rv_one[rv_one["group4"] == g])
        neg_row("S4 Sample", "All scored reviews (primary)", g, rv[rv["group4"] == g])
        neg_row("S4 Sample", "Written reviews only", g, rv_written[rv_written["group4"] == g])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 9. MAIN PAPER FIGURES (exactly three; each saved as vector PDF and 300-dpi PNG)
# ----------------------------------------------------------------------------
def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1_rating_distribution(rating_table: pd.DataFrame) -> pd.DataFrame:
    """Two-panel normalized 1-5 rating distribution: all scored vs written-only."""
    panels = [('All scored reviews', '(a) All scored reviews'),
              ('Written reviews only', '(b) Written reviews only')]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.1), sharex=True)
    export = []
    for ax, (sample, title) in zip(axes, panels):
        sub = rating_table[(rating_table['sample'] == sample) &
                           (rating_table['grouping'] == 'Delivery x review timing')]
        sub = sub.set_index('group').reindex(GROUP4_LEVELS)
        y = np.arange(len(sub))[::-1]
        left = np.zeros(len(sub))
        for star in range(1, 6):
            widths = sub[f'pct_{star}star'].to_numpy(dtype=float)
            ax.barh(y, widths, left=left, color=RATING_COLORS[star],
                    edgecolor='white', linewidth=0.6, height=.60,
                    label=f'{star} star' if star == 1 else f'{star} stars')
            for yi, w, l in zip(y, widths, left):
                if w >= 5:
                    ax.text(l + w / 2, yi, f'{w:.0f}%', ha='center', va='center',
                            fontsize=8, color='white' if star in (1, 5) else '#20242b')
            left += widths
        labels = [f'{GROUP4_DISPLAY[g]}\n(n = {int(n):,})'
                  for g, n in zip(sub.index, sub['n_reviews'])]
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 100)
        ax.set_ylim(-.7, len(sub) - .3)
        ax.set_title(title, loc='left', fontweight='bold', pad=10)
        ax.xaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
        ax.grid(axis='x', color='#e5e8ec', linewidth=.7)
        ax.set_axisbelow(True)
        for g, r in sub.iterrows():
            for star in range(1, 6):
                export.append({'sample': sample, 'group': g, 'n_reviews': int(r['n_reviews']),
                               'star': star, 'n_with_rating': int(r[f'n_{star}star']),
                               'pct': float(r[f'pct_{star}star']),
                               'ci_low': float(r[f'pct_{star}star_ci_low']),
                               'ci_high': float(r[f'pct_{star}star_ci_high']),
                               'mean_score': float(r['mean_score']),
                               'negative_pct': float(r['negative_pct'])})
    axes[1].set_xlabel('Share within each review group (each bar = 100%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc='upper center',
               bbox_to_anchor=(.57, 1.01), title='Rating')
    fig.suptitle('Figure 1. Ratings by delivery performance and review timing',
                 fontweight='bold', fontsize=12, y=1.075)
    fig.text(.06, -.015,
             'Sample sizes shown separately because group sizes differ. Percentages are within-group, '
             'not absolute frequencies.\nAll scored reviews include non-written ratings; written reviews '
             'are a potentially selected subset.', fontsize=8, ha='left', va='top', color='#3e4858')
    fig.tight_layout()
    save_figure(fig, 'Figure_1_Rating_Distribution')
    return pd.DataFrame(export)


def figure2_delay_severity(sev: pd.DataFrame) -> pd.DataFrame:
    """Wilson 95% CIs: all scored reviews vs the on/after-delivery subset."""
    specs = [('All scored reviews', 'All scored reviews', '#475569', -0.11, 'o'),
             ('Reviewed on/after delivery only', 'Reviewed on/after delivery', '#167a8c', 0.11, 's')]
    d4 = sev[sev['scheme'] == '4 categories (Figure 2)']
    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    x = np.arange(len(SEVERITY4_LEVELS))
    rows = []
    for sample, display, color, xoffset, symbol in specs:
        part = d4[d4['sample'] == sample].set_index('severity_category').reindex(SEVERITY4_LEVELS)
        pct, lo, hi = [part[c].to_numpy(dtype=float) for c in
                       ['negative_pct', 'negative_ci_low', 'negative_ci_high']]
        ns = part['n_reviews'].to_numpy(dtype=int)
        for i, (p, lower, upper, n) in enumerate(zip(pct, lo, hi, ns)):
            if not np.isfinite(p):
                continue
            unstable = n < SMALL_N
            ax.errorbar(x[i]+xoffset, p, yerr=[[p-lower], [upper-p]], fmt=symbol,
                        color=color, ecolor=color, markersize=7,
                        markerfacecolor='white' if unstable else color,
                        elinewidth=1.35, capsize=4, zorder=4,
                        label=display if i == 0 else None)
            rows.append({'sample': sample, 'severity_category': SEVERITY4_LEVELS[i],
                         'n_reviews': n, 'n_negative': int(part.iloc[i]['n_negative']),
                         'negative_pct': p, 'ci_low': lower, 'ci_high': upper,
                         'small_n': unstable})
        # Do not connect categories with lines: these are bins, not continuous-time points.
    ax.set_ylim(0, 100)
    ax.set_xlim(-.65, 3.65)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
    ax.set_xticks(x, [SEVERITY_DISPLAY[c] + (' late' if c != 'On time' else '')
                      for c in SEVERITY4_LEVELS])
    ax.set_ylabel('Negative ratings (1–2 stars)')
    ax.set_title('Figure 2. Negative ratings by delivery-delay severity', loc='left',
                 fontsize=12, fontweight='bold', pad=13)
    ax.grid(axis='y', color='#e4e8ed', linewidth=.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False)
    # Two rows of explicit denominators beneath the axis.
    for row_idx, (sample, display, color, _, _) in enumerate(specs):
        d = d4[d4['sample'] == sample].set_index('severity_category').reindex(SEVERITY4_LEVELS)
        yy = -.19 - .075*row_idx
        ax.text(-.06, yy, display, ha='right', va='center', transform=ax.transAxes,
                fontsize=7.5, color=color)
        for xi, r in zip(x, d.itertuples()):
            n = int(r.n_reviews)
            ax.text(xi, yy, f'{n:,}' + ('*' if n < SMALL_N else ''),
                    transform=ax.get_xaxis_transform(), ha='center', va='center',
                    fontsize=8, color=color)
    fig.text(.065, .015,
             '95% Wilson intervals. On/after-delivery group is a subset of all reviews; comparisons are descriptive. '
             '* n < 100, unstable.',
             fontsize=7.9, color='#455468')
    fig.subplots_adjust(left=.31, right=.98, top=.89, bottom=.34)
    save_figure(fig, 'Figure_2_Delay_Severity')
    return pd.DataFrame(rows)


def figure3_delivery_phrase_prevalence(prev: pd.DataFrame) -> pd.DataFrame:
    """Exploratory detected phrase prevalence for the written-review sample only."""
    subset = prev[(prev['grouping'] == 'Delivery x review timing') &
                  (prev['category'] == 'delivery_delay_nonreceipt')].copy()
    groups = ['On time | On/after delivery', 'Delayed | On/after delivery',
              'Delayed | Before delivery', 'On time | Before delivery']
    d = subset.set_index('group').reindex(groups).reset_index()
    fig, ax = plt.subplots(figsize=(9.4, 5.1))
    y = np.arange(len(d))[::-1]
    maxlim = 0
    for k, r in enumerate(d.itertuples()):
        if not np.isfinite(r.pct):
            continue
        color = '#0c7c86' if r.n_written_reviews >= 200 else '#8999a5'
        ax.errorbar(float(r.pct), y[k], xerr=[[float(r.pct-r.ci_low)], [float(r.ci_high-r.pct)]],
                    fmt='o', color=color, ecolor=color, markersize=7,
                    capsize=4, elinewidth=1.65)
        ax.text(float(r.ci_high) + 1.6, y[k],
                f'{r.pct:.1f}%  (n={int(r.n_written_reviews):,})',
                fontsize=9, va='center')
        maxlim = max(maxlim, float(r.ci_high))
    ax.set_yticks(y, [GROUP4_DISPLAY[g].replace('\n', ' ') for g in groups])
    ax.set_xlim(0, min(100, max(50, maxlim+20)))
    ax.set_xlabel('Written reviews matching delay/non-receipt phrases (%)')
    ax.set_title('Figure 3. Algorithmically detected delivery-related expressions',
                 loc='left', fontsize=12, fontweight='bold', pad=14)
    ax.grid(axis='x', color='#e4e8ed', linewidth=.75)
    ax.set_axisbelow(True)
    fig.text(.06, .025,
             '95% Wilson confidence intervals are for rule-detected phrase prevalence, NOT validated complaint '
             'rates. Small groups may be unstable.', fontsize=7.8, color='#455468')
    fig.subplots_adjust(left=.31, right=.96, top=.86, bottom=.22)
    save_figure(fig, 'Figure_3_Delivery_Related_Expressions')
    return d.rename(columns={'pct':'detected_pct', 'ci_low':'ci95_low', 'ci_high':'ci95_high'})


# ----------------------------------------------------------------------------
# 10. TEXT SUMMARY
# ----------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main_results_summary(ctx: dict) -> str:
    """Main-paper summary: descriptive analyses only; regressions are intentionally absent."""
    rv, clf, elig = ctx['rv'], ctx['clf'], ctx['eligible']
    rating, sev, prev = ctx['rating'], ctx['severity'], ctx['prevalence']
    lines = [
        'OLIST - MAIN RESEARCH METHODS AND RESULTS (THREE FIGURES)',
        '=' * 68,
        'Research question: How do customer ratings and delivery-related review',
        'expressions vary across delivery performance, delay severity, and',
        'review submission timing in Brazilian e-commerce?',
        '',
        'STUDY DESIGN AND SAMPLES',
        'Exploratory, retrospective observational analysis; NOT a causal study.',
        f'Eligible delivered orders: {len(elig):,}.',
        f'Eligible scored review records: {len(rv):,}, across {rv.order_id.nunique():,} orders.',
        f'Written review records (nonempty comment message): {len(clf):,}, across {clf.order_id.nunique():,} orders.',
        f'Unknown or invalid review timing: {int((rv.timing == "Unknown").sum()):,} scored review records.',
        f'Rule-based NLP unit tests on constructed sentences: {ctx["nlp_tests"][0]}/{ctx["nlp_tests"][1]}; NOT accuracy validation.',
        'Review-level observations can repeat within an order.',
        'Delivery delay = actual delivery calendar date after estimated delivery calendar date.',
        'Review timing uses review_answer_timestamp relative to actual delivery timestamp.',
        'Missing/chronologically impossible answer timestamps are Unknown, not before delivery.',
        'Negative rating = score 1-2; 3-5 is the complement for prevalence calculations.',
        '',
        'FIGURE 1 - DISTRIBUTION OF CUSTOMER RATINGS',
        'Why this method? Five-point ratings are ordinal; using 100% stacked bar charts',
        'preserves the full distribution and permits percentage comparisons between',
        'groups with very different sample sizes. All scored versus written-only',
        'panels expose potential selection associated with writing a comment.',
    ]
    for sample in ['All scored reviews','Written reviews only']:
        d = rating[(rating['sample'] == sample) & (rating['grouping'] == 'Delivery x review timing')]
        lines.append(f'Panel: {sample}')
        for _, r in d.iterrows():
            lines.append(f'  {r["group"]}: n={int(r.n_reviews):,}, mean score={r.mean_score:.2f}, negative={r.negative_pct:.2f}%.')
    lines += ['', 'FIGURE 2 - DELAY SEVERITY AND NEGATIVE RATINGS',
              'Why this method? Severity distinguishes short delays from long delays.',
              'Negative-rating proportions with Wilson 95% intervals show uncertainty',
              'and unequal denominators; the groups are delay bins, not a time series.',
              'On/after delivery is a SELECTED SUBSET of all scored reviews, so the two',
              'series cannot support a causal interpretation of review timing.']
    s4 = sev[sev['scheme'] == '4 categories (Figure 2)']
    for sample in ['All scored reviews','Reviewed on/after delivery only','Reviewed before delivery only']:
        lines.append(f'Sample: {sample}')
        for _, r in s4[s4['sample'] == sample].iterrows():
            lines.append(f'  {r.severity_category}: {r.negative_pct:.2f}% (n={int(r.n_reviews):,})'
                         + (' [small n < 100]' if r.small_group_flag else ''))
    lines += ['', 'FIGURE 3 - EXPLORATORY PORTUGUESE DELIVERY EXPRESSIONS',
              'Why this method? Star ratings show satisfaction but do not indicate',
              'what concern was articulated. Deterministic phrase rules make the',
              'lexical analysis transparent in the absence of manually labeled data.',
              'The denominator is written reviews only; the detector uses Portuguese',
              'comment text and, if present, its title.',
              'Labels are algorithmically detected phrases, NOT confirmed complaints.',
              'No manual ground truth: precision, recall and F1 are UNKNOWN.']
    p = prev[(prev['grouping'] == 'Delivery x review timing') &
             (prev['category'] == 'delivery_delay_nonreceipt')]
    for _, r in p.iterrows():
        lines.append(f'  {r["group"]}: {r.pct:.2f}% ({int(r.n_flagged):,}/{int(r.n_written_reviews):,})'
                     + (' [small n]' if r.small_group_flag else ''))
    lines += ['', 'MAIN LIMITATIONS',
              '- Differences in review timing and written-comment participation are',
              '  selected groups, not repeat measurements or randomized interventions.',
              '- Final delivery severity can be measured AFTER a before-delivery review:',
              '  its association with early reviews is retrospective, not a forecast.',
              '- Severe delays reviewed on/after delivery have very small cells.',
              '- Observed ratings may reflect sellers, products, regions, expectations,',
              '  the review-invitation process and other unobserved characteristics.',
              '- Wilson intervals treat review rows as independent; a minority of orders',
              '  contribute multiple reviews. Order-clustered regression is supplementary.',
              '- Phrase rules can include false positives and false negatives; they do',
              '  not estimate independently verified complaint prevalence.',
              '', 'SUPPLEMENTARY ONLY',
              'N1-N4 logistic regressions were computed and retained as separate',
              'numerical research checks. They were NOT used to generate or to claim',
              'adjustment of Figures 1-3; their outputs are NOT paper findings.',
              'The three figures and three main methods are the only main presentation.',
              '']
    return '\n'.join(lines)


def supplementary_regression_summary(models: pd.DataFrame, vifs: pd.DataFrame,
                                     diag: pd.DataFrame) -> str:
    """Internal-only model documentation: never merged into the paper results summary."""
    l = ['SUPPLEMENTARY REGRESSION / INTERNAL REFERENCE ONLY',
         '=' * 65,
         'The research paper contains THREE descriptive/exploratory figures.',
         'No logistic-regression plot is generated; these are not main paper results.',
         'Outcome for all N1-N4 = observed negative rating (1-2 vs 3-5 stars).',
         'Order-level clustering of standard errors; confidence intervals are odds ratios,',
         'not probability changes or causal estimates.',
         'N1 core: severity and review timing.',
         'N2 adjusted: severity, timing and observable order covariates.',
         'N3: adjusted, only reviews on/after delivery.',
         'N4: adjusted, most recently answered review per order.',
         'The fitted models have only main effects; severity x timing interactions',
         'are not specified. The effect of review timing may differ by severity.',
         'Timing and severity are tightly associated, so estimates can depend on',
         'specification, selection, sparse cells and unmeasured confounders.',
         'Neither adjusted associations nor sensitivity analyses establish causation.',
         '']
    for mid in ['N1','N2','N3','N4']:
        t = models[models.model_id == mid]
        d = diag[diag.model_id == mid].iloc[0]
        l += [f'{mid}: {t.model_label.iloc[0]}',
              f'  Formula: {t.formula.iloc[0]}',
              f'  n={int(d.n_model):,}, negative={int(d.n_negative):,}, unique orders={int(d.n_orders):,},',
              f'  excluded for unknown timing={int(d.n_excluded_unknown_timing):,},',
              f'  excluded for missing covariates={int(d.n_excluded_missing_covariates):,},',
              f'  max VIF={d.highest_vif:.2f} (not proof of no collinearity).']
        for _, r in t.iterrows():
            l.append(f'  {r.term_label}: OR={r.odds_ratio:.3f}, 95%CI={r.ci_low:.3f}-{r.ci_high:.3f}, p={r.p_value:.4g}')
        l.append('')
    return '\n'.join(l)


def main() -> None:
    import argparse
    global DATA_DIR
    parser = argparse.ArgumentParser(description='Olist - three research figures + supplementary regression')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR,
                        help='Directory of the four raw Olist CSV files; defaults to Olist datasets/ next to script')
    args = parser.parse_args()
    DATA_DIR = args.data_dir.resolve()
    for folder in (ROOT_OUT, OUT_DIR, FIG_DIR, REG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    print(f'Olist: reading raw CSVs from {DATA_DIR}', flush=True)
    raw, conversion_log = load_raw()
    audit = audit_raw(raw, conversion_log)
    save_csv(audit, 'data_audit.csv')
    eligible, rv, sample_log = construct_samples(raw)
    save_csv(sample_log, 'sample_construction.csv')
    rv_written = rv[rv.has_text].copy()
    rv_one = one_review_per_order(rv)
    assert len(rv) >= len(rv_written)

    # Rating and delivery analyses
    rating = analysis_a_rating_distribution(rv, rv_written)
    save_csv(rating, 'rating_distribution.csv')
    selection = (rating[(rating.grouping == 'Delivery x review timing') &
                        rating['sample'].isin(['All scored reviews','Written reviews only'])]
                 [['sample','group','n_reviews','negative_pct','mean_score']].copy())
    save_csv(selection,'written_review_selection.csv')
    timing = analysis_b_review_timing(rv)
    save_csv(timing, 'review_timing_summary.csv')
    severity = analysis_c_delay_severity(rv,rv_written)
    save_csv(severity, 'delay_severity_negative_ratings.csv')
    timing_severity = severity[severity['scheme']=='4 categories (Figure 2)'].copy()
    save_csv(timing_severity, 'review_timing_by_delay_severity.csv')

    # Portuguese review classification
    passed,total,failures = run_nlp_unit_tests()
    if failures:
        raise AssertionError('NLP unit tests failed: ' + '; '.join(failures))
    classified = classify_reviews(rv_written)
    prevalence = complaint_prevalence(classified)
    save_csv(prevalence, 'delivery_related_expression_prevalence.csv')
    text_cols = (['review_id','order_id','review_score','negative','delivery_status',
                  'timing','group4','severity4','days_vs_estimate','word_count',
                  'text','text_normalized'] + [f'flag_{c}' for c in CATEGORIES] +
                 ['flag_delivery_delay_nonreceipt','n_categories_flagged',
                  'any_service_concern_flag','delivery_topic_mention',
                  'delivery_mention_without_complaint','matched_rules'])
    save_csv(classified[text_cols], 'processed_written_reviews.csv')
    save_csv(nlp_rule_frequency(classified), 'nlp_rule_diagnostics.csv')
    save_csv(export_example_matches(classified), 'nlp_example_matches.csv')

    # Supplementary regression models
    models, vif, diag = run_regressions(rv,rv_one)
    for mid in ['N1','N2','N3','N4']:
        save_csv(models[models.model_id == mid], f'logistic_model_{mid}.csv', REG_DIR)
    save_csv(vif, 'regression_vif.csv', REG_DIR)
    save_csv(diag, 'regression_diagnostics.csv', REG_DIR)
    (REG_DIR / 'regression_summary.txt').write_text(
        supplementary_regression_summary(models, vif, diag), encoding='utf-8')

    # Preserve descriptive sensitivity checks separately, not as paper figures.
    assoc = association_tests({'All scored reviews':rv, 'One review per order':rv_one},
                              {'Written reviews':classified,
                               'One written review per order':one_review_per_order(classified)})
    save_csv(assoc, 'statistical_associations_supplementary.csv', REG_DIR)
    sensitivity = sensitivity_analyses(eligible,rv,rv_written,rv_one)
    save_csv(sensitivity, 'sensitivity_analyses.csv', REG_DIR)

   # Export research figures
    save_csv(figure1_rating_distribution(rating), 'Figure_1_data.csv')
    save_csv(figure2_delay_severity(severity), 'Figure_2_data.csv')
    save_csv(figure3_delivery_phrase_prevalence(prevalence), 'Figure_3_data.csv')
    ctx = {'eligible':eligible,'rv':rv,'clf':classified, 'rating':rating,
           'severity':severity, 'prevalence':prevalence, 'nlp_tests':(passed,total)}
    (ROOT_OUT / 'Research_Results_Summary.txt').write_text(
        main_results_summary(ctx),encoding='utf-8')
    figs = sorted(p.name for p in FIG_DIR.iterdir() if p.suffix in ('.pdf','.png'))
    pdfs = [x for x in figs if x.endswith('.pdf')]
    pngs = [x for x in figs if x.endswith('.png')]
    if len(pdfs)!=3 or len(pngs)!=3:
        raise RuntimeError(f'Expected exactly 3 PDF + 3 PNG figures, found: {figs}')
    if any('Regression' in f or 'Figure_4' in f for f in figs):
        raise RuntimeError('Regression visualization / Figure 4 is forbidden in research outputs')
    print('\n=== ANALYSIS COMPLETED ===', flush=True)
    print(f'Eligible delivered orders: {len(eligible):,}; scored reviews: {len(rv):,}; written reviews: {len(classified):,}')
    print(f'Rule unit tests: {passed}/{total}; three main figures: {", ".join(pdfs)}')
    print(f'Supplementary numerical regression models retained: N1, N2, N3, N4')
    print(f'Output folder: {ROOT_OUT}',flush=True)


if __name__ == '__main__':
    main()
