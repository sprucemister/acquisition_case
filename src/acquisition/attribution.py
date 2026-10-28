"""Digital mix tables and the direct-attribution baseline (one row per cookie)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .load_clean import Data
from .utils import banner, nonholdout_sendout_dates, save_table, to_day


# --- Digital interactions ----------------------------------------------------------
def link_sales_to_digital(sales: pd.DataFrame, digital: pd.DataFrame) -> pd.DataFrame:
    """Sale -> its converting interaction -> cookie -> every interaction of that cookie."""
    out = (
        sales.merge(
            digital[["interaction_id", "cookie_id"]],
            left_on="digital_interaction_id",
            right_on="interaction_id",
            how="inner",
        )
        .drop(columns="interaction_id")
        .merge(digital, on="cookie_id", how="left")
    )
    out["di_days_before_sale"] = (
        ((out["sale_timestamp_utc"] - out["timestamp_utc"]) / pd.Timedelta(days=1))
        .round(0)
        .astype("Int64")
    )
    return out


def within_lookback(df: pd.DataFrame, days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    return df[df["di_days_before_sale"] <= days]


def is_inorganic(df: pd.DataFrame) -> pd.Series:
    return df["channel"].notna() & ~df["channel"].isin(config.NON_INORGANIC_CHANNELS)


def pivot_counts(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """One row per sale, one column per category, interaction counts as values."""
    return df.pivot_table(
        index="sale_id",
        columns=dim,
        values="interaction_id",
        aggfunc="count",
        fill_value=0,
        dropna=False,
    ).reset_index()


def count_table(df_ix: pd.DataFrame, dim: str, scale: float = 1.0) -> pd.DataFrame:
    """Interactions per category with shares (fractions) and a scaled version."""
    t = (
        df_ix[dim]
        .fillna("Unknown")
        .value_counts()
        .rename_axis(dim.title())
        .reset_index(name="Interactions")
    )
    t["Share of Total"] = t["Interactions"] / t["Interactions"].sum()
    t["Scaled Interactions"] = (t["Interactions"] * scale).round().astype(int)
    t["Scaled Share of Total"] = t["Scaled Interactions"] / t["Scaled Interactions"].sum()
    return t


def digital_mix(data: Data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all interactions linked to sales within the lookback, one row per inorganic interaction)."""
    linked = link_sales_to_digital(data.sales, data.digital)
    in_window = within_lookback(linked)
    df_ix = (
        in_window[is_inorganic(in_window)]
        .dropna(subset=["interaction_id"])
        .drop_duplicates("interaction_id")
    )
    inorganic_rows = in_window[is_inorganic(in_window)]
    for dim in config.MIX_DIMS:
        save_table(pivot_counts(inorganic_rows, dim), f"sale_by_{dim}_counts.csv", index=False)
        save_table(count_table(df_ix, dim), f"digital_mix_{dim}.csv", index=False)
    return in_window, df_ix


# --- Direct attribution baseline ----------------------------------------------------
def latest_touch_per_cookie(in_window: pd.DataFrame) -> pd.DataFrame:
    """One row per cookie (its latest interaction) plus a flag for any inorganic touch."""
    return (
        in_window.assign(_inorg=is_inorganic(in_window))
        .assign(has_inorganic_digital=lambda d: d.groupby("cookie_id")["_inorg"].transform("any"))
        .drop(columns="_inorg")
        .sort_values(["timestamp_utc", "sale_timestamp_utc"])
        .drop_duplicates("cookie_id", keep="last")
    )


def build_baseline(data: Data, in_window: pd.DataFrame) -> pd.DataFrame:
    """The 'direct attribution baseline': one row per cookie with sendout info and a case."""
    latest = latest_touch_per_cookie(in_window)

    nonholdout = data.dm[data.dm["sendout_group"] != "Holdout"]
    latest_sendout_per_address = nonholdout.sort_values("sendout_date").drop_duplicates(
        "address_id", keep="last"
    )
    base = latest.merge(latest_sendout_per_address, on="address_id", how="left")

    # days_since_sendout is a calendar lookup, independent of the person.
    sendouts = nonholdout_sendout_dates(data.dm).to_frame()
    left = base.assign(sale_date=to_day(base["sale_timestamp_utc"]), _order=range(len(base)))
    base = (
        pd.merge_asof(
            left.sort_values("sale_date"),
            sendouts,
            left_on="sale_date",
            right_on="last_sendout_date",
            direction="backward",
        )
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    base["days_since_sendout"] = (base["sale_date"] - base["last_sendout_date"]).dt.days.astype(
        "Int64"
    )

    has_digital = base["has_inorganic_digital"] == True  # noqa: E712
    has_sendout = base["sendout_group"].notna()  # address matched a non-holdout sendout
    base["attribution_case"] = np.select(
        [has_digital & has_sendout, has_digital & ~has_sendout, ~has_digital & has_sendout],
        ["Both", "Digital only", "Sendout only"],
        default="No Direct Attribution",
    )
    return base


def case_summary(base: pd.DataFrame) -> pd.DataFrame:
    s = (
        base.groupby("attribution_case")
        .agg(
            sales=("sale_id", "nunique"), median_days_since_sendout=("days_since_sendout", "median")
        )
        .reindex(config.CASE_ORDER)
        .fillna({"sales": 0})
    )
    s["% of sales"] = s["sales"] / s["sales"].sum()
    total = pd.DataFrame(
        {
            "sales": [s["sales"].sum()],
            "median_days_since_sendout": [base["days_since_sendout"].median()],
            "% of sales": [1.0],
        },
        index=["Total"],
    )
    return pd.concat([s, total])


def print_case_summary(summary: pd.DataFrame, title: str = "Direct attribution summary") -> None:
    banner(title)
    print(
        summary.rename_axis("Attribution case").to_string(
            formatters={
                "sales": "{:,.0f}".format,
                "median_days_since_sendout": "{:.1f}".format,
                "% of sales": "{:.1%}".format,
            }
        )
    )


def reconcile(data: Data, base: pd.DataFrame) -> None:
    """Show how many sales the one-row-per-cookie baseline leaves out."""
    n_total = data.sales["sale_id"].nunique()
    n_kept = base["sale_id"].nunique()
    print(
        f"Sales in baseline: {n_kept:,} of {n_total:,} "
        f"({n_total - n_kept:,} not in the baseline: no linked cookie, or not the "
        f"latest interaction row for their cookie)"
    )
