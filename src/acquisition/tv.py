"""TV: active-day calendar, TV on/off tests and the monthly view."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from . import config
from .load_clean import Data
from .utils import banner, save_fig, save_table, to_day, year_dates


def build_tv_dates(tv_spots: pd.DataFrame) -> pd.DataFrame:
    """Every day of the year with a flag: spot aired that day or the day before."""
    spot_days = to_day(tv_spots["air_time"]).drop_duplicates()
    active = set(spot_days) | set(spot_days + pd.Timedelta(days=1))
    out = pd.DataFrame({"date": year_dates()})
    out["tv_spot_active"] = out["date"].isin(active)
    return out


def attach_tv_flag(base: pd.DataFrame, tv_dates: pd.DataFrame) -> pd.DataFrame:
    return (
        base.assign(sale_date=to_day(base["sale_timestamp_utc"]))
        .drop(columns="tv_spot_active", errors="ignore")
        .merge(
            tv_dates[["date", "tv_spot_active"]], left_on="sale_date", right_on="date", how="left"
        )
        .drop(columns="date")
    )


def daily_sales_frame(sales: pd.DataFrame, tv_dates: pd.DataFrame) -> pd.DataFrame:
    """Unique sales per calendar day (zero-sales days included) with the TV flag and month."""
    per_day = (
        sales.assign(date=to_day(sales["sale_timestamp_utc"])).groupby("date")["sale_id"].nunique()
    )
    return tv_dates.assign(
        tv_spot_active=lambda d: d["tv_spot_active"].astype(bool),
        sales=lambda d: d["date"].map(per_day).fillna(0).astype(int),
        month=lambda d: d["date"].dt.month,
    )


def welch_test(df_daily: pd.DataFrame) -> dict:
    on = df_daily.loc[df_daily["tv_spot_active"], "sales"]
    off = df_daily.loc[~df_daily["tv_spot_active"], "sales"]
    t, p = stats.ttest_ind(on, off, equal_var=False)  # Welch: no equal-variance assumption
    banner("TV active vs. not: daily sales (Welch t-test)")
    print(df_daily.groupby("tv_spot_active")["sales"].agg(["count", "mean", "std"]).round(2))
    print(f"\nDifference in means: {on.mean() - off.mean():.2f} sales/day")
    print(f"Welch t = {t:.3f}, p = {p:.4f}")
    return {"diff": on.mean() - off.mean(), "t": t, "p": p}


def tv_baseline_sales(tv_dates: pd.DataFrame) -> int:
    n_days = int(tv_dates["tv_spot_active"].sum())
    total = n_days * config.TV_SALES_PER_ACTIVE_DAY
    print(
        f"Baseline TV attribution in {config.YEAR}: {n_days} active days x "
        f"{config.TV_SALES_PER_ACTIVE_DAY} sales/day = {total:,} sales"
    )
    return total


def ztest_by_case(base: pd.DataFrame) -> pd.DataFrame:
    """For each attribution case: does its share of sales differ on TV-active vs. other days?"""
    df_test = (
        base.dropna(subset=["tv_spot_active"])
        .drop_duplicates("sale_id")
        .assign(tv_spot_active=lambda d: d["tv_spot_active"].astype(bool))
    )
    active = df_test["tv_spot_active"]
    n_on, n_off = int(active.sum()), int((~active).sum())
    banner("TV active vs. not: share of sales by attribution case (z-test)")
    print(df_test["attribution_case"].value_counts())
    print(f"Total sales: TV on = {n_on:,}, TV off = {n_off:,}\n")

    rows = []
    for case in config.CASE_ORDER:
        hit = df_test["attribution_case"] == case
        x_on, x_off = int((hit & active).sum()), int((hit & ~active).sum())
        if n_on == 0 or n_off == 0 or (x_on + x_off) in (0, n_on + n_off):
            print(f"Skipping '{case}': no variation to test (x_on={x_on}, x_off={x_off}).")
            continue
        z, p = proportions_ztest([x_on, x_off], [n_on, n_off])
        lo, hi = confint_proportions_2indep(x_on, n_on, x_off, n_off, method="wald", compare="diff")
        rows.append(
            {
                "Attribution case": case,
                "Sales (TV on)": x_on,
                "Sales (TV off)": x_off,
                "% of sales (TV on)": x_on / n_on,
                "% of sales (TV off)": x_off / n_off,
                "Diff (pp)": (x_on / n_on - x_off / n_off) * 100,
                "95% CI low (pp)": lo * 100,
                "95% CI high (pp)": hi * 100,
                "z": z,
                "p-value": p,
            }
        )
    out = pd.DataFrame(rows).set_index("Attribution case")
    print(
        out.to_string(
            formatters={
                "Sales (TV on)": "{:,.0f}".format,
                "Sales (TV off)": "{:,.0f}".format,
                "% of sales (TV on)": "{:.1%}".format,
                "% of sales (TV off)": "{:.1%}".format,
                "Diff (pp)": "{:+.2f}".format,
                "95% CI low (pp)": "{:+.2f}".format,
                "95% CI high (pp)": "{:+.2f}".format,
                "z": "{:.3f}".format,
                "p-value": "{:.4f}".format,
            }
        )
    )
    save_table(out, "tv_ztest_by_case.csv")
    return out


def sales_by_month(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Average daily sales by month and TV status (table + chart)."""
    month_names = {m: pd.Timestamp(config.YEAR, m, 1).strftime("%b") for m in range(1, 13)}
    g = df_daily.groupby(["month", "tv_spot_active"])["sales"]
    t = g.agg(days="size", total="sum", avg="mean", sd="std").unstack("tv_spot_active")

    def col(name: str, flag: bool) -> pd.Series:
        return t[(name, flag)] if (name, flag) in t.columns else pd.Series(np.nan, index=t.index)

    summary = pd.DataFrame(
        {
            "Days (TV on)": col("days", True),
            "Days (TV off)": col("days", False),
            "Total sales (TV on)": col("total", True),
            "Total sales (TV off)": col("total", False),
            "Avg/day (TV on)": col("avg", True),
            "Avg/day (TV off)": col("avg", False),
        }
    ).reindex(range(1, 13))
    summary["Lift (sales/day)"] = summary["Avg/day (TV on)"] - summary["Avg/day (TV off)"]
    summary["Lift (%)"] = summary["Lift (sales/day)"] / summary["Avg/day (TV off)"]

    def month_p(m: int) -> float:
        d = df_daily[df_daily["month"] == m]
        on, off = d.loc[d["tv_spot_active"], "sales"], d.loc[~d["tv_spot_active"], "sales"]
        return (
            np.nan
            if len(on) < 2 or len(off) < 2
            else stats.ttest_ind(on, off, equal_var=False).pvalue
        )

    summary["p-value"] = [month_p(m) for m in summary.index]
    summary.index = [month_names[m] for m in summary.index]
    summary = summary.fillna(
        {"Days (TV on)": 0, "Days (TV off)": 0, "Total sales (TV on)": 0, "Total sales (TV off)": 0}
    )
    banner("Sales by month and TV status")
    print(
        summary.rename_axis("Month").to_string(
            formatters={
                "Days (TV on)": "{:,.0f}".format,
                "Days (TV off)": "{:,.0f}".format,
                "Total sales (TV on)": "{:,.0f}".format,
                "Total sales (TV off)": "{:,.0f}".format,
                "Avg/day (TV on)": "{:.1f}".format,
                "Avg/day (TV off)": "{:.1f}".format,
                "Lift (sales/day)": "{:+.1f}".format,
                "Lift (%)": "{:+.1%}".format,
                "p-value": "{:.3f}".format,
            }
        )
    )
    save_table(summary, "sales_by_month_and_tv.csv")

    fig, ax = plt.subplots(figsize=(14, 6))
    x, w = np.arange(12), 0.38
    b_off = ax.bar(x - w / 2, summary["Avg/day (TV off)"], w, color="#9db4c0", label="No TV spot")
    b_on = ax.bar(x + w / 2, summary["Avg/day (TV on)"], w, color="#e4572e", label="TV spot active")
    ax.bar_label(b_off, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b_on, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{m}\n(off {int(summary.loc[m, 'Days (TV off)'])} d / on {int(summary.loc[m, 'Days (TV on)'])} d)"
            for m in summary.index
        ],
        fontsize=8,
    )
    ax.set_ylabel("Average daily sales")
    ax.set_title(
        f"Average Daily Sales by Month and TV Status, {config.YEAR}",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "sales_by_month_and_tv.png")
    return summary


def run(data: Data, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs all TV analyses. Returns (tv_dates, baseline with tv flag, daily sales frame)."""
    tv_dates = build_tv_dates(data.tv_spots)
    base = attach_tv_flag(base, tv_dates)
    df_daily = daily_sales_frame(data.sales, tv_dates)
    welch_test(df_daily)
    tv_baseline_sales(tv_dates)
    ztest_by_case(base)
    sales_by_month(df_daily)
    return tv_dates, base, df_daily
