"""Reallocate the excess of 'Both' sales to 'Sendout only'.

Idea: shortly after a sendout the 'Both' count is inflated by sendout-driven sales that
happen to have a digital touch too. Late in the cycle (days 24-27) the sendout effect has
faded, so the 'Both' level there is a flat baseline (per season). On earlier days the
excess over that baseline is credited to 'Sendout only'. The total number of sales is
conserved.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config
from .attribution import count_table
from .config import CASE_ORDER, SEASON_ORDER
from .load_clean import Data
from .utils import (
    banner,
    make_calendar,
    nonholdout_sendout_dates,
    save_fig,
    save_table,
    season_of,
    to_day,
    weighted_median,
)

NAVY = "#000080"
ORANGE = "#e85d2a"


def prepare_sales(base: pd.DataFrame) -> pd.DataFrame:
    """One row per sale with sale date and season."""
    df = base.drop_duplicates("sale_id").copy()
    df["sale_date"] = to_day(df["sale_timestamp_utc"])
    df["season"] = season_of(df["sale_date"])
    return df


def season_baselines(df_s: pd.DataFrame, cal: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flat 'Both' level per season: mean daily 'Both' sales on calendar days 24-27.

    Returns (baseline per season, daily 'Both' count indexed by sale date).
    """
    n_both = df_s[df_s["attribution_case"] == "Both"].groupby("sale_date").size()
    cal = cal.assign(
        season=season_of(cal["date"]),
        both_sales=lambda d: n_both.reindex(d["date"], fill_value=0).to_numpy(),
    )
    lo, hi = config.BASELINE_DAYS
    in_base = cal["days_since_sendout"].between(lo, hi)
    tbl = (
        cal[in_base]
        .groupby("season")["both_sales"]
        .agg(baseline="mean", days="size")
        .reindex(SEASON_ORDER)
    )
    tbl["days"] = tbl["days"].fillna(0).astype(int)
    if tbl["baseline"].isna().any():
        raise ValueError("A season has no days in the baseline window; widen BASELINE_DAYS.")

    banner(f"Baseline 'Both' sales per day (days {lo}-{hi}), by season")
    print(tbl.to_string(formatters={"baseline": "{:.2f}".format, "days": "{:,.0f}".format}))
    save_table(tbl, "both_baseline_by_season.csv")
    return tbl["baseline"], n_both


def reallocate(df_s: pd.DataFrame, baseline: pd.Series, n_both: pd.Series) -> pd.DataFrame:
    """Split each 'Both' sale into a kept share and a moved-to-'Sendout only' share.

    keep = season baseline / that day's 'Both' count (capped at 1), only before day
    REALLOC_BEFORE of the cycle. Weights sum to 1 per sale, so the total is conserved.
    """
    is_both = df_s["attribution_case"] == "Both"
    realloc_day = df_s["days_since_sendout"] < config.REALLOC_BEFORE  # NaN -> False
    day_total = df_s["sale_date"].map(n_both)
    day_base = df_s["season"].map(baseline)
    keep = (day_base / day_total).clip(upper=1.0).fillna(1.0)
    keep = keep.where(is_both & realloc_day, 1.0)

    kept = df_s.assign(
        attribution_case_new=df_s["attribution_case"], weight=keep.where(is_both, 1.0)
    )
    moved = (
        df_s[is_both]
        .assign(attribution_case_new="Sendout only", weight=(1 - keep[is_both]))
        .loc[lambda d: d["weight"] > 0]
    )
    return pd.concat([kept, moved], ignore_index=True)


def summarise(
    df: pd.DataFrame, case_col: str = "attribution_case", weight: str = "weight"
) -> pd.DataFrame:
    rows = {}
    for case in CASE_ORDER:
        d = df[df[case_col] == case]
        rows[case] = {
            "sales": d[weight].sum(),
            "median_days_since_sendout": weighted_median(d["days_since_sendout"], d[weight]),
        }
    out = pd.DataFrame(rows).T
    out["% of sales"] = out["sales"] / out["sales"].sum()
    return out


def plot_allocation(df_s: pd.DataFrame, realloc: pd.DataFrame, sendouts: pd.Series) -> None:
    """Original 'Both' sales per day, stacked: stays 'Both' (navy) / moved to 'Sendout only'."""
    all_days = pd.date_range(f"{config.YEAR}-01-01", f"{config.YEAR}-12-31", freq="D")
    split = (
        realloc[realloc["attribution_case"] == "Both"]  # original case
        .groupby(["sale_date", "attribution_case_new"])["weight"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["Both", "Sendout only"], fill_value=0)
        .reindex(all_days, fill_value=0)
    )
    orig = (
        df_s[df_s["attribution_case"] == "Both"]
        .groupby("sale_date")
        .size()
        .reindex(all_days, fill_value=0)
    )
    assert (split.sum(axis=1) - orig).abs().max() < 1e-6, "Stacks don't match original Both counts"

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(all_days, split["Both"], width=1.0, color=NAVY, label="Stays 'Both' (Digital + Sendout)")
    ax.bar(
        all_days,
        split["Sendout only"],
        width=1.0,
        color=ORANGE,
        bottom=split["Both"],
        label="Moved to 'Sendout only'",
    )
    for d in sendouts:
        ax.axvline(d, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_xlim(all_days[0], all_days[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Original 'Both' sales per day")
    ax.set_title(
        f"Original 'Both' Sales by Day, Split by Model Allocation, {config.YEAR}",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=0.8))
    labels.append("Sendout")
    ax.legend(handles, labels, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, f"both_sales_allocation_by_day_{config.YEAR}.png")

    tot = split.sum()
    print(f"Original 'Both' sales: {tot.sum():,.1f}")
    print(f"  stays Both:        {tot['Both']:,.1f} ({tot['Both'] / tot.sum():.1%})")
    print(
        f"  moved to Sendout:  {tot['Sendout only']:,.1f} ({tot['Sendout only'] / tot.sum():.1%})"
    )


def run(
    data: Data, base: pd.DataFrame, tv_dates: pd.DataFrame, df_ix: pd.DataFrame
) -> pd.DataFrame:
    sendouts = nonholdout_sendout_dates(data.dm)
    cal = make_calendar(tv_dates, sendouts)
    df_s = prepare_sales(base)
    baseline, n_both = season_baselines(df_s, cal)
    realloc = reallocate(df_s, baseline, n_both)

    # The reallocation must not create or destroy sales.
    total_before, total_after = float(len(df_s)), float(realloc["weight"].sum())
    assert np.isclose(total_before, total_after), (total_before, total_after)

    before = summarise(df_s.assign(weight=1.0))
    after = summarise(realloc, case_col="attribution_case_new")
    change = pd.DataFrame(
        {
            "before": before["sales"],
            "after (season's baseline)": after["sales"],
            "change": after["sales"] - before["sales"],
        }
    )
    change["change %"] = change["change"] / change["before"]
    banner("Attribution cases: before vs. after reallocation")
    for title, t in [("BEFORE", before), ("AFTER (season's baseline)", after)]:
        print(title)
        print(
            t.to_string(
                formatters={
                    "sales": "{:,.0f}".format,
                    "median_days_since_sendout": "{:.1f}".format,
                    "% of sales": "{:.1%}".format,
                }
            ),
            "\n",
        )
    print("CHANGE")
    print(
        change.to_string(
            formatters={
                "before": "{:,.0f}".format,
                "after (season's baseline)": "{:,.0f}".format,
                "change": "{:+,.0f}".format,
                "change %": "{:+.1%}".format,
            }
        )
    )
    save_table(before, "cases_before.csv")
    save_table(after, "cases_after.csv")
    save_table(change, "cases_change.csv")

    # Digital interactions scale with the digital-attributed share of sales.
    d_before = before.loc[["Digital only", "Both"], "sales"].sum()
    d_after = after.loc["Digital only", "sales"] + after.loc["Both", "sales"]
    scale = d_after / d_before
    banner(f"Digital scale after reallocation = {scale:.3f}")
    for dim in config.MIX_DIMS:
        t = count_table(df_ix, dim, scale=scale)
        save_table(t, f"digital_mix_scaled_{dim}.csv", index=False)
        print(
            t.head(8).to_string(
                index=False,
                formatters={
                    "Share of Total": "{:.1%}".format,
                    "Scaled Share of Total": "{:.1%}".format,
                },
            ),
            "\n",
        )

    plot_allocation(df_s, realloc, sendouts)
    return change
