"""Exploratory charts and the digital-mix comparison."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from . import config
from .config import CASE_COLORS, CASE_ORDER
from .load_clean import Data
from .utils import (
    banner,
    make_calendar,
    nonholdout_sendout_dates,
    ns,
    save_fig,
    save_table,
    to_day,
    year_dates,
)


# --- 1. Daily sales for the whole year ------------------------------------------------
def plot_daily_sales(df_daily: pd.DataFrame, sendouts: pd.Series) -> None:
    colors = df_daily["tv_spot_active"].map({True: "#e4572e", False: "#9db4c0"})
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(df_daily["date"], df_daily["sales"], width=1.0, color=colors)
    for d in sendouts:
        ax.axvline(d, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(f"Daily Sales, {config.YEAR}", fontsize=14, fontweight="bold", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total sales")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp(f"{config.YEAR}-01-01"), pd.Timestamp(f"{config.YEAR}-12-31"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(color="#e4572e", label="TV spot active"),
            Patch(color="#9db4c0", label="No TV spot"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Sendout"),
        ],
        frameon=False,
        loc="upper left",
        ncol=3,
    )
    fig.tight_layout()
    save_fig(fig, "daily_sales_2025.png")


# --- 2. People receiving a sendout each month --------------------------------------------
def plot_monthly_recipients(dm: pd.DataFrame) -> None:
    nonholdout = dm.loc[dm["sendout_group"] != "Holdout"].assign(
        month=lambda d: d["sendout_date"].dt.month
    )
    monthly = (
        nonholdout.groupby("month")
        .agg(people=("address_id", "nunique"), rows=("address_id", "size"))
        .reindex(range(1, 13), fill_value=0)
    )
    monthly.index = [pd.Timestamp(config.YEAR, m, 1).strftime("%b") for m in monthly.index]
    banner("People receiving a non-holdout sendout, by month")
    print(monthly.to_string(formatters={"people": "{:,.0f}".format, "rows": "{:,.0f}".format}))
    save_table(monthly, "people_receiving_sendout_by_month.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(12)
    bars = ax.bar(x, monthly["people"], color="#4c78a8")
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in monthly["people"]], fontsize=9, padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly.index)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of People")
    ax.set_title(
        "Number of People Receiving DM Sendout", fontsize=14, fontweight="bold", loc="left"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "people_receiving_sendout_by_month.png")


# --- 3. Digital interactions per channel over the year ---------------------------------------
def plot_digital_by_channel(digital: pd.DataFrame) -> None:
    ix = digital.assign(
        date=lambda d: to_day(d["timestamp_utc"]), channel=lambda d: d["channel"].fillna("Unknown")
    )
    days = year_dates()
    channels = ix["channel"].value_counts().index.tolist()
    daily = (
        ix.groupby(["date", "channel"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=days, columns=channels, fill_value=0)
    )
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(18, 6))
    for i, ch in enumerate(channels):
        ax.plot(
            daily.index,
            daily[ch],
            color=cmap(i % 10),
            linewidth=1.8,
            label=f"{ch} (total {daily[ch].sum():,})",
        )
    ax.set_xlim(days[0], days[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Digital interactions per day")
    ax.set_title(
        f"Daily Digital Interactions by Channel, {config.YEAR}",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "digital_interactions_daily_by_channel.png")
    print(
        f"Interactions on the {config.YEAR} calendar: {int(daily.values.sum()):,} of {len(ix):,} in the table"
    )


# --- 4. Average daily sales by attribution case and days since sendout ----------------------------
def plot_days_since_sendout(
    base: pd.DataFrame, tv_dates: pd.DataFrame, sendouts: pd.Series
) -> None:
    max_lag = 26
    df = base.drop_duplicates("sale_id").assign(sale_date=lambda d: to_day(d["sale_timestamp_utc"]))
    cal = make_calendar(tv_dates, sendouts, max_lag=max_lag)
    days = list(range(max_lag + 1))

    daily = (
        df.groupby(["sale_date", "attribution_case"])
        .size()
        .unstack(fill_value=0)
        .reindex(cal["date"], fill_value=0)
        .reindex(columns=CASE_ORDER, fill_value=0)
        .reset_index()
        .merge(cal[["date", "days_since_sendout", "tv_spot_active"]], on="date")
    )

    def style_x(ax) -> None:
        ax.set_xlim(0, max_lag)
        ax.xaxis.set_major_locator(MultipleLocator(3))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel("Days since sendout")
        ax.spines[["top", "right"]].set_visible(False)

    avg = daily.groupby("days_since_sendout")[CASE_ORDER].mean().reindex(days)
    save_table(avg, "avg_daily_sales_by_case_and_day.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    for case in CASE_ORDER:
        ax.plot(
            days,
            avg[case],
            color=CASE_COLORS[case],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=case,
        )
    style_x(ax)
    ax.set_ylabel("Average daily sales")
    ax.set_title(
        "Daily Sales by Day After a Sendout, by Attribution Case",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "sendout_daily_sales_by_attribution_case_daily.png")

    panels = [(True, "TV spot active"), (False, "No TV spot")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, (tv_flag, title) in zip(axes, panels):
        avg_tv = (
            daily[daily["tv_spot_active"] == tv_flag]
            .groupby("days_since_sendout")[CASE_ORDER]
            .mean()
            .reindex(days)
        )
        for case in CASE_ORDER:
            ax.plot(
                days,
                avg_tv[case],
                color=CASE_COLORS[case],
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=case,
            )
        style_x(ax)
        ax.set_title(
            f"{title} ({int((cal['tv_spot_active'] == tv_flag).sum())} days)",
            fontsize=13,
            fontweight="bold",
            loc="left",
        )
    axes[0].set_ylabel("Average daily sales")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Daily Sales by Day After a Sendout, by Attribution Case and TV Status",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save_fig(fig, "sendout_daily_sales_by_attribution_case_daily_by_tv.png")


# --- 5. Time to response after a sendout ('Sendout only' case) --------------------------------------
def plot_sendout_hazard(base: pd.DataFrame, data: Data) -> None:
    max_day, min_at_risk, smooth_days = (
        config.SURVIVAL_MAX_DAY,
        config.SURVIVAL_MIN_AT_RISK,
        config.SURVIVAL_SMOOTH_DAYS,
    )
    ex = (
        data.dm.loc[data.dm["sendout_group"].isin(["Variant A", "Variant B"])]
        .assign(sendout_date=lambda d: ns(d["sendout_date"]))
        .sort_values(["address_id", "sendout_date"])
        .reset_index(drop=True)
        .rename_axis("exposure_id")
        .reset_index()
    )
    ex["next_sendout"] = ex.groupby("address_id")["sendout_date"].shift(-1)

    so_ids = base.loc[base["attribution_case"] == "Sendout only", "sale_id"].unique()
    sales = (
        data.sales.loc[
            data.sales["sale_id"].isin(so_ids), ["sale_id", "address_id", "sale_timestamp_utc"]
        ]
        .dropna(subset=["address_id"])
        .assign(sale_date=lambda d: to_day(d["sale_timestamp_utc"]))
        .sort_values("sale_date")
    )
    end_date = to_day(data.sales["sale_timestamp_utc"]).max()
    sale_exp = pd.merge_asof(
        sales,
        ex[["exposure_id", "address_id", "sendout_date"]].sort_values("sendout_date"),
        left_on="sale_date",
        right_on="sendout_date",
        by="address_id",
        direction="backward",
    ).dropna(subset=["exposure_id"])
    sale_exp["t"] = (sale_exp["sale_date"] - sale_exp["sendout_date"]).dt.days
    first_resp = sale_exp.groupby("exposure_id")["t"].min()

    # Duration and event flag per exposure; non-responders censored at the next sendout / end of data.
    ex["event"] = ex["exposure_id"].isin(first_resp.index)
    cutoff = end_date + pd.Timedelta(days=1)
    cens_end = ex["next_sendout"].fillna(cutoff).clip(upper=cutoff)
    ex["duration"] = (
        np.where(
            ex["event"], ex["exposure_id"].map(first_resp), (cens_end - ex["sendout_date"]).dt.days
        )
        .astype(int)
        .clip(min=0)
    )
    ex["event"] = ex["event"] & (ex["duration"] <= max_day)
    ex["duration"] = ex["duration"].clip(upper=max_day)

    T = ex["duration"].to_numpy()
    events = np.bincount(T[ex["event"].to_numpy()], minlength=max_day + 1)
    at_risk = np.bincount(T, minlength=max_day + 1)[::-1].cumsum()[::-1]
    hazard = np.divide(events, at_risk, out=np.zeros(max_day + 1), where=at_risk > 0)
    df_h = pd.DataFrame(
        {"day": np.arange(max_day + 1), "events": events, "at_risk": at_risk, "hazard": hazard}
    )
    df_h["smooth"] = df_h["hazard"].rolling(smooth_days, center=True, min_periods=1).mean()
    save_table(df_h, "sendout_only_response_hazard.csv", index=False)

    ok = df_h["at_risk"] >= min_at_risk
    last_ok = ok[ok].index.max() if ok.any() else -1
    solid, dash = df_h.loc[:last_ok], df_h.loc[last_ok:]
    color = "#e4572e"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_h["day"], df_h["hazard"], color=color, alpha=0.2, linewidth=1, label="Daily (raw)")
    ax.plot(
        solid["day"],
        solid["smooth"],
        color=color,
        linewidth=2.5,
        label=f"{smooth_days}-day smoothed",
    )
    ax.plot(
        dash["day"],
        dash["smooth"],
        color=color,
        linewidth=1.5,
        linestyle="--",
        label=f"Fewer than {min_at_risk:,} exposures at risk",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=3))
    ax.set_xlim(0, max_day)
    ax.set_xlabel("Days since sendout")
    ax.set_ylabel("Responses per exposure still at risk, per day")
    ax.set_title(
        "Daily Response Rate After a Mailer, 'Sendout only' Case",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "sendout_only_response_hazard.png")
    print(f"Hazard: {len(ex):,} exposures, {int(events.sum()):,} responders within {max_day} days.")


# --- 6. Digital mix: early vs. late in the sendout cycle ---------------------------------------------
def digital_mix_by_window(digital: pd.DataFrame, sendouts: pd.Series) -> dict[str, pd.DataFrame]:
    windows = config.MIX_WINDOWS
    w_early, w_late = list(windows)
    w_colors = {w_early: "#e4572e", w_late: "#4c78a8"}

    ix = digital.assign(date=lambda d: to_day(d["timestamp_utc"])).sort_values("date")
    ix = pd.merge_asof(
        ix, sendouts.to_frame(), left_on="date", right_on="last_sendout_date", direction="backward"
    )
    ix["days_since_sendout"] = (ix["date"] - ix["last_sendout_date"]).dt.days
    ix["window"] = ""
    for name, (lo_d, hi_d) in windows.items():
        ix.loc[ix["days_since_sendout"].between(lo_d, hi_d), "window"] = name
    ix = ix[ix["window"] != ""]
    banner("Digital mix: " + " vs. ".join(windows))
    print("Interactions per window:")
    print(ix["window"].value_counts().reindex([w_early, w_late]).to_string(), "\n")

    results: dict[str, pd.DataFrame] = {}
    for dim in config.MIX_DIMS:
        counts = pd.crosstab(ix[dim].fillna("Unknown"), ix["window"]).reindex(
            columns=[w_early, w_late], fill_value=0
        )
        totals = counts.sum()
        shares = counts / totals
        chi2, p, _, _ = stats.chi2_contingency(counts)
        cramers_v = np.sqrt(chi2 / totals.sum())  # two windows -> min(r, c) - 1 = 1

        tbl = pd.DataFrame(
            {
                f"Share {w_early}": shares[w_early],
                f"Share {w_late}": shares[w_late],
                "Diff (pp)": (shares[w_late] - shares[w_early]) * 100,
            }
        )
        tbl["p (category)"] = [
            proportions_ztest(
                [counts.loc[c, w_early], counts.loc[c, w_late]], [totals[w_early], totals[w_late]]
            )[1]
            if counts.loc[c].min() > 0
            else np.nan
            for c in counts.index
        ]
        tbl = tbl.sort_values(f"Share {w_early}", ascending=False)
        results[dim] = tbl
        save_table(tbl, f"digital_mix_windows_{dim}.csv")
        print(f"=== {dim.upper()} ===  chi-square p = {p:.4g}, Cramér's V = {cramers_v:.3f}")
        print(
            tbl.to_string(
                formatters={
                    f"Share {w_early}": "{:.1%}".format,
                    f"Share {w_late}": "{:.1%}".format,
                    "Diff (pp)": "{:+.1f}".format,
                    "p (category)": "{:.4f}".format,
                }
            ),
            "\n",
        )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, dim in zip(axes, config.MIX_DIMS):
        tbl = results[dim].iloc[::-1]  # largest category on top
        y, h = np.arange(len(tbl)), 0.4
        for i, w in enumerate([w_early, w_late]):
            b = ax.barh(y + (i - 0.5) * h, tbl[f"Share {w}"], height=h, color=w_colors[w], label=w)
            ax.bar_label(b, labels=[f"{v:.0%}" for v in tbl[f"Share {w}"]], fontsize=8, padding=2)
        ax.set_yticks(y)
        ax.set_yticklabels(tbl.index, fontsize=9)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(dim.title(), fontsize=13, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].set_xlabel("Share of the window's digital interactions")
    fig.suptitle(
        "Digital Interaction Mix: " + " vs. ".join(windows) + " After a Sendout",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save_fig(fig, "digital_mix_days2_8_vs_21_27.png")
    return results


def run(data: Data, base: pd.DataFrame, tv_dates: pd.DataFrame, df_daily: pd.DataFrame) -> None:
    sendouts = nonholdout_sendout_dates(data.dm)
    plot_daily_sales(df_daily, sendouts)
    plot_monthly_recipients(data.dm)
    plot_digital_by_channel(data.digital)
    plot_days_since_sendout(base, tv_dates, sendouts)
    plot_sendout_hazard(base, data)
    digital_mix_by_window(data.digital, sendouts)
