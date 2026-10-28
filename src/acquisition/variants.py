"""Direct mail: holdout vs. Variant A vs. Variant B."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from . import config
from .config import GROUP_COLORS, GROUPS, RESPONSE_WINDOW_DAYS
from .load_clean import Data
from .utils import banner, ns, save_fig, save_table, to_day


def build_exposures(dm: pd.DataFrame) -> pd.DataFrame:
    """One row per address per sendout."""
    return (
        dm.loc[dm["sendout_group"].isin(GROUPS)]
        .assign(sendout_date=lambda d: ns(d["sendout_date"]))
        .reset_index(drop=True)
        .rename_axis("exposure_id")
        .reset_index()
    )


def link_sales_to_exposures(sales: pd.DataFrame, ex: pd.DataFrame, window: int) -> pd.DataFrame:
    """Each sale goes to the address's most recent sendout on or before it (within `window` days)."""
    s = (
        sales[["sale_id", "address_id", "sale_timestamp_utc"]]
        .dropna(subset=["address_id"])
        .assign(sale_date=lambda d: to_day(d["sale_timestamp_utc"]))
        .sort_values("sale_date")
    )
    linked = pd.merge_asof(
        s,
        ex[["exposure_id", "address_id", "sendout_group", "sendout_date"]].sort_values(
            "sendout_date"
        ),
        left_on="sale_date",
        right_on="sendout_date",
        by="address_id",
        direction="backward",
    )
    linked["lag"] = (linked["sale_date"] - linked["sendout_date"]).dt.days
    linked = linked[linked["lag"].between(0, window)].copy()
    linked["lag"] = linked["lag"].astype(int)
    return linked


def run(data: Data, tv_dates: pd.DataFrame) -> pd.DataFrame:
    window = RESPONSE_WINDOW_DAYS
    ex = build_exposures(data.dm)
    n = ex["sendout_group"].value_counts().reindex(GROUPS)
    sale_exp = link_sales_to_exposures(data.sales, ex, window)

    # --- Conversion rate per group (exposures with at least one sale) ---
    conv = sale_exp.groupby("sendout_group")["exposure_id"].nunique().reindex(GROUPS).fillna(0)
    rate = conv / n
    lo, hi = proportion_confint(conv, n, method="wilson")

    banner("Mailer groups: two-proportion z-tests")
    rows = []
    for a, b in [("Variant A", "Holdout"), ("Variant B", "Holdout"), ("Variant A", "Variant B")]:
        z, p = proportions_ztest([conv[a], conv[b]], [n[a], n[b]])
        print(f"  {a} vs {b}: diff = {(rate[a] - rate[b]) * 100:+.2f} pp, z = {z:.2f}, p = {p:.4f}")
        rows.append(
            {"comparison": f"{a} vs {b}", "diff_pp": (rate[a] - rate[b]) * 100, "z": z, "p": p}
        )
    out = pd.DataFrame(
        {"exposures": n, "responders": conv, "conversion_rate": rate, "ci_low": lo, "ci_high": hi}
    )
    save_table(out, "mailer_conversion.csv")
    save_table(pd.DataFrame(rows), "mailer_ztests.csv", index=False)

    # --- Chart 1: conversion rate with 95% CI ---
    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(GROUPS))
    bars = ax.bar(
        x,
        rate.to_numpy(),
        color=[GROUP_COLORS[g] for g in GROUPS],
        yerr=[(rate - lo).to_numpy(), (hi - rate).to_numpy()],
        capsize=5,
        error_kw={"elinewidth": 1.2},
    )
    ax.bar_label(bars, labels=[f"{v:.2%}" for v in rate], padding=14, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={int(n[g]):,})" for g in GROUPS])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylabel(f"% of sendouts with a sale within {window} days")
    ax.set_title(
        "Conversion Rate by Mailer Group (95% CI)", fontsize=14, fontweight="bold", loc="left"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "sendout_variant_conversion_rate.png")

    # --- Chart 2: daily sales per 1,000 addresses by days since sendout ---
    lags = np.arange(config.SHOWN_LAG_DAYS + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    for g in GROUPS:
        daily = (
            sale_exp[sale_exp["sendout_group"] == g]["lag"]
            .value_counts()
            .reindex(lags, fill_value=0)
        )
        ax.plot(
            lags,
            daily.sort_index() / n[g] * 1000,
            color=GROUP_COLORS[g],
            linewidth=2.5,
            marker="o",
            markersize=4,
            linestyle="--" if g == "Holdout" else "-",
            label=g,
        )
    ax.set_xlim(0, config.SHOWN_LAG_DAYS)
    ax.set_xlabel("Days since sendout")
    ax.set_ylabel("Daily sales per 1,000 addresses")
    ax.set_title("Daily Sales by Day After a Sendout", fontsize=14, fontweight="bold", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "sendout_variant_daily_sales.png")

    # --- Chart 3: sales rate by TV status and mailer group (per 1,000 exposure-days) ---
    plot_by_tv(ex, sale_exp, tv_dates, window)
    return out


def plot_by_tv(
    ex: pd.DataFrame, sale_exp: pd.DataFrame, tv_dates: pd.DataFrame, window: int
) -> None:
    tv_by_date = (
        tv_dates.assign(date=lambda d: ns(d["date"]))
        .set_index("date")["tv_spot_active"]
        .astype(bool)
    )
    tv_states = [False, True]
    tv_labels = ["No TV spot", "TV spot active"]

    tagged = sale_exp.assign(tv_spot_active=sale_exp["sale_date"].map(tv_by_date))
    sales_cnt = (
        tagged.dropna(subset=["tv_spot_active"])
        .groupby(["sendout_group", "tv_spot_active"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=GROUPS, columns=tv_states, fill_value=0)
    )

    # Exposure-days: each exposure contributes one day for every lag 0..window.
    per_date = ex.groupby(["sendout_group", "sendout_date"]).size().rename("n_exp").reset_index()
    expo_days = pd.concat(
        [
            per_date.assign(date=per_date["sendout_date"] + pd.Timedelta(days=k))
            for k in range(window + 1)
        ],
        ignore_index=True,
    )
    expo_days["tv_spot_active"] = expo_days["date"].map(
        tv_by_date
    )  # NaN outside the year -> dropped
    exposure_days = (
        expo_days.dropna(subset=["tv_spot_active"])
        .groupby(["sendout_group", "tv_spot_active"])["n_exp"]
        .sum()
        .unstack(fill_value=0)
        .reindex(index=GROUPS, columns=tv_states, fill_value=0)
    )

    rate_tv = sales_cnt / exposure_days
    ci_lo = pd.DataFrame(index=GROUPS, columns=tv_states, dtype=float)
    ci_hi = ci_lo.copy()
    for g in GROUPS:
        for t in tv_states:
            ci_lo.loc[g, t], ci_hi.loc[g, t] = proportion_confint(
                sales_cnt.loc[g, t], exposure_days.loc[g, t], method="wilson"
            )

    banner("Sales per 1,000 exposure-days by TV status")
    print((rate_tv * 1000).rename(columns=dict(zip(tv_states, tv_labels))).round(3).to_string())
    save_table(
        (rate_tv * 1000).rename(columns=dict(zip(tv_states, tv_labels))), "mailer_rate_by_tv.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(tv_states))
    w = 0.8 / len(GROUPS)
    for i, g in enumerate(GROUPS):
        vals = np.array([rate_tv.loc[g, t] for t in tv_states]) * 1000
        err_lo = vals - np.array([ci_lo.loc[g, t] for t in tv_states]) * 1000
        err_hi = np.array([ci_hi.loc[g, t] for t in tv_states]) * 1000 - vals
        pos = x + (i - (len(GROUPS) - 1) / 2) * w
        bars = ax.bar(
            pos,
            vals,
            width=w,
            color=GROUP_COLORS[g],
            label=g,
            yerr=[err_lo, err_hi],
            capsize=4,
            error_kw={"elinewidth": 1.1},
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{lab}\n({int(exposure_days[t].sum()):,} exposure-days)"
            for lab, t in zip(tv_labels, tv_states)
        ]
    )
    ax.set_ylabel(f"Sales per 1,000 exposure-days (within {window} days of a sendout)")
    ax.set_title(
        "Sales Rate by TV Status and Mailer Group (95% CI)",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "sendout_variant_sales_rate_by_tv.png")
