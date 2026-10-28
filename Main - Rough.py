from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

BASE = "data/raw/"

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# Read Data ----
# ---------------------------------------------------------------

df_sales_raw = pd.read_csv(
    BASE + "sales.csv",
    dtype={"sale_timestamp_utc": str},
).assign(
    sale_timestamp_utc=lambda d: pd.to_datetime(d["sale_timestamp_utc"], utc=True)
)

df_digital_interactions_raw = pd.read_csv(
    BASE + "digital_interactions.csv"
).assign(
    timestamp_utc=lambda d: pd.to_datetime(d["timestamp_utc"], utc=True)
)

df_tv_spots = pd.read_csv(BASE + "tv_spots.csv").assign(
    air_time=lambda d: pd.to_datetime(d["air_time"], utc=True)
)

df_dm_sendouts_raw = pd.read_csv(BASE + "dm_sendouts.csv").assign(
    sendout_date=lambda d: pd.to_datetime(d["sendout_date"]).dt.date
)

# ---------------------------------------------------------------
# Clean Data ----
# ---------------------------------------------------------------

def clean_key(s: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and uppercase. Nulls stay null."""
    return s.astype("string").str.strip().str.upper()

## Sales ---------------------------------------------------------

# Normalize keys first so dedup works on the cleaned values
df_sales = df_sales_raw.copy()
for col in ["digital_interaction_id", "address_id"]:
    df_sales[col] = clean_key(df_sales[col])

# Drop exact duplicates, then for each sale_id keep the earliest timestamp
df_sales = df_sales.drop_duplicates()
min_ts = df_sales.groupby("sale_id")["sale_timestamp_utc"].transform("min")
df_sales = df_sales[df_sales["sale_timestamp_utc"] == min_ts].reset_index(drop=True)

## Digital -------------------------------------------------------

df_digital_interactions = df_digital_interactions_raw.copy()

# interaction_id is the table's own key (sales.digital_interaction_id points to it)
for col in ["interaction_id", "cookie_id"]:
    df_digital_interactions[col] = clean_key(df_digital_interactions[col])

df_digital_interactions["campaign"] = np.select(
    [
        df_digital_interactions["campaign"].isna() & (df_digital_interactions["channel"] == "Direct"),
        df_digital_interactions["campaign"].isna() & (df_digital_interactions["channel"] == "Organic Search"),
    ],
    ["Direct", "Organic Search"],
    default=df_digital_interactions["campaign"],
)

df_digital_interactions["platform"] = df_digital_interactions["platform"].fillna("Direct")

## TV Spots ------------------------------------------------------
# Nothing to clean, df_tv_spots is final as read

## Direct Mail Sendouts -------------------------------------------
grp = df_dm_sendouts_raw["sendout_group"].astype("string").str.upper()

df_dm_sendouts = df_dm_sendouts_raw.copy()
df_dm_sendouts["address_id"] = clean_key(df_dm_sendouts["address_id"])
df_dm_sendouts["sendout_group"] = np.select(
    [
        grp.str.contains("HOLD", na=False),
        grp.str.contains("CONT", na=False),
        grp.str.contains("B", na=False),
    ],
    ["Holdout", "Holdout", "Variant B"],
    default="Variant A",
)

# Check: key columns are clean
for name, s in {
    "sales.digital_interaction_id": df_sales["digital_interaction_id"],
    "sales.address_id": df_sales["address_id"],
    "digital.interaction_id": df_digital_interactions["interaction_id"],
    "digital.cookie_id": df_digital_interactions["cookie_id"],
    "dm.address_id": df_dm_sendouts["address_id"],
}.items():
    s = s.dropna()
    print(f"{name}: {s.nunique():,} unique, "
          f"{(s != s.str.strip()).sum()} with stray whitespace, "
          f"{(s != s.str.upper()).sum()} not uppercase")


# ---------------------------------------------------------------
# Baselines ----
# ---------------------------------------------------------------

## Digital Interactions Mix ----

# Join sales & digital interactions
df_all_digital_with_sales = (
    df_sales
    .merge(
        df_digital_interactions[["interaction_id", "cookie_id"]],
        left_on="digital_interaction_id",
        right_on="interaction_id",
        how="inner",
    )
    .drop(columns="interaction_id")  
    .merge(df_digital_interactions, on="cookie_id", how="left")
)

# Calculate days between interaction and sale
df_all_digital_with_sales["di_days_before_sale"] = (
    (df_all_digital_with_sales["sale_timestamp_utc"]
     - df_all_digital_with_sales["timestamp_utc"])
    / pd.Timedelta(days=1)
).round(0).astype("Int64")

# Drop sales older than 7 days
df_all_digital_with_sales_in7days=df_all_digital_with_sales[df_all_digital_with_sales['di_days_before_sale']<=7]

# Isolate inorganic/organic sales
df_all_inorganic_digital_with_sales_in7days=df_all_digital_with_sales_in7days[~df_all_digital_with_sales_in7days["channel"].isin(["Direct", "Organic Search","Paid Search"])]

# Make pivot counts for each channel/platform/campaign
def pivot_counts(df, dim):
    return (
        df.pivot_table(
            index="sale_id",
            columns=dim,
            values="interaction_id",
            aggfunc="count",
            fill_value=0,
            dropna=False,
        )
        .reset_index()
    )

df_joined_by_channel  = pivot_counts(df_all_inorganic_digital_with_sales_in7days, "channel")
df_joined_by_platform = pivot_counts(df_all_inorganic_digital_with_sales_in7days, "platform")
df_joined_by_campaign = pivot_counts(df_all_inorganic_digital_with_sales_in7days, "campaign")



## Digital Interactions Mix - Make Tables ----

# One row per interaction
df_ix = (
    df_all_inorganic_digital_with_sales_in7days
    .dropna(subset=["interaction_id"])
    .drop_duplicates("interaction_id")
)


def count_table_for_digital_channels_with_scale(df, dim, scale=1.0):
    t = (
        df[dim].fillna("Unknown")
        .value_counts()
        .rename_axis(dim.title())
        .reset_index(name="Interactions")
    )
    t["% of Total"] = t["Interactions"] / t["Interactions"].sum()*100
    t["Scaled Interactions"] = (t["Interactions"] * scale).round().astype(int)
    t["Scaled % of Total"] = t["Scaled Interactions"] / t["Scaled Interactions"].sum()*100
    return t

count_table_for_digital_channels_with_scale(df_ix, "channel", 1)
count_table_for_digital_channels_with_scale(df_ix, "platform", 1)
count_table_for_digital_channels_with_scale(df_ix, "campaign", 1)


## Direct Attribution ----

# One row per cookie: the one with the latest interaction
df_latest_di_with_sales_in7days = (
    df_all_digital_with_sales_in7days # df_all_inorganic_digital_with_sales_in7days
    .sort_values(["timestamp_utc", "sale_timestamp_utc"])
    .drop_duplicates("cookie_id", keep="last")
)
df_latest_di_with_sales_in7days = (
    df_all_digital_with_sales_in7days
    .assign(
        _is_inorganic=lambda d: d["channel"].notna()
            & ~d["channel"].isin(["Direct", "Organic Search","Paid Search"])
    )
    .assign(
        has_inorganic_digital=lambda d: d.groupby("cookie_id")["_is_inorganic"].transform("any")
    )
    .drop(columns="_is_inorganic")
    .sort_values(["timestamp_utc", "sale_timestamp_utc"])
    .drop_duplicates("cookie_id", keep="last")
)

# Non-holdout sendouts, latest per address (for address-level fields like variant)
df_dm_nonholdout_all = (
    df_dm_sendouts[df_dm_sendouts["sendout_group"] != "Holdout"]
    .assign(sendout_date=lambda d: pd.to_datetime(d["sendout_date"]))
)

df_dm_nonholdout_latest = (
    df_dm_nonholdout_all
    .sort_values("sendout_date")
    .drop_duplicates("address_id", keep="last")
)

# Left join the address-level sendout info
df_direct_attribution_baseline = df_latest_di_with_sales_in7days.merge(
    df_dm_nonholdout_latest, on="address_id", how="left"
)

# days_since_sendout: calendar lookup, independent of the person
sendout_dates = (
    df_dm_nonholdout_all["sendout_date"]
    .astype("datetime64[ns]")
    .drop_duplicates()
    .sort_values()
    .rename("last_sendout_date")
    .to_frame()
)

left = (
    df_direct_attribution_baseline
    .assign(
        sale_date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"),
        _order=lambda d: range(len(d)),
    )
    .sort_values("sale_date")
)

df_direct_attribution_baseline = (
    pd.merge_asof(
        left, sendout_dates,
        left_on="sale_date", right_on="last_sendout_date",
        direction="backward",
    )
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

df_direct_attribution_baseline["days_since_sendout"] = (
    (df_direct_attribution_baseline["sale_date"]
     - df_direct_attribution_baseline["last_sendout_date"])
    .dt.days.astype("Int64")
)


## Direct Attribution Summary Table ----

# Add attribution_case to the baseline dataframe
has_digital = df_direct_attribution_baseline["has_inorganic_digital"] == True
has_sendout = df_direct_attribution_baseline["sendout_group"].notna()   # address matched a non-holdout sendout

df_direct_attribution_baseline["attribution_case"] = np.select(
    [has_digital & has_sendout, has_digital & ~has_sendout, ~has_digital & has_sendout],
    ["Both", "Digital only", "Sendout only"],
    default="No Direct Attribution",
)

# Summary table
order = ["Digital only", "Sendout only", "Both", "No Direct Attribution"]

summary = (
    df_direct_attribution_baseline.groupby("attribution_case")
    .agg(
        sales=("sale_id", "nunique"),
        median_days_since_sendout=("days_since_sendout", "median"),
    )
    .reindex(order)
    .fillna({"sales": 0})
)
summary["% of sales"] = summary["sales"] / summary["sales"].sum()

total = pd.DataFrame(
    {
        "sales": [summary["sales"].sum()],
        "median_days_since_sendout": [df_direct_attribution_baseline["days_since_sendout"].median()],
        "% of sales": [1.0],
    },
    index=["Total"],
)
summary = pd.concat([summary, total])

print(
    summary.rename_axis("Attribution case")
    .to_string(formatters={
        "sales": "{:,.0f}".format,
        "median_days_since_sendout": "{:.1f}".format,
        "% of sales": "{:.1%}".format,
    })
)

## TV Spots - Wrangling ----

spot_days = (
    pd.to_datetime(df_tv_spots["air_time"])
    .dt.tz_localize(None)
    .dt.normalize()
    .drop_duplicates()
)
active_days = set(spot_days) | set(spot_days + pd.Timedelta(days=1))

df_tv_spot_dates_2025 = pd.DataFrame(
    {"date": pd.date_range("2025-01-01", "2025-12-31", freq="D")}
)
df_tv_spot_dates_2025["tv_spot_active"] = df_tv_spot_dates_2025["date"].isin(active_days)

df_direct_attribution_baseline = (
    df_direct_attribution_baseline
    .assign(
        sale_date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    )
    .drop(columns="tv_spot_active", errors="ignore")
    .merge(
        df_tv_spot_dates_2025[["date", "tv_spot_active"]],
        left_on="sale_date", right_on="date", how="left",
    )
    .drop(columns="date")
)

## TV Spots - Mean Differences & t-Test ----

# Daily sales count, including days with zero sales
daily_sales = (
    df_sales
    .assign(date=lambda d: d["sale_timestamp_utc"].dt.tz_localize(None).dt.normalize())
    .groupby("date")["sale_id"].nunique()
)

df_daily = df_tv_spot_dates_2025.assign(
    sales=lambda d: d["date"].map(daily_sales).fillna(0).astype(int)
)

tv_on  = df_daily.loc[df_daily["tv_spot_active"],  "sales"]
tv_off = df_daily.loc[~df_daily["tv_spot_active"], "sales"]

# Welch's t-test (doesn't assume equal variances)
t_stat, p_value = stats.ttest_ind(tv_on, tv_off, equal_var=False)

print(df_daily.groupby("tv_spot_active")["sales"].agg(["count", "mean", "std"]).round(2))
print(f"\nDifference in means: {tv_on.mean() - tv_off.mean():.2f} sales/day")
print(f"Welch t = {t_stat:.3f}, p = {p_value:.4f}")


baseline_tv=df_tv_spot_dates_2025[df_tv_spot_dates_2025['tv_spot_active']==True].shape[0]*15
print('Baseline TV Spot Attribution in 2025 = '+str(baseline_tv)+' sales')


## TV Spots - Z Test Proportions ----

CASES = df_direct_attribution_baseline['attribution_case'].unique()

# One row per sale, known TV flag, only the cases being compared
df_test = (
    df_direct_attribution_baseline
    .dropna(subset=["tv_spot_active"])
    .drop_duplicates("sale_id")
    .assign(tv_spot_active=lambda d: d["tv_spot_active"].astype(bool))
    .loc[lambda d: d["attribution_case"].isin(CASES)]
)

active = df_test["tv_spot_active"]
n_on, n_off = int(active.sum()), int((~active).sum())

print(df_test["attribution_case"].value_counts())
print(f"Total sales: TV on = {n_on:,}, TV off = {n_off:,}\n")

rows = []
for case in CASES:
    hit = df_test["attribution_case"] == case
    x_on, x_off = int((hit & active).sum()), int((hit & ~active).sum())

    if n_on == 0 or n_off == 0 or (x_on + x_off) == 0 or (x_on + x_off) == (n_on + n_off):
        print(f"Skipping '{case}': no variation to test (x_on={x_on}, x_off={x_off}).")
        continue

    z, p = proportions_ztest([x_on, x_off], [n_on, n_off])
    lo, hi = confint_proportions_2indep(x_on, n_on, x_off, n_off, method="wald", compare="diff")

    rows.append({
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
    })

summary_ztest = pd.DataFrame(rows).set_index("Attribution case")
print(summary_ztest.to_string(formatters={
    "Sales (TV on)": "{:,.0f}".format,
    "Sales (TV off)": "{:,.0f}".format,
    "% of sales (TV on)": "{:.1%}".format,
    "% of sales (TV off)": "{:.1%}".format,
    "Diff (pp)": "{:+.2f}".format,
    "95% CI low (pp)": "{:+.2f}".format,
    "95% CI high (pp)": "{:+.2f}".format,
    "z": "{:.3f}".format,
    "p-value": "{:.4f}".format,
}))


# ------------------------------------------------------------------
# Exploration ----
# ------------------------------------------------------------------

## Plot Direct Attributions for 365 Days ----

# Daily sales on the full 2025 calendar (zero-sales days included)
daily_sales = (
    df_sales
    .assign(date=lambda d: d["sale_timestamp_utc"].dt.tz_localize(None).dt.normalize())
    .groupby("date")["sale_id"].nunique()
)
df_daily = df_tv_spot_dates_2025.assign(
    sales=lambda d: d["date"].map(daily_sales).fillna(0).astype(int)
)

# Distinct sendout dates (non-holdout)
sendout_dates = (
    pd.to_datetime(
        df_dm_sendouts.loc[df_dm_sendouts["sendout_group"] != "Holdout", "sendout_date"]
    )
    .drop_duplicates()
    .sort_values()
)

colors = df_daily["tv_spot_active"].map({True: "#e4572e", False: "#9db4c0"})

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(df_daily["date"], df_daily["sales"], width=1.0, color=colors)

for d in sendout_dates:
    ax.axvline(d, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

ax.set_title("Daily Sales, 2025", fontsize=14, fontweight="bold", loc="left")
ax.set_xlabel("Date")
ax.set_ylabel("Total sales")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
ax.spines[["top", "right"]].set_visible(False)

ax.legend(handles=[
    Patch(color="#e4572e", label="TV spot active"),
    Patch(color="#9db4c0", label="No TV spot"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Sendout"),
], frameon=False, loc="upper left", ncol=3)

plt.tight_layout()
plt.savefig(OUT / "daily_sales_2025.png", dpi=200)
plt.show()


## A vs B Variant ----

WINDOW = 30   # days after a sendout in which a sale counts as a response
GROUPS = ["Holdout", "Variant A", "Variant B"]
G_COLORS = {"Holdout": "#9d9d9d", "Variant A": "#4c78a8", "Variant B": "#e4572e"}

# Exposures: one row per address per sendout
ex = (
    df_dm_sendouts
    .loc[lambda d: d["sendout_group"].isin(GROUPS)]
    .assign(sendout_date=lambda d: pd.to_datetime(d["sendout_date"]).astype("datetime64[ns]"))
    .reset_index(drop=True)
    .rename_axis("exposure_id").reset_index()
)
n = ex["sendout_group"].value_counts().reindex(GROUPS)

# Each sale goes to the address's most recent sendout on or before the sale
sales = (
    df_sales[["sale_id", "address_id", "sale_timestamp_utc"]]
    .dropna(subset=["address_id"])
    .assign(sale_date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
    .sort_values("sale_date")
)
sale_exp = pd.merge_asof(
    sales,
    ex[["exposure_id", "address_id", "sendout_group", "sendout_date"]].sort_values("sendout_date"),
    left_on="sale_date", right_on="sendout_date", by="address_id", direction="backward",
)
sale_exp["lag"] = (sale_exp["sale_date"] - sale_exp["sendout_date"]).dt.days
sale_exp = sale_exp[sale_exp["lag"].between(0, WINDOW)].copy()
sale_exp["lag"] = sale_exp["lag"].astype(int)

# Conversion rate per group (exposures with at least one sale)
conv = sale_exp.groupby("sendout_group")["exposure_id"].nunique().reindex(GROUPS).fillna(0)
rate = conv / n
lo, hi = proportion_confint(conv, n, method="wilson")

print("Two-proportion z-tests")
for a, b in [("Variant A", "Holdout"), ("Variant B", "Holdout"), ("Variant A", "Variant B")]:
    z, p = proportions_ztest([conv[a], conv[b]], [n[a], n[b]])
    print(f"  {a} vs {b}: diff = {(rate[a]-rate[b])*100:+.2f} pp, z = {z:.2f}, p = {p:.4f}")

# Chart 1: Conversion rate with 95% CI
fig1, ax1 = plt.subplots(figsize=(7, 6))
x = np.arange(len(GROUPS))
bars = ax1.bar(x, rate.to_numpy(), color=[G_COLORS[g] for g in GROUPS],
               yerr=[(rate - lo).to_numpy(), (hi - rate).to_numpy()],
               capsize=5, error_kw={"elinewidth": 1.2})
ax1.bar_label(bars, labels=[f"{v:.2%}" for v in rate], padding=14, fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{g}\n(n={int(n[g]):,})" for g in GROUPS])
ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
ax1.set_ylabel(f"% of sendouts with a sale within {WINDOW} days")
ax1.set_title("Conversion Rate by Mailer Group (95% CI)",
              fontsize=14, fontweight="bold", loc="left")
ax1.spines[["top", "right"]].set_visible(False)

fig1.tight_layout()
fig1.savefig(OUT / "sendout_variant_conversion_rate.png", dpi=200)
plt.show()

# Chart 2: Daily sales per 1,000 addresses, by days since sendout
MAX_LAG = 27

fig2, ax2 = plt.subplots(figsize=(9, 6))
lags = np.arange(MAX_LAG + 1)
for g in GROUPS:
    daily = (sale_exp[sale_exp["sendout_group"] == g]["lag"]
             .value_counts().reindex(lags, fill_value=0).sort_index())
    ax2.plot(lags, daily / n[g] * 1000, color=G_COLORS[g], linewidth=2.5,
             marker="o", markersize=4,
             linestyle="--" if g == "Holdout" else "-", label=g)
ax2.set_xlim(0, MAX_LAG)
ax2.set_xlabel("Days since sendout")
ax2.set_ylabel("Daily sales per 1,000 addresses")
ax2.set_title("Daily Sales by Day After a Sendout",
              fontsize=14, fontweight="bold", loc="left")
ax2.legend(frameon=False)
ax2.spines[["top", "right"]].set_visible(False)

fig2.tight_layout()
fig2.savefig(OUT / "sendout_variant_daily_sales.png", dpi=200)
plt.show()

## Monthly Sendout Recipients ----

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Non-holdout sendouts, with calendar month
dm = (
    df_dm_sendouts
    .loc[lambda d: d["sendout_group"] != "Holdout"]
    .assign(sendout_date=lambda d: pd.to_datetime(d["sendout_date"]))
    .assign(month=lambda d: d["sendout_date"].dt.month)
)

# People who received a sendout, per month
monthly = (
    dm.groupby("month")
      .agg(people=("address_id", "nunique"), rows=("address_id", "size"))
      .reindex(range(1, 13), fill_value=0)
)
monthly.index = [pd.Timestamp(2025, m, 1).strftime("%b") for m in monthly.index]

print(monthly.to_string(formatters={"people": "{:,.0f}".format, "rows": "{:,.0f}".format}))

# Chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(12)
bars = ax.bar(x, monthly["people"], color="#4c78a8")
ax.bar_label(bars, labels=[f"{v:,.0f}" for v in monthly["people"]], fontsize=9, padding=3)

ax.set_xticks(x)
ax.set_xticklabels(monthly.index)
ax.set_xlabel("Month")
ax.set_ylabel("Number of People")
ax.set_title("Number of People Receiving DM Sendout",
             fontsize=14, fontweight="bold", loc="left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "people_receiving_sendout_by_month.png", dpi=200)
plt.show()

## DI Channels over the Calender ----

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# All digital interactions, by calendar date
ix = df_digital_interactions.assign(
    date=lambda d: d["timestamp_utc"].dt.tz_localize(None).dt.normalize(),
    channel=lambda d: d["channel"].fillna("Unknown"),
)

all_days = pd.date_range("2025-01-01", "2025-12-31", freq="D")

# One column per channel, ordered by total volume (largest first)
channels = ix["channel"].value_counts().index.tolist()
daily = (
    ix.groupby(["date", "channel"]).size()
      .unstack(fill_value=0)
      .reindex(index=all_days, columns=channels, fill_value=0)
)

# Chart
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(18, 6))
for i, ch in enumerate(channels):
    ax.plot(daily.index, daily[ch], color=cmap(i % 10), linewidth=1.8,
            label=f"{ch} (total {daily[ch].sum():,})")

ax.set_xlim(all_days[0], all_days[-1])
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlabel("Date")
ax.set_ylabel("Digital interactions per day")
ax.set_title("Daily Digital Interactions by Channel, 2025",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "digital_interactions_daily_by_channel.png", dpi=200)
plt.show()

print(daily.sum().astype(int).to_string())
print(f"\nTotal interactions on the 2025 calendar: {int(daily.values.sum()):,} "
      f"of {len(ix):,} in the table")



## A vs B Variant with TV ----

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# TV status by calendar date
tv_by_date = (
    df_tv_spot_dates_2025
    .assign(date=lambda d: pd.to_datetime(d["date"]).astype("datetime64[ns]"))
    .set_index("date")["tv_spot_active"].astype(bool)
)

# Sales (already within WINDOW days of a sendout), tagged with TV status of the sale date
sale_exp["tv_spot_active"] = sale_exp["sale_date"].map(tv_by_date)
sales_cnt = (
    sale_exp.dropna(subset=["tv_spot_active"])
    .groupby(["sendout_group", "tv_spot_active"]).size()
    .unstack(fill_value=0)
    .reindex(index=GROUPS, columns=[False, True], fill_value=0)
)

# Exposure-days: each exposure contributes one day per lag
ex_per_date = ex.groupby(["sendout_group", "sendout_date"]).size().rename("n_exp").reset_index()
expo_days = pd.concat(
    [ex_per_date.assign(date=ex_per_date["sendout_date"] + pd.Timedelta(days=k))
     for k in range(WINDOW + 1)],
    ignore_index=True,
)
expo_days["tv_spot_active"] = expo_days["date"].map(tv_by_date)   # NaN outside 2025 -> dropped
exposure_days = (
    expo_days.dropna(subset=["tv_spot_active"])
    .groupby(["sendout_group", "tv_spot_active"])["n_exp"].sum()
    .unstack(fill_value=0)
    .reindex(index=GROUPS, columns=[False, True], fill_value=0)
)

# Rate per 1,000 exposure-days with Wilson 95% CI
rate_tv = sales_cnt / exposure_days
ci_lo = pd.DataFrame(index=GROUPS, columns=[False, True], dtype=float)
ci_hi = ci_lo.copy()
for g in GROUPS:
    for t in [False, True]:
        lo_, hi_ = proportion_confint(sales_cnt.loc[g, t], exposure_days.loc[g, t], method="wilson")
        ci_lo.loc[g, t], ci_hi.loc[g, t] = lo_, hi_

print("Sales per 1,000 exposure-days")
print((rate_tv * 1000).rename(columns={False: "No TV spot", True: "TV spot active"}).round(3).to_string())
print("\nExposure-days")
print(exposure_days.rename(columns={False: "No TV spot", True: "TV spot active"}).to_string(
    formatters={False: "{:,.0f}".format, True: "{:,.0f}".format}))

# Clustered bars: x = TV status, bars = mailer group
fig, ax = plt.subplots(figsize=(9, 6))
tv_states = [False, True]
tv_labels = ["No TV spot", "TV spot active"]
x = np.arange(len(tv_states))
w = 0.8 / len(GROUPS)

for i, g in enumerate(GROUPS):
    vals = np.array([rate_tv.loc[g, t] for t in tv_states]) * 1000
    err_lo = vals - np.array([ci_lo.loc[g, t] for t in tv_states]) * 1000
    err_hi = np.array([ci_hi.loc[g, t] for t in tv_states]) * 1000 - vals
    pos = x + (i - (len(GROUPS) - 1) / 2) * w
    bars = ax.bar(pos, vals, width=w, color=G_COLORS[g], label=g,
                  yerr=[err_lo, err_hi], capsize=4, error_kw={"elinewidth": 1.1})
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=12)

ax.set_xticks(x)
ax.set_xticklabels([
    f"{lab}\n({int(exposure_days[t].sum()):,} exposure-days)" for lab, t in zip(tv_labels, tv_states)
])
ax.set_ylabel(f"Sales per 1,000 exposure-days (within {WINDOW} days of a sendout)")
ax.set_title("Sales Rate by TV Status and Mailer Group (95% CI)",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "sendout_variant_sales_rate_by_tv.png", dpi=200)
plt.show()


## Days Since DM Sendout ----

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

MAX_LAG = 26    # last day included
CASES = ["Digital only", "Sendout only", "Both", "No Direct Attribution"]
COLORS = {
    "Digital only":          "#4c78a8",
    "Sendout only":          "#59a14f",
    "Both":                  "#e4572e",
    "No Direct Attribution": "#9d9d9d",
}

# One row per sale, tagged with its attribution case
df = (
    df_direct_attribution_baseline
    .drop_duplicates("sale_id")
    .assign(sale_date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
)

# 2025 calendar with days_since_sendout and tv_spot_active
sendout_dates = (
    pd.to_datetime(
        df_dm_sendouts.loc[df_dm_sendouts["sendout_group"] != "Holdout", "sendout_date"]
    )
    .astype("datetime64[ns]").drop_duplicates().sort_values()
    .rename("last_sendout_date").to_frame()
)
calendar = pd.merge_asof(
    df_tv_spot_dates_2025[["date", "tv_spot_active"]]
        .astype({"date": "datetime64[ns]"}).sort_values("date"),
    sendout_dates, left_on="date", right_on="last_sendout_date", direction="backward",
)
calendar["days_since_sendout"] = (calendar["date"] - calendar["last_sendout_date"]).dt.days
calendar = calendar.dropna(subset=["days_since_sendout"])
calendar["days_since_sendout"] = calendar["days_since_sendout"].astype(int)
calendar["tv_spot_active"] = calendar["tv_spot_active"].astype(bool)
calendar = calendar[calendar["days_since_sendout"] <= MAX_LAG]

days = list(range(MAX_LAG + 1))

# Daily sales per case on every calendar day (zeros included)
daily = (
    df.groupby(["sale_date", "attribution_case"]).size()
      .unstack(fill_value=0)
      .reindex(calendar["date"], fill_value=0)
      .reindex(columns=CASES, fill_value=0)
      .reset_index()
      .merge(calendar[["date", "days_since_sendout", "tv_spot_active"]], on="date")
)

def style_x(ax):
    ax.set_xlim(0, MAX_LAG)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("Days since sendout")
    ax.spines[["top", "right"]].set_visible(False)

# Chart 1: all days
avg = daily.groupby("days_since_sendout")[CASES].mean().reindex(days)

fig, ax = plt.subplots(figsize=(10, 6))
for case in CASES:
    ax.plot(days, avg[case], color=COLORS[case], linewidth=2.2,
            marker="o", markersize=4, label=case)

style_x(ax)
ax.set_ylabel("Average daily sales")
ax.set_title("Daily Sales by Day After a Sendout, by Attribution Case",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(OUT / "sendout_daily_sales_by_attribution_case_daily.png", dpi=200)
plt.show()

# Chart 2: split by TV status
panels = [(True, "TV spot active"), (False, "No TV spot")]
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, (tv_flag, panel_title) in zip(axes, panels):
    d_tv = daily[daily["tv_spot_active"] == tv_flag]
    avg_tv = d_tv.groupby("days_since_sendout")[CASES].mean().reindex(days)
    n_days = int((calendar["tv_spot_active"] == tv_flag).sum())

    for case in CASES:
        ax.plot(days, avg_tv[case], color=COLORS[case], linewidth=2.2,
                marker="o", markersize=4, label=case)

    style_x(ax)
    ax.set_title(f"{panel_title} ({n_days} days)", fontsize=13, fontweight="bold", loc="left")

axes[0].set_ylabel("Average daily sales")
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("Daily Sales by Day After a Sendout, by Attribution Case and TV Status",
             fontsize=15, fontweight="bold", x=0.01, ha="left")

plt.tight_layout()
plt.savefig(OUT / "sendout_daily_sales_by_attribution_case_daily_by_tv.png", dpi=200)
plt.show()


## Survival Analysis for DM Sendout ----

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

MAX_DAY = 150         # x-axis cap
MIN_AT_RISK = 1000    # curve is dashed once fewer exposures than this are still being followed
SMOOTH_DAYS = 7       # rolling window for the smoothed line
COLOR = "#e4572e"

# Exposures: one row per address per non-holdout sendout (A and B pooled)
ex = (
    df_dm_sendouts
    .loc[lambda d: d["sendout_group"].isin(["Variant A", "Variant B"])]
    .assign(sendout_date=lambda d: pd.to_datetime(d["sendout_date"]).astype("datetime64[ns]"))
    .sort_values(["address_id", "sendout_date"])
    .reset_index(drop=True)
    .rename_axis("exposure_id").reset_index()
)
ex["next_sendout"] = ex.groupby("address_id")["sendout_date"].shift(-1)

# 'Sendout only' sales, each assigned to the address's latest sendout on or before the sale
so_ids = (df_direct_attribution_baseline
          .loc[lambda d: d["attribution_case"] == "Sendout only", "sale_id"].unique())
sales = (
    df_sales.loc[lambda d: d["sale_id"].isin(so_ids), ["sale_id", "address_id", "sale_timestamp_utc"]]
    .dropna(subset=["address_id"])
    .assign(sale_date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
    .sort_values("sale_date")
)
end_date = df_sales["sale_timestamp_utc"].dt.tz_localize(None).dt.normalize().max()

sale_exp = pd.merge_asof(
    sales,
    ex[["exposure_id", "address_id", "sendout_date"]].sort_values("sendout_date"),
    left_on="sale_date", right_on="sendout_date", by="address_id", direction="backward",
).dropna(subset=["exposure_id"])
sale_exp["t"] = (sale_exp["sale_date"] - sale_exp["sendout_date"]).dt.days

first_resp = sale_exp.groupby("exposure_id")["t"].min()

# Duration and event flag per exposure, censored at MAX_DAY
ex["event"] = ex["exposure_id"].isin(first_resp.index)
cens_end = ex["next_sendout"].fillna(end_date + pd.Timedelta(days=1)).clip(upper=end_date + pd.Timedelta(days=1))
ex["duration"] = np.where(
    ex["event"],
    ex["exposure_id"].map(first_resp),
    (cens_end - ex["sendout_date"]).dt.days,
).astype(int).clip(min=0)
ex["event"] = ex["event"] & (ex["duration"] <= MAX_DAY)
ex["duration"] = ex["duration"].clip(upper=MAX_DAY)

# Daily hazard: responders on day t / exposures still at risk on day t
T = ex["duration"].to_numpy()
events = np.bincount(T[ex["event"].to_numpy()], minlength=MAX_DAY + 1)
at_risk = np.bincount(T, minlength=MAX_DAY + 1)[::-1].cumsum()[::-1]
hazard = np.divide(events, at_risk, out=np.zeros(MAX_DAY + 1), where=at_risk > 0)

df_h = pd.DataFrame({"day": np.arange(MAX_DAY + 1), "events": events,
                     "at_risk": at_risk, "hazard": hazard})
df_h["smooth"] = df_h["hazard"].rolling(SMOOTH_DAYS, center=True, min_periods=1).mean()

# Chart
ok = df_h["at_risk"] >= MIN_AT_RISK
last_ok = ok[ok].index.max() if ok.any() else -1
solid, dash = df_h.loc[:last_ok], df_h.loc[last_ok:]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_h["day"], df_h["hazard"], color=COLOR, alpha=0.2, linewidth=1, label="Daily (raw)")
ax.plot(solid["day"], solid["smooth"], color=COLOR, linewidth=2.5,
        label=f"{SMOOTH_DAYS}-day smoothed")
ax.plot(dash["day"], dash["smooth"], color=COLOR, linewidth=1.5, linestyle="--",
        label=f"Fewer than {MIN_AT_RISK:,} exposures at risk")

ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=3))
ax.set_xlim(0, MAX_DAY)
ax.set_xlabel("Days since sendout")
ax.set_ylabel("Responses per exposure still at risk, per day")
ax.set_title("Daily Response Rate After a Mailer, 'Sendout only' Case",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "sendout_only_response_hazard.png", dpi=200)
plt.show()

print(f"Exposures: {len(ex):,}, responders within {MAX_DAY} days: {int(events.sum()):,}")
if ok.any():
    print(f"At-risk count stays above {MIN_AT_RISK:,} through day {last_ok}.")
    

## TV Sales by Month ----

# Daily sales on the full 2025 calendar, with month and TV status
daily_sales = (
    df_sales
    .assign(date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
    .groupby("date")["sale_id"].nunique()
)

df_daily = df_tv_spot_dates_2025.assign(
    date=lambda d: pd.to_datetime(d["date"]).astype("datetime64[ns]"),
    tv_spot_active=lambda d: d["tv_spot_active"].astype(bool),
).assign(
    sales=lambda d: d["date"].map(daily_sales).fillna(0).astype(int),
    month=lambda d: d["date"].dt.month,
)

MONTHS = {m: pd.Timestamp(2025, m, 1).strftime("%b") for m in range(1, 13)}

# Summary table: month x TV status
g = df_daily.groupby(["month", "tv_spot_active"])["sales"]
stats_tbl = g.agg(days="size", total="sum", avg="mean", sd="std").unstack("tv_spot_active")

summary = pd.DataFrame({
    "Days (TV on)":   stats_tbl[("days", True)],
    "Days (TV off)":  stats_tbl[("days", False)],
    "Total sales (TV on)":  stats_tbl[("total", True)],
    "Total sales (TV off)": stats_tbl[("total", False)],
    "Avg/day (TV on)":  stats_tbl[("avg", True)],
    "Avg/day (TV off)": stats_tbl[("avg", False)],
}).reindex(range(1, 13))
summary["Lift (sales/day)"] = summary["Avg/day (TV on)"] - summary["Avg/day (TV off)"]
summary["Lift (%)"] = summary["Lift (sales/day)"] / summary["Avg/day (TV off)"]

# Welch t-test per month (needs at least 2 days in each group)
def month_p(m):
    d = df_daily[df_daily["month"] == m]
    on, off = d.loc[d["tv_spot_active"], "sales"], d.loc[~d["tv_spot_active"], "sales"]
    if len(on) < 2 or len(off) < 2:
        return np.nan
    return stats.ttest_ind(on, off, equal_var=False).pvalue

summary["p-value"] = [month_p(m) for m in summary.index]
summary.index = [MONTHS[m] for m in summary.index]
summary = summary.fillna({"Days (TV on)": 0, "Days (TV off)": 0,
                          "Total sales (TV on)": 0, "Total sales (TV off)": 0})

print(summary.rename_axis("Month").to_string(formatters={
    "Days (TV on)": "{:,.0f}".format, "Days (TV off)": "{:,.0f}".format,
    "Total sales (TV on)": "{:,.0f}".format, "Total sales (TV off)": "{:,.0f}".format,
    "Avg/day (TV on)": "{:.1f}".format, "Avg/day (TV off)": "{:.1f}".format,
    "Lift (sales/day)": "{:+.1f}".format, "Lift (%)": "{:+.1%}".format,
    "p-value": "{:.3f}".format,
}))

# Chart: average daily sales by month, TV on vs. off
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(12)
w = 0.38
b_off = ax.bar(x - w / 2, summary["Avg/day (TV off)"], w, color="#9db4c0", label="No TV spot")
b_on  = ax.bar(x + w / 2, summary["Avg/day (TV on)"],  w, color="#e4572e", label="TV spot active")
ax.bar_label(b_off, fmt="%.1f", fontsize=8, padding=2)
ax.bar_label(b_on,  fmt="%.1f", fontsize=8, padding=2)

ax.set_xticks(x)
ax.set_xticklabels([
    f"{m}\n(off {int(summary.loc[m, 'Days (TV off)'])} d / on {int(summary.loc[m, 'Days (TV on)'])} d)"
    for m in summary.index
], fontsize=8)
ax.set_ylabel("Average daily sales")
ax.set_title("Average Daily Sales by Month and TV Status, 2025",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "sales_by_month_and_tv.png", dpi=200)
plt.show()


# -----------------------------------------------------------------
# Deep Dive Model for DI ----
# -----------------------------------------------------------------

## Make Model ----

BASELINE_DAYS = (24, 27)   # inclusive: days used to set the flat 'Both' level
REALLOC_BEFORE = 24        # days_since_sendout < 24 can have excess moved

order = ["Digital only", "Sendout only", "Both", "No Direct Attribution"]

# --- Rough seasons ---
SEASON_ORDER = ["May-Aug", "Sep-Oct", "Nov-Apr"]

def season_of(dates):
    m = pd.DatetimeIndex(dates).month
    return pd.Series(
        np.where((m >= 5) & (m <= 8), "May-Aug",
                 np.where((m >= 9) & (m <= 10), "Sep-Oct", "Nov-Apr")),
        index=getattr(dates, "index", None),
    )

# --- 2025 calendar with days_since_sendout and season ---
sendout_dates = (
    pd.to_datetime(
        df_dm_sendouts.loc[df_dm_sendouts["sendout_group"] != "Holdout", "sendout_date"]
    )
    .astype("datetime64[ns]").drop_duplicates().sort_values()
    .rename("last_sendout_date").to_frame()
)
calendar = pd.merge_asof(
    df_tv_spot_dates_2025[["date"]]
        .astype({"date": "datetime64[ns]"}).sort_values("date"),
    sendout_dates, left_on="date", right_on="last_sendout_date", direction="backward",
)
calendar["days_since_sendout"] = (calendar["date"] - calendar["last_sendout_date"]).dt.days
calendar = calendar.dropna(subset=["days_since_sendout"])
calendar["days_since_sendout"] = calendar["days_since_sendout"].astype(int)
calendar["season"] = season_of(calendar["date"]).to_numpy()

# --- One row per sale; season from the sale date ---
df_s = (
    df_direct_attribution_baseline
    .drop_duplicates("sale_id")
    .assign(
        sale_date=lambda d: d["sale_timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"),
    )
)
df_s["season"] = season_of(df_s["sale_date"]).to_numpy()

# --- Daily 'Both' sales ---
n_both = df_s[df_s["attribution_case"] == "Both"].groupby("sale_date").size()
calendar["both_sales"] = n_both.reindex(calendar["date"], fill_value=0).to_numpy()

# --- Three baselines: one per season ---
in_base = calendar["days_since_sendout"].between(*BASELINE_DAYS)
base_tbl = (
    calendar[in_base].groupby("season")["both_sales"]
    .agg(baseline="mean", days="size")
    .reindex(SEASON_ORDER)
)
base_tbl["days"] = base_tbl["days"].fillna(0).astype(int)
if base_tbl["baseline"].isna().any():
    raise ValueError("A season has no days in the baseline window; widen BASELINE_DAYS.")

# --- Table of the 3 thresholds ---
print(f"Baseline 'Both' sales per day (days {BASELINE_DAYS[0]}-{BASELINE_DAYS[1]}), by season:")
print(base_tbl.to_string(formatters={"baseline": "{:.2f}".format, "days": "{:,.0f}".format}), "\n")

baseline = base_tbl["baseline"]   # index: season

# --- Split each 'Both' sale into kept share and moved-to-Sendout-only share ---
is_both = df_s["attribution_case"] == "Both"
realloc_day = df_s["days_since_sendout"] < REALLOC_BEFORE      # NaN -> False

day_total = df_s["sale_date"].map(n_both)
day_base = df_s["season"].map(baseline)                        # each sale gets its own season's baseline
keep = (day_base / day_total).clip(upper=1.0).fillna(1.0)
keep = keep.where(is_both & realloc_day, 1.0)

kept_rows = df_s.assign(attribution_case_new=df_s["attribution_case"], weight=keep.where(is_both, 1.0))
moved_rows = (
    df_s[is_both]
    .assign(attribution_case_new="Sendout only", weight=(1 - keep[is_both]))
    .loc[lambda d: d["weight"] > 0]
)
df_realloc = pd.concat([kept_rows, moved_rows], ignore_index=True)



## Plot Model ----

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# --- Original 'Both' sales only, split by where the model put them ---
both_split = (
    df_realloc[df_realloc["attribution_case"] == "Both"]      # original case
    .groupby(["sale_date", "attribution_case_new"])["weight"].sum()
    .unstack(fill_value=0)
    .reindex(columns=["Both", "Sendout only"], fill_value=0)
)

# --- Full 2025 calendar, zero-sales days included ---
all_days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
both_split = both_split.reindex(all_days, fill_value=0)

# Check: stacked height per day should equal the original 'Both' count
orig = (df_s[df_s["attribution_case"] == "Both"].groupby("sale_date").size()
        .reindex(all_days, fill_value=0))
assert abs((both_split.sum(axis=1) - orig).abs().max()) < 1e-6, "Stacks don't match original Both counts"

# --- Chart ---
NAVY, ORANGE = "#000080", "#e85d2a"

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(all_days, both_split["Both"], width=1.0, color=NAVY,
       label="Stays 'Both' (Digital + Sendout)")
ax.bar(all_days, both_split["Sendout only"], width=1.0, color=ORANGE,
       bottom=both_split["Both"], label="Moved to 'Sendout only'")

ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlabel("Date")
ax.set_ylabel("Original 'Both' sales per day")
ax.set_title("Original 'Both' Sales by Day, Split by Model Allocation, 2025",
             fontsize=14, fontweight="bold", loc="left")
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# --- Vertical lines at each sendout (drawn behind the bars) ---
for d in sendout_dates["last_sendout_date"]:
    ax.axvline(d, color="black", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)

# Rebuild the legend so the sendout line appears in it
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], color="black", linestyle="--", linewidth=0.8))
labels.append("Sendout")
ax.legend(handles, labels, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig(OUT / "both_sales_allocation_by_day_2025.png", dpi=200)
plt.show()

tot = both_split.sum()
print(f"Original 'Both' sales: {tot.sum():,.1f}")
print(f"  stays Both:        {tot['Both']:,.1f} ({tot['Both']/tot.sum():.1%})")
print(f"  moved to Sendout:  {tot['Sendout only']:,.1f} ({tot['Sendout only']/tot.sum():.1%})")

## Before & After Tables ----

## Summary helper (weighted so fractional sales work) 
def weighted_median(values, weights):
    m = values.notna()
    v, w = values[m].to_numpy(), weights[m].to_numpy()
    if len(v) == 0:
        return np.nan
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, 0.5 * cw[-1])]

def make_summary(d, case_col, w_col):
    med = {c: weighted_median(x["days_since_sendout"], x[w_col]) for c, x in d.groupby(case_col)}
    s = pd.DataFrame({
        "sales": d.groupby(case_col)[w_col].sum(),
        "median_days_since_sendout": pd.Series(med),
    }).reindex(order)
    s["sales"] = s["sales"].fillna(0)
    s["% of sales"] = s["sales"] / s["sales"].sum()
    total = pd.DataFrame({
        "sales": [s["sales"].sum()],
        "median_days_since_sendout": [weighted_median(d["days_since_sendout"], d[w_col])],
        "% of sales": [1.0],
    }, index=["Total"])
    return pd.concat([s, total])

fmt = {
    "sales": "{:,.1f}".format,
    "median_days_since_sendout": "{:.1f}".format,
    "% of sales": "{:.1%}".format,
}

summary_before = make_summary(df_s.assign(weight=1.0), "attribution_case", "weight")
summary_after  = make_summary(df_realloc, "attribution_case_new", "weight")

print("BEFORE: original attribution")
print(summary_before.rename_axis("Attribution case").to_string(formatters=fmt))
print("\nAFTER: 'Both' above each TV status's baseline moved to 'Sendout only'")
print(summary_after.rename_axis("Attribution case").to_string(formatters=fmt))

## Digital Mix VS Days Since Sendout ----

WINDOWS = {"Days 2-8": (2, 8), "Days 21-27": (21, 27)}
W_EARLY, W_LATE = list(WINDOWS)
DIMS = ["channel", "platform", "campaign"]
W_COLORS = {W_EARLY: "#e4572e", W_LATE: "#4c78a8"}

# --- Sendout dates (non-holdout) ---
sendout_dates = (
    pd.to_datetime(
        df_dm_sendouts.loc[df_dm_sendouts["sendout_group"] != "Holdout", "sendout_date"]
    )
    .astype("datetime64[ns]").drop_duplicates().sort_values()
    .rename("last_sendout_date").to_frame()
)

# --- Days since sendout for every digital interaction, then keep the two windows ---
ix = (
    df_digital_interactions
    .assign(date=lambda d: d["timestamp_utc"]
            .dt.tz_localize(None).dt.normalize().astype("datetime64[ns]"))
    .sort_values("date")
)
ix = pd.merge_asof(ix, sendout_dates, left_on="date", right_on="last_sendout_date",
                   direction="backward")
ix["days_since_sendout"] = (ix["date"] - ix["last_sendout_date"]).dt.days

ix["window"] = ""
for name, (lo_d, hi_d) in WINDOWS.items():
    ix.loc[ix["days_since_sendout"].between(lo_d, hi_d), "window"] = name
ix = ix[ix["window"] != ""]

print("Interactions per window:")
print(ix["window"].value_counts().reindex([W_EARLY, W_LATE]).to_string(), "\n")

# --- Shares, tests, and chart data per dimension ---
results = {}
for dim in DIMS:
    counts = (
        pd.crosstab(ix[dim].fillna("Unknown"), ix["window"])
        .reindex(columns=[W_EARLY, W_LATE], fill_value=0)
    )
    totals = counts.sum()
    shares = counts / totals

    chi2, p, _, _ = chi2_contingency(counts)
    cramers_v = np.sqrt(chi2 / totals.sum())     # 2 windows -> min(r,c)-1 = 1

    tbl = pd.DataFrame({
        f"Share {W_EARLY}": shares[W_EARLY],
        f"Share {W_LATE}": shares[W_LATE],
        "Diff (pp)": (shares[W_LATE] - shares[W_EARLY]) * 100,
    })
    tbl["p (category)"] = [
        proportions_ztest([counts.loc[c, W_EARLY], counts.loc[c, W_LATE]],
                          [totals[W_EARLY], totals[W_LATE]])[1]
        if counts.loc[c].min() > 0 else np.nan
        for c in counts.index
    ]
    tbl = tbl.sort_values(f"Share {W_EARLY}", ascending=False)
    results[dim] = tbl

    print(f"=== {dim.upper()} ===  chi-square p = {p:.4g}, Cramér's V = {cramers_v:.3f}")
    print(tbl.to_string(formatters={
        f"Share {W_EARLY}": "{:.1%}".format,
        f"Share {W_LATE}": "{:.1%}".format,
        "Diff (pp)": "{:+.1f}".format,
        "p (category)": "{:.4f}".format,
    }), "\n")

# --- Chart: one panel per dimension, bars = share of window's interactions ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, dim in zip(axes, DIMS):
    tbl = results[dim].iloc[::-1]           # largest category on top
    y = np.arange(len(tbl))
    h = 0.4
    for i, w in enumerate([W_EARLY, W_LATE]):
        b = ax.barh(y + (i - 0.5) * h, tbl[f"Share {w}"], height=h,
                    color=W_COLORS[w], label=w)
        ax.bar_label(b, labels=[f"{v:.0%}" for v in tbl[f"Share {w}"]],
                     fontsize=8, padding=2)
    ax.set_yticks(y)
    ax.set_yticklabels(tbl.index, fontsize=9)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(dim.title(), fontsize=13, fontweight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(frameon=False, loc="lower right")
axes[0].set_xlabel("Share of the window's digital interactions")
fig.suptitle("Digital Interaction Mix: Days 2-8 vs. Days 21-27 After a Sendout",
             fontsize=15, fontweight="bold", x=0.01, ha="left")

plt.tight_layout()
plt.savefig(OUT / "digital_mix_days2_8_vs_21_27.png", dpi=200)
plt.show()

## Comparison to Baseline Table ----

change = pd.DataFrame({
    "Sales before": summary_before["sales"],
    "Sales after":  summary_after["sales"],
    "% of sales before": summary_before["% of sales"],
    "% of sales after":  summary_after["% of sales"],
})
change["Change (sales)"] = change["Sales after"] - change["Sales before"]
change["Change (%)"] = change["Change (sales)"] / change["Sales before"].replace(0, np.nan)
change["Change (pp of total)"] = (change["% of sales after"] - change["% of sales before"]) * 100

print("\nCHANGE")
print(change[["Sales before", "Sales after", "Change (sales)", "Change (%)",
              "% of sales before", "% of sales after", "Change (pp of total)"]]
      .rename_axis("Attribution case")
      .to_string(formatters={
          "Sales before": "{:,.1f}".format,
          "Sales after": "{:,.1f}".format,
          "Change (sales)": "{:+,.1f}".format,
          "Change (%)": "{:+.1%}".format,
          "% of sales before": "{:.1%}".format,
          "% of sales after": "{:.1%}".format,
          "Change (pp of total)": "{:+.1f}".format,
      }))


## Mix Change ----

dig = ["Digital only", "Both"]
scale = change.loc[dig, "Sales after"].sum() / change.loc[dig, "Sales before"].sum()
print(scale)

df_di_final_mix_channel = count_table_for_digital_channels_with_scale(df_ix, "channel", scale=scale)
df_di_final_mix_platform = count_table_for_digital_channels_with_scale(df_ix, "platform", scale=scale)
df_di_final_mix_campaign = count_table_for_digital_channels_with_scale(df_ix, "campaign", scale=scale)
