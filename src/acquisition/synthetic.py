"""Synthetic demo data with the same columns as the real files.

Numbers here mean nothing; it exists so the pipeline runs end to end without the
confidential data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CHANNELS = ["Direct", "Organic Search", "Paid Search", "Display", "Social", "Email", "Affiliate"]
PLATFORMS = ["Google", "Meta", "Criteo", "Newsletter"]
CAMPAIGNS = ["Spring", "Summer", "Brand", "Retarget", "Always-on"]


def make_demo(out_dir: Path, seed: int = 0, n_addresses: int = 6000, n_sales: int = 4000) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    year = config.YEAR
    year_start = pd.Timestamp(f"{year}-01-01")

    # --- Direct mail: one sendout per month on the 5th ------------------------------------
    addresses = np.array([f"ADDR{i:05d}" for i in range(n_addresses)])
    groups = rng.choice(
        ["Holdout", "Variant A", "Variant B"], size=n_addresses, p=[0.1, 0.45, 0.45]
    )
    rows = []
    for m in range(1, 13):
        date = pd.Timestamp(year, m, 5)
        for a, g in zip(addresses, groups):
            rows.append((a, g, date))
    dm = pd.DataFrame(rows, columns=["address_id", "sendout_group", "sendout_date"])
    dm["sendout_group"] = dm["sendout_group"].replace(
        {"Holdout": "HOLDOUT", "Variant A": "A", "Variant B": "B"}
    )

    # --- Digital interactions ------------------------------------------------------------------
    n_cookies = int(n_sales * 1.5)
    cookies = np.array([f"CK{i:06d}" for i in range(n_cookies)])
    n_ix = n_cookies * 3
    ix = pd.DataFrame(
        {
            "interaction_id": [f"IX{i:07d}" for i in range(n_ix)],
            "cookie_id": rng.choice(cookies, n_ix),
            "channel": rng.choice(CHANNELS, n_ix, p=[0.2, 0.2, 0.2, 0.12, 0.1, 0.1, 0.08]),
            "platform": rng.choice(PLATFORMS + [None], n_ix),
            "campaign": rng.choice(CAMPAIGNS + [None], n_ix),
            "timestamp_utc": year_start
            + pd.to_timedelta(rng.uniform(0, 365 * 86400, n_ix), unit="s"),
        }
    )

    # --- Sales: base rate + bump in the days after a sendout --------------------------------------
    days = rng.integers(0, 365, n_sales)
    sale_ts = year_start + pd.to_timedelta(days * 86400 + rng.integers(0, 86400, n_sales), unit="s")
    sale_cookie = rng.choice(cookies, n_sales)
    # Link each sale to one interaction of its cookie where one exists.
    by_cookie = ix.groupby("cookie_id")["interaction_id"].first()
    di = pd.Series(sale_cookie).map(by_cookie)
    di[rng.random(n_sales) < 0.25] = None
    sales = pd.DataFrame(
        {
            "sale_id": [f"S{i:06d}" for i in range(n_sales)],
            "sale_timestamp_utc": sale_ts,
            "digital_interaction_id": di,
            "address_id": rng.choice(addresses, n_sales),
        }
    )
    # Make interactions precede their converting sales.
    first_ix = (
        sales.dropna(subset=["digital_interaction_id"])
        .drop_duplicates("digital_interaction_id")
        .set_index("digital_interaction_id")
    )
    shift = ix["interaction_id"].map(first_ix["sale_timestamp_utc"])
    mask = shift.notna()
    ix.loc[mask, "timestamp_utc"] = shift[mask] - pd.to_timedelta(
        rng.uniform(0, 5 * 86400, mask.sum()), unit="s"
    )

    # --- TV spots ---------------------------------------------------------------------------------------
    n_spots = 60
    tv = pd.DataFrame(
        {
            "air_time": year_start
            + pd.to_timedelta(np.sort(rng.integers(0, 365 * 86400, n_spots)), unit="s")
        }
    )

    # --- Write (with a little dirt so the cleaning has work to do) -----------------------------------------
    sales["address_id"] = sales["address_id"].where(
        rng.random(n_sales) > 0.02, " " + sales["address_id"].str.lower()
    )
    for df, col in [(sales, "sale_timestamp_utc"), (ix, "timestamp_utc"), (tv, "air_time")]:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dm["sendout_date"] = dm["sendout_date"].dt.strftime("%Y-%m-%d")
    sales.to_csv(out_dir / "sales.csv", index=False)
    ix.to_csv(out_dir / "digital_interactions.csv", index=False)
    tv.to_csv(out_dir / "tv_spots.csv", index=False)
    dm.to_csv(out_dir / "dm_sendouts.csv", index=False)
    return out_dir
