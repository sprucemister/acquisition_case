"""Read the four raw files and clean them into the frames used everywhere else."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import clean_key, ns, parse_utc


@dataclass
class Data:
    sales: pd.DataFrame
    digital: pd.DataFrame
    tv_spots: pd.DataFrame
    dm: pd.DataFrame


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    missing = [
        f
        for f in ("sales.csv", "digital_interactions.csv", "tv_spots.csv", "dm_sendouts.csv")
        if not (data_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. See README ('Data') or run with --demo."
        )
    # Timestamps are read as text first, then parsed, because formats can be mixed.
    sales = pd.read_csv(data_dir / "sales.csv", dtype={"sale_timestamp_utc": str})
    sales["sale_timestamp_utc"] = parse_utc(sales["sale_timestamp_utc"])

    digital = pd.read_csv(data_dir / "digital_interactions.csv", dtype={"timestamp_utc": str})
    digital["timestamp_utc"] = parse_utc(digital["timestamp_utc"])

    tv = pd.read_csv(data_dir / "tv_spots.csv", dtype={"air_time": str})
    tv["air_time"] = parse_utc(tv["air_time"])

    dm = pd.read_csv(data_dir / "dm_sendouts.csv")
    dm["sendout_date"] = ns(dm["sendout_date"])
    return {"sales": sales, "digital": digital, "tv_spots": tv, "dm": dm}


def clean(raw: dict[str, pd.DataFrame]) -> Data:
    # --- Sales -----------------------------------------------------------------
    # Normalise keys first so the dedup works on the cleaned values.
    sales = raw["sales"].copy()
    for col in ("digital_interaction_id", "address_id"):
        sales[col] = clean_key(sales[col])
    # Drop exact duplicates, then keep the earliest timestamp per sale_id.
    sales = sales.drop_duplicates()
    min_ts = sales.groupby("sale_id")["sale_timestamp_utc"].transform("min")
    sales = sales[sales["sale_timestamp_utc"] == min_ts].reset_index(drop=True)

    # --- Digital -----------------------------------------------------------------
    digital = raw["digital"].copy()
    for col in ("interaction_id", "cookie_id"):
        digital[col] = clean_key(digital[col])
    digital["campaign"] = np.select(
        [
            digital["campaign"].isna() & (digital["channel"] == "Direct"),
            digital["campaign"].isna() & (digital["channel"] == "Organic Search"),
        ],
        ["Direct", "Organic Search"],
        default=digital["campaign"],
    )
    digital["platform"] = digital["platform"].fillna("Direct")

    # --- Direct mail ---------------------------------------------------------------
    dm = raw["dm"].copy()
    grp = dm["sendout_group"].astype("string").str.upper()
    dm["address_id"] = clean_key(dm["address_id"])
    dm["sendout_group"] = np.select(
        [
            grp.str.contains("HOLD", na=False),
            grp.str.contains("CONT", na=False),
            grp.str.contains("B", na=False),
        ],
        ["Holdout", "Holdout", "Variant B"],
        default="Variant A",
    )

    data = Data(sales=sales, digital=digital, tv_spots=raw["tv_spots"].copy(), dm=dm)
    check_keys(data)
    return data


def check_keys(data: Data) -> None:
    """Report (and warn about) dirty keys or non-unique sale ids."""
    keys = {
        "sales.digital_interaction_id": data.sales["digital_interaction_id"],
        "sales.address_id": data.sales["address_id"],
        "digital.interaction_id": data.digital["interaction_id"],
        "digital.cookie_id": data.digital["cookie_id"],
        "dm.address_id": data.dm["address_id"],
    }
    print("Key check:")
    for name, s in keys.items():
        s = s.dropna()
        stray = int((s != s.str.strip()).sum())
        lower = int((s != s.str.upper()).sum())
        print(f"  {name}: {s.nunique():,} unique, {stray} stray whitespace, {lower} not uppercase")
        if stray or lower:
            warnings.warn(f"{name} still has dirty values", stacklevel=2)
    if not data.sales["sale_id"].is_unique:
        warnings.warn("sale_id is not unique after cleaning", stacklevel=2)
