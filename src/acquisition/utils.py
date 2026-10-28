"""Small helpers shared by every module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import config
from .config import PATHS


# --- Keys and dates -----------------------------------------------------------
def clean_key(s: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and uppercase. Nulls stay null."""
    return s.astype("string").str.strip().str.upper()


def parse_utc(s: pd.Series) -> pd.Series:
    """Parse timestamps as UTC. Falls back to mixed-format parsing if needed."""
    try:
        return pd.to_datetime(s, utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(s, utc=True, format="mixed")


def ns(s) -> pd.Series:
    """Naive datetime64[ns] (needed so merge_asof keys have the same resolution)."""
    return pd.to_datetime(s).astype("datetime64[ns]")


def to_day(s: pd.Series) -> pd.Series:
    """UTC timestamp -> naive midnight datetime64[ns] (the calendar day)."""
    return s.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")


def season_of(dates) -> np.ndarray:
    """Rough season label for each date (see config.SEASON_MONTHS)."""
    month_to_season = {m: s for s, months in config.SEASON_MONTHS.items() for m in months}
    return pd.DatetimeIndex(dates).month.map(month_to_season).to_numpy()


# --- Sendout calendar ------------------------------------------------------------
def nonholdout_sendout_dates(dm: pd.DataFrame) -> pd.Series:
    """Distinct, sorted dates of non-holdout sendouts."""
    d = dm.loc[dm["sendout_group"] != "Holdout", "sendout_date"]
    return ns(d).drop_duplicates().sort_values().rename("last_sendout_date").reset_index(drop=True)


def year_dates(year: int = config.YEAR) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def make_calendar(
    tv_dates: pd.DataFrame, sendouts: pd.Series, max_lag: int | None = None
) -> pd.DataFrame:
    """One row per day of the year with days since the latest non-holdout sendout.

    Days before the first sendout are dropped (the measure is undefined for them).
    """
    cal = pd.merge_asof(
        tv_dates[["date", "tv_spot_active"]]
        .assign(date=lambda d: ns(d["date"]))
        .sort_values("date"),
        sendouts.to_frame(),
        left_on="date",
        right_on="last_sendout_date",
        direction="backward",
    )
    cal["days_since_sendout"] = (cal["date"] - cal["last_sendout_date"]).dt.days
    cal = cal.dropna(subset=["days_since_sendout"])
    cal["days_since_sendout"] = cal["days_since_sendout"].astype(int)
    cal["tv_spot_active"] = cal["tv_spot_active"].astype(bool)
    if max_lag is not None:
        cal = cal[cal["days_since_sendout"] <= max_lag]
    return cal.reset_index(drop=True)


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna()
    v, w = values[m].to_numpy(dtype=float), weights[m].to_numpy(dtype=float)
    if len(v) == 0:
        return np.nan
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


# --- Output ---------------------------------------------------------------------
def save_fig(fig: Figure, name: str) -> None:
    """Save into outputs/, optionally show, then close to free memory."""
    import matplotlib.pyplot as plt

    PATHS.ensure()
    fig.savefig(PATHS.out_dir / name, dpi=200, bbox_inches="tight")
    if PATHS.show:
        plt.show()
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    PATHS.ensure()
    df.to_csv(PATHS.tables_dir / name, index=index)


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")
