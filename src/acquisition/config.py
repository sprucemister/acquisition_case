"""Paths, assumptions and plot settings.

Every modelling assumption lives here so a reviewer can see them all in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    """Run-time locations. Set by ``run_all`` (or ``ACQ_DATA_DIR`` for the data folder)."""

    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("ACQ_DATA_DIR", ROOT / "data" / "raw"))
    )
    out_dir: Path = ROOT / "outputs"
    show: bool = False  # True: also open each figure window (blocks a plain script run)

    @property
    def tables_dir(self) -> Path:
        return self.out_dir / "tables"

    def ensure(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

# --- Calendar ------------------------------------------------------------
YEAR = 2025

# --- Digital ---------------------------------------------------------------
# Channels that do not count as "inorganic" digital interactions.
NON_INORGANIC_CHANNELS = ("Direct", "Organic Search", "Paid Search")
# Interactions up to this many (rounded) days before the sale are kept.
# Note: negative values (interactions *after* the sale) are also kept, as in the
# original analysis. See README "Known limitations".
LOOKBACK_DAYS = 7

# --- Attribution cases -----------------------------------------------------
CASE_ORDER = ["Digital only", "Sendout only", "Both", "No Direct Attribution"]
CASE_COLORS = {
    "Digital only": "#4c78a8",
    "Sendout only": "#59a14f",
    "Both": "#e4572e",
    "No Direct Attribution": "#9d9d9d",
}

# --- Direct mail ---------------------------------------------------------
GROUPS = ["Holdout", "Variant A", "Variant B"]
GROUP_COLORS = {"Holdout": "#9d9d9d", "Variant A": "#4c78a8", "Variant B": "#e4572e"}
RESPONSE_WINDOW_DAYS = 30  # a sale within this many days of a sendout counts as a response
SHOWN_LAG_DAYS = 27  # last day shown in the daily-sales-after-sendout charts
SURVIVAL_MAX_DAY = 150
SURVIVAL_MIN_AT_RISK = 1000
SURVIVAL_SMOOTH_DAYS = 7

# --- 'Both' reallocation model -------------------------------------------------
# Days (inclusive) after a sendout used to estimate the flat 'Both' level. By then the
# sendout effect is assumed to have faded.
BASELINE_DAYS = (24, 27)
# Only 'Both' sales before this day can have their excess moved to 'Sendout only'.
REALLOC_BEFORE = 24
SEASON_MONTHS = {
    "May-Aug": (5, 6, 7, 8),
    "Sep-Oct": (9, 10),
    "Nov-Apr": (11, 12, 1, 2, 3, 4),
}
SEASON_ORDER = list(SEASON_MONTHS)

# --- TV ----------------------------------------------------------------------
# Assumption from the case brief: sales credited per TV-active day when TV cannot be
# attributed directly. Confirm the value against the brief before sharing.
TV_SALES_PER_ACTIVE_DAY = 15

# --- Digital-mix windows (days since sendout) ----------------------------------
MIX_WINDOWS = {"Days 2-8": (2, 8), "Days 21-27": (21, 27)}
MIX_DIMS = ["channel", "platform", "campaign"]
