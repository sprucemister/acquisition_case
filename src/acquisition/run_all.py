"""Single entry point: ``uv run run-analysis``."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

from . import attribution, config, explore, load_clean, model, synthetic, tv, variants


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Multi-channel acquisition attribution case study")
    ap.add_argument("--data-dir", type=Path, help="folder with the four raw CSV files")
    ap.add_argument("--out-dir", type=Path, help="where charts and tables are written")
    ap.add_argument("--demo", action="store_true", help="run on generated synthetic data")
    ap.add_argument("--show", action="store_true", help="also open each figure window")
    args = ap.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")  # no GUI needed, works on any machine

    paths = config.PATHS
    paths.show = args.show
    if args.out_dir:
        paths.out_dir = args.out_dir
    if args.demo:
        paths.data_dir = synthetic.make_demo(Path(tempfile.mkdtemp(prefix="acq_demo_")))
        if not args.out_dir:
            paths.out_dir = config.ROOT / "outputs_demo"
    elif args.data_dir:
        paths.data_dir = args.data_dir
    paths.ensure()

    data = load_clean.clean(load_clean.load_raw(paths.data_dir))

    in_window, df_ix = attribution.digital_mix(data)
    base = attribution.build_baseline(data, in_window)
    attribution.reconcile(data, base)
    attribution.print_case_summary(attribution.case_summary(base))

    tv_dates, base, df_daily = tv.run(data, base)
    variants.run(data, tv_dates)
    explore.run(data, base, tv_dates, df_daily)
    model.run(data, base, tv_dates, df_ix)

    print(f"\nDone. Charts: {paths.out_dir}  Tables: {paths.tables_dir}")


if __name__ == "__main__":
    main()
