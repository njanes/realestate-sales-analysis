"""Reproduce the README results without opening Jupyter.

    python train.py                 # print the results table
    python train.py --markdown      # print it as Markdown, ready for the README
    python train.py --save-figures  # also regenerate the charts in images/
"""

import argparse
import sys
from pathlib import Path

from kc_housing.config import DATA_PATH, IMAGES_DIR
from kc_housing.data import load_clean
from kc_housing.evaluate import format_table, results_table, to_markdown

DATASET_URL = "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH,
        help="path to kc_house_data.csv (default: repo root)",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="print the table as Markdown",
    )
    parser.add_argument(
        "--save-figures", action="store_true",
        help=f"regenerate the charts in {IMAGES_DIR.name}/",
    )
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(
            f"Dataset not found at {args.data}.\n"
            f"Download kc_house_data.csv from {DATASET_URL} "
            "and put it in the repo root, or pass --data."
        )

    df = load_clean(args.data)
    table = results_table(df)
    if args.markdown:
        print(to_markdown(table))
    else:
        print(format_table(table).to_string(index=False))

    if args.save_figures:
        import matplotlib
        matplotlib.use("Agg")
        from kc_housing.plots import save_figures

        save_figures(df, IMAGES_DIR)
        print(f"\nSaved charts to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
