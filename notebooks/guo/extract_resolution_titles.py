import argparse
from pathlib import Path

import pandas as pd


def extract_titles(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, usecols=["undl_id", "title","date"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df):,} rows to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract undl_id and title from resolution_table.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/resolution_table.csv"),
        help="Path to source resolution table CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/resolution_titles.csv"),
        help="Path to output CSV file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_titles(args.input, args.output)
