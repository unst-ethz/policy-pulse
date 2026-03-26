import argparse
from pathlib import Path

import pandas as pd


def extract_missing_titles(
    titles_csv: Path,
    keywords_csv: Path,
    output_csv: Path,
) -> int:
    titles_df = pd.read_csv(titles_csv, usecols=["undl_id", "title","date"])
    keywords_df = pd.read_csv(keywords_csv, usecols=["undl_id"])

    titles_df["undl_id"] = titles_df["undl_id"].astype(str)
    keyword_ids = set(keywords_df["undl_id"].astype(str))

    missing_df = titles_df[~titles_df["undl_id"].isin(keyword_ids)].copy()
    missing_df = missing_df.drop_duplicates(subset=["undl_id"]).sort_values("undl_id")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(output_csv, index=False, encoding="utf-8")
    return len(missing_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract undl_id,title where undl_id exists in titles CSV but not in keywords CSV."
    )
    parser.add_argument(
        "--titles-csv",
        type=Path,
        default=Path("data/resolution_titles.csv"),
        help="Path to resolution titles CSV (must contain undl_id,title)",
    )
    parser.add_argument(
        "--keywords-csv",
        type=Path,
        default=Path("app/assets/undlid_keywords.csv"),
        help="Path to keywords CSV (must contain undl_id)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/resolution_titles_missing_in_keywords.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = extract_missing_titles(
        titles_csv=args.titles_csv,
        keywords_csv=args.keywords_csv,
        output_csv=args.output_csv,
    )
    print(f"Saved {count} rows to {args.output_csv}")
