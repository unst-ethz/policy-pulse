import argparse
from pathlib import Path

import pandas as pd


def append_keywords_update(
    base_csv: Path,
    update_csv: Path,
    output_csv: Path | None = None,
    deduplicate_by_undl_id: bool = False,
) -> None:
    base_df = pd.read_csv(base_csv)
    update_df = pd.read_csv(update_csv)

    required_cols = {"undl_id", "keywords"}
    if not required_cols.issubset(base_df.columns):
        missing = required_cols - set(base_df.columns)
        raise ValueError(f"Base CSV missing columns: {sorted(missing)}")
    if not required_cols.issubset(update_df.columns):
        missing = required_cols - set(update_df.columns)
        raise ValueError(f"Update CSV missing columns: {sorted(missing)}")

    merged_df = pd.concat([base_df, update_df], ignore_index=True)

    if deduplicate_by_undl_id:
        # Keep the latest (updated) value when undl_id duplicates exist.
        merged_df["undl_id"] = merged_df["undl_id"].astype(str)
        merged_df = merged_df.drop_duplicates(subset=["undl_id"], keep="last")

    target_path = output_csv if output_csv else base_csv
    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(target_path, index=False, encoding="utf-8")

    print(f"Base rows: {len(base_df):,}")
    print(f"Update rows: {len(update_df):,}")
    print(f"Saved rows: {len(merged_df):,}")
    print(f"Saved to: {target_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append app/assets/undlid_keywords_updated.csv into app/assets/undlid_keywords.csv"
    )
    parser.add_argument(
        "--base-csv",
        type=Path,
        default=Path("app/assets/undlid_keywords.csv"),
        help="Target/base keywords CSV",
    )
    parser.add_argument(
        "--update-csv",
        type=Path,
        default=Path("app/assets/undlid_keywords_updated.csv"),
        help="Update CSV to append",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, overwrite base CSV.",
    )
    parser.add_argument(
        "--deduplicate-by-undl-id",
        action="store_true",
        help="Drop duplicated undl_id after append, keeping updated rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    append_keywords_update(
        base_csv=args.base_csv,
        update_csv=args.update_csv,
        output_csv=args.output_csv,
        deduplicate_by_undl_id=args.deduplicate_by_undl_id,
    )
