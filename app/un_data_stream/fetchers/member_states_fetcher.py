"""
Member State authority list fetcher.

Fetches the UN member-state authority CSV (multi-language country names,
M49/ISO codes, membership periods, earlier/later name links) from the UN
Digital Library and returns it as a cleaned DataFrame.
"""

import pandas as pd

from ..core.abstractions import DatasetFetcher


class MemberStatesFetcher(DatasetFetcher):
    """Fetches the UN member-state authority list."""

    def _fetch_and_parse(self, url: str) -> pd.DataFrame:
        df = pd.read_csv(url)

        # Drop the synthetic "Unknown" row (NaN ISO Code) and any other
        # rows missing the ISO code, since ISO is our join key everywhere.
        before = len(df)
        df = df[df["ISO Code"].notna()].copy()
        dropped = before - len(df)
        if dropped:
            self.logger.info(
                f"Dropped {dropped} member-state row(s) with no ISO code"
            )

        # Drop the unnamed pandas index column if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        df = df.reset_index(drop=True)

        self.logger.info(
            f"Successfully fetched {len(df)} member-state records "
            f"({df['ISO Code'].nunique()} distinct ISO codes)"
        )

        return df

    def get_dataset_type(self) -> str:
        return "member_states"
