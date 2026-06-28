"""
Generate high-resolution word cloud images matching all five tabs of the
interactive word cloud on the Trends page (Default, Geopolitical, Thematic,
Action, Subjects), using Consensus colour mode (pink → amber → green).

Usage — generate all tabs:
    python -m app.generate_wordcloud_hd

Usage — generate a specific tab:
    python -m app.generate_wordcloud_hd --mode thematic
    python -m app.generate_wordcloud_hd --mode all --dpi 300 --max-words 20

Output files: app/assets/wordcloud_hd_<mode>.png
"""
import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud


_CONSENSUS_COLORS = ["#ff66cc", "#e6b24b", "#33cc33"]  # matches wordcloud_interactive.py
_IGNORE_WORDS = {"resolution", "general assembly"}
_FONT_PATH = r"C:\Windows\Fonts\segoeuib.ttf"
_WC_WIDTH, _WC_HEIGHT = 1200, 800  # fixed canvas — keeps font proportions identical to interactive

MODES = ["default", "geopolitical", "thematic", "action", "subjects"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_keywords(mode: str, assets_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with columns [undl_id, keyword_text] for the given mode."""

    if mode == "default":
        df = pd.read_csv(assets_dir / "undlid_keywords.csv")
        df = df.rename(columns={"keywords": "keyword_text"})
        return df[["undl_id", "keyword_text"]]

    if mode in ("geopolitical", "thematic", "action"):
        col = mode.capitalize()  # Geopolitical / Thematic / Action
        df = pd.read_csv(assets_dir / "undlid_keywords_3d_noun_fixed.csv")
        df = df.rename(columns={"Original_ID": "undl_id", col: "keyword_text"})
        return df[["undl_id", "keyword_text"]]

    if mode == "subjects":
        res_subj = pd.read_csv(data_dir / "resolution_subject_table.csv")
        subj = pd.read_csv(data_dir / "subject_table.csv", usecols=["subject_id", "label_en"])
        merged = res_subj.merge(subj, on="subject_id", how="left")
        # Group all subject labels for each resolution into a semicolon-separated string
        grouped = (
            merged.groupby("undl_id")["label_en"]
            .apply(lambda s: ";".join(s.dropna().astype(str)))
            .reset_index()
            .rename(columns={"label_en": "keyword_text"})
        )
        return grouped[["undl_id", "keyword_text"]]

    raise ValueError(f"Unknown mode: {mode}")


def _build_word_stats(
    keywords_df: pd.DataFrame,
    score_by_id: dict[str, float],
) -> tuple[dict[str, int], dict[str, float]]:
    """Return (word → frequency, word → mean consensus_score)."""
    split_pat = re.compile(r"[;,]")
    word_to_ids: dict[str, list[str]] = {}

    for undl_id, keyword_text in keywords_df[["undl_id", "keyword_text"]].itertuples(index=False):
        if pd.isna(keyword_text):
            continue
        uid = str(undl_id)
        tokens: set[str] = set()
        for kw in split_pat.split(str(keyword_text)):
            tok = re.sub(r"\s+", " ", kw.strip().lower())
            if tok and tok not in _IGNORE_WORDS:
                tokens.add(tok)
        for tok in tokens:
            word_to_ids.setdefault(tok, []).append(uid)

    frequencies: dict[str, int] = {}
    consensus_by_word: dict[str, float] = {}

    for word, ids in word_to_ids.items():
        frequencies[word] = len(ids)
        scores = [score_by_id[i] for i in ids if i in score_by_id]
        consensus_by_word[word] = float(np.mean(scores)) if scores else float("nan")

    return frequencies, consensus_by_word


# ---------------------------------------------------------------------------
# Colour mapping
# ---------------------------------------------------------------------------

def _make_consensus_color_func(consensus_by_word: dict, lo: float, hi: float):
    span = hi - lo if hi > lo else 1.0
    stops = [mpl.colors.to_rgb(c) for c in _CONSENSUS_COLORS]

    def color_func(word, *_args, **_kwargs):
        score = consensus_by_word.get(word, float("nan"))
        if isinstance(score, float) and np.isnan(score):
            return "#aaaaaa"
        t = float(np.clip((score - lo) / span, 0.0, 1.0))
        if t <= 0.5:
            frac, c0, c1 = t / 0.5, stops[0], stops[1]
        else:
            frac, c0, c1 = (t - 0.5) / 0.5, stops[1], stops[2]
        r = int((c0[0] + frac * (c1[0] - c0[0])) * 255)
        g = int((c0[1] + frac * (c1[1] - c0[1])) * 255)
        b = int((c0[2] + frac * (c1[2] - c0[2])) * 255)
        return f"rgb({r},{g},{b})"

    return color_func


# ---------------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------------

def generate_one(
    mode: str,
    assets_dir: Path,
    data_dir: Path,
    output_dir: Path,
    max_words: int,
    width: int,
    height: int,
    dpi: int,
):
    resolutions_df = pd.read_csv(
        data_dir / "resolution_table.csv", usecols=["undl_id", "consensus_score"]
    )
    score_by_id = (
        resolutions_df.set_index(resolutions_df["undl_id"].astype(str))["consensus_score"]
        .dropna()
        .to_dict()
    )

    keywords_df = _load_keywords(mode, assets_dir, data_dir)
    frequencies, consensus_by_word = _build_word_stats(keywords_df, score_by_id)

    top_words = sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))[:max_words]
    top_freq = {w: f for w, f in top_words}
    top_consensus = {w: consensus_by_word[w] for w in top_freq}

    valid_scores = [s for s in top_consensus.values() if not np.isnan(s)]
    lo = float(np.percentile(valid_scores, 1)) if valid_scores else 0.0
    hi = float(np.percentile(valid_scores, 99)) if valid_scores else 1.0

    color_func = _make_consensus_color_func(top_consensus, lo=lo, hi=hi)

    wc = WordCloud(
        width=_WC_WIDTH,
        height=_WC_HEIGHT,
        background_color="white",
        prefer_horizontal=1.0,
        relative_scaling=0.5,
        min_font_size=16,
        max_font_size=100,
        max_words=max_words,
        random_state=42,
        collocations=False,
        font_path=_FONT_PATH,
    ).generate_from_frequencies(top_freq)
    wc.recolor(color_func=color_func, random_state=42)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    cmap = mpl.colors.LinearSegmentedColormap.from_list("consensus", _CONSENSUS_COLORS, N=256)
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, orientation="vertical")
    cbar.set_label("Avg. consensus score", fontsize=max(8, dpi // 40))
    cbar.ax.tick_params(labelsize=max(7, dpi // 50))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"wordcloud_hd_{mode}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[{mode:12s}] Saved → {out_path}  ({len(top_freq)} words)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Generate HD consensus word clouds for all tabs")
    p.add_argument(
        "--mode",
        choices=MODES + ["all"],
        default="all",
        help="Which tab to generate (default: all)",
    )
    p.add_argument("--assets-dir", type=Path, default=Path("app/assets"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("app/assets"))
    p.add_argument("--max-words", type=int, default=20)
    p.add_argument("--width", type=int, default=3600)
    p.add_argument("--height", type=int, default=2000)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    modes_to_run = MODES if args.mode == "all" else [args.mode]
    for m in modes_to_run:
        generate_one(
            mode=m,
            assets_dir=args.assets_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_words=args.max_words,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
