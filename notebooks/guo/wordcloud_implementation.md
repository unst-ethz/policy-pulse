# Word Cloud Feature — Implementation Documentation

## Overview

The word cloud visualises the most frequent keywords found in UN General Assembly resolutions, filtered by the active user selections (country, year range, subject, keyword search). It supports five distinct **keyword panels** (modes), each derived from a different extraction pipeline. Users can hover over a word to preview matching resolutions, and click a word to push it into the keyword-search filter and switch to the Resolutions tab.

---

## Architecture

```
data/resolution_table.csv
        │
        ▼
[extract_resolution_titles.py]
        │  produces
        ▼
data/resolution_titles.csv            (undl_id, title, date)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[T5 Model Pipeline]                       [GPT-4.1-mini Pipeline]
notebooks/guo/1113.ipynb                  notebooks/guo/keyword_openai.ipynb
                                          notebooks/guo/extract_keywords_3d.py
                                          notebooks/guo/extract_keywords_3d_noun.py
        │                                              │
        ▼                                              ▼
app/assets/undlid_keywords.csv         app/assets/undlid_keywords_3d_noun_fixed.csv
   (undl_id, keywords)                    (Original_ID, Geopolitical, Thematic, Action)
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
              app/features/wordcloud_interactive.py
                    (_init_wc_data, _build_wordcloud)
                           │
                           ▼
                  Plotly Scatter figure
                  rendered in the app
```

---

## Step 1 — Title Extraction

**Script:** `notebooks/guo/extract_resolution_titles.py`

Reads `data/resolution_table.csv` and writes `data/resolution_titles.csv` containing only the columns `undl_id`, `title`, and `date`. This lightweight file is the shared input for both downstream extraction pipelines.

---

## Step 2 — Keyword Extraction Pipelines

### 2A — Default Panel: T5 Model (`Voicelab/vlt5-base-keywords`)

**Notebook:** `notebooks/guo/1113.ipynb`  
**Output:** `app/assets/undlid_keywords.csv` (columns: `undl_id`, `keywords`)

This pipeline uses the fine-tuned T5 model [`Voicelab/vlt5-base-keywords`](https://huggingface.co/Voicelab/vlt5-base-keywords), which is specialised for unsupervised keyword extraction from short texts. Each resolution title is prefixed with `"Keywords: "` and passed to the model.

**Sliding-window strategy for long titles:**

```python
task_prefix = "Keywords: "
window_size = 24   # tokens per window
stride      = 20   # step size between windows

# Short titles (≤ window_size words) → processed in one pass
# Long titles → overlapping windows are processed independently;
#               all keyword sets are union-merged
```

**Generation parameters:**

```python
output = model.generate(
    input_ids,
    no_repeat_ngram_size=3,
    num_beams=4,
    max_length=100
)
```

Keywords are decoded, split on commas, deduplicated (per-resolution set), and filtered to a minimum length of 3 characters. The final result is comma-separated and written to CSV.

#### Known issue: T5 token hallucinations

During batch runs, the model was observed producing corrupted tokens — most notably, **"nuclear" consistently rendered as "ruclear"** across many resolutions. This appears to be a systematic tokenisation artefact: the T5 tokeniser splits "nuclear" at a subword boundary, and during beam-search decoding the leading subword is occasionally dropped, causing the misspelling to reproduce faithfully across independent inputs.

> I suspect this is a typical T5 "hallucination," where the model mispredicts tokens during generation rather than copying text exactly. It is interesting that "nuclear" consistently becomes "ruclear" (and similarly "observer" appeared as "obever") across many cases. For now, the corrections are applied manually to the CSV and documented in the git history (`feat: wc - Manual Keyword Correction: rulear -> nuclear`, commit `423150a`; `feat: wc : obever -> observer`, commit `e045803`) while we look for a long-term fix.

**Workaround applied:** a post-processing pass over `app/assets/undlid_keywords.csv` manually replaced all occurrences of `ruclear` → `nuclear` and `obever` → `observer`. The corrections are tracked as dedicated commits.

**Long-term options to investigate:**
- Pin the exact model revision and tokeniser version to reproduce the bug deterministically.
- Add a post-processing regex correction step in the extraction script.
- Switch to an instruction-following LLM (see Pipeline 2B) for all titles.

---

### 2B — 3D Panels (Geopolitical / Thematic / Action): GPT-4.1-mini

**Scripts:**
- `notebooks/guo/extract_keywords_3d.py` — first version (strict copy-from-title rules)
- `notebooks/guo/extract_keywords_3d_noun.py` — revised version (adds noun normalisation)
- `notebooks/guo/keyword_openai.ipynb` — interactive notebook wrapping the same logic with incremental write-to-disk

**Output:** `app/assets/undlid_keywords_3d_noun_fixed.csv` (columns: `Original_ID`, `Geopolitical`, `Thematic`, `Action`)

Each resolution title is sent to `gpt-4.1-mini` via the OpenAI chat-completions API with `response_format={"type": "json_object"}` and `temperature=0` for deterministic output.

#### Prompt design (noun-normalisation version)

The system prompt instructs the model to act as an international-relations expert and populate three fields:

| Field | Description | Example output |
|-------|-------------|----------------|
| `Geopolitical` | Specific countries, regions, or territories mentioned | `Israel;Iraq` |
| `Thematic` | Core subject matter or technical domain | `Nuclear installations;Human Rights` |
| `Action` | Diplomatic means or course-of-action nouns | `Aggression;Investigation` |

Key prompt rules enforced:

1. **Noun normalisation** — gerunds/verbs are converted to noun forms (`"Promoting"` → `"Promotion"`, `"Combating"` → `"Combat"`).
2. **Semantic aggregation (critical)** — established multi-word phrases such as `"National or Ethnic, Religious and Linguistic Minorities"` must **not** be split across semicolons.
3. **Entity normalisation** — `"Syrian Arab Republic"` → `"Syria"`, `"territories of Ukraine"` → `"Ukraine"`.
4. **Denoising** — a long explicit stop-list removes administrative boilerplate (`"General Assembly"`, `"Situation in"`, `"resolution"`, `"adopted by"`, etc.).
5. **Semicolon delimiter** — multiple values within a field are separated by `;` (not `,`) to avoid conflicts with CSV structure.

**Incremental write strategy (notebook version):**

```python
# Writes each row to disk immediately after API response
with open(output_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Original_ID", "Geopolitical", "Thematic", "Action"])
    if not file_exists:
        writer.writeheader()
    for i, row in df.iterrows():
        result = extract_keywords_3d(row)
        writer.writerow(result)
        f.flush()          # force to disk — crash-safe for long runs
        time.sleep(0.5)    # avoid API rate-limit errors
```

This pattern makes the extraction resumable: re-running with a `df[30:]` slice skips already-processed rows.

---

### 2C — Subjects Panel: Live Database Query

**Source:** `data.query_engine.resolution_subject_table` and `data.query_engine.subject_table`

No offline extraction file is needed. At initialisation time, `_init_wc_data()` joins `resolution_subject_table` (mapping `undl_id → subject_id`) with `subject_table` (mapping `subject_id → label_en`) to build a per-resolution string of subject labels. This live join means the Subjects panel automatically reflects any updates to the underlying subject catalogue.

---

## Step 3 — Index Building (`_init_wc_data`)

**File:** `app/features/wordcloud_interactive.py`, function `_init_wc_data()`

On first app load, the function builds two in-memory indices for each of the five modes:

```
_resolution_wc_data_by_mode[mode][undl_id] = {"word_freq": {term: count}}
_wc_word_undlid_map_by_mode[mode][term]    = [undl_id, undl_id, ...]
```

**Token normalisation applied during indexing:**

- Split on `/[;,]/` (for Default / 3D modes) or `/\||--|;/` (for Subjects mode).
- Strip and lowercase each token.
- Collapse internal whitespace.
- Ignore a global stop-list (`"resolution"`, `"general assembly"`).
- Mode-specific exclusions: in Geopolitical mode, generic terms `{"peoples", "states", "united nations"}` are excluded at render time (see `_EXCLUDED_TERMS_BY_MODE`).

---

## Step 4 — Rendering (`_build_wordcloud`)

**File:** `app/features/wordcloud_interactive.py`, function `_build_wordcloud()`

### 4.1 Frequency aggregation

For the currently filtered set of resolution IDs, word frequencies are summed across all matching resolutions using `_aggregate_word_freq`.

### 4.2 Re-weighting by clickable result count

The raw frequency is **not** used directly as the display weight. Instead, the top 30 candidates by raw frequency are re-scored by how many resolutions would actually be returned if the user clicked that word (computed via `_count_click_search_results`). This aligns visual size with searchability rather than raw occurrence count.

### 4.3 Layout generation

Word positions are computed by the [`wordcloud`](https://github.com/amueller/word_cloud) Python library on a 1200 × 800 px canvas with `prefer_horizontal=1.0` and `relative_scaling=0.5`. The resulting pixel coordinates are then normalised to Plotly's `[-1.1, 1.1]` axis range.

```python
wordcloud = WordCloud(
    width=1200, height=800,
    prefer_horizontal=1.0,
    relative_scaling=0.5,
    min_font_size=16, max_font_size=100,
    max_words=len(word_freq_dict),
    random_state=42,
)
wordcloud.generate_from_frequencies(word_freq_dict)
```

The layout uses a fixed `random_state=42` so the word arrangement is deterministic across page reloads.

### 4.4 Colour mapping

Word colours are drawn from matplotlib's `Blues` colormap, normalised to the `[0.3, 1.0]` range to avoid near-white colours for low-frequency words.

### 4.5 Plotly figure

Two `go.Scatter` traces are layered:

- **`text_trace`** — visible words, positioned and sized from the WordCloud layout.
- **`hover_trace`** — invisible marker trace centred on each word (with a size proportional to the word's rendered area) that provides the hover tooltip showing `"Click to search <word> — N resolutions"`.

---

## Step 5 — Interaction

### Hover

`update_resolution_table` callback: on hover, the word is looked up in `_wc_word_undlid_map_by_mode` and the intersection with the currently filtered resolution set is displayed as resolution cards below the chart.

### Click (click-to-search)

`click_word_to_search` callback appends the clicked word as a quoted exact-match term (`"<word>"`) to the keyword-search input using `&` (AND) semantics. The app then switches to the Resolutions tab. In **Subjects** mode, clicking a word sets the subject dropdown filter instead of the keyword field.

---

## Data Files Reference

| File | Format | Produced by | Consumed by |
|------|--------|-------------|-------------|
| `data/resolution_titles.csv` | `undl_id, title, date` | `extract_resolution_titles.py` | Both extraction pipelines |
| `app/assets/undlid_keywords.csv` | `undl_id, keywords` | `notebooks/guo/1113.ipynb` (T5) | `wordcloud_interactive.py` — Default mode |
| `app/assets/undlid_keywords_3d_noun_fixed.csv` | `Original_ID, Geopolitical, Thematic, Action` | `extract_keywords_3d_noun.py` / `keyword_openai.ipynb` (GPT-4.1-mini) | `wordcloud_interactive.py` — 3D modes |
| `app/assets/undlid_keywords_updated.csv` | `undl_id, keywords` | manual corrections | `append_undlid_keywords_update.py` → merged into base |

### Updating the default keywords CSV

Use `notebooks/guo/append_undlid_keywords_update.py` to merge a correction file into the base:

```bash
python notebooks/guo/append_undlid_keywords_update.py \
    --base-csv app/assets/undlid_keywords.csv \
    --update-csv app/assets/undlid_keywords_updated.csv \
    --deduplicate-by-undl-id
```

The `--deduplicate-by-undl-id` flag keeps the updated row when an `undl_id` appears in both files.

---

## Wordcloud Mode Summary

| Tab label | Mode key | Keyword source file | Column |
|-----------|----------|---------------------|--------|
| Default | `default` | `undlid_keywords.csv` | `keywords` |
| Geopolitical | `geopolitical` | `undlid_keywords_3d_noun_fixed.csv` | `Geopolitical` |
| Thematic | `thematic` | `undlid_keywords_3d_noun_fixed.csv` | `Thematic` |
| Action | `action` | `undlid_keywords_3d_noun_fixed.csv` | `Action` |
| Subjects | `category` | live DB join | `subject_table.label_en` |

---

## Known Issues & Notes

### T5 subword hallucinations (Default panel)

The `Voicelab/vlt5-base-keywords` model systematically mis-generates certain tokens. Observed cases:

| Corrupted output | Correct term | Commits fixing it |
|-----------------|--------------|-------------------|
| `ruclear` | `nuclear` | `423150a` |
| `obever` | `observer` | `e045803` |

The pattern is consistent across many independent inputs, suggesting a systematic tokeniser boundary issue rather than random noise. The leading subword of the correct token is dropped during beam-search decoding, resulting in a stable misspelling. Manual post-processing of the CSV is the current mitigation.

### Geopolitical mode generic-term exclusion

Terms `{"peoples", "states", "united nations"}` are filtered out at render time for the Geopolitical panel because they appear so frequently across the corpus that they dominate the chart without adding geographic specificity (`_EXCLUDED_TERMS_BY_MODE` in `wordcloud_interactive.py:27`).

### Subjects panel dependency

The Subjects (category) panel requires both `resolution_subject_table` and `subject_table` to be available on the query engine at startup. If either is missing the mode is silently disabled and logs a warning.
