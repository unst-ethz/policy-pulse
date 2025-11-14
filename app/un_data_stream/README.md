# unDataStream

A modular Python framework for processing, analyzing, and querying UN resolution data from multiple sources (General Assembly, Security Council, UNBIS thesaurus).

## Overview

**unDataStream** provides a clean, maintainable architecture for working with UN voting data. It handles data fetching, processing, normalization, and analysis while supporting extensibility for new data sources.

### Key Features

- 🔄 **Automatic Data Pipeline**: Fetches, processes, and caches UN resolution data
- 📊 **Agreement Matrix Analysis**: Pre-computed voting agreement scores between all country pairs
- 🔍 **Advanced Querying**: Filter resolutions by date, subject, and analyze voting patterns
- 💾 **Smart Caching**: Persistent storage to avoid re-processing data
- 🌳 **Subject Hierarchy**: Full UNBIS thesaurus integration with ancestor/descendant relationships
- 🧩 **Modular Design**: Extensible architecture for adding new data sources

## Architecture

```
unDataStream/
├── core/              # Base classes and abstractions
│   └── abstractions.py
├── fetchers/          # Data acquisition from remote sources
│   ├── ga_fetcher.py
│   ├── sc_fetcher.py
│   └── thesaurus_fetcher.py
├── processors/        # Data transformation and normalization
│   ├── ga_processor.py
│   ├── sc_processor.py
│   └── thesaurus_processor.py
├── data/              # Data orchestration and storage
│   ├── fetcher.py
│   ├── processor.py
│   ├── merger.py
│   └── repository.py  # Main entry point
└── analysis/          # Query engine and analytics
    ├── query_engine.py
    └── analyzer.py
```

## Installation

### Prerequisites

```bash
pip install pandas numpy pyyaml rdflib requests
```

### Configuration

Create a `config/data_sources.yaml` file with your data source URLs:

```yaml
debug: false
logs: true

paths:
  data: "data/"
  logs: "logs/"

data_sources:
  ga_resolutions:
    url: "https://example.com/ga_voting.csv"
  sc_resolutions:
    url: "https://example.com/sc_voting.csv"
  unbis_thesaurus:
    url: "https://example.com/unbist.ttl"
```

## Quick Start

### 1. Initialize the Data Repository

The `DataRepository` is your main entry point. It handles all data fetching, processing, and caching.

```python
from pathlib import Path
from unDataStream import DataRepository

# Initialize repository (will fetch and process data on first run)
config_path = Path('config/data_sources.yaml')
repo = DataRepository(config_path)

# On first run: fetches data, processes it, calculates agreement matrices, saves cache
# On subsequent runs: loads from cache (much faster!)
```

### 2. Access Processed Data

```python
# Get all processed data as DataFrames
data = repo.get_data()

resolution_table = data['resolution']           # All resolutions with metadata
resolution_subject_table = data['resolution_subject']  # Resolution-subject links
subject_table = data['subject']                 # UNBIS subjects
closure_table = data['closure']                 # Subject hierarchy

print(f"Loaded {len(resolution_table)} resolutions")
print(f"Loaded {len(subject_table)} subjects")
```

### 3. Access Agreement Matrices

```python
# Get pre-computed agreement matrices
agreement_matrices, country_columns = repo.get_agreement_data()

print(f"Agreement matrices for {len(agreement_matrices)} resolutions")
print(f"Countries: {country_columns[:5]}...")  # Show first 5 countries

# Access specific resolution's agreement matrix
resolution_id = 'A/78/L.1'
if resolution_id in agreement_matrices:
    matrix = agreement_matrices[resolution_id]
    print(f"Agreement matrix shape: {matrix.shape}")  # e.g., (193, 193)
```

### 4. Query Resolutions

Use the `ResolutionQueryEngine` for advanced filtering and analysis.

```python
from unDataStream import ResolutionQueryEngine

# Initialize query engine
query_engine = ResolutionQueryEngine(repo)

# Query resolutions by date range
recent_resolutions = query_engine.query_resolutions(
    start_date='2023-01-01',
    end_date='2023-12-31'
)
print(f"Found {len(recent_resolutions)} resolutions in 2023")

# Query resolutions by subject (with hierarchical expansion)
climate_resolutions = query_engine.query_resolutions(
    subject_ids=['http://metadata.un.org/thesaurus/1001156'],  # Climate change
    include_descendants=True  # Include narrower subjects
)
print(f"Found {len(climate_resolutions)} climate-related resolutions")

# Combined filters
filtered_resolutions = query_engine.query_resolutions(
    start_date='2020-01-01',
    end_date='2024-12-31',
    subject_ids=['http://metadata.un.org/thesaurus/1001156'],
    include_descendants=True
)
```

### 5. Analyze Country Agreement Patterns

```python
# Get detailed agreement scores (one row per resolution)
usa_agreements = query_engine.query_agreement_between_countries(
    country_code='US',
    average=False
)
print("Agreement scores shape:", usa_agreements.shape)
print(usa_agreements.head())
# Output:
#     undl_id    CA    UK    DE    FR    CN   ...
# 0  A/78/L.1  0.95  0.90  0.85  0.88  0.45  ...
# 1  A/78/L.2  0.92  0.88  0.80  0.85  0.50  ...

# Get average agreement scores (single row)
usa_avg_agreements = query_engine.query_agreement_between_countries(
    country_code='US',
    average=True
)
print("Average agreements:")
print(usa_avg_agreements)
# Output:
#   source_country  resolution_count    CA    UK    DE    FR    CN   ...
# 0             US               150  0.93  0.89  0.82  0.86  0.47  ...

# Analyze agreement for specific resolutions only
climate_res_ids = climate_resolutions['undl_id'].tolist()
usa_climate_agreements = query_engine.query_agreement_between_countries(
    country_code='US',
    resolution_ids=climate_res_ids,
    average=True
)

# Get summary statistics
stats = query_engine.get_agreement_summary_stats('US')
print("Top 5 countries agreeing with USA:")
print(stats.head())
# Output:
#   target_country  mean_agreement  std_agreement  count_resolutions  min  max
# 0             CA            0.93           0.12                145  0.65 1.00
# 1             UK            0.89           0.15                143  0.50 1.00
# 2             AU            0.87           0.16                144  0.48 1.00
```

### 6. Combine Queries for Advanced Analysis

```python
# Example: Analyze how USA and China vote on climate resolutions
climate_resolutions = query_engine.query_resolutions(
    start_date='2020-01-01',
    subject_ids=['http://metadata.un.org/thesaurus/1001156'],
    include_descendants=True
)

climate_ids = climate_resolutions['undl_id'].tolist()

# Get USA's agreement with all countries on climate issues
usa_climate = query_engine.query_agreement_between_countries(
    country_code='US',
    resolution_ids=climate_ids,
    average=True
)

# Get China's agreement with all countries on climate issues  
china_climate = query_engine.query_agreement_between_countries(
    country_code='CN',
    resolution_ids=climate_ids,
    average=True
)

# Compare USA-UK vs China-UK agreement on climate
print(f"USA-UK climate agreement: {usa_climate['UK'].values[0]:.3f}")
print(f"China-UK climate agreement: {china_climate['UK'].values[0]:.3f}")
```

## Data Structures

### Resolution Table
Contains all resolution metadata and voting data:
- `undl_id`: Unique resolution identifier
- `date`: Resolution date
- `title`: Resolution title
- `session`: Session number
- Country columns: `US`, `UK`, `FR`, etc. with votes (`Y`, `N`, `A`)
- Aggregate counts: `total_yes`, `total_no`, `total_abstentions`

### Agreement Matrices
3D structure: Dictionary mapping `undl_id` → 2D numpy array (n_countries × n_countries)

**Agreement Scoring:**
- Same vote (Y-Y, A-A, N-N): **1.0** (perfect agreement)
- One abstention (Y-A, A-N): **0.5** (partial agreement)
- Opposite votes (Y-N): **0.0** (complete disagreement)
- Missing votes: **NaN**

### Subject Tables
- `subject_table`: All UNBIS thesaurus subjects with labels
- `closure_table`: Hierarchical relationships (ancestor-descendant pairs)
- `resolution_subject_table`: Links resolutions to subjects

## Advanced Features

### Force Recalculation

If data sources are updated, force recalculation:

```python
# Check if agreement matrices are available
if repo.has_agreement_matrices():
    print("Agreement matrices ready!")

# Force recalculation (e.g., after updating data sources)
repo.force_recalculate_agreement_matrices()
```

### Working with Cached Data

The repository automatically caches processed data in the configured data directory:

```
data/
├── resolution_table.csv
├── resolution_subject_table.csv
├── subject_table.csv
├── closure_table.csv
└── agreement_matrices.pkl  # Binary format for fast loading
```

**Cache Behavior:**
- First run: Fetches from URLs → Processes → Saves cache (**slow**, ~1-5 minutes)
- Subsequent runs: Loads from cache (**fast**, ~1-5 seconds)

To force re-fetching, delete the cache files.

## Logging

Enable detailed logging in your config:

```yaml
debug: true  # Enable debug-level console output
logs: true   # Enable file logging
```

Logs are saved to `logs/un_resolution_analyzer.log` with detailed timestamps and function traces.

## Performance Tips

1. **Use caching**: Let the repository save processed data
2. **Filter early**: Use `query_resolutions()` to filter before agreement analysis
3. **Average mode**: Use `average=True` when you don't need per-resolution details
4. **Batch queries**: Query multiple resolutions at once instead of looping

## Example Workflow

```python
from pathlib import Path
from unDataStream import DataRepository, ResolutionQueryEngine

# 1. Initialize (first run will take a few minutes)
repo = DataRepository(Path('config/data_sources.yaml'))

# 2. Create query engine
query_engine = ResolutionQueryEngine(repo)

# 3. Find resolutions of interest
recent_resolutions = query_engine.query_resolutions(
    start_date='2023-01-01',
    subject_ids=['http://metadata.un.org/thesaurus/1001156'],  # Climate
    include_descendants=True
)

print(f"Analyzing {len(recent_resolutions)} recent climate resolutions")

# 4. Analyze voting patterns
resolution_ids = recent_resolutions['undl_id'].tolist()

for country in ['US', 'CN', 'DE', 'IN', 'BR']:
    agreements = query_engine.query_agreement_between_countries(
        country_code=country,
        resolution_ids=resolution_ids,
        average=True
    )
    print(f"\n{country} average agreements on climate:")
    print(agreements.iloc[0, 2:7])  # Show first 5 countries
```

## Extending the Framework

The modular design allows easy extension:

```python
from unDataStream.core import DatasetFetcher, DatasetProcessor

# Add new data source (e.g., Human Rights Council)
class HRCFetcher(DatasetFetcher):
    def fetch(self) -> pd.DataFrame:
        # Implement fetching logic
        pass

class HRCProcessor(DatasetProcessor):
    def process(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # Implement processing logic
        pass
```

## Troubleshooting

**Issue: "Agreement matrices not available"**
- Ensure data is processed: Check if `agreement_matrices.pkl` exists in data folder
- Force recalculation: `repo.force_recalculate_agreement_matrices()`

**Issue: "Country not found in country columns"**
- Check available countries: `query_engine.country_columns`
- Use exact column names from the data source

**Issue: Slow initialization**
- First run processes all data (expected)
- Check if cache files exist in data folder
- Verify network connection for remote data sources

## License

Part of the UN-ETH Project for analyzing UN resolution voting patterns.

## Contact

For questions or contributions, please contact the UN-ETH Project Team.
