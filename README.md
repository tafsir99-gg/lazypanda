# AI Data Cleaner

> Production-grade CLI tool for automated ML dataset cleaning.

**AI Data Cleaner** (`adc`) analyzes and cleans CSV datasets for Kaggle competitions and machine learning projects. It uses deterministic Python for all detection and cleaning tasks, and optionally calls Gemini AI only for semantic reasoning and human-readable explanations.

## Quick Start

```bash
# Install
uv pip install -e .

# Analyze a dataset
adc analyze data/train.csv

# Clean a dataset
adc clean data/train.csv

# Clean without any AI calls (fully free, fully deterministic)
adc clean data/train.csv --no-ai
```

## Features (v1.0)

- ✅ Missing value detection and imputation
- ✅ Duplicate row detection and removal
- ✅ Data type inference and correction
- ✅ Outlier detection (IQR + Z-score)
- ✅ High-cardinality column detection
- ✅ Constant and near-constant column detection
- ✅ Category consistency normalization
- ✅ Suspicious value detection
- ✅ Markdown + JSON reports
- ✅ Reproducible pipeline script export
- ✅ Optional Gemini AI explanations (with hard token budget)

## Documentation

See `docs/` for full usage guide and architecture details.

---

*Built with Python, pandas, Typer, Rich, and Google Gemini.*
