# LazyPanda 🐼 — The Interactive Terminal-First Dataset Sanitizer

**LazyPanda** is a professional, production-grade command-line application that analyzes, cleans, and purifies CSV datasets for machine learning projects, Kaggle competitions, and data engineering pipelines. 

## 💻 Live CLI Preview

**1. The Interactive Wizard Setup Block**
```shell
┌──────────────────────────────────────────────────────────────┐
│ LazyPanda 🐼 – Interactive Wizard                            │
│ Step-by-step guided cleaning mode                            │
│ Answer each prompt to configure your run.                    │
└──────────────────────────────────────────────────────────────┘

? Select AI model tier: 
  gemini-3.5-flash - Blazing fast, high-efficiency default
❯ gemini-3.1-pro   - Premium Flagship (Deep reasoning)
  gemini-2.5-flash - Stable production baseline
  Skip AI / 100% Offline
✔ Model: gemini-3.1-pro
[?] Max output tokens (default 8192): 8192
✔ Token budget: 8,192

⚠ GEMINI_API_KEY not found in environment.
[?] Enter your Gemini API key (input hidden): ***************************************
✔ API key set for this session
[?] Path to your CSV file: /Users/tafsiradm/Downloads/customer_churn_dirty.csv
✔ Input: /Users/tafsiradm/Downloads/customer_churn_dirty.csv
[?] Output directory (default ./outputs): /Users/tafsiradm/Downloads/cleaned2
✔ Output: /Users/tafsiradm/Downloads/cleaned2
```

**2. The Pre-Cleaning Analysis Progress Phase**
```shell
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Pre-Cleaning Analysis                               │
│ Scanning dataset for anomalies and missing values...         │
└──────────────────────────────────────────────────────────────┘

Running dataset profiling...
[████████████████████████░░░░░░░░░░░░] 65%

✔ type_inference......... DONE
✔ missing_values......... DONE
✔ cardinality............ DONE
✔ outliers............... DONE
✔ duplicates............. DONE
➔ category_consistency... RUNNING
  suspicious_values...... PENDING
```

**3. The Final Success Dashboard Matrix**
```shell
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: Cleaning Pipeline + AI (gemini-3.1-pro)             │
│ Running all cleaners and Gemini AI enrichment...             │
└──────────────────────────────────────────────────────────────┘

✨ AI Insights: Quality score 85/100 · 5 recommendation(s) · 2890 tokens used

                             Output Artifacts                              
┌──────────────┬────────────────────────────────────────────────────────┬────────┐
│ Artifact     │ Path                                                   │ Status │
├──────────────┼────────────────────────────────────────────────────────┼────────┤
│ Cleaned CSV  │ .../cleaned2/customer_churn_dirty_cleaned.csv          │ ✔ DONE │
│ Audit Report │ .../cleaned2/customer_churn_dirty_audit_report.md      │ ✔ DONE │
│ Dirty Report │ .../cleaned2/customer_churn_dirty_dirty_report.md      │ ✔ DONE │
└──────────────┴────────────────────────────────────────────────────────┴────────┘

✔ Wizard complete — 3 artifact(s) written to /Users/tafsiradm/Downloads/cleaned2
```


## ✨ Key Features

1. **Deterministic-First Pipeline:** All detection and cleaning operations are executed locally using strict pandas/numpy rules. Outliers are bounded mathematically, missing values are imputed statistically, and cardinality is resolved deterministically for reproducible, high-speed execution.
2. **Interactive Terminal Wizard UI:** A beautifully crafted, step-by-step CLI prompt interface powered by Rich and Questionary that guides you through safely preparing your dataset.
3. **3-Layer AI Truncation Defense:** An opt-in semantic enrichment layer powered by Gemini AI. We transmit only lightweight structural metadata (never raw rows) to generate human-readable insights. Our robust token budgeting and dynamic override system guarantees responses are never truncated mid-sentence, no matter how complex the data anomaly.

---

## 🚀 Global One-Line Installation

Install LazyPanda globally to your terminal instantly using `uv`:

```bash
uv tool install git+https://github.com/yourusername/lazypanda.git
```

Verify your installation:
```bash
lazypanda --help
```

---

## 🪄 Complete Interactive Walkthrough

The quickest way to sanitize a dataset is using the **Interactive Wizard**. Simply run:

```bash
lazypanda wizard
```

The wizard will guide you through the following steps:
1. **AI Model Selection:** Choose your semantic reasoning tier. You can select standard models (`gemini-3.5-flash`), premium flagship models for deep reasoning (`gemini-3.1-pro`), or skip AI entirely for 100% offline execution.
2. **Custom Token Constraints:** Define a custom `max_output_tokens` threshold to accommodate highly verbose structural audits without hitting Google SDK truncation limits.
3. **Path Routing:** Easily specify the input CSV file and securely declare the target directory for all output artifacts.
4. **Safety Review:** The wizard runs an initial diagnostic scan, presenting a beautiful terminal-based analysis table so you can safely review the damage before committing to destructive operations.

---

## 📦 The 3 Clean Artifacts

When you run a cleaning pass, LazyPanda generates three definitive enterprise-ready artifacts in your target directory, adhering to a strict file naming convention:

- **`{original_name}_cleaned.csv`**
  *(The purified production matrix)*
  Your fully sanitized dataset, ready for machine learning ingestion, training, or database uploads.

- **`{original_name}_dirty_report.md`**
  *(The "Before" baseline error profile)*
  A comprehensive Markdown report profiling the exact anomalies, outliers, and missing values detected before any transformations occurred.

- **`{original_name}_audit_report.md`**
  *(The "After" post-cleaning verification and semantic audit)*
  A human-readable semantic verification report containing the exact steps taken to purify the dataset, alongside deep natural-language insights generated by Gemini AI.

---

## ⚙️ Configuration Guidelines

To unlock the optional AI semantic reasoning layer, you need to provide a Google Gemini API Key.

1. Create a `.env` file in the directory where you run LazyPanda (or set it globally).
2. Add your API key:
   ```env
   GEMINI_API_KEY="your-google-gemini-api-key"
   ```

You can also create a local `config.yaml` to override mathematical thresholds (e.g., z-score caps, imputation strategies). 

*Happy Cleaning! 🐼*
