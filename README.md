# Trader Performance vs Market Sentiment Analysis

**Primetrade.ai Data Science Intern Assignment – Version 4**

Analyzes how Bitcoin Fear & Greed sentiment relates to trader behavior and performance on Hyperliquid. Features binary classification for next-day profitability prediction with SHAP-based local explanations, feature importance interpretation, and actionable trading strategies.

---

## Key Features

* **Binary Classification Model** - Predicts next-day profitability (loss = bottom 25%, profit = top 75%)
* **Interactive Prediction Page** - Select any daily record and get probability predictions with adjustable threshold
* **Feature Importance** - Global model explanations with ranked contributions
* **Loss Detection Focus** - Balanced class weighting and minority class amplification
* **Actionable Strategies** - Data-driven rules based on trader segments and model insights

---

## Project Structure

```
primetrade_assignment/
|-- config/                     # Configuration
|-- data/                       # Data (raw, processed)
|-- src/
|   |-- data/                   # Data loading & preparation
|   |-- features/               # Feature engineering
|   |-- analysis/               # Analysis & visualizations
|   |-- models/                 # Predictive model (binary classification + SHAP)
|   |-- actionable/             # Strategies
|-- scripts/                    # Download utilities
|-- outputs/                    # Generated metrics, charts, reports
|-- models/                     # Trained model pickle
|-- run_pipeline.py             # Full pipeline runner
|-- streamlit_app.py            # Interactive dashboard
|-- requirements.txt            # Dependencies
|-- .gitignore                  # Git ignore rules
|-- README.md                   # This file
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/Mangesh1998/Trader-Performance-vs-Market-Sentiment-version-4.git
cd Trader-Performance-vs-Market-Sentiment-version-4
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/download_data.py
```

Or manually place CSVs in `data/raw/`:
- Fear & Greed: https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view
- Trader Data: https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view

### 3. Run Pipeline

```bash
python run_pipeline.py
```

Generates outputs in:
- `outputs/daily_metrics.csv` - Processed metrics
- `outputs/trader_segments.csv` - Trader segments
- `outputs/classification_report.csv` - Model metrics
- `outputs/confusion_matrix.csv` - Confusion matrix
- `outputs/feature_importances.csv` - Feature importance
- `models/model.pkl` - Trained model

### 4. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501

---

## Dashboard Pages

* **Overview** - Project info
* **Data Summary** - Metrics & segments
* **Analysis Charts** - Q1/Q2/Q3 visualizations
* **Segments** - Trader breakdown
* **Strategies** - Rules + model insights
* **Prediction** - Predict & explain with SHAP
* **Model Performance** - Metrics + feature importance

---

## Binary Classification Model

**Target Classes:**
- `loss`: Bottom 25% of next-day PnL (risky days)
- `profit`: Top 75% of next-day PnL (profitable days)

**Why binary?** Clearer boundaries, reduces imbalance, improves loss detection.

**Imbalance Handling:**
- SMOTE: Oversamples minority class during training
- Class weights: Balanced + 2x minority amplification
- Baseline: Most-frequent classifier for comparison

**Key Metrics:**
- Accuracy, Precision, Loss Recall (critical!), F1-score
- ROC-AUC: Probability ranking quality

---

## Prediction Page Features

1. **Record Selection** - Pick account + date from daily metrics
2. **Threshold Slider** - Adjust loss probability threshold (0.0-1.0)
3. **Predictions** - Model probability + threshold-adjusted label
4. **SHAP Explanations** - Feature-level impact for prediction
5. **Recommendations** - Risk actions based on predicted probability
6. **Feature Values** - Top 5 global importance features displayed

---

## Configuration

Edit `config/config.yaml`:

```yaml
model:
  imbalance_strategy: "smote"      # smote, class_weight, undersample
  profitability_buckets: 2         # Binary: loss/profit
  random_state: 42
  test_size: 0.2
```

---

## Dependencies

* pandas, numpy
* scikit-learn, imbalanced-learn (SMOTE)
* matplotlib, seaborn (visualization)
* streamlit (dashboard)
* shap (local explanations)
* pyyaml (config)

See `requirements.txt` for exact versions.

---

## Key Insights

1. Binary classification significantly outperforms multi-class for imbalanced trader data
2. Loss recall is critical - use threshold slider to catch risky days
3. SHAP local explanations personalize predictions (what matters for this trader?)
4. Trader segments matter - high-leverage traders behave differently

---

## Contact

GitHub: Mangesh1998
https://github.com/Mangesh1998/Trader-Performance-vs-Market-Sentiment-version-4

Last updated: February 26, 2026
