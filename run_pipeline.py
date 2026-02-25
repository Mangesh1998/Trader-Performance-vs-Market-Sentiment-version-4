"""
Primetrade Assignment - Full Pipeline Runner

Runs all stages: Data → Prepare → Features → Analysis → Model → Actionable Strategies
"""

from pathlib import Path
import sys

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import yaml
import pandas as pd

from src.data.load_data import load_fear_greed, load_trader_data
from src.data.prepare_data import prepare_merged_data, create_daily_metrics
from src.features.metrics import create_analysis_features, create_trader_segments, create_model_features
from src.analysis.run_analysis import run_full_analysis
from src.models.train_model import train_profitability_model, evaluate_model
from src.actionable.strategies import generate_actionable_strategies


def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("Primetrade Assignment - Full Pipeline")
    print("=" * 60)

    config = load_config()
    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_dir"]
    outputs_dir = PROJECT_ROOT / config["paths"]["outputs_dir"]
    figures_dir = PROJECT_ROOT / config["paths"]["figures_dir"]
    models_dir = PROJECT_ROOT / config["paths"]["models_dir"]

    for d in [data_dir, processed_dir, outputs_dir, figures_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Part A: Data Loading ----
    print("\n[1/7] Loading datasets...")
    try:
        fear_greed = load_fear_greed(str(data_dir))
        trader = load_trader_data(str(data_dir))
    except FileNotFoundError as e:
        print(f"\n⚠️  Data files not found. Please download from the assignment links and place in: {data_dir}")
        print("  Fear & Greed: https://drive.google.com/file/d/1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf/view")
        print("  Trader:       https://drive.google.com/file/d/1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs/view")
        print(f"\nError: {e}")
        return 1

    print(f"  Fear & Greed: {fear_greed.shape[0]} rows, {fear_greed.shape[1]} cols")
    print(f"  Trader:       {trader.shape[0]} rows, {trader.shape[1]} cols")
    print(f"  Missing: FG={fear_greed.isnull().sum().sum()}, Trader={trader.isnull().sum().sum()}")
    print(f"  Duplicates: FG={fear_greed.duplicated().sum()}, Trader={trader.duplicated().sum()}")

    # ---- Part A: Prepare & Merge ----
    print("\n[2/7] Preparing and merging data (align by date)...")
    fg_prepared, trader_merged = prepare_merged_data(fear_greed, trader)
    daily = create_daily_metrics(trader_merged)
    daily = create_analysis_features(trader_merged)
    segments = create_trader_segments(daily)

    print(f"  Merged daily metrics: {daily.shape[0]} rows")
    print(f"  Unique accounts: {daily['Account'].nunique()}")

    # ---- Part B: Analysis ----
    print("\n[3/7] Running Part B analysis (charts & tables)...")
    results = run_full_analysis(trader_merged, figures_dir)
    print(f"  Charts saved to: {figures_dir}")

    # ---- Part C: Actionable Strategies ----
    print("\n[4/7] Generating actionable strategies...")
    strategies = generate_actionable_strategies(daily, results.get("q3", {}), results, outputs_dir)
    for s in strategies:
        print(f"  - {s['name']}: {s['rule'][:80]}...")

    # ---- Bonus: Model ----
    print("\n[5/7] Training predictive model (next-day profitability bucket)...")
    X, y = create_model_features(daily, n_buckets=config["model"]["profitability_buckets"])

    if len(X) > 50 and y.nunique() >= 2:
        clf, metrics, le = train_profitability_model(
            X, y,
            imbalance_strategy=config["model"]["imbalance_strategy"],
            random_state=config["model"]["random_state"],
            save_path=models_dir,
        )
        print(f"  Model metrics: F1-weighted={metrics.get('f1_weighted', 0):.4f}, F1-macro={metrics.get('f1_macro', 0):.4f}")
        # Highlight loss recall (if available) and ROC-AUC
        if 'recall_loss' in metrics:
            print(f"  Recall (loss): {metrics['recall_loss']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        # Baseline comparison
        baseline = metrics.get('baseline') or {}
        if baseline:
            print(f"  Baseline (most-frequent) - Accuracy: {baseline.get('accuracy', 0):.4f}, F1-weighted: {baseline.get('f1_weighted', 0):.4f}")

        # Evaluate and save report (use scaler from saved model)
        from sklearn.model_selection import train_test_split
        with open(models_dir / "model.pkl", "rb") as f:
            saved = pickle.load(f)
        scaler = saved["scaler"]
        # Save feature importances (if present) to outputs
        try:
            fi = saved.get("feature_importances")
            if fi:
                pd.DataFrame.from_dict(fi, orient="index", columns=["importance"]).sort_values(
                    "importance", ascending=False
                ).to_csv(outputs_dir / "feature_importances.csv")
                print(f"  Feature importances saved to: {outputs_dir / 'feature_importances.csv'}")
        except Exception:
            pass
        X_train, X_test, y_train, y_test = train_test_split(
            X, le.transform(y.astype(str)), test_size=config["model"]["test_size"],
            random_state=config["model"]["random_state"], stratify=le.transform(y.astype(str))
        )
        X_test_scaled = scaler.transform(X_test)
        evaluate_model(clf, X_test_scaled, y_test, le, outputs_dir)
    else:
        print("  Skipped: insufficient data or classes for modeling")

    print("\n[6/7] Saving processed outputs...")
    daily.to_csv(outputs_dir / "daily_metrics.csv", index=False)
    segments.to_csv(outputs_dir / "trader_segments.csv", index=False)
    trader_merged.to_csv(processed_dir / "trader_merged.csv", index=False)

    print("\n[7/7] Done!")
    print("=" * 60)
    print("Outputs:")
    print(f"  - Figures:   {figures_dir}")
    print(f"  - Metrics:   {outputs_dir / 'daily_metrics.csv'}")
    print(f"  - Segments:  {outputs_dir / 'trader_segments.csv'}")
    print(f"  - Model:     {models_dir / 'model.pkl'}")
    print("\nRun dashboard: streamlit run streamlit_app.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
