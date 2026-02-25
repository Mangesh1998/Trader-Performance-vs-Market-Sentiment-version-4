"""Feature engineering - daily metrics, trader segments, profitability buckets."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from src.data.prepare_data import create_daily_metrics


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_analysis_features(trader_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Create all metrics for Part B analysis:
    - daily PnL per account
    - win rate, avg trade size
    - leverage distribution (proxy)
    - num trades per day
    - long/short ratio
    """
    daily = create_daily_metrics(trader_merged)
    
    # Drawdown proxy: rolling max PnL - current cumulative PnL
    daily = daily.sort_values(["Account", "date"])
    daily["cumulative_pnl"] = daily.groupby("Account")["daily_pnl"].cumsum()
    daily["rolling_max_pnl"] = daily.groupby("Account")["cumulative_pnl"].cummax()
    daily["drawdown_proxy"] = daily["rolling_max_pnl"] - daily["cumulative_pnl"]
    
    # Fill NaN leverage with median
    if "avg_leverage" in daily.columns:
        daily["avg_leverage"] = daily["avg_leverage"].fillna(daily["avg_leverage"].median())
    
    return daily


def create_trader_segments(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Segment traders for Part B.3:
    - high vs low leverage
    - frequent vs infrequent
    - consistent winners vs inconsistent
    """
    agg = daily.groupby("Account").agg(
        avg_pnl=("daily_pnl", "mean"),
        total_trades=("num_trades", "sum"),
        avg_leverage=("avg_leverage", "mean"),
        win_rate_avg=("win_rate", "mean"),
        pnl_std=("daily_pnl", "std"),
    ).reset_index()
    
    agg["pnl_std"] = agg["pnl_std"].fillna(0)
    
    # Leverage segments: high vs low (median split)
    lev_med = agg["avg_leverage"].median()
    agg["leverage_segment"] = np.where(agg["avg_leverage"] > lev_med, "high", "low")
    
    # Frequency: frequent vs infrequent (median split on total_trades)
    freq_med = agg["total_trades"].median()
    agg["frequency_segment"] = np.where(agg["total_trades"] > freq_med, "frequent", "infrequent")
    
    # Consistent winners: high win rate + low PnL std vs inconsistent
    wr_med = agg["win_rate_avg"].median()
    std_med = agg["pnl_std"].median()
    agg["consistency_segment"] = np.where(
        (agg["win_rate_avg"] >= wr_med) & (agg["pnl_std"] <= std_med),
        "consistent_winner",
        "inconsistent",
    )
    
    return agg


def create_model_features(
    daily: pd.DataFrame, n_buckets: int = 2
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create features and target for predictive model.
    Target: next-day BINARY profitability classification.
    Classes: 'profit' (PnL >= 0) vs 'loss' (PnL < 0).
    Binary classification typically yields better results than multi-class.
    """
    config = _load_config()
    daily = daily.sort_values(["Account", "date"])
    
    # Shift daily_pnl to get next-day profitability
    daily["next_day_pnl"] = daily.groupby("Account")["daily_pnl"].shift(-1)
    daily = daily.dropna(subset=["next_day_pnl"])
    
    # Balanced target design: label the bottom 25% of next-day PnL as 'loss',
    # the remaining as 'profit'. This creates a clearer minority class
    # (loss ~25%) which helps the model focus on detecting risky days.
    q25 = daily["next_day_pnl"].quantile(0.25)
    daily["profitability_bucket"] = np.where(
        daily["next_day_pnl"] <= q25,
        "loss",
        "profit",
    )
    daily["profitability_bucket"] = daily["profitability_bucket"].astype(str)
    
    # Feature columns
    feature_cols = [
        "daily_pnl", "win_rate", "avg_trade_size_usd", "num_trades",
        "avg_leverage", "long_short_ratio", "drawdown_proxy",
        "classification",
    ]
    
    available = [c for c in feature_cols if c in daily.columns]
    X = daily[available].copy()
    
    # Encode classification (Fear=0, Greed=1, Neutral/Unknown=0.5)
    if "classification" in X.columns:
        m = {"Fear": 0.0, "Greed": 1.0, "Neutral": 0.5, "Unknown": 0.5}
        X["classification"] = X["classification"].astype(str).map(m).fillna(0.5)
    
    # Fill NaN
    X = X.fillna(X.median(numeric_only=True))
    
    y = daily["profitability_bucket"]
    
    return X, y
