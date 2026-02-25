"""Data preparation pipeline - align datasets by date, clean, merge."""

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_fear_greed(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare Fear & Greed: normalize date, handle duplicates by date."""
    config = _load_config()
    date_col = config["data"]["date_column"]
    class_col = config["data"]["classification_column"]
    
    df = df.copy()
    
    # Ensure date column
    if "date" not in df.columns:
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
        else:
            raise ValueError("No date or timestamp column in Fear & Greed data")
    
    df["date"] = pd.to_datetime(df["date"])
    
    # Map classification (ensure Fear/Greed) - handles "Extreme Fear", "Fear", "Greed", etc.
    if class_col in df.columns:
        s = df[class_col].astype(str).str.strip()
        df[class_col] = "Neutral"
        df.loc[s.str.lower().str.contains("fear"), class_col] = "Fear"
        df.loc[s.str.lower().str.contains("greed"), class_col] = "Greed"
    
    # Deduplicate by date (keep last)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    
    return df


def prepare_trader_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare Trader data: parse Timestamp IST, derive date, clean."""
    config = _load_config()
    ts_col = config["data"]["timestamp_ist_column"]
    
    df = df.copy()
    
    if ts_col not in df.columns:
        raise ValueError(f"Column '{ts_col}' not found in Trader data")
    
    # Parse Timestamp IST
    df["datetime_ist"] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = df["datetime_ist"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    
    # Filter rows with valid dates
    df = df.dropna(subset=["date", "datetime_ist"])
    
    # Normalize column names
    if "Closed PnL" in df.columns:
        df["closed_pnl"] = df["Closed PnL"].astype(float)
    else:
        df["closed_pnl"] = 0.0
    
    return df


def prepare_merged_data(
    fear_greed: pd.DataFrame, trader: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align datasets by date and merge.
    Returns: (merged_daily_metrics, trader_daily_with_sentiment)
    """
    config = _load_config()
    fg = prepare_fear_greed(fear_greed)
    tr = prepare_trader_data(trader)
    
    # Merge trader with sentiment by date
    sentiment_lookup = fg[["date", config["data"]["classification_column"]]].drop_duplicates()
    tr_merged = tr.merge(sentiment_lookup, on="date", how="left")
    tr_merged["classification"] = tr_merged["classification"].fillna("Unknown")
    
    return fg, tr_merged


def create_daily_metrics(trader_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily metrics per account per date for analysis.
    Called from feature engineering pipeline.
    """
    # Group by account and date
    grp = trader_merged.groupby(["Account", "date"])
    
    daily = (
        grp.agg(
            daily_pnl=("closed_pnl", "sum"),
            num_trades=("Order ID", "nunique"),
            avg_trade_size_usd=("Size USD", "mean"),
            total_volume_usd=("Size USD", "sum"),
        )
        .reset_index()
    )
    
    # Win rate: trades with Closed PnL > 0
    winning = trader_merged[trader_merged["closed_pnl"] > 0].groupby(["Account", "date"]).size().reset_index(name="winning_trades")
    total = trader_merged.groupby(["Account", "date"]).size().reset_index(name="total_trades")
    wr = total.merge(winning, on=["Account", "date"], how="left").fillna(0)
    wr["win_rate"] = wr["winning_trades"] / wr["total_trades"].replace(0, 1)
    daily = daily.merge(wr[["Account", "date", "win_rate"]], on=["Account", "date"], how="left")
    
    # Long/Short ratio
    trader_merged["is_long"] = trader_merged["Side"].str.upper().eq("BUY").astype(int)
    ls = trader_merged.groupby(["Account", "date"]).agg(long_count=("is_long", "sum"), total=("is_long", "count")).reset_index()
    ls["long_short_ratio"] = ls["long_count"] / (ls["total"] - ls["long_count"] + 1e-6)
    daily = daily.merge(ls[["Account", "date", "long_short_ratio"]], on=["Account", "date"], how="left")
    
    # Leverage proxy: |Start Position| / (Size USD + 1) per trade, then mean per day
    trader_merged["leverage_proxy"] = (
        trader_merged["Start Position"].abs() / (trader_merged["Size USD"].abs() + 1)
    ).clip(upper=100)  # cap outliers
    lev = trader_merged.groupby(["Account", "date"])["leverage_proxy"].mean().reset_index(name="avg_leverage")
    daily = daily.merge(lev, on=["Account", "date"], how="left")

    # Add sentiment
    sent = trader_merged[["date", "classification"]].drop_duplicates()
    daily = daily.merge(sent, on="date", how="left")

    return daily
