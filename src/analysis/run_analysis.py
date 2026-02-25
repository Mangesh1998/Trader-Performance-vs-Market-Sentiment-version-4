"""Part B Analysis - Fear vs Greed performance, behavior, segments, insights."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.features.metrics import create_analysis_features, create_trader_segments


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_output_dir() -> Path:
    config = _load_config()
    out = Path(__file__).resolve().parent.parent.parent / config["paths"]["figures_dir"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def q1_performance_by_sentiment(daily: pd.DataFrame, out_dir: Path) -> dict:
    """Q1: Does performance (PnL, win rate, drawdown) differ between Fear vs Greed days?"""
    sent = daily[daily["classification"].isin(["Fear", "Greed"])]
    if sent.empty:
        return {}
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # PnL by sentiment
    sns.boxplot(data=sent, x="classification", y="daily_pnl", ax=axes[0])
    axes[0].set_title("Daily PnL: Fear vs Greed")
    axes[0].axhline(0, color="gray", ls="--")
    
    # Win rate by sentiment
    sns.boxplot(data=sent, x="classification", y="win_rate", ax=axes[1])
    axes[1].set_title("Win Rate: Fear vs Greed")
    
    # Drawdown proxy by sentiment
    sns.boxplot(data=sent, x="classification", y="drawdown_proxy", ax=axes[2])
    axes[2].set_title("Drawdown Proxy: Fear vs Greed")
    
    plt.tight_layout()
    plt.savefig(out_dir / "q1_performance_by_sentiment.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    stats = sent.groupby("classification").agg(
        mean_pnl=("daily_pnl", "mean"),
        median_pnl=("daily_pnl", "median"),
        mean_win_rate=("win_rate", "mean"),
        mean_drawdown=("drawdown_proxy", "mean"),
    ).round(4)
    
    return stats.to_dict()


def q2_behavior_by_sentiment(daily: pd.DataFrame, out_dir: Path) -> dict:
    """Q2: Do traders change behavior (frequency, leverage, long/short, position sizes)?"""
    sent = daily[daily["classification"].isin(["Fear", "Greed"])]
    if sent.empty:
        return {}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.boxplot(data=sent, x="classification", y="num_trades", ax=axes[0, 0])
    axes[0, 0].set_title("Trade Frequency")
    
    sns.boxplot(data=sent, x="classification", y="avg_leverage", ax=axes[0, 1])
    axes[0, 1].set_title("Leverage (proxy)")
    
    sns.boxplot(data=sent, x="classification", y="long_short_ratio", ax=axes[1, 0])
    axes[1, 0].set_title("Long/Short Ratio")
    
    sns.boxplot(data=sent, x="classification", y="avg_trade_size_usd", ax=axes[1, 1])
    axes[1, 1].set_title("Avg Position Size (USD)")
    
    plt.tight_layout()
    plt.savefig(out_dir / "q2_behavior_by_sentiment.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    return sent.groupby("classification")[["num_trades", "avg_leverage", "long_short_ratio", "avg_trade_size_usd"]].mean().to_dict()


def q3_segments(daily: pd.DataFrame, segments: pd.DataFrame, out_dir: Path) -> dict:
    """Q3: Identify segments - high/low leverage, frequent/infrequent, consistent vs inconsistent."""
    daily_with_seg = daily.merge(
        segments[["Account", "leverage_segment", "frequency_segment", "consistency_segment"]],
        on="Account",
        how="left",
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    sns.barplot(
        data=daily_with_seg.groupby("leverage_segment")["daily_pnl"].mean().reset_index(),
        x="leverage_segment", y="daily_pnl", ax=axes[0],
    )
    axes[0].set_title("Avg PnL: High vs Low Leverage")
    
    sns.barplot(
        data=daily_with_seg.groupby("frequency_segment")["daily_pnl"].mean().reset_index(),
        x="frequency_segment", y="daily_pnl", ax=axes[1],
    )
    axes[1].set_title("Avg PnL: Frequent vs Infrequent")
    
    sns.barplot(
        data=daily_with_seg.groupby("consistency_segment")["daily_pnl"].mean().reset_index(),
        x="consistency_segment", y="daily_pnl", ax=axes[2],
    )
    axes[2].set_title("Avg PnL: Consistent vs Inconsistent")
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15, ha="right")
    
    plt.tight_layout()
    plt.savefig(out_dir / "q3_segments.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    return {
        "leverage": daily_with_seg.groupby("leverage_segment")["daily_pnl"].mean().to_dict(),
        "frequency": daily_with_seg.groupby("frequency_segment")["daily_pnl"].mean().to_dict(),
        "consistency": daily_with_seg.groupby("consistency_segment")["daily_pnl"].mean().to_dict(),
    }


def q4_insights_charts(daily: pd.DataFrame, out_dir: Path) -> None:
    """Q4: At least 3 insights backed by charts/tables."""
    # Insight 1: Sentiment distribution over time
    fig, ax = plt.subplots(figsize=(12, 4))
    daily["date_only"] = pd.to_datetime(daily["date"]).dt.date
    ct = daily.groupby(["date_only", "classification"]).size().unstack(fill_value=0)
    ct.plot(kind="area", stacked=True, ax=ax)
    ax.set_title("Insight 1: Sentiment Distribution Over Time")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "insight1_sentiment_over_time.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    # Insight 2: PnL distribution by sentiment (violin)
    fig, ax = plt.subplots(figsize=(8, 4))
    sent = daily[daily["classification"].isin(["Fear", "Greed"])]
    if not sent.empty:
        sns.violinplot(data=sent, x="classification", y="daily_pnl", ax=ax)
        ax.set_title("Insight 2: PnL Distribution - Fear vs Greed")
        ax.axhline(0, color="gray", ls="--")
    plt.tight_layout()
    plt.savefig(out_dir / "insight2_pnl_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    
    # Insight 3: Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ["daily_pnl", "win_rate", "num_trades", "avg_leverage", "long_short_ratio", "avg_trade_size_usd", "drawdown_proxy"]
    num_cols = [c for c in num_cols if c in daily.columns]
    corr = daily[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Insight 3: Feature Correlations")
    plt.tight_layout()
    plt.savefig(out_dir / "insight3_correlation.png", dpi=120, bbox_inches="tight")
    plt.close()


def run_full_analysis(trader_merged: pd.DataFrame, out_dir: Optional[Path] = None) -> dict:
    """Run Part B analysis end-to-end."""
    out_dir = out_dir or _ensure_output_dir()
    
    daily = create_analysis_features(trader_merged)
    segments = create_trader_segments(daily)
    
    results = {}
    results["q1"] = q1_performance_by_sentiment(daily, out_dir)
    results["q2"] = q2_behavior_by_sentiment(daily, out_dir)
    results["q3"] = q3_segments(daily, segments, out_dir)
    q4_insights_charts(daily, out_dir)
    
    # Save summary tables
    daily.to_csv(out_dir.parent / "daily_metrics.csv", index=False)
    segments.to_csv(out_dir.parent / "trader_segments.csv", index=False)
    
    return results
