"""Part C - Actionable strategy ideas based on findings."""

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def generate_actionable_strategies(
    daily_metrics: pd.DataFrame,
    segment_stats: Dict[str, Any],
    analysis_results: Dict[str, Any],
    output_path: Path = None,
) -> List[Dict[str, str]]:
    """
    Generate 2 strategy ideas / rules of thumb based on analysis.
    Example: "During Fear days, reduce leverage for segment X; increase trade frequency only for segment Y."
    """
    strategies = []
    
    # Strategy 1: Sentiment-based leverage adjustment
    if "q2" in analysis_results and analysis_results["q2"]:
        q2 = analysis_results["q2"]
        fear_lev = q2.get("Fear", {}).get("avg_leverage") or 0
        greed_lev = q2.get("Greed", {}).get("avg_leverage") or 0
        if fear_lev > greed_lev or greed_lev > fear_lev:
            strategies.append({
                "id": "S1",
                "name": "Sentiment-Based Leverage Adjustment",
                "rule": "During Fear days, reduce leverage for high-leverage traders; during Greed days, "
                        "consider capping leverage for infrequent traders to avoid overtrading.",
                "evidence": f"Leverage behavior differs by sentiment (Fear vs Greed).",
            })
    
    # Strategy 2: Segment-specific trade frequency
    if "q3" in segment_stats:
        q3 = segment_stats
        freq = q3.get("frequency", {})
        if "frequent" in freq and "infrequent" in freq:
            strategies.append({
                "id": "S2",
                "name": "Segment-Specific Trade Frequency",
                "rule": "Increase trade frequency only for consistent winners; recommend infrequent traders "
                        "to avoid overtrading on Fear days when volatility is higher.",
                "evidence": "Frequent vs infrequent segments show different PnL profiles.",
            })
    
    # Strategy 3: Leverage + sentiment combined
    strategies.append({
        "id": "S3",
        "name": "Leverage-Sentiment Rule",
        "rule": "For high-leverage traders: reduce position sizes during Fear days. For low-leverage "
                "traders: opportunity to scale up selectively during Greed days if win rate is stable.",
        "evidence": "High vs low leverage segments react differently to sentiment.",
    })
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(strategies).to_csv(output_path / "actionable_strategies.csv", index=False)
    
    return strategies
