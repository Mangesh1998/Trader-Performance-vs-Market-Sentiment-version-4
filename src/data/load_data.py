"""Data loading pipeline - loads Fear/Greed and Trader datasets."""

import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
import yaml


def _load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _download_file(url: str, save_path: Path) -> Path:
    """Download file from URL (handles Google Drive)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if url.startswith("https://drive.google.com"):
        # Google Drive - use gdown or direct download
        try:
            import gdown
            gdown.download(url, str(save_path), quiet=False, fuzzy=True)
        except ImportError:
            # Fallback: direct request
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return save_path


def load_fear_greed(data_dir: str = None) -> pd.DataFrame:
    """
    Load Bitcoin Fear & Greed dataset.
    Returns DataFrame with columns: timestamp, value, classification, date
    """
    config = _load_config()
    data_dir = data_dir or Path(__file__).resolve().parent.parent.parent / config["paths"]["data_dir"]
    data_dir = Path(data_dir)
    file_path = data_dir / config["data"]["fear_greed_file"]
    
    if not file_path.exists():
        url = config["data"]["fear_greed_url"]
        # Use gdown for Google Drive
        try:
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id=1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf",
                str(file_path), quiet=False
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Fear & Greed data not found at {file_path}. "
                f"Please download manually from the assignment link and place at {file_path}. Error: {e}"
            )
    
    df = pd.read_csv(file_path)
    
    # Ensure date column exists (Fear & Greed: timestamp int, value, classification, date object)
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_trader_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load Hyperliquid Trader dataset.
    Returns DataFrame with trader execution data.
    """
    config = _load_config()
    data_dir = data_dir or Path(__file__).resolve().parent.parent.parent / config["paths"]["data_dir"]
    data_dir = Path(data_dir)
    file_path = data_dir / config["data"]["trader_file"]
    
    if not file_path.exists():
        try:
            import gdown
            gdown.download(
                f"https://drive.google.com/uc?id=1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs",
                str(file_path), quiet=False
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Trader data not found at {file_path}. "
                f"Please download manually from the assignment link and place at {file_path}. Error: {e}"
            )
    
    df = pd.read_csv(file_path)
    return df


def document_dataset(df: pd.DataFrame, name: str) -> dict:
    """Document dataset statistics: rows, columns, missing, duplicates."""
    return {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_per_column": df.isnull().sum().to_dict(),
        "total_missing": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def load_all_data(data_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load both datasets and return documentation."""
    fear_greed = load_fear_greed(data_dir)
    trader = load_trader_data(data_dir)
    
    docs = {
        "fear_greed": document_dataset(fear_greed, "Fear & Greed"),
        "trader": document_dataset(trader, "Trader Data"),
    }
    return fear_greed, trader, docs
