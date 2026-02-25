"""Download Fear & Greed and Trader datasets from Google Drive."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def main():
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown")
        return 1

    urls = {
        "fear_greed": ("1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf", data_dir / config["data"]["fear_greed_file"]),
        "trader": ("1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs", data_dir / config["data"]["trader_file"]),
    }

    for name, (file_id, path) in urls.items():
        if path.exists():
            print(f"Already exists: {path}")
            continue
        print(f"Downloading {name}...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(path), quiet=False)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
