"""
从 Kaggle 下载真实 Elo 评分数据并缓存到本地。
数据源：International Football Elo Ratings (1872-2025)
https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_and_cache():
    """Download latest Elo ratings from Kaggle and save to cache."""
    try:
        import kagglehub
    except ImportError:
        print("需要安装 kagglehub: pip install kagglehub")
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("需要安装 pandas: pip install pandas")
        sys.exit(1)

    # Import local cache saver
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.model.data import save_ratings

    print("正在从 Kaggle 下载 Elo 评分数据...")
    path = kagglehub.dataset_download("saifalnimri/international-football-elo-ratings")
    csv_path = Path(path) / "eloratings.csv"

    df = pd.read_csv(csv_path)
    df["date_parsed"] = pd.to_datetime(df["date"], format="mixed")

    # Only use data from 2023 onwards for current ratings
    modern = df[df["date_parsed"] > "2023-01-01"].copy()
    print(f"  原始数据: {len(df)} 条记录")
    print(f"  2023年后数据: {len(modern)} 条记录")

    # Normalize non-breaking spaces in team names
    modern["team_clean"] = modern["team"].str.replace("\xa0", " ", regex=False)

    # Get latest rating per team
    latest = (
        modern.sort_values("date_parsed")
        .groupby("team_clean")
        .last()
        .reset_index()
        .sort_values("rating", ascending=False)
    )
    latest = latest[latest["rating"] > 0]

    # Convert to dict
    ratings = dict(zip(latest["team_clean"], latest["rating"].astype(int)))
    print(f"  获取到 {len(ratings)} 支球队的数据")
    print(f"  最新数据截止: {latest['date_parsed'].max().date()}")

    # Print top 10
    print("\n  Top 10 Elo 评分:")
    for i, (_, row) in enumerate(latest.head(10).iterrows(), 1):
        print(f"    {i}. {row.team_clean:25s} {int(row.rating)}")

    # Save to cache
    save_ratings(ratings)
    print(f"\n✅ 已缓存 {len(ratings)} 支球队的 Elo 评分")
    print(f"   缓存位置: {Path(__file__).parent / '.elo_cache.pkl'}")
    print("   下次运行模型时会自动使用缓存数据")


def main():
    parser = argparse.ArgumentParser(description="下载 Elo 评分数据")
    args = parser.parse_args()
    fetch_and_cache()


if __name__ == "__main__":
    main()
