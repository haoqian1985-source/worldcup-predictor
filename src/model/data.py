"""球队真实 Elo 数据 + 2026 世界杯实际分组（数据来源：Kaggle Elo Ratings Dataset）。"""

import logging
import os
import pickle
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 2026 世界杯 12 个小组（每组 4 队，前 2 名出线→24队+8个第三名晋级→32强）
# 数据来源：openfootball/worldcup.json
GROUPS = {
    "A组": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B组": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C组": ["Brazil", "Haiti", "Morocco", "Scotland"],
    "D组": ["Australia", "Paraguay", "Turkey", "USA"],
    "E组": ["Curacao", "Ecuador", "Germany", "Ivory Coast"],
    "F组": ["Japan", "Netherlands", "Sweden", "Tunisia"],
    "G组": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H组": ["Cape Verde", "Saudi Arabia", "Spain", "Uruguay"],
    "I组": ["France", "Iraq", "Norway", "Senegal"],
    "J组": ["Algeria", "Argentina", "Austria", "Jordan"],
    "K组": ["Colombia", "DR Congo", "Portugal", "Uzbekistan"],
    "L组": ["Croatia", "England", "Ghana", "Panama"],
}


def _get_this_dir() -> Path:
    """Get the directory where fetch_elo.py should save data."""
    return Path(__file__).parent


# Cache file for Elo ratings
_ELO_CACHE_FILE = _get_this_dir() / ".elo_cache.pkl"


def _get_default_ratings() -> dict[str, int]:
    """Fallback Elo ratings if Kaggle data can't be loaded."""
    return {
        "Algeria": 1726, "Argentina": 2113, "Australia": 1774,
        "Austria": 1818, "Belgium": 1849, "Bosnia and Herzegovina": 1571,
        "Brazil": 1979, "Canada": 1802, "Cape Verde": 1560,
        "Colombia": 1998, "Croatia": 1933, "Curacao": 1467,
        "Czech Republic": 1731, "DR Congo": 1616, "Ecuador": 1933,
        "Egypt": 1591, "England": 2042, "France": 2062,
        "Germany": 1910, "Ghana": 1509, "Haiti": 1542,
        "Iran": 1754, "Iraq": 1582, "Ivory Coast": 1607,
        "Japan": 1878, "Jordan": 1687, "Mexico": 1835,
        "Morocco": 1830, "Netherlands": 1959, "New Zealand": 1586,
        "Norway": 1922, "Panama": 1742, "Paraguay": 1833,
        "Portugal": 1976, "Qatar": 1427, "Saudi Arabia": 1612,
        "Scotland": 1790, "Senegal": 1803, "South Africa": 1531,
        "South Korea": 1784, "Spain": 2171, "Sweden": 1660,
        "Switzerland": 1897, "Tunisia": 1641, "Turkey": 1880,
        "USA": 1747, "Uruguay": 1890, "Uzbekistan": 1735,
    }


# Load ratings: try cache first, fall back to defaults
_ELO_RATINGS: dict[str, int] | None = None


def _load_ratings() -> dict[str, int]:
    """Load Elo ratings from cache or fallback."""
    global _ELO_RATINGS
    if _ELO_RATINGS is not None:
        return _ELO_RATINGS

    cache_path = _ELO_CACHE_FILE
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                _ELO_RATINGS = pickle.load(f)
            logger.info("Loaded Elo ratings from cache (%d teams)", len(_ELO_RATINGS))
            return _ELO_RATINGS
        except Exception as e:
            logger.warning("Failed to load Elo cache: %s", e)

    _ELO_RATINGS = _get_default_ratings()
    logger.info("Using default Elo ratings (%d teams)", len(_ELO_RATINGS))
    return _ELO_RATINGS


def save_ratings(ratings: dict[str, int]):
    """Save Elo ratings to cache. Called by fetch_elo.py."""
    global _ELO_RATINGS
    _ELO_RATINGS = ratings
    cache_path = _ELO_CACHE_FILE
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(ratings, f)
        logger.info("Saved Elo ratings to cache (%d teams)", len(ratings))
    except Exception as e:
        logger.error("Failed to save Elo cache: %s", e)


def get_elo(team_name: str) -> int:
    """Get Elo rating for a team."""
    ratings = _load_ratings()
    return ratings.get(team_name, 1500)


def get_all_teams() -> list[str]:
    """Get all unique teams from all groups."""
    teams = set()
    for group_teams in GROUPS.values():
        for team in group_teams:
            teams.add(team)
    return sorted(teams)


def get_group_names() -> list[str]:
    """Get sorted group names."""
    return sorted(GROUPS.keys())
