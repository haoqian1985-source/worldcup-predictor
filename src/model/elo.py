"""Elo 评分系统 — 根据球队实力分差预测比赛胜负概率。"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.model.data import get_elo


@dataclass
class EloMatchPrediction:
    team_a_win: float = 0.0
    draw: float = 0.0
    team_b_win: float = 0.0
    team_a_new_elo: Optional[int] = None
    team_b_new_elo: Optional[int] = None


class EloCalculator:
    """Elo rating calculator for football match prediction."""

    # K-factors by match stage
    K_GROUP = 20      # 小组赛
    K_KNOCKOUT = 30   # 淘汰赛
    K_FINAL = 40      # 决赛

    # Home advantage (not used for neutral World Cup matches)
    HOME_ADVANTAGE = 0

    def __init__(self, k_factor: int = 20):
        self.k_factor = k_factor
        self.ratings: dict[str, int] = {}

    def get_rating(self, team: str) -> int:
        """Get current rating for a team."""
        return self.ratings.get(team, get_elo(team))

    def set_rating(self, team: str, rating: int):
        self.ratings[team] = rating

    def expected_score(self, elo_a: int, elo_b: int) -> float:
        """Expected score for team A against team B (0-1)."""
        diff = elo_b - elo_a
        return 1.0 / (1.0 + 10.0 ** (diff / 400.0))

    def predict_match(self, team_a: str, team_b: str,
                      stage: str = "group") -> EloMatchPrediction:
        """Predict match outcome using Elo ratings."""
        elo_a = self.get_rating(team_a)
        elo_b = self.get_rating(team_b)

        exp_a = self.expected_score(elo_a, elo_b)
        exp_b = 1.0 - exp_a

        # Convert expected score to win/draw/loss probabilities
        # Empirical adjustment for football
        draw_prob = 0.25 * (1.0 - abs(exp_a - exp_b))
        team_a_win = exp_a * (1.0 - draw_prob)
        team_b_win = exp_b * (1.0 - draw_prob)

        total = team_a_win + draw_prob + team_b_win
        team_a_win /= total
        draw_prob /= total
        team_b_win /= total

        # Calculate new Elo if we were to update
        k = self._get_k_factor(stage)
        new_a = elo_a + k * (0.5 - exp_a)  # assuming draw
        new_b = elo_b + k * (0.5 - exp_b)

        return EloMatchPrediction(
            team_a_win=round(team_a_win, 4),
            draw=round(draw_prob, 4),
            team_b_win=round(team_b_win, 4),
            team_a_new_elo=int(round(new_a)),
            team_b_new_elo=int(round(new_b)),
        )

    def update_ratings(self, team_a: str, team_b: str,
                       goals_a: int, goals_b: int,
                       stage: str = "group"):
        """Update Elo ratings after a match."""
        elo_a = self.get_rating(team_a)
        elo_b = self.get_rating(team_b)
        exp_a = self.expected_score(elo_a, elo_b)
        exp_b = 1.0 - exp_a

        if goals_a > goals_b:
            actual_a, actual_b = 1.0, 0.0
        elif goals_a < goals_b:
            actual_a, actual_b = 0.0, 1.0
        else:
            actual_a, actual_b = 0.5, 0.5

        k = self._get_k_factor(stage)
        self.ratings[team_a] = int(round(elo_a + k * (actual_a - exp_a)))
        self.ratings[team_b] = int(round(elo_b + k * (actual_b - exp_b)))

    def _get_k_factor(self, stage: str) -> int:
        if stage == "final":
            return self.K_FINAL
        elif stage in ("round_of_32", "round_of_16", "quarter", "semi"):
            return self.K_KNOCKOUT
        return self.K_GROUP

    def get_team_strength(self, team: str) -> float:
        """Normalized attack strength from Elo (used by Poisson model)."""
        elo = self.get_rating(team)
        # Map Elo 1400-2100 → strength 0.6-1.4
        return 0.6 + (elo - 1400) * (0.8 / 700)

    def get_team_defense(self, team: str) -> float:
        """Normalized defense weakness from Elo (lower = better defense)."""
        elo = self.get_rating(team)
        # Map Elo 1400-2100 → defense 1.4-0.6 (inverse)
        return 1.4 - (elo - 1400) * (0.8 / 700)
