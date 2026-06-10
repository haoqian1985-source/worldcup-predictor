"""泊松分布进球预测模型 — 预测比分和比赛结果概率。"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import poisson

from src.model.elo import EloCalculator


@dataclass
class ScoreProbability:
    score: str = ""
    probability: float = 0.0


@dataclass
class MatchPrediction:
    home_goals_lambda: float = 0.0
    away_goals_lambda: float = 0.0
    score_matrix: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    most_likely_scores: list[ScoreProbability] = field(default_factory=list)
    home_win_prob: float = 0.0
    draw_prob: float = 0.0
    away_win_prob: float = 0.0
    over_2_5_prob: float = 0.0
    both_teams_score_prob: float = 0.0


class PoissonModel:
    """Poisson-based goal prediction model."""

    # Base goals per match (World Cup average)
    BASE_GOALS = 1.2

    def __init__(self, elo_calculator: EloCalculator):
        self.elo = elo_calculator

    def predict(self, team_a: str, team_b: str,
                max_goals: int = 5) -> MatchPrediction:
        """Predict match score distribution using Poisson."""
        # Get team strengths from Elo
        attack_a = self.elo.get_team_strength(team_a)
        attack_b = self.elo.get_team_strength(team_b)
        defense_a = self.elo.get_team_defense(team_a)
        defense_b = self.elo.get_team_defense(team_b)

        # Calculate expected goals (lambda) for each team
        lambda_a = attack_a * defense_b * self.BASE_GOALS
        lambda_b = attack_b * defense_a * self.BASE_GOALS

        # Build score probability matrix (0 to max_goals)
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_a = poisson.pmf(i, lambda_a)
                prob_b = poisson.pmf(j, lambda_b)
                score_matrix[i, j] = prob_a * prob_b

        # Normalize
        score_matrix /= score_matrix.sum()

        # Most likely scores
        flat_indices = np.argsort(score_matrix.flatten())[::-1][:5]
        most_likely = []
        for idx in flat_indices:
            i, j = np.unravel_index(idx, score_matrix.shape)
            most_likely.append(ScoreProbability(
                score=f"{i}-{j}",
                probability=round(float(score_matrix[i, j]), 4),
            ))

        # Win/draw probabilities
        home_win = sum(score_matrix[i, j].item()
                       for i in range(max_goals + 1)
                       for j in range(max_goals + 1)
                       if i > j)
        draw = sum(score_matrix[i, i].item()
                   for i in range(max_goals + 1))
        away_win = sum(score_matrix[i, j].item()
                       for i in range(max_goals + 1)
                       for j in range(max_goals + 1)
                       if i < j)

        # Over 2.5 goals probability
        over_2_5 = sum(score_matrix[i, j].item()
                       for i in range(max_goals + 1)
                       for j in range(max_goals + 1)
                       if i + j > 2)

        # Both teams score probability
        both_score = sum(score_matrix[i, j].item()
                         for i in range(1, max_goals + 1)
                         for j in range(1, max_goals + 1))

        return MatchPrediction(
            home_goals_lambda=round(lambda_a, 3),
            away_goals_lambda=round(lambda_b, 3),
            score_matrix=score_matrix,
            most_likely_scores=most_likely,
            home_win_prob=round(home_win, 4),
            draw_prob=round(draw, 4),
            away_win_prob=round(away_win, 4),
            over_2_5_prob=round(over_2_5, 4),
            both_teams_score_prob=round(both_score, 4),
        )

    def simulate_score(self, team_a: str, team_b: str) -> tuple[int, int]:
        """Simulate a single match score using Poisson distribution."""
        attack_a = self.elo.get_team_strength(team_a)
        attack_b = self.elo.get_team_strength(team_b)
        defense_a = self.elo.get_team_defense(team_a)
        defense_b = self.elo.get_team_defense(team_b)

        lambda_a = attack_a * defense_b * self.BASE_GOALS
        lambda_b = attack_b * defense_a * self.BASE_GOALS

        goals_a = int(np.random.poisson(lambda_a))
        goals_b = int(np.random.poisson(lambda_b))

        return goals_a, goals_b
