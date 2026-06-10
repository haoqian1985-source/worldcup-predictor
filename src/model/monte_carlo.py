"""蒙特卡洛模拟 — 模拟整个世界杯赛程（12组×4队→32强淘汰赛），统计各队晋级概率。"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.model.data import GROUPS, get_all_teams
from src.model.elo import EloCalculator
from src.model.poisson import PoissonModel

logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class SimulationResult:
    team: str = ""
    round_of_32_count: int = 0
    round_of_16_count: int = 0
    quarter_count: int = 0
    semi_count: int = 0
    final_count: int = 0
    champion_count: int = 0
    total_simulations: int = 0

    @property
    def round_of_32_prob(self) -> float:
        return self.round_of_32_count / max(self.total_simulations, 1)

    @property
    def round_of_16_prob(self) -> float:
        return self.round_of_16_count / max(self.total_simulations, 1)

    @property
    def quarter_prob(self) -> float:
        return self.quarter_count / max(self.total_simulations, 1)

    @property
    def semi_prob(self) -> float:
        return self.semi_count / max(self.total_simulations, 1)

    @property
    def final_prob(self) -> float:
        return self.final_count / max(self.total_simulations, 1)

    @property
    def champion_prob(self) -> float:
        return self.champion_count / max(self.total_simulations, 1)


class TournamentSimulator:
    """Monte Carlo tournament simulator for 2026 World Cup (12 groups × 4 teams → R32)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.elo = EloCalculator()
        self.poisson = PoissonModel(self.elo)
        self.results: dict[str, SimulationResult] = {}

    def _simulate_group_stage(self) -> dict[str, dict[str, TeamStats]]:
        """Simulate all group matches. Returns {group_name: {team: TeamStats}}."""
        group_stats: dict[str, dict[str, TeamStats]] = {}

        for group_name, teams in GROUPS.items():
            stats = {t: TeamStats() for t in teams}
            # Each team plays every other team (4 teams → 6 matches per group)
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    team_a = teams[i]
                    team_b = teams[j]
                    goals_a, goals_b = self.poisson.simulate_score(team_a, team_b)
                    self.elo.update_ratings(team_a, team_b, goals_a, goals_b, "group")

                    stats[team_a].goals_for += goals_a
                    stats[team_a].goals_against += goals_b
                    stats[team_b].goals_for += goals_b
                    stats[team_b].goals_against += goals_a

                    if goals_a > goals_b:
                        stats[team_a].wins += 1
                        stats[team_a].points += 3
                        stats[team_b].losses += 1
                    elif goals_a < goals_b:
                        stats[team_b].wins += 1
                        stats[team_b].points += 3
                        stats[team_a].losses += 1
                    else:
                        stats[team_a].draws += 1
                        stats[team_a].points += 1
                        stats[team_b].draws += 1
                        stats[team_b].points += 1

            group_stats[group_name] = stats

        return group_stats

    def _select_knockout_teams(self, group_stats: dict[str, dict[str, TeamStats]]) -> list[str]:
        """Select 32 teams: top 2 from each group + 8 best third-placed."""
        qualified = []
        third_placed: list[tuple[str, dict[str, TeamStats]]] = []

        for group_name, teams in GROUPS.items():
            stats = group_stats[group_name]
            sorted_teams = sorted(
                teams,
                key=lambda t: (stats[t].points, stats[t].goal_diff, stats[t].goals_for),
                reverse=True,
            )
            qualified.extend(sorted_teams[:2])  # Top 2 advance
            third_placed.append((sorted_teams[2], stats))  # 3rd place

        # Sort third-placed teams across all groups by performance
        third_placed.sort(
            key=lambda x: (x[1][x[0]].points, x[1][x[0]].goal_diff, x[1][x[0]].goals_for),
            reverse=True,
        )
        # Top 8 third-placed teams advance
        qualified.extend(t[0] for t in third_placed[:8])

        return qualified

    def _simulate_knockout_match(self, team_a: str, team_b: str,
                                  stage: str) -> tuple[str, int, int]:
        """Simulate a single knockout match. Returns (winner, goals_a, goals_b)."""
        goals_a, goals_b = self.poisson.simulate_score(team_a, team_b)

        if goals_a == goals_b:
            # Penalties: stronger team slightly more likely to win
            elo_a = self.elo.get_rating(team_a)
            elo_b = self.elo.get_rating(team_b)
            pen_win_prob = 0.5 + (elo_a - elo_b) / 2000
            if self.rng.random() < pen_win_prob:
                goals_a += 1
            else:
                goals_b += 1

        self.elo.update_ratings(team_a, team_b, goals_a, goals_b, stage)

        if goals_a > goals_b:
            return team_a, goals_a, goals_b
        return team_b, goals_a, goals_b

    def _simulate_tournament(self) -> dict[str, int]:
        """Simulate one full tournament. Returns {team: max_round_reached}."""
        # Round mapping: 0=group exit, 1=R32, 2=R16, 3=QF, 4=SF, 5=Final, 6=Champion
        max_round: dict[str, int] = {t: 0 for t in get_all_teams()}

        # Group stage
        group_stats = self._simulate_group_stage()

        # Select 32 knockout teams
        qualified = self._select_knockout_teams(group_stats)
        for team in qualified:
            max_round[team] = 1  # at least R32

        # Seed teams by group performance for balanced bracket
        def perf_key(team):
            for gs in group_stats.values():
                if team in gs:
                    s = gs[team]
                    return (s.points, s.goal_diff, s.goals_for)
            return (0, 0, 0)

        qualified.sort(key=perf_key, reverse=True)

        # Round of 32: pair 1v32, 2v31, 3v30, ..., 16v17
        r32_winners = []
        for i in range(16):
            winner, _, _ = self._simulate_knockout_match(
                qualified[i], qualified[31 - i], "round_of_32")
            r32_winners.append(winner)
            max_round[winner] = 2

        # Round of 16
        r16_winners = []
        for i in range(0, 16, 2):
            winner, _, _ = self._simulate_knockout_match(
                r32_winners[i], r32_winners[i + 1], "round_of_16")
            r16_winners.append(winner)
            max_round[winner] = 3

        # Quarter-finals
        qf_winners = []
        for i in range(0, 8, 2):
            winner, _, _ = self._simulate_knockout_match(
                r16_winners[i], r16_winners[i + 1], "quarter")
            qf_winners.append(winner)
            max_round[winner] = 4

        # Semi-finals
        sf_winners = []
        for i in range(0, 4, 2):
            winner, _, _ = self._simulate_knockout_match(
                qf_winners[i], qf_winners[i + 1], "semi")
            sf_winners.append(winner)
            max_round[winner] = 5

        # Final
        champion, _, _ = self._simulate_knockout_match(
            sf_winners[0], sf_winners[1], "final")
        max_round[champion] = 6

        return max_round

    def run_simulation(self, n: int = 10000) -> dict[str, SimulationResult]:
        """Run N tournament simulations and aggregate results."""
        logger.info("Running %d tournament simulations...", n)

        all_teams = get_all_teams()
        self.results = {t: SimulationResult(team=t, total_simulations=n) for t in all_teams}

        for sim in range(n):
            if (sim + 1) % 1000 == 0:
                logger.info("  Simulation %d / %d", sim + 1, n)

            self.elo = EloCalculator()
            max_round = self._simulate_tournament()

            for team, round_reached in max_round.items():
                if round_reached >= 1:
                    self.results[team].round_of_32_count += 1
                if round_reached >= 2:
                    self.results[team].round_of_16_count += 1
                if round_reached >= 3:
                    self.results[team].quarter_count += 1
                if round_reached >= 4:
                    self.results[team].semi_count += 1
                if round_reached >= 5:
                    self.results[team].final_count += 1
                if round_reached >= 6:
                    self.results[team].champion_count += 1

        return self.results

    def get_ranked_results(self, top_n: Optional[int] = None) -> list[SimulationResult]:
        """Get results sorted by champion probability."""
        sorted_results = sorted(
            self.results.values(),
            key=lambda r: r.champion_prob,
            reverse=True,
        )
        if top_n:
            return sorted_results[:top_n]
        return sorted_results
