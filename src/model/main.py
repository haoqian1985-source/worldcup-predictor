#!/usr/bin/env python3
"""世界杯比赛预测工具 — 数据模型版（Elo + 泊松 + 蒙特卡洛）"""

import argparse
import logging
import sys

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.model.data import GROUPS, get_all_teams
from src.model.elo import EloCalculator
from src.model.poisson import PoissonModel
from src.model.monte_carlo import TournamentSimulator
from src.model.display import (
    display_single_match,
    display_tournament_results,
    display_group_stage,
    display_team_info,
)

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="世界杯比赛预测工具（数据模型版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单场预测
  python main.py "France vs Brazil"
  python main.py --team1 Argentina --team2 Germany

  # 蒙特卡洛模拟（整个赛程）
  python main.py --simulate
  python main.py --simulate --n 5000
  python main.py --simulate --top 10

  # 查看信息
  python main.py --groups
  python main.py --team Brazil
        """,
    )
    parser.add_argument("match", nargs="?", help="比赛对阵，格式如 'France vs Brazil'")
    parser.add_argument("--team1", help="球队 A")
    parser.add_argument("--team2", help="球队 B")
    parser.add_argument("--simulate", action="store_true",
                        help="运行完整蒙特卡洛赛程模拟")
    parser.add_argument("--n", type=int, default=10000,
                        help="蒙特卡洛模拟次数 (默认 10000)")
    parser.add_argument("--top", type=int, default=16,
                        help="显示前 N 名 (默认 16)")
    parser.add_argument("--groups", action="store_true",
                        help="显示小组赛分组")
    parser.add_argument("--team", help="查询球队信息")
    return parser.parse_args()


def parse_match_arg(match_str: str):
    for sep in [" vs ", " VS ", " v ", " V "]:
        if sep in match_str:
            parts = match_str.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    raise ValueError(f"无法解析比赛对阵: {match_str}")


def do_match_prediction(team_a: str, team_b: str):
    """Run single match prediction."""
    elo = EloCalculator()
    poisson = PoissonModel(elo)
    elo_pred = elo.predict_match(team_a, team_b)
    poisson_pred = poisson.predict(team_a, team_b)
    display_single_match(team_a, team_b, elo_pred, poisson_pred)


def do_simulation(n: int, top_n: int):
    """Run full tournament simulation."""
    sim = TournamentSimulator()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"模拟 {n} 次世界杯赛程...", total=None)
        sim.run_simulation(n)

    results = sim.get_ranked_results(top_n=top_n)
    display_tournament_results(results, top_n=top_n)
    return results


def do_team_info(team: str):
    """Show team info."""
    if team not in get_all_teams():
        available = [t for t in get_all_teams() if team in t]
        if available:
            team = available[0]
        else:
            console.print(f"[red]未找到球队 '{team}'[/red]")
            return

    elo = EloCalculator()
    rating = elo.get_rating(team)
    champion_prob = 0.0
    best_round_prob = 0.0

    logger.info("Running quick simulation for team data...")
    sim = TournamentSimulator()
    sim.run_simulation(5000)
    results = sim.get_ranked_results()
    for r in results:
        if r.team == team:
            champion_prob = r.champion_prob
            best_round_prob = r.round_of_32_prob
            break

    display_team_info(team, rating, champion_prob, best_round_prob)


def main():
    args = parse_args()

    if args.groups:
        display_group_stage()
        return

    if args.team:
        do_team_info(args.team)
        return

    if args.simulate:
        do_simulation(args.n, args.top)
        return

    # Single match prediction
    if args.match:
        team_a, team_b = parse_match_arg(args.match)
    elif args.team1 and args.team2:
        team_a, team_b = args.team1, args.team2
    else:
        console.print("[red]错误: 请指定比赛、使用 --simulate、--groups 或 --team[/red]")
        console.print("试试: python main.py \"巴西 vs 法国\"")
        sys.exit(1)

    do_match_prediction(team_a, team_b)


if __name__ == "__main__":
    main()
