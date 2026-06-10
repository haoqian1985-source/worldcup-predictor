"""展示模块 — Rich 终端美化输出。"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.bar import Bar
from rich import box
from rich.text import Text

from src.model.poisson import MatchPrediction
from src.model.monte_carlo import SimulationResult
from src.model.data import GROUPS

console = Console()


def display_single_match(team_a: str, team_b: str,
                         elo_pred, poisson_pred: MatchPrediction):
    """Show single match prediction with Elo + Poisson results."""

    # Header
    console.print()
    console.print(Panel(
        f"[bold yellow]{team_a}[/bold yellow] vs [bold yellow]{team_b}[/bold yellow]",
        subtitle="数据模型预测 (Elo + 泊松)",
        box=box.ROUNDED,
    ))

    # Elo probabilities
    console.print("\n[bold underline]📊 Elo 评分预测[/bold underline]")
    console.print(f"  {team_a} 胜: [cyan]{elo_pred.team_a_win*100:.1f}%[/cyan]  |  "
                  f"平局: [white]{elo_pred.draw*100:.1f}%[/white]  |  "
                  f"{team_b} 胜: [green]{elo_pred.team_b_win*100:.1f}%[/green]")

    # Poisson probabilities
    console.print(f"\n[bold underline]⚽ 泊松进球模型预测[/bold underline]")
    console.print(f"  {team_a} 预期进球 λ={poisson_pred.home_goals_lambda:.3f}")
    console.print(f"  {team_b} 预期进球 λ={poisson_pred.away_goals_lambda:.3f}")
    console.print()

    probs = poisson_pred
    bar_a = "█" * max(1, int(probs.home_win_prob * 40))
    bar_d = "█" * max(1, int(probs.draw_prob * 40))
    bar_b = "█" * max(1, int(probs.away_win_prob * 40))

    t = Table(box=box.SIMPLE, padding=(0, 2))
    t.add_column(team_a, style="cyan", justify="center")
    t.add_column("平局", style="white", justify="center")
    t.add_column(team_b, style="green", justify="center")
    t.add_row(
        f"{probs.home_win_prob*100:.1f}%\n{bar_a}",
        f"{probs.draw_prob*100:.1f}%\n{bar_d}",
        f"{probs.away_win_prob*100:.1f}%\n{bar_b}",
    )
    console.print(t)
    console.print()

    # Most likely scores
    console.print("[bold underline]🏅 最可能比分[/bold underline]")
    for s in probs.most_likely_scores[:3]:
        console.print(f"  {s.score:>5}  — {s.probability*100:.1f}%")
    console.print()

    # Additional stats
    stats_text = (
        f"大2.5球概率: {probs.over_2_5_prob*100:.1f}%  |  "
        f"两队进球概率: {probs.both_teams_score_prob*100:.1f}%"
    )
    console.print(Panel(stats_text, box=box.SQUARE))
    console.print()


def display_tournament_results(results: list[SimulationResult], top_n: int = 16):
    """Show tournament simulation results as a ranked table."""
    console.print()
    console.print(Panel(
        f"[bold]世界杯 2026 蒙特卡洛模拟结果[/bold] ({results[0].total_simulations} 次模拟)",
        box=box.ROUNDED,
    ))
    console.print()

    table = Table(box=box.ROUNDED, header_style="bold", title="夺冠概率排行")
    table.add_column("排名", style="dim", width=4)
    table.add_column("球队", width=14)
    table.add_column("夺冠", style="yellow", width=8)
    table.add_column("决赛", style="cyan", width=8)
    table.add_column("四强", style="blue", width=8)
    table.add_column("八强", style="magenta", width=8)
    table.add_column("16强", width=8)
    table.add_column("32强", width=8)

    for i, r in enumerate(results[:top_n], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
        team_display = f"{medal} {r.team}" if medal else r.team

        table.add_row(
            str(i),
            team_display,
            f"{r.champion_prob*100:.1f}%",
            f"{r.final_prob*100:.1f}%",
            f"{r.semi_prob*100:.1f}%",
            f"{r.quarter_prob*100:.1f}%",
            f"{r.round_of_16_prob*100:.1f}%",
            f"{r.round_of_32_prob*100:.1f}%",
        )

    console.print(table)
    console.print()


def display_group_stage():
    """Show group stage overview."""
    console.print()
    console.print(Panel("[bold]2026 世界杯小组赛分组[/bold]", box=box.ROUNDED))
    console.print()

    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("小组", width=6)
    table.add_column("球队", width=40)

    for g_name in sorted(GROUPS.keys()):
        teams = "  vs  ".join(GROUPS[g_name])
        table.add_row(g_name, teams)

    console.print(table)
    console.print()


def display_team_info(team: str, elo_rating: int, champion_prob: float,
                      best_round_prob: float):
    """Show information about a specific team."""
    console.print()
    panel = Panel(
        f"[bold yellow]{team}[/bold yellow]\n\n"
        f"初始 Elo 评分: [cyan]{elo_rating}[/cyan]\n"
        f"模拟夺冠概率: [yellow]{champion_prob*100:.1f}%[/yellow]\n"
        f"小组出线概率: [green]{best_round_prob*100:.1f}%[/green]",
        title="球队信息",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()
