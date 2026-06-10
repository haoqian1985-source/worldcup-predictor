#!/usr/bin/env python3
"""世界杯比赛预测工具 - AI 驱动版"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from src.ai.config import Config
from src.ai.data_gatherer import gather_match_context
from src.ai.predictor import predict_match

console = Console()
logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="世界杯比赛预测工具（AI 驱动版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py "巴西 vs 法国"
  python main.py --team1 阿根廷 --team2 德国
  python main.py "阿根廷 vs 法国" --model deepseek/deepseek-chat
        """,
    )
    parser.add_argument("match", nargs="?", help="比赛对阵，格式如 '巴西 vs 法国'")
    parser.add_argument("--team1", help="球队 A")
    parser.add_argument("--team2", help="球队 B")
    parser.add_argument("--model", help="LLM 模型名")
    return parser.parse_args()


def parse_match_arg(match_str: str):
    """Parse '巴西 vs 法国' into (巴西, 法国)."""
    for sep in [" vs ", " VS ", " v ", " V ", " vs ", " VS "]:
        if sep in match_str:
            parts = match_str.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    raise ValueError(f"无法解析比赛对阵: {match_str}，请使用如 '巴西 vs 法国' 的格式")


def display_prediction(result, team_a_name, team_b_name):
    """Display prediction results in a rich terminal UI."""
    # Header
    console.print()
    console.print(Panel(
        f"[bold yellow]{team_a_name}[/bold yellow]  vs  [bold yellow]{team_b_name}[/bold yellow]",
        subtitle="AI 足球分析预测",
        box=box.ROUNDED,
    ))
    console.print()

    # Probability bar
    prob_table = Table(box=box.SIMPLE, padding=(0, 2))
    prob_table.add_column(team_a_name, style="cyan", justify="center")
    prob_table.add_column("平局", style="white", justify="center")
    prob_table.add_column(team_b_name, style="green", justify="center")

    prob_a = getattr(result, "team_a_win_probability", 33)
    prob_draw = getattr(result, "draw_probability", 34)
    prob_b = getattr(result, "team_b_win_probability", 33)

    bar_a = "█" * max(1, prob_a // 5)
    bar_draw = "█" * max(1, prob_draw // 5)
    bar_b = "█" * max(1, prob_b // 5)

    prob_table.add_row(
        f"{prob_a}%\n{bar_a}",
        f"{prob_draw}%\n{bar_draw}",
        f"{prob_b}%\n{bar_b}",
    )
    console.print(prob_table)
    console.print()

    # Core prediction
    winner = getattr(result, "predicted_winner", "")
    score = getattr(result, "predicted_score", "")
    confidence = getattr(result, "confidence", "")

    confidence_color = {"高": "green", "中": "yellow", "低": "red"}.get(confidence, "white")
    winner_text = f"[bold]{winner}[/bold]" if winner else "待定"

    info = Panel(
        f"[bold]预测胜者:[/bold] {winner_text}\n"
        f"[bold]预测比分:[/bold] {score}\n"
        f"[bold]信心指数:[/bold] [{confidence_color}]{confidence}[/{confidence_color}]",
        box=box.ROUNDED,
    )
    console.print(info)
    console.print()

    # Key analysis
    analysis = getattr(result, "key_analysis", [])
    if analysis:
        console.print("[bold underline]核心分析[/bold underline]")
        for i, point in enumerate(analysis, 1):
            console.print(f"  {i}. {point}")
        console.print()

    # Tactical analysis
    tactical = getattr(result, "tactical_analysis", "")
    if tactical:
        console.print(f"[bold underline]战术分析[/bold underline]\n  {tactical}")
        console.print()

    # Players to watch
    players = getattr(result, "key_players_to_watch", [])
    if players:
        console.print("[bold underline]关键球员[/bold underline]")
        console.print("  " + ", ".join(players))
        console.print()

    # Risk factors
    risks = getattr(result, "risk_factors", [])
    if risks:
        console.print("[bold underline]风险因素[/bold underline]")
        for risk in risks:
            console.print(f"  ⚠ {risk}")
        console.print()

    # Betting advice
    betting = getattr(result, "betting_advice", "")
    if betting:
        console.print(Panel(f"💡 {betting}", title="投注建议", box=box.SQUARE))
        console.print()


def main():
    args = parse_args()

    # Determine teams
    if args.match:
        team_a, team_b = parse_match_arg(args.match)
    elif args.team1 and args.team2:
        team_a, team_b = args.team1, args.team2
    else:
        console.print("[red]错误: 请指定比赛，如 'python main.py \"巴西 vs 法国\"'[/red]")
        sys.exit(1)

    # Load config
    config = Config.from_env()
    if args.model:
        config.llm_model = args.model

    if not config.llm_api_key:
        console.print("[red]错误: 未配置 API Key。请在 .env 文件中设置 DEEPSEEK_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY[/red]")
        sys.exit(1)

    console.print(f"⚽ [bold]世界杯预测: {team_a} vs {team_b}[/bold]")
    console.print(f"  模型: {config.llm_model}")
    console.print()

    # Step 1: Gather data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("正在收集球队数据...", total=None)
        context = gather_match_context(team_a, team_b, config)

    # Step 2: Predict
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("AI 分析预测中...", total=None)
        result = predict_match(context, config)

    # Step 3: Display
    display_prediction(result, team_a, team_b)


if __name__ == "__main__":
    main()
