#!/usr/bin/env python3
"""
世界杯预测工具 — 双模型对决
AI 分析师 vs 数据科学家
"""

import argparse
import sys
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="世界杯预测 — 双模型对决",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py ai "巴西 vs 法国"     # AI 分析师模式
  python main.py model "巴西 vs 法国"  # 数据科学家模式
  python main.py both "巴西 vs 法国"   # 双模型对决（推荐）
  python main.py simulate              # 蒙特卡洛赛程模拟
        """,
    )
    parser.add_argument("mode", choices=["ai", "model", "both", "simulate"],
                        help="ai=AI分析师  model=数据科学家  both=双模型对决  simulate=赛程模拟")
    parser.add_argument("match", nargs="?", help="比赛对阵，如 '巴西 vs 法国'")
    parser.add_argument("--n", type=int, default=10000, help="模拟次数")
    parser.add_argument("--top", type=int, default=16, help="显示前N名")
    return parser.parse_args()


def _run_python(mod: str, args: list[str]):
    """Run a Python module with PYTHONPATH set correctly."""
    root = Path(__file__).parent
    env = {**__import__("os").environ, "PYTHONPATH": str(root)}
    subprocess.run([sys.executable, "-m", mod] + args, cwd=root, env=env)


def run_ai(match: str):
    """Run AI analyst mode."""
    _run_python("src.ai.main", [match])


def run_model(match: str):
    """Run data scientist mode."""
    _run_python("src.model.main", [match])


def run_both(match: str):
    """Run both models and compare."""
    from rich.console import Console
    from rich.panel import Panel
    from rich import box

    console = Console()

    console.print(Panel(
        f"[bold yellow]⚡ 双模型对决: {match}[/bold yellow]\n\n"
        "[cyan]🤖 AI 分析师[/cyan]  vs  [green]📊 数据科学家[/green]",
        box=box.DOUBLE,
    ))
    console.print()

    # Round 1: AI
    console.print("[bold cyan]─── 第一回合: AI 分析师 ───[/bold cyan]")
    run_ai(match)

    # Round 2: Data Model
    console.print("\n[bold green]─── 第二回合: 数据科学家 ───[/bold green]")
    run_model(match)


def run_simulation(n: int, top: int):
    """Run tournament simulation."""
    _run_python("src.model.main", ["--simulate", "--n", str(n), "--top", str(top)])


def main():
    args = parse_args()

    if args.mode == "simulate":
        run_simulation(args.n, args.top)
    elif args.mode == "both":
        if not args.match:
            print("错误: both 模式需要指定比赛，如 '巴西 vs 法国'")
            sys.exit(1)
        run_both(args.match)
    elif args.mode == "ai":
        if not args.match:
            print("错误: ai 模式需要指定比赛")
            sys.exit(1)
        run_ai(args.match)
    elif args.mode == "model":
        if not args.match:
            print("错误: model 模式需要指定比赛")
            sys.exit(1)
        run_model(args.match)


if __name__ == "__main__":
    main()
