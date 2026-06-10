"""每日复盘 — 输入实际比分，对比预测，生成小红书文案。"""

import json
import sys
from datetime import datetime, date
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.model.predict_all import PREDICTIONS_FILE, load_predictions

console = Console()


def input_result(match_idx: int | None = None, date_filter: str | None = None):
    """Input actual match results interactively or by date."""
    data = load_predictions()
    matches = data["matches"]

    if date_filter:
        day_matches = [m for m in matches if m["date"] == date_filter]
        if not day_matches:
            print(f"  该日期没有比赛: {date_filter}")
            return
    elif match_idx is not None:
        day_matches = [matches[match_idx]]
    else:
        # Interactive: show unmatched matches
        unmatched = [(i, m) for i, m in enumerate(matches) if m["actual"] is None]
        if not unmatched:
            console.print("[green]所有比赛都已录入！[/green]")
            return

        console.print(f"\n共 {len(unmatched)} 场未录入的比赛:")
        for idx, (i, m) in enumerate(unmatched[:20]):
            console.print(f"  {idx}. [{m['date']}] {m['team_a']} vs {m['team_b']}")
        console.print()

        choice = input("选择编号（或输入日期 YYYY-MM-DD）: ").strip()
        if "-" in choice:
            input_result(date_filter=choice)
            return
        try:
            idx = int(choice)
            if idx < 0 or idx >= len(unmatched):
                print("无效编号")
                return
            day_matches = [unmatched[idx][1]]
            actual_idx = unmatched[idx][0]
        except ValueError:
            return

    for m in day_matches:
        _input_single(m, data, date_filter is not None)


def _input_single(m: dict, data: dict, batch_mode: bool = False):
    """Input result for a single match."""
    p = m["predicted"]
    console.print(f"\n[bold]{m['team_a']} vs {m['team_b']}[/bold] ({m['date']})")
    console.print(f"  模型预测: {p['score']}  ({p['home_win_prob']*100:.0f}%/{p['draw_prob']*100:.0f}%/{p['away_win_prob']*100:.0f}%)")

    if batch_mode:
        # Auto-input from command line arguments format: "team_a-team_b score_a-score_b"
        return

    while True:
        try:
            result = input("  实际比分 (格式: 2-1, 或回车跳过): ").strip()
            if not result:
                return
            a, b = result.split("-")
            score_a, score_b = int(a.strip()), int(b.strip())
            if score_a < 0 or score_b < 0:
                raise ValueError
            m["actual"] = {
                "score_a": score_a,
                "score_b": score_b,
                "result": "home_win" if score_a > score_b else "away_win" if score_a < score_b else "draw",
            }
            break
        except (ValueError, TypeError):
            print("  格式错误，请用 2-1 格式输入")

    # Save after each input
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    _show_match_verdict(m)


def _show_match_verdict(m: dict):
    """Show verdict for a single match."""
    p = m["predicted"]
    a = m["actual"]

    score_correct = a["score_a"] == p["score_a"] and a["score_b"] == p["score_b"]

    # Check result direction
    def result_type(goals_a, goals_b):
        if goals_a > goals_b: return "home"
        if goals_a < goals_b: return "away"
        return "draw"

    pred_result = result_type(p["score_a"], p["score_b"])
    actual_result = result_type(a["score_a"], a["score_b"])
    direction_correct = pred_result == actual_result

    if score_correct:
        verdict = "✅ 比分完全准确！"
    elif direction_correct:
        verdict = "⚠️ 方向正确，比分偏差"
    else:
        verdict = "❌ 预测错误"

    # Upset factor
    if actual_result == "home" and p["home_win_prob"] < 0.3:
        upset = "🔥 冷门！"
    elif actual_result == "away" and p["away_win_prob"] < 0.3:
        upset = "🔥 冷门！"
    elif actual_result == "draw" and p["draw_prob"] < 0.2:
        upset = "🔥 小冷门"
    else:
        upset = ""

    console.print(f"  → {verdict} {upset}")
    console.print()


def daily_report(date_str: str | None = None):
    """Generate a daily report for Xiaohongshu."""
    data = load_predictions()
    matches = data["matches"]

    if date_str is None:
        date_str = date.today().isoformat()

    day_matches = [m for m in matches if m["date"] == date_str and m["actual"]]
    day_all = [m for m in matches if m["date"] == date_str]

    if not day_matches and not day_all:
        console.print(f"[yellow]{date_str} 没有比赛[/yellow]")
        return

    # Stats
    total = len(day_all)
    played = len(day_matches)
    pending = total - played

    correct_score = sum(1 for m in day_matches
                        if m["actual"]["score_a"] == m["predicted"]["score_a"]
                        and m["actual"]["score_b"] == m["predicted"]["score_b"])

    correct_direction = sum(1 for m in day_matches
                            if m["predicted"].get("result", _result_type(m["predicted"]["score_a"], m["predicted"]["score_b"]))
                            == m["actual"]["result"])

    upsets = [m for m in day_matches if _is_upset(m)]

    # Print report
    console.print()
    console.print(Panel(
        f"[bold]📅 世界杯日报 — {date_str}[/bold]",
        box=box.ROUNDED,
    ))

    if pending > 0:
        console.print(f"[yellow]⏳ 还有 {pending}/{total} 场比赛未录入[/yellow]\n")

    # Results table
    if day_matches:
        table = Table(box=box.ROUNDED, header_style="bold")
        table.add_column("对阵", width=42)
        table.add_column("预测", width=8)
        table.add_column("实际", width=8)
        table.add_column("判定", width=12)
        table.add_column("备注", width=10)

        for m in day_matches:
            p, a = m["predicted"], m["actual"]
            pred_s = f"{p['score_a']}-{p['score_b']}"
            actual_s = f"{a['score_a']}-{a['score_b']}"
            sc = p["score_a"] == a["score_a"] and p["score_b"] == a["score_b"]
            dc = p.get("result", _result_type(p["score_a"], p["score_b"])) == a["result"]
            if sc:
                tag = "✅ 准确"
            elif dc:
                tag = "⚠️ 方向"
            else:
                tag = "❌ 错误"
            note = "🔥 冷门" if _is_upset(m) else ""
            table.add_row(
                f"{m['team_a']} vs {m['team_b']}",
                pred_s, actual_s, tag, note,
            )
        console.print(table)

    # Summary
    if played > 0:
        console.print(f"\n[bold]今日统计:[/bold]")
        console.print(f"  比分准确: {correct_score}/{played} ({correct_score/played*100:.0f}%)")
        console.print(f"  方向准确: {correct_direction}/{played} ({correct_direction/played*100:.0f}%)")
        if upsets:
            upset_str = ", ".join(f"{m['team_a']} vs {m['team_b']}" for m in upsets)
            console.print(f"  冷门: {upset_str}")

    # Overall stats
    _print_overall_stats(matches)


def _print_overall_stats(matches: list):
    """Print overall prediction accuracy."""
    played = [m for m in matches if m["actual"]]
    if not played:
        return

    correct_score = sum(1 for m in played
                        if m["actual"]["score_a"] == m["predicted"]["score_a"]
                        and m["actual"]["score_b"] == m["predicted"]["score_b"])
    correct_dir = sum(1 for m in played
                      if m["predicted"].get("result", _result_type(m["predicted"]["score_a"], m["predicted"]["score_b"]))
                      == m["actual"]["result"])

    console.print(f"\n[bold]累计统计 ({len(played)}/{len(matches)} 场已完赛):[/bold]")
    console.print(f"  比分准确率: {correct_score}/{len(played)} ({correct_score/len(played)*100:.0f}%)")
    console.print(f"  方向准确率: {correct_dir}/{len(played)} ({correct_dir/len(played)*100:.0f}%)")


def _result_type(ga: int, gb: int) -> str:
    if ga > gb: return "home_win"
    if ga < gb: return "away_win"
    return "draw"


def _is_upset(m: dict) -> bool:
    """Check if result was an upset."""
    p, a = m["predicted"], m["actual"]
    if a["result"] == "home_win" and p["home_win_prob"] < 0.30:
        return True
    if a["result"] == "away_win" and p["away_win_prob"] < 0.30:
        return True
    if a["result"] == "draw" and p["draw_prob"] < 0.20:
        return True
    return False


def batch_input(date_str: str, results: list[tuple[str, str, int, int]]):
    """Batch input results for a specific date.

    Args:
        date_str: Date string like "2026-06-11"
        results: List of (team_a, team_b, score_a, score_b)
    """
    data = load_predictions()
    matches = data["matches"]
    count = 0

    for team_a, team_b, sa, sb in results:
        for m in matches:
            if (m["date"] == date_str and m["team_a"] == team_a
                    and m["team_b"] == team_b and m["actual"] is None):
                m["actual"] = {
                    "score_a": sa,
                    "score_b": sb,
                    "result": "home_win" if sa > sb else "away_win" if sa < sb else "draw",
                }
                count += 1
                break

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]✅ 已录入 {count} 场比赛结果[/green]")
    return count


def generate_xiaohongshu_post(date_str: str | None = None) -> str:
    """Generate a Xiaohongshu-ready post from daily report."""
    data = load_predictions()
    matches = data["matches"]

    if date_str is None:
        date_str = date.today().isoformat()

    day_matches = [m for m in matches if m["date"] == date_str and m["actual"]]
    if not day_matches:
        return ""

    played = len(day_matches)
    correct_score = sum(1 for m in day_matches
                        if m["actual"]["score_a"] == m["predicted"]["score_a"]
                        and m["actual"]["score_b"] == m["predicted"]["score_b"])
    correct_dir = sum(1 for m in day_matches
                      if m["predicted"].get("result", _result_type(m["predicted"]["score_a"], m["predicted"]["score_b"]))
                      == m["actual"]["result"])
    upsets = [m for m in day_matches if _is_upset(m)]

    score_rate = correct_score / played * 100 if played else 0
    dir_rate = correct_dir / played * 100 if played else 0

    lines = [f"📅 世界杯日报 {date_str}"]
    lines.append("")
    lines.append(f"今日 {played} 场战罢，预测方向准确率 {dir_rate:.0f}%，比分准确率 {score_rate:.0f}%")
    lines.append("")

    for m in day_matches:
        p, a = m["predicted"], m["actual"]
        sc = p["score_a"] == a["score_a"] and p["score_b"] == a["score_b"]
        dc = p.get("result", _result_type(p["score_a"], p["score_b"])) == a["result"]
        icon = "✅" if sc else ("⚠️" if dc else "❌")
        lines.append(f"{icon} {m['team_a']} vs {m['team_b']}: 预测 {p['score_a']}-{p['score_b']} → 实际 {a['score_a']}-{a['score_b']}")

    if upsets:
        lines.append("")
        lines.append("🔥 今日冷门：")
        for m in upsets:
            p = m["predicted"]
            lines.append(f"   {m['team_a']} vs {m['team_b']}（预测 {p['score_a']}-{p['score_b']}，仅 {max(p['home_win_prob'], p['away_win_prob'])*100:.0f}% 概率）")

    lines.append("")
    lines.append("📊 累计：数据模型 vs 真实赛果，持续更新中")
    lines.append("#世界杯 #世界杯2026 #足球预测 #数据科学")

    return "\n".join(lines)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="世界杯每日复盘工具")
    parser.add_argument("action", nargs="?", default="report",
                        choices=["input", "report", "post", "status"])
    parser.add_argument("--date", help="日期，如 2026-06-11")
    parser.add_argument("--batch", nargs="+", help='批量输入: --date 2026-06-11 --batch "Mexico,South Korea,2,1" "..."')
    args = parser.parse_args()

    if args.action == "input":
        input_result(date_filter=args.date)
    elif args.action == "report":
        daily_report(args.date)
    elif args.action == "post":
        post = generate_xiaohongshu_post(args.date)
        if post:
            print(post)
        else:
            print("当日暂无已录入的比赛")
    elif args.action == "status":
        data = load_predictions()
        played = sum(1 for m in data["matches"] if m["actual"])
        total = len(data["matches"])
        print(f"已录入: {played}/{total} 场")
        # Group by date
        from collections import Counter
        date_counts = Counter(m["date"] for m in data["matches"] if m["actual"])
        for d in sorted(date_counts):
            print(f"  {d}: {date_counts[d]} 场")


if __name__ == "__main__":
    main()
