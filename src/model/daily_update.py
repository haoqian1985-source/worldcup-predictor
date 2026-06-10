"""每日复盘 — 输入实际比分，对比预测，生成日报和小红书文案。

用法:
  python -m src.model.daily_update input --date 2026-06-11   # 交互输入比分
  python -m src.model.daily_update report --date 2026-06-11   # 日报看板
  python -m src.model.daily_update post --date 2026-06-11     # 小红书文案
  python -m src.model.daily_update status                     # 进度查询
"""

import json
import sys
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.model.predict_all import PREDICTIONS_FILE, load_predictions

console = Console()

RESULT_ICONS = {
    "home_win": {True: "✅", False: "⚠️"},
    "draw": {True: "✅", False: "⚠️"},
    "away_win": {True: "✅", False: "⚠️"},
}
RESULT_LABELS = {
    "home_win": "主胜", "draw": "平局", "away_win": "客胜",
}


def _predicted_result(p: dict) -> str:
    """Get the predicted result (based on highest probability)."""
    return p.get("result", "draw")


def _actual_result(a: dict) -> str:
    if a["score_a"] > a["score_b"]:
        return "home_win"
    if a["score_a"] < a["score_b"]:
        return "away_win"
    return "draw"


def _is_upset(p: dict, a: dict) -> bool:
    """Check if actual result was an upset vs prediction probabilities."""
    actual = _actual_result(a)
    if actual == "home_win" and p["home_win_prob"] < 0.30:
        return True
    if actual == "away_win" and p["away_win_prob"] < 0.30:
        return True
    if actual == "draw" and p["draw_prob"] < 0.20:
        return True
    return False


def _judge_match(p: dict, a: dict) -> tuple[str, str]:
    """Judge a match prediction vs actual. Returns (tag, reason).

    Tags: ✅ 准确 / ⚠️ 方向 / ❌ 错误
    """
    score_correct = (int(round(p["xg_a"])) == a["score_a"]
                     and int(round(p["xg_b"])) == a["score_b"])
    if score_correct:
        return "✅ 准确", "比分完全命中"

    pred_res = _predicted_result(p)
    actual_res = _actual_result(a)
    direction_correct = pred_res == actual_res

    if direction_correct:
        return "⚠️ 方向", f"方向对（预测{RESULT_LABELS[pred_res]}→实际{RESULT_LABELS[actual_res]}）"

    return "❌ 错误", f"方向错（预测{RESULT_LABELS[pred_res]}→实际{RESULT_LABELS[actual_res]}）"


def _match_label(p: dict) -> str:
    """Short summary of a prediction."""
    probs = f"{p['home_win_prob']*100:.0f}%/{p['draw_prob']*100:.0f}%/{p['away_win_prob']*100:.0f}%"
    return f"预测 {p['score']}  (λ={p['xg_a']:.2f}-{p['xg_b']:.2f})  {probs}  [{p.get('confidence', '?')}]"


# ─── Input ────────────────────────────────────────────────────

def input_result(date_filter: str | None = None):
    """Interactive input of match results."""
    data = load_predictions()
    matches = data["matches"]

    if date_filter:
        day_matches = [m for m in matches if m["date"] == date_filter]
        if not day_matches:
            console.print(f"[yellow]该日期没有比赛: {date_filter}[/yellow]")
            return
    else:
        unmatched = [(i, m) for i, m in enumerate(matches) if m["actual"] is None]
        if not unmatched:
            console.print("[green]所有比赛都已录入！[/green]")
            return
        console.print(f"\n共 {len(unmatched)} 场未录入:")
        for idx, (i, m) in enumerate(unmatched[:20]):
            p = m["predicted"]
            console.print(f"  {idx}. [{m['date']}] {m['team_a']} vs {m['team_b']}  ({_match_label(p)})")
        console.print()
        choice = input("选择编号（或输入日期 YYYY-MM-DD）: ").strip()
        if "-" in choice:
            input_result(date_filter=choice)
            return
        try:
            idx = int(choice)
            day_matches = [unmatched[idx][1]]
            actual_idx = unmatched[idx][0]
        except (ValueError, IndexError):
            return

    for m in day_matches:
        _input_single(m, data)


def _input_single(m: dict, data: dict):
    """Input result for a single match."""
    p = m["predicted"]
    console.print(f"\n[bold]{m['team_a']} vs {m['team_b']}[/bold] ({m['date']})")
    console.print(f"  预测: {_match_label(p)}")
    scores_str = " | ".join(f"{s['score']}({s['prob']*100:.0f}%)" for s in p.get("top_scores", []))
    console.print(f"  前3比分: {scores_str}")

    while True:
        try:
            result = input("  实际比分 (格式: 2-1, 或回车跳过): ").strip()
            if not result:
                return
            a_str, b_str = result.split("-")
            score_a, score_b = int(a_str.strip()), int(b_str.strip())
            if score_a < 0 or score_b < 0:
                raise ValueError
            actual = {"score_a": score_a, "score_b": score_b}
            m["actual"] = actual
            break
        except (ValueError, TypeError):
            console.print("  格式错误，请用 2-1 格式输入")

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tag, reason = _judge_match(p, actual)
    upset = " 🔥冷门!" if _is_upset(p, actual) else ""
    console.print(f"  → {tag} {reason}{upset}")
    console.print()


def batch_input(date_str: str, results: list[tuple[str, str, int, int]]) -> int:
    """Batch input results. Returns count of matches entered."""
    data = load_predictions()
    matches = data["matches"]
    count = 0

    for team_a, team_b, sa, sb in results:
        for m in matches:
            if (m["date"] == date_str and m["team_a"] == team_a
                    and m["team_b"] == team_b and m["actual"] is None):
                m["actual"] = {"score_a": sa, "score_b": sb}
                count += 1
                p = m["predicted"]
                tag, reason = _judge_match(p, m["actual"])
                upset = " 🔥冷门!" if _is_upset(p, m["actual"]) else ""
                console.print(f"  {tag} {m['team_a']} {sa}-{sb} {m['team_b']}{upset}")
                break

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    console.print(f"[green]✅ 已录入 {count} 场[/green]")
    return count


# ─── Report ───────────────────────────────────────────────────

def daily_report(date_str: str | None = None):
    """Generate a daily report."""
    data = load_predictions()
    matches = data["matches"]
    if date_str is None:
        date_str = date.today().isoformat()

    day_matches = [m for m in matches if m["date"] == date_str and m["actual"]]
    day_all = [m for m in matches if m["date"] == date_str]

    if not day_matches and not day_all:
        console.print(f"[yellow]{date_str} 没有比赛[/yellow]")
        return

    total = len(day_all)
    played = len(day_matches)
    pending = total - played

    console.print()
    console.print(Panel(f"[bold]📅 世界杯日报 — {date_str}[/bold]", box=box.ROUNDED))
    if pending > 0:
        console.print(f"[yellow]⏳ 还有 {pending}/{total} 场未录入[/yellow]\n")

    if day_matches:
        table = Table(box=box.ROUNDED, header_style="bold")
        table.add_column("对阵", width=42)
        table.add_column("预测比分", width=8)
        table.add_column("期望进球", width=12)
        table.add_column("概率", width=14)
        table.add_column("实际", width=6)
        table.add_column("判定", width=12)
        table.add_column("", width=8)

        for m in day_matches:
            p, a = m["predicted"], m["actual"]
            actual_s = f"{a['score_a']}-{a['score_b']}"
            tag, _ = _judge_match(p, a)
            upset_note = "🔥冷门" if _is_upset(p, a) else ""
            table.add_row(
                f"{m['team_a']} vs {m['team_b']}",
                p["score"],
                f"{p['xg_a']:.2f}-{p['xg_b']:.2f}",
                f"{p['home_win_prob']*100:.0f}%/{p['draw_prob']*100:.0f}%/{p['away_win_prob']*100:.0f}%",
                actual_s, tag, upset_note,
            )
        console.print(table)

    if played > 0:
        correct_score = sum(1 for m in day_matches
                            if int(round(m["predicted"]["xg_a"])) == m["actual"]["score_a"]
                            and int(round(m["predicted"]["xg_b"])) == m["actual"]["score_b"])
        correct_dir = sum(1 for m in day_matches
                          if _predicted_result(m["predicted"]) == _actual_result(m["actual"]))
        console.print(f"\n[bold]今日统计:[/bold]")
        console.print(f"  比分准确: {correct_score}/{played} ({correct_score/played*100:.0f}%)")
        console.print(f"  方向准确: {correct_dir}/{played} ({correct_dir/played*100:.0f}%)")

    _print_overall_stats(matches)


def _print_overall_stats(matches: list):
    """Print cumulative accuracy stats."""
    played = [m for m in matches if m["actual"]]
    if not played:
        return
    correct_score = sum(1 for m in played
                        if int(round(m["predicted"]["xg_a"])) == m["actual"]["score_a"]
                        and int(round(m["predicted"]["xg_b"])) == m["actual"]["score_b"])
    correct_dir = sum(1 for m in played
                      if _predicted_result(m["predicted"]) == _actual_result(m["actual"]))
    console.print(f"\n[bold]累计统计 ({len(played)}/{len(matches)} 场已完赛):[/bold]")
    console.print(f"  比分准确率: {correct_score}/{len(played)} ({correct_score/len(played)*100:.0f}%)")
    console.print(f"  方向准确率: {correct_dir}/{len(played)} ({correct_dir/len(played)*100:.0f}%)")


# ─── Xiaohongshu Post ─────────────────────────────────────────

def generate_xiaohongshu_post(date_str: str | None = None) -> str:
    """Generate a Xiaohongshu-ready post."""
    data = load_predictions()
    matches = data["matches"]
    if date_str is None:
        date_str = date.today().isoformat()

    day_matches = [m for m in matches if m["date"] == date_str and m["actual"]]
    if not day_matches:
        return ""

    played = len(day_matches)
    correct_score = sum(1 for m in day_matches
                        if int(round(m["predicted"]["xg_a"])) == m["actual"]["score_a"]
                        and int(round(m["predicted"]["xg_b"])) == m["actual"]["score_b"])
    correct_dir = sum(1 for m in day_matches
                      if _predicted_result(m["predicted"]) == _actual_result(m["actual"]))
    upsets = [m for m in day_matches if _is_upset(m["predicted"], m["actual"])]

    score_rate = correct_score / played * 100 if played else 0
    dir_rate = correct_dir / played * 100 if played else 0

    lines = [f"📅 世界杯日报 {date_str}"]
    lines.append("")
    lines.append(f"今日 {played} 场战罢，方向准确率 {dir_rate:.0f}%，比分准确率 {score_rate:.0f}%")
    lines.append("")

    for m in day_matches:
        p, a = m["predicted"], m["actual"]
        tag, _ = _judge_match(p, a)
        icon = "✅" if "准确" in tag else ("⚠️" if "方向" in tag else "❌")
        lines.append(
            f"{icon} {m['team_a']} vs {m['team_b']}: "
            f"预测 {p['score']} → 实际 {a['score_a']}-{a['score_b']}  {tag}"
        )

    if upsets:
        lines.append("")
        lines.append("🔥 今日冷门：")
        for m in upsets:
            p = m["predicted"]
            lines.append(
                f"   {m['team_a']} vs {m['team_b']} "
                f"（预测客胜概率仅 {min(p['home_win_prob'], p['away_win_prob'])*100:.0f}%）"
            )

    lines.append("")
    lines.append("📊 数据模型（Elo + 泊松）vs 真实赛果，每日更新")
    lines.append("#世界杯 #世界杯2026 #足球预测 #数据科学")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="世界杯每日复盘工具")
    parser.add_argument("action", nargs="?", default="report",
                        choices=["input", "report", "post", "status"])
    parser.add_argument("--date", help="日期，如 2026-06-11")
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
        from collections import Counter
        date_counts = Counter(m["date"] for m in data["matches"] if m["actual"])
        for d in sorted(date_counts):
            print(f"  {d}: {date_counts[d]} 场")


if __name__ == "__main__":
    main()
