"""生成并保存全部 72 场小组赛预测，供 daily_update.py 每日复盘使用。

使用期望进球（λ）作为主要预测指标，而非四舍五入的整数比分。
"""

import json
from pathlib import Path

from src.model.elo import EloCalculator
from src.model.poisson import PoissonModel

# 72 场小组赛赛程（12 组 × 6 场）
SCHEDULE = {
    "A组": [
        ("Mexico", "South Korea", "2026-06-11"),
        ("Czech Republic", "South Africa", "2026-06-11"),
        ("South Korea", "Czech Republic", "2026-06-18"),
        ("Mexico", "South Africa", "2026-06-18"),
        ("South Africa", "South Korea", "2026-06-24"),
        ("Czech Republic", "Mexico", "2026-06-24"),
    ],
    "B组": [
        ("Canada", "Bosnia and Herzegovina", "2026-06-12"),
        ("Qatar", "Switzerland", "2026-06-13"),
        ("Switzerland", "Bosnia and Herzegovina", "2026-06-18"),
        ("Canada", "Qatar", "2026-06-18"),
        ("Switzerland", "Canada", "2026-06-24"),
        ("Bosnia and Herzegovina", "Qatar", "2026-06-24"),
    ],
    "C组": [
        ("Brazil", "Haiti", "2026-06-13"),
        ("Morocco", "Scotland", "2026-06-14"),
        ("Scotland", "Morocco", "2026-06-19"),
        ("Brazil", "Morocco", "2026-06-19"),
        ("Scotland", "Brazil", "2026-06-24"),
        ("Haiti", "Morocco", "2026-06-24"),
    ],
    "D组": [
        ("USA", "Paraguay", "2026-06-12"),
        ("Australia", "Turkey", "2026-06-14"),
        ("USA", "Australia", "2026-06-19"),
        ("Turkey", "Paraguay", "2026-06-20"),
        ("Turkey", "USA", "2026-06-25"),
        ("Paraguay", "Australia", "2026-06-25"),
    ],
    "E组": [
        ("Germany", "Curacao", "2026-06-14"),
        ("Ivory Coast", "Ecuador", "2026-06-15"),
        ("Germany", "Ivory Coast", "2026-06-20"),
        ("Ecuador", "Curacao", "2026-06-21"),
        ("Curacao", "Ivory Coast", "2026-06-25"),
        ("Ecuador", "Germany", "2026-06-25"),
    ],
    "F组": [
        ("Netherlands", "Japan", "2026-06-14"),
        ("Sweden", "Tunisia", "2026-06-15"),
        ("Netherlands", "Sweden", "2026-06-20"),
        ("Tunisia", "Japan", "2026-06-21"),
        ("Tunisia", "Netherlands", "2026-06-25"),
        ("Japan", "Sweden", "2026-06-25"),
    ],
    "G组": [
        ("Belgium", "Egypt", "2026-06-15"),
        ("Iran", "New Zealand", "2026-06-16"),
        ("Belgium", "Iran", "2026-06-21"),
        ("New Zealand", "Egypt", "2026-06-22"),
        ("New Zealand", "Belgium", "2026-06-26"),
        ("Egypt", "Iran", "2026-06-26"),
    ],
    "H组": [
        ("Spain", "Cape Verde", "2026-06-15"),
        ("Saudi Arabia", "Uruguay", "2026-06-15"),
        ("Spain", "Saudi Arabia", "2026-06-21"),
        ("Uruguay", "Cape Verde", "2026-06-21"),
        ("Cape Verde", "Saudi Arabia", "2026-06-26"),
        ("Uruguay", "Spain", "2026-06-26"),
    ],
    "I组": [
        ("France", "Iraq", "2026-06-16"),
        ("Senegal", "Norway", "2026-06-16"),
        ("France", "Senegal", "2026-06-22"),
        ("Norway", "Iraq", "2026-06-23"),
        ("Norway", "France", "2026-06-26"),
        ("Iraq", "Senegal", "2026-06-26"),
    ],
    "J组": [
        ("Argentina", "Algeria", "2026-06-16"),
        ("Austria", "Jordan", "2026-06-17"),
        ("Argentina", "Austria", "2026-06-22"),
        ("Jordan", "Algeria", "2026-06-23"),
        ("Algeria", "Austria", "2026-06-27"),
        ("Jordan", "Argentina", "2026-06-27"),
    ],
    "K组": [
        ("Portugal", "DR Congo", "2026-06-17"),
        ("Uzbekistan", "Colombia", "2026-06-17"),
        ("Portugal", "Uzbekistan", "2026-06-23"),
        ("Colombia", "DR Congo", "2026-06-23"),
        ("Colombia", "Portugal", "2026-06-27"),
        ("DR Congo", "Uzbekistan", "2026-06-27"),
    ],
    "L组": [
        ("Croatia", "England", "2026-06-17"),
        ("Ghana", "Panama", "2026-06-17"),
        ("Croatia", "Ghana", "2026-06-23"),
        ("Panama", "England", "2026-06-23"),
        ("Panama", "Croatia", "2026-06-27"),
        ("England", "Ghana", "2026-06-27"),
    ],
}

PREDICTIONS_FILE = Path(__file__).parent / "predictions.json"


def _score_label(ga: float, gb: float) -> str:
    """Round expected goals to nearest integer for a clear predicted score."""
    return f"{int(round(ga))}-{int(round(gb))}"


def _confidence_label(home_win: float, draw: float, away_win: float) -> str:
    """Label the predicted result: 稳/略占优/势均/悬念."""
    best = max(home_win, draw, away_win)
    if best > 0.65:
        return "稳" if best > 0.75 else "略占优"
    elif best > 0.40:
        return "势均"
    return "悬念"


def generate_all_predictions() -> list[dict]:
    """Generate predictions for all 72 group matches."""
    elo = EloCalculator()
    poisson = PoissonModel(elo)

    predictions = []
    for group_name in sorted(SCHEDULE.keys()):
        for team_a, team_b, date in SCHEDULE[group_name]:
            elo_pred = elo.predict_match(team_a, team_b)
            poisson_pred = poisson.predict(team_a, team_b)

            la = poisson_pred.home_goals_lambda
            lb = poisson_pred.away_goals_lambda

            # Predicted result based on highest probability
            probs = {
                "home_win": poisson_pred.home_win_prob,
                "draw": poisson_pred.draw_prob,
                "away_win": poisson_pred.away_win_prob,
            }
            predicted_result = max(probs, key=probs.get)

            # Top 3 most likely scores (no duplicate probabilities)
            top_scores = []
            seen_probs = set()
            for s in poisson_pred.most_likely_scores:
                if s.probability not in seen_probs and len(top_scores) < 3:
                    top_scores.append({"score": s.score, "prob": round(s.probability, 4)})
                    seen_probs.add(s.probability)

            confidence = _confidence_label(
                poisson_pred.home_win_prob,
                poisson_pred.draw_prob,
                poisson_pred.away_win_prob,
            )

            predictions.append({
                "group": group_name,
                "team_a": team_a,
                "team_b": team_b,
                "date": date,
                "predicted": {
                    # Expected goals (continuous, not rounded)
                    "xg_a": round(la, 2),
                    "xg_b": round(lb, 2),
                    # Human-readable score (rounds if close to integer)
                    "score": _score_label(la, lb),
                    # Predicted result (home_win / draw / away_win)
                    "result": predicted_result,
                    # Win/draw/loss probabilities
                    "home_win_prob": round(poisson_pred.home_win_prob, 4),
                    "draw_prob": round(poisson_pred.draw_prob, 4),
                    "away_win_prob": round(poisson_pred.away_win_prob, 4),
                    # Over 2.5 goals probability
                    "over_2_5_prob": round(poisson_pred.over_2_5_prob, 4),
                    # Top 3 most likely scores
                    "top_scores": top_scores,
                    # Confidence label
                    "confidence": confidence,
                },
                "actual": None,
            })

    return predictions


def save_predictions():
    """Generate and save predictions to JSON file."""
    predictions = generate_all_predictions()
    data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_matches": len(predictions),
        "matches": predictions,
    }
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 {len(predictions)} 场小组赛预测到 {PREDICTIONS_FILE}")
    return predictions


def load_predictions() -> dict:
    """Load saved predictions."""
    if not PREDICTIONS_FILE.exists():
        print("❌ 未找到预测文件，请先运行 python -m src.model.predict_all")
        return {"matches": []}
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary():
    """Print a human-readable summary of all predictions."""
    data = load_predictions()
    if not data["matches"]:
        return

    print(f"\n🌍 2026 世界杯小组赛预测总览")
    print(f"   生成时间: {data['generated_at']}")
    print(f"   共 {data['total_matches']} 场比赛\n")

    current_group = None
    for m in data["matches"]:
        if m["group"] != current_group:
            current_group = m["group"]
            print(f"\n{'─' * 65}")
            print(f"  {current_group}")
            print(f"{'─' * 65}")

        p = m["predicted"]
        probs = f"{p['home_win_prob']*100:.0f}%/{p['draw_prob']*100:.0f}%/{p['away_win_prob']*100:.0f}%"
        scores_str = " | ".join(f"{s['score']}({s['prob']*100:.0f}%)" for s in p["top_scores"])
        elo_a = __import__("src.model.data", fromlist=["get_elo"]).get_elo(m["team_a"])
        elo_b = __import__("src.model.data", fromlist=["get_elo"]).get_elo(m["team_b"])

        status_str = ""
        if m["actual"]:
            a = m["actual"]
            actual_str = f"{a['score_a']}-{a['score_b']}"
            from src.model.daily_update import _judge_match
            tag, _ = _judge_match(p, a)
            status_str = f" → 实际 {actual_str} {tag}"

        print(f"  {m['date']}  {m['team_a']:20s} vs {m['team_b']:20s}{status_str}")
        print(f"    预测 {p['score']:>5s}  (λ={p['xg_a']:.2f}-{p['xg_b']:.2f})  |  {probs}  |  [{p['confidence']}]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print_summary()
    else:
        save_predictions()
        print_summary()
