import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.ai.config import Config
from src.ai.data_gatherer import MatchContext
from src.ai.llm import call_llm

logger = logging.getLogger(__name__)

PREDICTION_SYSTEM_PROMPT = """你是一位顶级的足球比赛分析师。你精通足球战术、球队历史、球员能力和比赛心理。
你的任务是基于提供的比赛数据给出专业、客观的预测分析。

分析框架：
1. **球队实力对比**：排名、阵容深度、近期状态
2. **战术分析**：两队风格匹配度、关键对位、可能的战术安排
3. **关键因素**：伤病影响、心理因素、比赛环境
4. **历史交锋**：过往战绩的心理影响
5. **综合预测**：基于以上分析的结论

请以JSON格式输出预测结果。不要使用markdown代码块标记。"""


@dataclass
class PredictionResult:
    team_a_win_probability: float = 0.0
    draw_probability: float = 0.0
    team_b_win_probability: float = 0.0
    predicted_score: str = ""
    confidence: str = ""          # 高 / 中 / 低
    predicted_winner: str = ""    # 队名 或 "平局"
    key_analysis: list[str] = field(default_factory=list)
    tactical_analysis: str = ""
    key_players_to_watch: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    betting_advice: str = ""


def _clean_json(text: str) -> str:
    """Extract JSON from LLM response, handling code fences."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_prediction(text: str, team_a: str, team_b: str) -> PredictionResult:
    """Parse LLM JSON response into PredictionResult."""
    text = _clean_json(text)
    try:
        data = json.loads(text)
        return PredictionResult(
            team_a_win_probability=data.get("team_a_win_probability", 33),
            draw_probability=data.get("draw_probability", 34),
            team_b_win_probability=data.get("team_b_win_probability", 33),
            predicted_score=data.get("predicted_score", ""),
            confidence=data.get("confidence", "中"),
            predicted_winner=data.get("predicted_winner", ""),
            key_analysis=data.get("key_analysis", []),
            tactical_analysis=data.get("tactical_analysis", ""),
            key_players_to_watch=data.get("key_players_to_watch", []),
            risk_factors=data.get("risk_factors", []),
            betting_advice=data.get("betting_advice", ""),
        )
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to parse prediction JSON: %s", e)
        return PredictionResult(predicted_winner=text[:200])


def predict_match(context: MatchContext, config: Config) -> PredictionResult:
    """Run AI prediction for a match."""
    team_a = context.team_a
    team_b = context.team_b
    h2h = context.head_to_head

    team_a_info = (
        f"球队: {team_a.name}\n"
        f"FIFA排名: {team_a.fifa_ranking}\n"
        f"近期状态: {team_a.recent_form}\n"
        f"风格: {team_a.style_description}\n"
        f"核心球员: {', '.join(team_a.key_players)}\n"
        f"伤病: {', '.join(team_a.injuries) if team_a.injuries else '无重大伤病'}\n"
    )

    team_b_info = (
        f"球队: {team_b.name}\n"
        f"FIFA排名: {team_b.fifa_ranking}\n"
        f"近期状态: {team_b.recent_form}\n"
        f"风格: {team_b.style_description}\n"
        f"核心球员: {', '.join(team_b.key_players)}\n"
        f"伤病: {', '.join(team_b.injuries) if team_b.injuries else '无重大伤病'}\n"
    )

    h2h_info = f"历史交锋: {h2h.summary}\n"
    if h2h.matches:
        h2h_info += "最近交锋:\n" + "\n".join(f"  - {m}" for m in h2h.matches)

    user_prompt = f"""请分析以下世界杯比赛并给出预测。

## 比赛信息
赛事: {context.tournament_info}
阶段: {context.match_stage or '待定'}

## 球队 A 数据
{team_a_info}

## 球队 B 数据
{team_b_info}

## 历史交锋
{h2h_info}

## 输出格式要求
请以JSON格式返回分析结果（不要使用markdown代码块），包含以下字段：

{{
    "team_a_win_probability": <整数, 球队A胜率百分比>,
    "draw_probability": <整数, 平局概率百分比>,
    "team_b_win_probability": <整数, 球队B胜率百分比>,
    "predicted_score": "<预测比分, 如 '2-1'>",
    "confidence": "<高/中/低>",
    "predicted_winner": "<预测胜者队名或'平局'>",
    "key_analysis": ["<要点1>", "<要点2>", "<要点3>"],
    "tactical_analysis": "<战术分析, 50字以内>",
    "key_players_to_watch": ["<关键球员1>", "<关键球员2>"],
    "risk_factors": ["<风险1>", "<风险2>"],
    "betting_advice": "<投注建议, 30字以内>"
}}

请确保三项概率之和等于100。"""

    logger.info("Running AI prediction for %s vs %s...", team_a.name, team_b.name)
    text = call_llm(
        messages=[{"role": "user", "content": user_prompt}],
        config=config,
        system_prompt=PREDICTION_SYSTEM_PROMPT,
        temperature=0.5,
    )

    return _parse_prediction(text, team_a.name, team_b.name)
