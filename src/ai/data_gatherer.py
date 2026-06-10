"""数据采集模块 - 利用 LLM 知识 + 搜索获取比赛相关信息。"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.ai.config import Config
from src.ai.llm import call_llm

logger = logging.getLogger(__name__)

WORLD_CUP_TEAMS = [
    "巴西", "阿根廷", "法国", "英格兰", "西班牙", "葡萄牙", "德国", "荷兰",
    "意大利", "乌拉圭", "比利时", "克罗地亚", "摩洛哥", "日本", "韩国",
    "美国", "墨西哥", "塞内加尔", "瑞士", "哥伦比亚", "丹麦", "尼日利亚",
    "喀麦隆", "澳大利亚", "伊朗", "沙特阿拉伯", "波兰", "塞尔维亚",
    "加纳", "厄瓜多尔", "加拿大", "威尔士", "突尼斯", "阿尔及利亚", "埃及",
]


@dataclass
class TeamData:
    name: str
    fifa_ranking: int = 0
    recent_form: str = ""        # e.g. "WWDLW" (最近5场)
    recent_matches: list[str] = field(default_factory=list)
    key_players: list[str] = field(default_factory=list)
    injuries: list[str] = field(default_factory=list)
    style_description: str = ""
    notes: str = ""


@dataclass
class HeadToHead:
    matches: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class MatchContext:
    team_a: TeamData
    team_b: TeamData
    head_to_head: HeadToHead
    match_stage: str = ""        # e.g. "小组赛", "1/8决赛", "半决赛", "决赛"
    tournament_info: str = ""


def _parse_team_data(text: str, team_name: str) -> TeamData:
    """Parse LLM JSON output into TeamData."""
    try:
        data = json.loads(text)
        return TeamData(
            name=team_name,
            fifa_ranking=data.get("fifa_ranking", 0),
            recent_form=data.get("recent_form", ""),
            recent_matches=data.get("recent_matches", []),
            key_players=data.get("key_players", []),
            injuries=data.get("injuries", []),
            style_description=data.get("style_description", ""),
            notes=data.get("notes", ""),
        )
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse team data for %s, using defaults", team_name)
        return TeamData(name=team_name, notes=text[:200] if text else "")


def gather_team_data(team_name: str, config: Config) -> TeamData:
    """Use LLM knowledge to gather info about a team."""
    system_prompt = "你是一个专业的足球数据专家。请以JSON格式返回数据，不要包含markdown代码块标记。"
    user_prompt = f"""请提供关于球队「{team_name}」的以下信息，以JSON格式返回（不要用markdown代码块）：

{{
    "fifa_ranking": <整数, FIFA世界排名>,
    "recent_form": "<最近5场比赛结果, 如 WWDLW, W=胜 D=平 L=负>",
    "recent_matches": ["<比赛1描述>", "<比赛2描述>", ...],
    "key_players": ["<核心球员1>", "<核心球员2>", ...],
    "injuries": ["<伤病球员1>", ...],
    "style_description": "<球队风格描述, 50字以内>",
    "notes": "<其他值得注意的信息>"
}}

请确保数据准确，基于你的知识库回答。最近比赛请提供2026年最新的比赛信息。"""
    try:
        text = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            config=config,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        return _parse_team_data(text, team_name)
    except Exception as e:
        logger.error("Failed to gather data for %s: %s", team_name, e)
        return TeamData(name=team_name)


def gather_head_to_head(team_a: str, team_b: str, config: Config) -> HeadToHead:
    """Use LLM knowledge to gather head-to-head history."""
    system_prompt = "你是一个专业的足球数据专家。请以JSON格式返回数据。"
    user_prompt = f"""请提供球队「{team_a}」和「{team_b}」的历史交锋记录，以JSON格式返回：

{{
    "matches": [
        "2022-12-18 阿根廷 3-3 法国 (点球 4-2) - 世界杯决赛",
        ...
    ],
    "summary": "<总体交锋概况, 50字以内>"
}}

请尽量提供2022年之后的最新交锋记录。"""
    try:
        text = call_llm(
            messages=[{"role": "user", "content": user_prompt}],
            config=config,
            system_prompt=system_prompt,
            temperature=0.3,
        )
        data = json.loads(text)
        return HeadToHead(
            matches=data.get("matches", []),
            summary=data.get("summary", ""),
        )
    except Exception as e:
        logger.error("Failed to gather head-to-head: %s", e)
        return HeadToHead()


def gather_match_context(team_a: str, team_b: str, config: Config) -> MatchContext:
    """Gather complete context for a match prediction."""
    logger.info("Gathering data for %s vs %s...", team_a, team_b)
    team_a_data = gather_team_data(team_a, config)
    team_b_data = gather_team_data(team_b, config)
    h2h = gather_head_to_head(team_a, team_b, config)

    return MatchContext(
        team_a=team_a_data,
        team_b=team_b_data,
        head_to_head=h2h,
        tournament_info="FIFA World Cup 2026",
    )
