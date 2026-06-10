# 世界杯预言家 · 双模型对决

⚽️🤖⚡📊

同一个比赛，让 **AI 分析师** 和 **数据科学家** 分别预测，对比着看。

## 项目结构

```
worldcup_redskill/
├── SKILL.md            # REDSkill 核心文件（小红书投稿用）
├── main.py             # 统一入口
├── src/ai/             # AI 驱动方案（基于大模型）
├── src/model/          # 数据模型方案（Elo+泊松+蒙特卡洛）
└── .env                # API 配置
```

## 使用

```bash
# 双模型对决
python main.py both "巴西 vs 法国"

# AI 分析
python main.py ai "阿根廷 vs 德国"

# 数据模型
python main.py model "英格兰 vs 西班牙"

# 赛程模拟
python main.py simulate --n 10000 --top 16
```

## 参赛

本工具参与小红书 REDSkill 大赏「世界杯预言家」赛道。
