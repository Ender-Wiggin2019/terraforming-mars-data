# TFM 数据分析指南

本文档说明 TFM 数据分析的数据结构和使用方法。

---

## 数据流程概览

```
远程 PostgreSQL          本地存储                  分析数据库
┌─────────────┐         ┌─────────────┐           ┌──────────────────┐
│ games       │         │ local_data/ │           │ tfm_analysis.db  │
│ game_results│  ───►   │ tfm.db      │   ───►    │                  │
│ user_*      │ preprocess│ *.csv      │ transform │ 结构化分析表     │
└─────────────┘         └─────────────┘           └──────────────────┘
```

**关键脚本**：
- `preprocess.py` - 从远程 PostgreSQL 获取数据到 `local_data/`
- `data_transform.ipynb` - 将原始数据转换为结构化分析表

---

## 分析数据库结构 (`local_data/tfm_analysis.db`)

运行 `data_transform.ipynb` 后生成的分析数据库包含以下表：

### 核心表

| 表名 | 说明 | 数据来源 |
|------|------|----------|
| `flat_game_results` | 展开后的游戏结果（每行一个玩家） | `game_results.scores` JSON 展开 |
| `processed_user_game_results` | 处理后的用户游戏记录 | `user_game_results` + 公司拆分 |
| `user_rank` | 用户 TrueSkill 排名 | CSV 导入 |
| `users` | 用户信息 | CSV 导入 |

### 从 games.game 提取的表

| 表名 | 说明 | 字段 |
|------|------|------|
| `played_cards` | 打出的卡牌 | game_id, user_id, card_name, card_order, resource_count, corporation_1/2 |
| `player_stats` | 玩家统计数据 | 资源、产量、全局贡献、时间、卡牌数等 |
| `milestones` | 达成的里程碑 | game_id, milestone_name, player_id, claim_order |
| `awards` | 资助的奖励 | game_id, award_name, funder_player_id, fund_order |
| `global_parameters` | 每世代全局参数 | game_id, generation, temperature, oxygen, oceans, venus |

### 表结构详情

#### flat_game_results
```
game_id, seed_game_id, players, generations, createtime,
position, rank, player_name, player_name_raw, player_score,
corporation_raw, corporation_1, corporation_2, corporation_3,
corporation_count, is_bot
```

#### played_cards
```
game_id, createtime, user_id, player_id, player_name, player_name_raw,
terraform_rating, corporation_1, corporation_2,
card_order, card_name, resource_count, clone_tag, bonus_resource, is_disabled
```

#### player_stats
```
game_id, createtime, generation, user_id, player_id, player_name,
corporation_1, corporation_2, terraform_rating, victory_points_final,
mega_credits, mc_production, steel, steel_production, titanium, titanium_production,
plants, plant_production, energy, energy_production, heat, heat_production,
actions_taken, delegates_placed,
oceans_contributed, oxygen_contributed, temperature_contributed, venus_contributed,
time_elapsed_ms, fleet_size, cards_played_count, cards_in_hand_count
```

---

## ⚠️ 数据完整性说明

### games 与 game_results 的关系

- **game_results 中存在没有对应 games 记录的数据**（约 5-6%）
- 这些通常是放弃/投降/异常结束的游戏（低世代数、低分数）
- `data_transform.ipynb` 使用 INNER JOIN 只处理同时存在于两表的数据
- `flat_game_results` 保留了所有 game_results 记录（不依赖 games）

### 处理策略

```python
# 从 games 提取数据时，只处理有 game_results 的游戏
sql = """
    SELECT g.* FROM games g
    INNER JOIN (SELECT DISTINCT game_id FROM game_results) gr
    ON g.game_id = gr.game_id
    WHERE g.status IN ('finished', 'end')
"""
```

---

## ⚠️ 重要数据处理注意事项

### 1. games 表存在多个分表，需要合并去重

数据库中存在多个 games 相关表，结构相同但存储不同时间段的数据：

| 表名 | 说明 |
|------|------|
| `games` | 主表/最新数据 |
| `games_YYYY.MM_YYYY.MM` | 历史分表，如 `games_2024.12_2025.02`、`games_2025.03_2025.06` |

**处理方式**：需要合并所有 games 表并去重：

```python
# 获取所有 games 相关表
tables = pg.listTables()
games_tables = [t for t in tables if t.startswith('games')]

# 合并所有表
all_games = []
for table in games_tables:
    df = pg.readTable(table)
    all_games.append(df)

games_combined = pd.concat(all_games, ignore_index=True)

# 按 game_id 去重，保留最新的 save_id
games_dedup = games_combined.loc[games_combined.groupby('game_id')['save_id'].idxmax()]
```

### 2. 获取最新游戏状态

同一个 `game_id` 会有多条记录（不同 `save_id`），**必须取 `save_id` 最大值**才能获取游戏最终状态：

```python
# 获取每个游戏的最新快照
latest_games = games_df.loc[games_df.groupby('game_id')['save_id'].idxmax()]
```

或使用 SQL（单表）：
```sql
SELECT g.*
FROM games g
INNER JOIN (
    SELECT game_id, MAX(save_id) as max_save_id
    FROM games
    GROUP BY game_id
) latest ON g.game_id = latest.game_id AND g.save_id = latest.max_save_id
```

### 3. 公司名称分隔处理

`user_game_results.corporation` 字段中，如果玩家使用双公司，会用 `|` 分隔：

```
"Teractor (breakthrough)|MorningStar Inc (breakthrough)"
```

**处理方式**：
```python
# 拆分公司名称
df['corporations'] = df['corporation'].str.split('|')
df_exploded = df.explode('corporations')
```

---

## 核心可分析数据

### 1. 公司使用数据 (Corporations)

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `players[].corporations` | 完整公司对象（含 name, resourceCount, isDisabled） |
| `games.game` | `players[].pickedCorporationCard` | 第一公司 |
| `games.game` | `players[].pickedCorporationCard2` | 第二公司（双公司模式） |
| `user_game_results` | `corporation` | 公司名称，多个用 `\|` 分隔 |
| `game_results` | `scores[].corporation` | 结果中的公司名称 |

**关联表获取更多上下文**:

```sql
-- 关联 user_game_results 获取玩家使用公司的胜负情况
SELECT
    ugr.corporation,
    ugr.position,
    ugr.player_score,
    ugr.players AS player_count,
    ugr.generations,
    u.name AS player_name,
    ur.trueskill
FROM user_game_results ugr
JOIN users u ON ugr.user_id = u.id
LEFT JOIN user_rank ur ON ugr.user_id = ur.id
WHERE ugr.phase = 'end'
```

**可分析维度**:
- 公司使用率（按玩家人数 `players` 分组）
- 公司胜率（`position = 1` 的比例）
- 公司平均分数（`player_score` 均值）
- 公司与世代数的关系（`generations`）
- 双公司组合分析（解析 `|` 分隔后的组合）
- 公司与玩家技能等级的关系（关联 `user_rank.trueskill`）

### 2. 前序卡 (Preludes)

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `players[].dealtPreludeCards` | 发到的前序牌 |
| `games.game` | `players[].preludeCardsInHand` | 手中的前序牌（通常游戏结束时为空） |
| `games.game` | `players[].playedCards` | 已打出的卡牌（包含前序） |
| `games.game` | `preludeDeck.drawPile` | 抽牌堆 |
| `games.game` | `preludeDeck.discardPile` | 弃牌堆 |

**注意**: 已打出的前序会在 `playedCards` 中，需要通过卡牌名称或类型判断哪些是前序卡。

**关联分析**:

```sql
-- 结合 user_game_results 分析前序卡与胜率
-- 需要先从 game.json 提取前序数据存入本地表
SELECT
    prelude_name,
    COUNT(*) as total_uses,
    SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) as wins,
    AVG(player_score) as avg_score
FROM local_prelude_usage lpu
JOIN user_game_results ugr ON lpu.game_id = ugr.game_id AND lpu.user_id = ugr.user_id
GROUP BY prelude_name
```

**可分析维度**:
- 前序使用率
- 前序胜率（关联 `user_game_results.position`）
- 前序平均得分贡献（关联 `user_game_results.player_score`）
- 前序与公司组合分析
- 前序与玩家人数的关系（关联 `user_game_results.players`）

### 3. 打出的卡牌 (Played Cards)

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `players[].playedCards` | 已打出卡牌数组 |
| `games.game` | `players[].playedCards[].name` | 卡牌名称 |
| `games.game` | `players[].playedCards[].resourceCount` | 卡牌上的资源数量 |
| `games.game` | `players[].cardsInHand` | 手牌（未打出） |
| `games.game` | `players[].draftedCards` | 轮抽的卡牌 |
| `games.game` | `projectDeck.discardPile` | 项目卡弃牌堆 |

**关联表获取胜负上下文**:

```sql
-- 需要先从 game.json 提取卡牌数据存入本地表 local_played_cards
-- 结构: game_id, user_id, card_name, resource_count

SELECT
    lpc.card_name,
    COUNT(*) as play_count,
    COUNT(DISTINCT lpc.game_id) as game_count,
    SUM(CASE WHEN ugr.position = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(SUM(CASE WHEN ugr.position = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
    AVG(ugr.player_score) as avg_score,
    AVG(lpc.resource_count) as avg_resources
FROM local_played_cards lpc
JOIN user_game_results ugr
    ON lpc.game_id = ugr.game_id AND lpc.user_id = ugr.user_id
WHERE ugr.phase = 'end'
GROUP BY lpc.card_name
HAVING COUNT(*) >= 10  -- 最小样本量
ORDER BY win_rate DESC
```

**可分析维度**:
- **卡牌打出率** = 某卡被打出次数 / 总局数（从 `game_results` 获取总局数）
- **卡牌胜率** = 打出某卡并获胜的次数 / 打出某卡的总次数（关联 `user_game_results.position`）
- **卡牌平均得分** = 打出某卡玩家的平均分（关联 `user_game_results.player_score`）
- 卡牌平均资源数（`resourceCount`）
- 卡牌与公司的关联（关联 `user_game_results.corporation`）
- 卡牌与玩家人数的关系（关联 `user_game_results.players`）
- 高分局的常见卡牌组合
- 卡牌与世代数的关系（关联 `user_game_results.generations`）

### 4. 玩家统计数据

**主要数据来源**:

| 来源 | 字段分类 | 具体字段 |
|------|----------|----------|
| `games.game` | 资源 | `megaCredits`, `steel`, `titanium`, `plants`, `energy`, `heat` |
| `games.game` | 产量 | `megaCreditProduction`, `steelProduction`, `titaniumProduction`, `plantProduction`, `energyProduction`, `heatProduction` |
| `games.game` | 评分 | `terraformRating`, `victoryPointsByGeneration[]`, `colonyVictoryPoints` |
| `games.game` | 行动 | `actionsTakenThisGame`, `totalDelegatesPlaced` |
| `games.game` | 全局贡献 | `globalParameterSteps.oceans/oxygen/temperature/venus/moon-*` |
| `games.game` | 时间 | `timer.sumElapsed`（毫秒） |
| `games.game` | 卡牌成本 | `cardCost`, `cardDiscount` |
| `games.game` | 殖民地 | `fleetSize`, `tradesThisGeneration`, `colonyTradeOffset`, `colonyTradeDiscount` |
| `user_game_results` | 结果 | `position`, `player_score`, `generations` |
| `user_rank` | 技能 | `mu`, `sigma`, `trueskill` |
| `users` | 账户 | `name`, `prop`（VIP 等） |

#### 4.1 资源数据（游戏结束时快照）
```json
{
  "megaCredits": 49, "megaCreditProduction": 11,
  "steel": 4, "steelProduction": 1,
  "titanium": 8, "titaniumProduction": 6,
  "plants": 2, "plantProduction": 0,
  "energy": 5, "energyProduction": 5,
  "heat": 4, "heatProduction": 1
}
```

#### 4.2 评分数据
```json
{
  "terraformRating": 37,
  "victoryPointsByGeneration": [21, 35, 39, 58, 77],  // 每世代 VP 快照
  "colonyVictoryPoints": 3
}
```

#### 4.3 行动与贡献数据
```json
{
  "actionsTakenThisGame": 132,
  "totalDelegatesPlaced": 7,
  "globalParameterSteps": {
    "oceans": 3, "oxygen": 2, "temperature": 5, "venus": 6,
    "moon-habitat": 0, "moon-mining": 0, "moon-logistics": 0
  }
}
```

#### 4.4 时间数据
```json
{
  "timer": {
    "sumElapsed": 3068894,  // 总用时（毫秒）
    "running": false,
    "afterFirstAction": true
  }
}
```

**关联分析示例**:

```sql
-- 关联 user_rank 分析高技能玩家的游戏特征
-- 需要先从 game.json 提取玩家统计数据存入本地表

SELECT
    CASE
        WHEN ur.trueskill >= 40 THEN 'High'
        WHEN ur.trueskill >= 25 THEN 'Medium'
        ELSE 'Low'
    END as skill_tier,
    AVG(lps.actions_taken) as avg_actions,
    AVG(lps.total_production) as avg_production,
    AVG(ugr.player_score) as avg_score,
    AVG(lps.timer_elapsed / 1000 / 60) as avg_time_minutes
FROM local_player_stats lps
JOIN user_game_results ugr ON lps.game_id = ugr.game_id AND lps.user_id = ugr.user_id
JOIN user_rank ur ON lps.user_id = ur.id
GROUP BY skill_tier
```

**可分析维度**:
- 平均游戏时长（`timer.sumElapsed`）
- 每行动平均用时（`timer.sumElapsed / actionsTakenThisGame`）
- 各全局参数贡献分布（`globalParameterSteps`）
- TR 增长曲线（`victoryPointsByGeneration`）
- 产量与胜率关系（各 `*Production` 字段与 `position` 的相关性）
- 技能等级与游戏风格（关联 `user_rank.trueskill`）
- VIP 玩家 vs 普通玩家对比（关联 `users.prop`）

### 5. 里程碑与奖励

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `milestones` | 本局可用的里程碑列表 |
| `games.game` | `claimedMilestones` | 已达成的里程碑（含 milestone.name, player.id） |
| `games.game` | `awards` | 本局可用的奖项列表 |
| `games.game` | `fundedAwards` | 已资助的奖项（含 award.name, player.id） |
| `games.game` | `players[].id` | 玩家 ID（用于关联） |

```json
{
  "milestones": [
    { "name": "Irrigator" }, { "name": "Rim Settler" }, { "name": "Minimalist" },
    { "name": "Networker" }, { "name": "Economizer" }, { "name": "Martian" }
  ],
  "claimedMilestones": [
    { "milestone": { "name": "Irrigator" }, "player": { "id": "p2966131cfd3e" } }
  ],
  "awards": [{ "name": "Biologist" }, { "name": "Scientist" }],
  "fundedAwards": [
    { "award": { "name": "Space Baron" }, "player": { "id": "p84ab1f1a973e" } }
  ]
}
```

**关联分析示例**:

```sql
-- 分析里程碑/奖励与胜率的关系
-- 需要先从 game.json 提取里程碑数据存入本地表

SELECT
    lm.milestone_name,
    COUNT(*) as claim_count,
    SUM(CASE WHEN ugr.position = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(SUM(CASE WHEN ugr.position = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as win_rate,
    AVG(ugr.player_score) as avg_score
FROM local_milestones lm
JOIN user_game_results ugr ON lm.game_id = ugr.game_id AND lm.player_id = ugr.user_id
GROUP BY lm.milestone_name
ORDER BY win_rate DESC
```

**可分析维度**:
- 里程碑达成率（各里程碑被达成次数 / 可用次数）
- 里程碑与胜率关系（关联 `user_game_results.position`）
- 奖励投资回报率（资助者最终分数排名）
- 常见里程碑/奖励组合
- 里程碑达成顺序分析（第一个 vs 第二个 vs 第三个）
- 不同玩家人数下的里程碑竞争情况（关联 `user_game_results.players`）

### 6. 殖民地数据

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `colonies` | 殖民地数组 |
| `games.game` | `colonies[].name` | 殖民地名称 |
| `games.game` | `colonies[].colonies` | 已建立殖民地的玩家 ID 列表 |
| `games.game` | `colonies[].isActive` | 是否激活 |
| `games.game` | `colonies[].trackPosition` | 轨道位置（影响贸易收益） |
| `games.game` | `colonies[].visitor` | 当前访客（最后贸易的玩家） |
| `games.game` | `players[].fleetSize` | 玩家舰队大小 |
| `games.game` | `players[].tradesThisGeneration` | 本世代贸易次数 |
| `games.game` | `players[].colonyTradeOffset` | 贸易偏移量 |
| `games.game` | `players[].colonyTradeDiscount` | 贸易折扣 |
| `games.game` | `players[].colonyVictoryPoints` | 殖民地胜利点数 |

```json
{
  "colonies": [
    {
      "name": "Pluto",
      "colonies": [{ "id": "p23b3c09f8fd3" }, { "id": "p84ab1f1a973e" }],
      "isActive": true,
      "trackPosition": 3,
      "visitor": { "id": "pbdbba220edf1" }
    }
  ]
}
```

**关联分析示例**:

```sql
-- 分析殖民地策略与胜率
SELECT
    lc.colony_name,
    COUNT(*) as settlement_count,
    SUM(CASE WHEN ugr.position = 1 THEN 1 ELSE 0 END) as wins,
    AVG(lps.colony_victory_points) as avg_colony_vp
FROM local_colonies lc
JOIN user_game_results ugr ON lc.game_id = ugr.game_id AND lc.player_id = ugr.user_id
JOIN local_player_stats lps ON lc.game_id = lps.game_id AND lc.player_id = lps.user_id
GROUP BY lc.colony_name
```

**可分析维度**:
- 殖民地使用率（各殖民地被建立次数）
- 殖民地与胜率关系（关联 `user_game_results.position`）
- 贸易频率分析（`tradesThisGeneration`）
- 殖民地 VP 贡献占比（`colonyVictoryPoints / player_score`）
- 舰队大小与贸易策略的关系（`fleetSize`）
- 多殖民地策略 vs 单殖民地策略的效果

### 7. 议会数据 (Turmoil)

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `turmoil.chairman` | 当前主席（玩家 ID） |
| `games.game` | `turmoil.rulingParty` | 执政党名称 |
| `games.game` | `turmoil.dominantParty` | 主导党名称 |
| `games.game` | `turmoil.parties` | 各党派详情 |
| `games.game` | `turmoil.parties[].name` | 党派名称 |
| `games.game` | `turmoil.parties[].delegates` | 代表列表（玩家 ID） |
| `games.game` | `turmoil.parties[].partyLeader` | 党派领袖（玩家 ID） |
| `games.game` | `turmoil.playersInfluenceBonus` | 玩家影响力加成 |
| `games.game` | `players[].totalDelegatesPlaced` | 玩家放置的代表总数 |
| `games.game` | `players[].turmoilPolicyActionUsed` | 是否使用了政策行动 |
| `games.game` | `players[].hasTurmoilScienceTagBonus` | 是否有议会科学标签加成 |

```json
{
  "turmoil": {
    "chairman": "pbdbba220edf1",
    "rulingParty": "Mars First",
    "dominantParty": "Kelvinists",
    "parties": [
      { "name": "Mars First", "delegates": ["p2966131cfd3e"], "partyLeader": "p2966131cfd3e" },
      { "name": "Scientists", "delegates": ["p23b3c09f8fd3"], "partyLeader": "p23b3c09f8fd3" },
      { "name": "Kelvinists", "delegates": ["p2966131cfd3e", "p2966131cfd3e", "NEUTRAL", "pbdbba220edf1"], "partyLeader": "p2966131cfd3e" }
    ]
  }
}
```

**关联分析示例**:

```sql
-- 分析政治参与度与胜率
SELECT
    CASE
        WHEN lps.total_delegates_placed >= 10 THEN 'Heavy'
        WHEN lps.total_delegates_placed >= 5 THEN 'Medium'
        ELSE 'Light'
    END as political_involvement,
    COUNT(*) as player_count,
    SUM(CASE WHEN ugr.position = 1 THEN 1 ELSE 0 END) as wins,
    AVG(ugr.player_score) as avg_score
FROM local_player_stats lps
JOIN user_game_results ugr ON lps.game_id = ugr.game_id AND lps.user_id = ugr.user_id
GROUP BY political_involvement
```

**可分析维度**:
- 主席当选次数与胜率
- 党派偏好分析（各党派代表数分布）
- 党派领袖与胜率关系
- 政治参与度（`totalDelegatesPlaced`）与胜率
- 执政党与游戏结果的相关性
- NEUTRAL 代表的分布情况

### 8. 全局参数进度

**主要数据来源**:

| 来源 | 字段 | 说明 |
|------|------|------|
| `games.game` | `globalsPerGeneration` | 每世代全局参数快照数组 |
| `games.game` | `globalsPerGeneration[].temperature` | 温度（-30 到 +8） |
| `games.game` | `globalsPerGeneration[].oxygen` | 氧气（0-14%） |
| `games.game` | `globalsPerGeneration[].oceans` | 海洋数量（0-9） |
| `games.game` | `globalsPerGeneration[].venus` | 金星等级（0-30） |
| `games.game` | `temperature` | 最终温度 |
| `games.game` | `oxygenLevel` | 最终氧气 |
| `games.game` | `venusScaleLevel` | 最终金星等级 |
| `games.game` | `generation` | 游戏世代数 |
| `games.game` | `gameAge` | 游戏年龄 |
| `user_game_results` | `generations` | 游戏世代数 |
| `user_game_results` | `players` | 玩家人数 |

```json
{
  "globalsPerGeneration": [
    { "temperature": -28, "oxygen": 2, "oceans": 1, "venus": 4 },
    { "temperature": -28, "oxygen": 3, "oceans": 4, "venus": 10 },
    { "temperature": -20, "oxygen": 8, "oceans": 6, "venus": 14 },
    { "temperature": -12, "oxygen": 12, "oceans": 9, "venus": 16 },
    { "temperature": 8, "oxygen": 14, "oceans": 9, "venus": 22 }
  ],
  "temperature": 8,
  "oxygenLevel": 14,
  "venusScaleLevel": 22,
  "generation": 5
}
```

**关联分析示例**:

```sql
-- 分析不同玩家数的地球化速度
SELECT
    ugr.players as player_count,
    AVG(ugr.generations) as avg_generations,
    AVG(lg.avg_temp_increase_per_gen) as avg_temp_speed,
    AVG(lg.avg_oxygen_increase_per_gen) as avg_oxygen_speed
FROM local_globals lg
JOIN user_game_results ugr ON lg.game_id = ugr.game_id
WHERE ugr.phase = 'end'
GROUP BY ugr.players
ORDER BY player_count
```

**可分析维度**:
- 地球化进度曲线（每世代参数变化）
- 各参数推进速度（参数变化量 / 世代数）
- 不同玩家数的地球化速度差异（关联 `user_game_results.players`）
- 游戏时长与地球化速度的关系
- 哪些扩展会加速/减缓地球化（关联 `game_options`）
- 金星扩展的影响分析

---

## 表关系与数据关联

### 核心关联关系

```
users                  user_rank              user_game_results
┌─────────────┐       ┌─────────────┐        ┌──────────────────┐
│ id (PK)     │──────▶│ id (FK)     │        │ user_id (FK)     │──┐
│ name        │       │ mu          │        │ game_id (FK)     │  │
│ prop        │       │ sigma       │        │ corporation      │  │
│ createtime  │       │ trueskill   │        │ position         │  │
└─────────────┘       └─────────────┘        │ player_score     │  │
                                              │ players          │  │
                                              │ generations      │  │
                                              └──────────────────┘  │
                                                                    │
games                  game_results                                 │
┌─────────────┐       ┌──────────────────┐                         │
│ game_id (PK)│◀──────│ game_id (FK)     │◀────────────────────────┘
│ save_id     │       │ scores (JSON)    │
│ game (JSON) │       │ players          │
│ status      │       │ generations      │
│ createtime  │       │ game_options     │
│ prop        │       └──────────────────┘
└─────────────┘
```

### 关联查询示例

```sql
-- 完整的玩家游戏分析查询
SELECT
    u.name as player_name,
    u.id as user_id,
    ur.trueskill,
    ur.mu,
    ur.sigma,
    ugr.game_id,
    ugr.corporation,
    ugr.position,
    ugr.player_score,
    ugr.players as player_count,
    ugr.generations,
    ugr.createtime as game_time,
    gr.game_options,
    gr.scores as all_scores
FROM users u
LEFT JOIN user_rank ur ON u.id = ur.id
JOIN user_game_results ugr ON u.id = ugr.user_id
JOIN game_results gr ON ugr.game_id = gr.game_id
WHERE ugr.phase = 'end'
ORDER BY ugr.createtime DESC
```

### games.game 字段与其他表的映射关系

| games.game 内字段 | 对应表/字段 | 说明 |
|-------------------|-------------|------|
| `id` | `games.game_id`, `game_results.game_id` | 游戏 ID |
| `players[].id` | 内部玩家 ID | 以 'p' 开头，用于 games.game 内部关联 |
| `players[].userId` | `users.id`, `user_game_results.user_id` | 以 'u' 开头，用于跨表关联 |
| `players[].name` | `users.name` | 玩家名称 |
| `generation` | `user_game_results.generations`, `game_results.generations` | 世代数 |
| `phase` | `games.status`, `user_game_results.phase` | 游戏状态 |

---

## 年度报告建议指标

### 玩家维度
- [ ] 总对局数（`COUNT(DISTINCT game_id) FROM user_game_results`）
- [ ] 胜率排行（`position = 1` 的比例，关联 `user_rank.trueskill` 排序）
- [ ] 最常用公司（`corporation` 分组统计）
- [ ] 平均分数（`AVG(player_score)`）
- [ ] 平均游戏时长（`game.json` 中 `timer.sumElapsed`）
- [ ] TrueSkill 变化曲线（`user_game_results` 中的 mu, sigma 历史）
- [ ] 活跃玩家排行（按 `createtime` 统计游戏频率）

### 卡牌维度
- [ ] 卡牌打出率 TOP N（打出次数 / 总局数）
- [ ] 卡牌胜率 TOP N（需设置最低打出次数门槛，如 ≥10 次）
- [ ] 卡牌平均得分贡献
- [ ] 最被忽视的卡牌（低打出率但高胜率）
- [ ] 最常组合的卡牌（共现分析）
- [ ] 卡牌与公司的最佳搭配

### 公司维度
- [ ] 公司使用率（按玩家人数分组）
- [ ] 公司胜率（`position = 1` 的比例）
- [ ] 公司平均分数
- [ ] 公司与特定卡牌的关联
- [ ] 双公司组合分析
- [ ] 公司与 TrueSkill 等级的关系

### 对局维度
- [ ] 按月对局数趋势（`createtime` 按月分组）
- [ ] 平均世代数（`AVG(generations)`）
- [ ] 玩家人数分布（`players` 字段分组）
- [ ] 游戏模式分布（从 `game_options` 提取）
- [ ] 排位赛 vs 普通赛（`is_rank` 字段）
- [ ] 游戏完成率（`phase = 'end'` 的比例）

---

## 数据提取示例

### 从 games 表提取数据的完整流程

**数据来源**：`games` 表（及其分表）的 `game` 字段存储 JSON 格式的完整游戏状态。

```python
import json
import pandas as pd
from pgOperation import PgOperation

# 1. 合并所有 games 表并获取最新快照
def get_all_latest_games(pg: PgOperation):
    """
    合并所有 games 分表，并获取每个 game_id 的最新 save_id

    注意：存在多个分表如 games, games_2024.12_2025.02, games_2025.03_2025.06 等
    """
    # 获取所有 games 相关表
    tables = pg.listTables()
    games_tables = [t for t in tables if t.startswith('games')]

    print(f"Found {len(games_tables)} games tables: {games_tables}")

    # 合并所有表
    all_games = []
    for table in games_tables:
        df = pg.readTable(table)
        df['source_table'] = table  # 记录来源表
        all_games.append(df)

    games_combined = pd.concat(all_games, ignore_index=True)

    # 按 game_id 去重，保留最新的 save_id
    games_dedup = games_combined.loc[games_combined.groupby('game_id')['save_id'].idxmax()]

    # 只保留已结束的游戏
    games_ended = games_dedup[games_dedup['status'] == 'end']

    return games_ended

# 2. 解析 games.game 字段的 JSON
def parse_game_json(game_str):
    """安全解析 games.game 字段的 JSON"""
    if isinstance(game_str, str):
        try:
            return json.loads(game_str)
        except:
            return None
    return game_str
```

### 提取玩家打出的所有卡牌

```python
def extract_played_cards(game_data):
    """
    从 games.game 字段解析的 JSON 中提取打出的卡牌
    包含 userId 用于跨表关联
    """
    cards = []
    for player in game_data.get('players', []):
        user_id = player.get('userId')  # 用于关联 user_game_results
        player_id = player.get('id')     # game.json 内部 ID

        for card in player.get('playedCards', []):
            cards.append({
                'game_id': game_data['id'],
                'user_id': user_id,
                'player_id': player_id,
                'player_name': player.get('name'),
                'card_name': card['name'],
                'resource_count': card.get('resourceCount', 0)
            })
    return cards
```

### 提取公司使用情况

```python
def extract_corporations(game_data):
    """
    从 games.game 字段解析的 JSON 中提取公司数据
    区分主副公司
    """
    corps = []
    for player in game_data.get('players', []):
        user_id = player.get('userId')
        primary_corp = player.get('pickedCorporationCard', {}).get('name')
        secondary_corp = player.get('pickedCorporationCard2', {}).get('name')

        for corp in player.get('corporations', []):
            corps.append({
                'game_id': game_data['id'],
                'user_id': user_id,
                'player_name': player.get('name'),
                'corporation': corp['name'],
                'resource_count': corp.get('resourceCount', 0),
                'is_disabled': corp.get('isDisabled', False),
                'is_primary': corp['name'] == primary_corp,
                'is_secondary': corp['name'] == secondary_corp
            })
    return corps
```

### 提取玩家统计数据

```python
def extract_player_stats(game_data):
    """从 games.game 字段解析的 JSON 中提取玩家统计数据"""
    stats = []
    for player in game_data.get('players', []):
        gps = player.get('globalParameterSteps', {})
        timer = player.get('timer', {})

        stats.append({
            'game_id': game_data['id'],
            'user_id': player.get('userId'),
            'player_name': player.get('name'),
            # 评分
            'terraform_rating': player.get('terraformRating'),
            'victory_points': player.get('victoryPointsByGeneration', [])[-1] if player.get('victoryPointsByGeneration') else None,
            # 资源
            'mega_credits': player.get('megaCredits'),
            'mc_production': player.get('megaCreditProduction'),
            'steel': player.get('steel'),
            'steel_production': player.get('steelProduction'),
            'titanium': player.get('titanium'),
            'titanium_production': player.get('titaniumProduction'),
            # 行动
            'actions_taken': player.get('actionsTakenThisGame'),
            'delegates_placed': player.get('totalDelegatesPlaced'),
            # 全局贡献
            'oceans_contributed': gps.get('oceans', 0),
            'oxygen_contributed': gps.get('oxygen', 0),
            'temp_contributed': gps.get('temperature', 0),
            'venus_contributed': gps.get('venus', 0),
            # 时间
            'time_elapsed_ms': timer.get('sumElapsed'),
            # 殖民地
            'colony_vp': player.get('colonyVictoryPoints', 0),
            'fleet_size': player.get('fleetSize'),
            # 卡牌
            'cards_played': len(player.get('playedCards', [])),
            'cards_in_hand': len(player.get('cardsInHand', []))
        })
    return stats
```

### 提取里程碑与奖励

```python
def extract_milestones_awards(game_data):
    """从 games.game 字段解析的 JSON 中提取里程碑和奖励数据"""
    result = {'milestones': [], 'awards': []}

    # 里程碑
    for idx, m in enumerate(game_data.get('claimedMilestones', [])):
        result['milestones'].append({
            'game_id': game_data['id'],
            'milestone_name': m['milestone']['name'],
            'player_id': m['player']['id'],
            'claim_order': idx + 1
        })

    # 奖励
    for idx, a in enumerate(game_data.get('fundedAwards', [])):
        result['awards'].append({
            'game_id': game_data['id'],
            'award_name': a['award']['name'],
            'funder_id': a['player']['id'],
            'fund_order': idx + 1
        })

    return result
```

### 计算卡牌胜率（关联 user_game_results）

```python
def calculate_card_win_rate(cards_df, ugr_df, min_plays=10):
    """
    计算卡牌胜率
    cards_df: 从 games.game 字段提取的卡牌数据
    ugr_df: user_game_results 表
    """
    # 合并获取胜负信息
    merged = cards_df.merge(
        ugr_df[['user_id', 'game_id', 'position', 'player_score']],
        on=['user_id', 'game_id'],
        how='left'
    )
    merged['is_winner'] = (merged['position'] == 1).astype(int)

    # 按卡牌分组统计
    card_stats = merged.groupby('card_name').agg(
        total_plays=('user_id', 'count'),
        wins=('is_winner', 'sum'),
        avg_score=('player_score', 'mean')
    ).reset_index()

    card_stats['win_rate'] = (card_stats['wins'] / card_stats['total_plays'] * 100).round(2)

    # 过滤最小样本量
    return card_stats[card_stats['total_plays'] >= min_plays].sort_values('win_rate', ascending=False)
```

### 处理双公司分隔符

```python
def split_corporations(ugr_df):
    """拆分 user_game_results 中的双公司"""
    # 创建副本
    df = ugr_df.copy()

    # 拆分公司名称
    df['corporations_list'] = df['corporation'].str.split('|')

    # 展开为多行
    df_exploded = df.explode('corporations_list')
    df_exploded = df_exploded.rename(columns={'corporations_list': 'single_corporation'})

    return df_exploded
```

---

## 未来规划

### 数据处理改进

1. **前序卡 (Preludes) 单独提取**
   - 目前前序卡混在 `playedCards` 中
   - 可通过卡牌类型或打出顺序（前2张）识别
   - 建议新增 `preludes` 表

2. **殖民地数据提取**
   - 从 `games.game.colonies` 提取
   - 包含殖民地建设、贸易记录
   - 建议新增 `colonies` 和 `trades` 表

3. **议会数据提取**
   - 从 `games.game.turmoil` 提取
   - 包含党派、代表、政策使用
   - 建议新增 `turmoil_stats` 表

4. **VP 增长曲线**
   - 从 `players[].victoryPointsByGeneration` 提取
   - 每世代 VP 快照
   - 用于分析得分节奏

### 分析方向

1. **年度报告指标**
   - 玩家活跃度（游戏数、胜率、TrueSkill 变化）
   - 公司/卡牌/前序 META 分析
   - 游戏时长与难度分析

2. **卡牌分析**
   - 卡牌打出率、胜率、平均分数
   - 卡牌组合分析（共现矩阵）
   - 卡牌与公司的最佳搭配

3. **玩家风格分析**
   - 全局参数贡献偏好
   - 资源/产量策略
   - 游戏时长分布

4. **时间序列分析**
   - 每世代资源/产量变化
   - 地球化进度曲线
   - 玩家 TrueSkill 变化趋势

### 使用建议

```python
# 快速开始分析
import sqlite3
import pandas as pd

conn = sqlite3.connect('./local_data/tfm_analysis.db')

# 公司胜率
pd.read_sql('''
    SELECT corporation_1, players,
           COUNT(*) as games,
           ROUND(AVG(CASE WHEN rank=1 THEN 1.0 ELSE 0 END)*100, 1) as win_rate
    FROM flat_game_results
    WHERE is_bot = 0 AND corporation_1 IS NOT NULL
    GROUP BY corporation_1, players
    HAVING games >= 10
    ORDER BY win_rate DESC
''', conn)

# 卡牌胜率（关联查询）
pd.read_sql('''
    SELECT pc.card_name, COUNT(*) as plays,
           ROUND(AVG(CASE WHEN ugr.position=1 THEN 1.0 ELSE 0 END)*100, 1) as win_rate
    FROM played_cards pc
    JOIN processed_user_game_results ugr
        ON pc.game_id = ugr.game_id AND pc.user_id = ugr.user_id
    WHERE ugr.phase = 'end'
    GROUP BY pc.card_name
    HAVING plays >= 20
    ORDER BY win_rate DESC
''', conn)

conn.close()
```
