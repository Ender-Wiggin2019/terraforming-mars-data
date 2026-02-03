---
name: tfm-data-processing
description: Terraforming Mars (TFM) game data processing skill. Use when working with TFM PostgreSQL database, analyzing game data, processing player statistics, or handling game JSON structures. This skill covers database safety rules, table schemas, and game data structure.
---

# TFM Data Processing

## Critical Safety Rules

### Remote PostgreSQL Database - READ ONLY

**严禁对远程 PostgreSQL 数据库执行任何写操作！**

- ✅ 允许: `SELECT`, `readTable()`, `readSql()`
- ❌ 禁止: `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, `DROP`, `CREATE`, `ALTER`
- ❌ 禁止: `writeDfToPg()`, `appendPgTable()`, `deleteTableData()`, `run()` 等写入方法

使用 `pgOperation.py` 中的 `PgOperation` 类时，只能调用：
- `readTable(table)` - 读取整表
- `readSql(sql)` - 执行自定义 SELECT 查询
- `listTables()` - 列出所有表
- `tableToCsv()` / `exportAllTablesToSnapshot()` - 导出数据到本地

### Local SQLite Database - For Results Storage

处理后的结果数据应存储在本地 SQLite 数据库中，而非远程 PostgreSQL。

## Local Data Structure (`./local_data/`)

本地数据存储目录，通过 `preprocess.py` 从远程 PostgreSQL 获取并存储。

### 目录结构

```
./local_data/
├── tfm.db          # SQLite 数据库（主要数据存储）
├── user_rank.csv   # 用户排名数据
└── users.csv       # 用户信息数据
```

### SQLite 数据库 (`tfm.db`)

| 表名 | 说明 | 数据来源 |
|------|------|----------|
| `games` | 游戏数据（已合并去重） | 合并 `games`, `games_2024.12_2025.02`, `games_2025.03_2025.06` |
| `game_results` | 游戏结果汇总 | `game_results` 表 |
| `user_game_results` | 用户游戏记录 | `user_game_results` 表 |

**数据范围**：仅包含 2025 年及以后的数据（`createtime >= '2025-01-01'`）

### CSV 文件

| 文件 | 说明 |
|------|------|
| `user_rank.csv` | 用户 TrueSkill 排名数据（全量） |
| `users.csv` | 用户账户信息（全量） |

### 读取本地数据

```python
import sqlite3
import pandas as pd

# 读取 SQLite 数据
conn = sqlite3.connect('./local_data/tfm.db')
games_df = pd.read_sql('SELECT * FROM games', conn)
game_results_df = pd.read_sql('SELECT * FROM game_results', conn)
user_game_results_df = pd.read_sql('SELECT * FROM user_game_results', conn)
conn.close()

# 读取 CSV 数据
user_rank_df = pd.read_csv('./local_data/user_rank.csv')
users_df = pd.read_csv('./local_data/users.csv')
```

### 生成/更新本地数据

运行预处理脚本从远程数据库获取最新数据：

```bash
# 使用 uv 运行
uv run preprocess

# 或直接运行 Python 文件
uv run python preprocess.py
```

## Database Schema

### users 表
用户信息表，存储玩家账户信息。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 用户唯一标识 |
| name | string | 用户名 |
| password | string | 密码 |
| prop | json/string | 用户属性（VIP状态、回滚次数等） |
| createtime | datetime | 创建时间 |

### user_rank 表
用户排名/TrueSkill 评分表。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 用户 ID |
| rank_value | float | 排名值 |
| mu | float | TrueSkill μ 值（技能均值） |
| sigma | float | TrueSkill σ 值（不确定性） |
| trueskill | float | TrueSkill 分数 |
| created_at | datetime | 创建时间 |
| updated_at | datetime | 更新时间 |

### user_game_results 表
用户游戏结果历史记录。

| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | string | 用户 ID |
| game_id | string | 游戏 ID |
| players | int | 玩家人数 |
| generations | int | 游戏世代数 |
| createtime | datetime | 游戏创建时间 |
| corporation | string | 使用的公司（可能多个，用 `\|` 分隔） |
| position | int | 最终排名 |
| player_score | int | 玩家分数 |
| rank_value, mu, sigma, trueskill | float | 该局的 TrueSkill 变化 |
| is_rank | int | 是否为排位赛 |
| phase | string | 游戏阶段（end 表示已结束） |

### games 系列表
包括 `games`, `games_YYYY.MM_YYYY.MM` 等，结构相同但存储不同时间段数据。

**⚠️ 注意**：存在多个分表，需要合并去重！

| 表名示例 | 说明 |
|----------|------|
| `games` | 主表/最新数据 |
| `games_2024.12_2025.02` | 历史分表 |
| `games_2025.03_2025.06` | 历史分表 |

| 字段 | 类型 | 说明 |
|------|------|------|
| game_id | string | 游戏唯一标识 |
| save_id | int | 存档 ID（同一 game_id 取最大值） |
| game | json | **完整游戏状态 JSON**（非常大） |
| status | string | 游戏状态（running/end） |
| createtime | datetime | 创建时间 |
| prop | json | 游戏简要属性（玩家列表、阶段等） |

**处理要点**：
1. 合并所有 `games*` 表
2. 按 `game_id` 分组取 `save_id` 最大值
3. 过滤 `status = 'end'` 的已完成游戏

### game_results 表
游戏结果汇总表。

| 字段 | 类型 | 说明 |
|------|------|------|
| game_id | string | 游戏 ID |
| seed_game_id | string | 种子游戏 ID（用于复盘） |
| players | int | 玩家人数 |
| generations | int | 世代数 |
| game_options | json | 游戏选项配置 |
| scores | json array | 各玩家分数，包含 corporation, playerScore, player |
| createtime | datetime | 创建时间 |

## Game Data Structure (JSON)

`game` 字段包含完整游戏状态，详见 `snapshot/clean_game_data.json`。主要模块：

### 游戏基础信息
- `id` - 游戏 ID
- `phase` - 当前阶段（action, end 等）
- `generation` - 当前世代
- `gameAge` - 游戏年龄
- `seed` / `currentSeed` - 随机种子
- `createtime` / `updatetime` - 时间戳
- `gameOptions` - 游戏配置选项

### 全局参数
- `temperature` - 温度（-30 到 +8）
- `oxygenLevel` - 氧气等级（0-14%）
- `venusScaleLevel` - 金星等级
- `globalsPerGeneration` - 每世代全局参数快照

### 玩家数据 (players 数组)
每个玩家对象包含：

**基础信息**
- `id`, `name`, `color`, `userId`
- `corporations` - 使用的公司
- `terraformRating` - TR 值

**资源**
- `megaCredits`, `steel`, `titanium`, `plants`, `energy`, `heat`
- `*Production` - 对应资源的产量

**卡牌**
- `cardsInHand` - 手牌
- `playedCards` - 已打出的卡牌（含资源数）
- `draftedCards` - 轮抽的卡

**统计**
- `victoryPointsByGeneration` - 每世代 VP
- `actionsTakenThisGame` - 本局行动次数
- `globalParameterSteps` - 全局参数贡献
- `timer` - 计时器数据

### 里程碑与奖励
- `milestones` - 可用里程碑列表
- `claimedMilestones` - 已达成的里程碑
- `awards` - 可用奖项列表
- `fundedAwards` - 已资助的奖项

### 殖民地 (colonies)
- `name` - 殖民地名称
- `colonies` - 已建立殖民地的玩家
- `trackPosition` - 轨道位置
- `visitor` - 当前访客

### 议会 (turmoil)
- `chairman` - 主席
- `rulingParty` - 执政党
- `dominantParty` - 主导党
- `parties` - 各党派及代表

### 卡组状态
- `corporationDeck`, `projectDeck`, `preludeDeck`, `ceoDeck`
- 各含 `drawPile` 和 `discardPile`

## Workflow

### 数据处理流程

1. **从远程 PostgreSQL 读取数据**
   ```python
   from pgOperation import PgOperation
   import os
   from dotenv import load_dotenv

   load_dotenv()
   pg = PgOperation(
       ip=os.getenv('PG_HOST'),
       port=os.getenv('PG_PORT'),
       user=os.getenv('PG_USER'),
       pwd=os.getenv('PG_PASSWORD'),
       db=os.getenv('PG_DATABASE')
   )

   df = pg.readTable('users')  # 只读！
   ```

2. **在 Jupyter 中处理，保留中间态**
   - 使用 `.ipynb` 文件进行数据处理
   - 保留每个处理步骤的输出
   - 使用变量保存中间结果便于调试

3. **结果存储到本地 SQLite**
   ```python
   import sqlite3

   conn = sqlite3.connect('results.db')
   df_processed.to_sql('table_name', conn, if_exists='replace', index=False)
   conn.close()
   ```

### 环境配置

复制 `.env.example` 为 `.env` 并填入数据库连接信息。

## 数据分析文档

**进行数据分析时，请参阅 `analyze/` 目录下的详细文档：**

| 文档 | 说明 |
|------|------|
| [`analyze/PREPROCESSING.md`](./analyze/PREPROCESSING.md) | 数据预处理指南：如何从远程 PG 获取数据并存储到本地 |
| [`analyze/DATA_ANALYSIS.md`](./analyze/DATA_ANALYSIS.md) | 数据分析指南：可分析维度、表关系、示例代码 |

### PREPROCESSING.md 内容概要

- 预处理脚本运行方法
- games 表分片获取策略（减轻数据库负载）
- 本地存储方案（SQLite + CSV）
- 增量更新方法

### DATA_ANALYSIS.md 内容概要

- **数据处理注意事项**：games 表合并去重、公司名称分隔处理
- **核心可分析数据**：
  - 公司使用数据 (Corporations)
  - 前序卡 (Preludes)
  - 打出的卡牌 (Played Cards)
  - 玩家统计数据
  - 里程碑与奖励
  - 殖民地数据
  - 议会数据 (Turmoil)
  - 全局参数进度
- **表关系与数据关联**：跨表查询示例
- **年度报告建议指标**：玩家/卡牌/公司/对局维度
- **数据提取示例代码**：完整的 Python 函数

## Reference Files

### 数据分析文档
- `analyze/PREPROCESSING.md` - 数据预处理完整指南
- `analyze/DATA_ANALYSIS.md` - 数据分析完整指南

### 数据快照
- `snapshot/clean_game_data.json` - 清洗后的游戏 JSON 示例
- `snapshot/raw_game_data.json` - 原始游戏 JSON
- `snapshot/*.csv` - 各表的前几行数据快照

### 本地数据
- `local_data/tfm.db` - 本地 SQLite 数据库
- `local_data/user_rank.csv` - 用户排名 CSV
- `local_data/users.csv` - 用户信息 CSV

### 代码文件
- `pgOperation.py` - PostgreSQL 操作封装类
- `preprocess.py` - 数据预处理脚本
