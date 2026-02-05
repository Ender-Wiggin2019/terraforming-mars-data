# Terraforming Mars Data Processing

殖民火星数据处理与分析项目

## Description

这个项目用于处理和分析殖民火星 (Terraforming Mars) 的游戏数据，包括数据获取、清洗、分析和可视化。

## Prerequisites

### 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Python 包管理器

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync
```

### 环境配置

复制 `.env.example` 为 `.env`，并配置数据库连接信息：

```bash
cp .env.example .env
# 编辑 .env 文件，填入数据库连接信息
```

## Project Structure

```
tfm-data/
├── local_data/           # 本地数据存储目录
│   ├── tfm.db           # 原始数据 SQLite 数据库
│   ├── tfm_analysis.db  # 分析用数据库
│   └── users.csv        # 用户信息
├── display/              # 分析结果输出目录
│   ├── *.csv            # 数据分析结果
│   ├── *.png            # 可视化图表
│   └── *.json           # JSON 格式结果
├── data/                 # 静态数据文件
│   └── cn_merged.json   # 中英文名称映射
├── fonts/                # 字体文件
└── snapshot/             # 数据快照
```

## Workflow

按以下顺序执行脚本完成完整的数据分析流程：

---

### Step 1: 数据预处理 - `preprocess.py`

从远程 PostgreSQL 数据库获取游戏数据并存储到本地 SQLite。

**功能**：
- 从远程数据库获取 games、game_results、user_game_results 表
- 支持分步执行和断点续传
- 数据存储到 `local_data/tfm.db`

**使用方法**：

```bash
# 运行所有步骤（获取全部数据）
uv run python preprocess.py

# 只运行指定步骤
uv run python preprocess.py --step 1           # 只获取 games
uv run python preprocess.py --step 2           # 只获取 game_results
uv run python preprocess.py --step 3           # 只获取 user_game_results
uv run python preprocess.py --step 1,2,3       # 运行指定步骤

# 强制重新获取（忽略已有数据）
uv run python preprocess.py --force
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--step` | 指定运行步骤 (1=games, 2=game_results, 3=user_game_results) | 全部 |
| `--force` | 强制重新获取，忽略已有数据 | False |

**输出**：`local_data/tfm.db`

---

### Step 2: 数据转换 - `data_transform.ipynb`

将原始数据转换为分析友好的结构化格式。

**功能**：
- 解析 JSON 格式的游戏数据
- 提取关键字段并结构化存储
- 生成分析用数据库

**使用方法**：

```bash
# 方式1: 在 Jupyter/VS Code 中直接运行 notebook
# 打开 data_transform.ipynb 并运行所有单元格

# 方式2: 使用命令行运行
uv run jupyter execute data_transform.ipynb
```

**输出**：`local_data/tfm_analysis.db`

---

### Step 3: 卡牌分析 - `card_analysis.py`

分析卡牌（公司、前序、项目卡）的使用情况和胜率。

**功能**：
- 分析公司、前序、项目卡的使用次数和胜率
- 计算贝叶斯平均顺位（平滑低样本数据）
- 生成 2P 和 4P 分开的分析结果
- 生成可视化图表

**使用方法**：

```bash
uv run python card_analysis.py
```

**无命令行参数**，直接运行即可。

**输出文件**：
- `display/corporation_stats_2p.csv` / `display/corporation_stats_4p.csv` - 公司统计
- `display/prelude_stats_2p.csv` / `display/prelude_stats_4p.csv` - 前序统计
- `display/card_stats_2p.csv` / `display/card_stats_4p.csv` - 项目卡统计
- `display/played_cards_raw.csv` - 原始打出卡牌数据
- `display/*_weighted_ranking.png` - 加权排名图表

---

### Step 4: 游戏分析 - `game_analysis.py`

分析游戏维度的数据，包括里程碑、奖项、殖民地等。

**功能**：
- 游戏基础统计（对局数、平均分数、平均时代等）
- 里程碑达成率和胜率分析
- 奖项获取率和胜率分析
- 殖民地使用率和胜率分析
- 按时代分布的分数统计

**使用方法**：

```bash
uv run python game_analysis.py
```

**无命令行参数**，直接运行即可。

**输出文件**：
- `display/game_basic_stats.png` - 基础统计图
- `display/milestone_stats_2p.csv` / `display/milestone_stats_4p.csv` - 里程碑统计
- `display/award_stats_2p.csv` / `display/award_stats_4p.csv` - 奖项统计
- `display/colony_stats_2p.csv` / `display/colony_stats_4p.csv` - 殖民地统计
- `display/*_weighted_ranking.png` - 加权排名图表
- `display/score_by_generation_*.png` - 分数分布图
- `display/tr_by_generation_*.png` - TR 分布图

---

### Step 5: 用户分析 - `user_analysis.py`

从用户维度分析玩家数据。

**功能**：
- 玩家平均统计（胜率、顺位、分数、时代、TR、打牌数）
- 玩家各时代最高纪录（最高分、最多打牌）
- 玩家公司/前序/卡牌使用统计
- 每个公司的 Top 100 玩家排名（贝叶斯平均顺位）

**使用方法**：

```bash
# 分析 4 人局数据（默认）
uv run python user_analysis.py

# 分析 2 人局数据
uv run python user_analysis.py -p 2

# 分析指定人数对局
uv run python user_analysis.py --players 3
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p`, `--players` | 筛选指定人数的对局 | 4 |

**输出文件**（以 4p 为例）：
- `display/user_player_stats_4p.csv` - 玩家综合统计
- `display/user_records_by_generation_4p.csv` - 各时代最高纪录
- `display/user_corporation_stats_4p.csv` - 玩家公司使用统计
- `display/user_prelude_stats_4p.csv` - 玩家前序使用统计
- `display/user_card_stats_4p.csv` - 玩家卡牌使用统计
- `display/user_corp_top100_players_4p.csv` - 各公司 Top 100 玩家
- `display/user_stats_distribution_4p.png` - 统计分布图
- `display/user_top_players_4p.png` - 顶尖玩家图表

---

### Step 6: 用户数据聚合 - `user_aggregate.py`

聚合多个用户的数据，生成综合统计 JSON。

**功能**：
- 聚合指定用户的加权平均统计
- 汇总各时代最高纪录
- 列出用户在 Top 100 的公司排名

**使用方法**：

```bash
# 聚合单个用户数据
uv run python user_aggregate.py -u "user_id_1"

# 聚合多个用户数据（逗号分隔）
uv run python user_aggregate.py -u "user_id_1,user_id_2,user_id_3"

# 指定人数筛选（默认4人局）
uv run python user_aggregate.py -u "user_id_1,user_id_2" -p 2

# 输出到文件
uv run python user_aggregate.py -u "user_id_1,user_id_2" -o ./output.json
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-u`, `--users` | 用户 ID 列表（逗号分隔）**必填** | - |
| `-p`, `--players` | 筛选指定人数的对局 | 4 |
| `-o`, `--output` | 输出 JSON 文件路径 | stdout |

**输出格式**（JSON）：
```json
{
  "user_ids": ["user_id_1", "user_id_2"],
  "players_filter": 4,
  "player_stats": {
    "total_games": 250,
    "win_rate": 47.6,
    "avg_position": 1.796,
    ...
  },
  "records_by_generation": {
    "4": {"max_score": 134, "max_cards_played": 75},
    ...
  },
  "top100_corporations": [
    {"corporation": "公司名", "rank": 5, "avg_score": 95.2, ...},
    ...
  ]
}
```

---

### Step 7: 验证测试 - `test_user_aggregate.py`

验证用户数据聚合的准确性。

**功能**：
- 验证单用户数据与 CSV 一致
- 验证多用户加权平均计算正确
- 验证时代最高纪录取最大值正确

**使用方法**：

```bash
# 运行前先生成测试数据
uv run python user_aggregate.py -u "69215eb418a5" -p 4 -o ./display/test_user1.json
uv run python user_aggregate.py -u "9007426e7f53" -p 4 -o ./display/test_user2.json
uv run python user_aggregate.py -u "69215eb418a5,9007426e7f53" -p 4 -o ./display/test_combined.json

# 运行测试
uv run python test_user_aggregate.py
```

**输出**：测试结果（PASS/FAIL）

---

### Utility: 检查未匹配玩家 - `check_unmatched_players.py`

检查 game_results 中的玩家名是否能与 users.csv 匹配。

**功能**：
- 从 game_results.scores JSON 提取所有玩家名
- 与 users.csv 的用户名进行匹配
- 输出无法匹配且出现次数 >= 2 的名字列表

**使用方法**：

```bash
# 默认运行（出现次数 >= 2）
uv run python check_unmatched_players.py

# 指定最小出现次数
uv run python check_unmatched_players.py -m 5

# 指定输出文件
uv run python check_unmatched_players.py -o ./display/unmatched.csv
```

**参数说明**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m`, `--min-occurrences` | 最小出现次数 | 2 |
| `-o`, `--output` | 输出 CSV 文件路径 | `display/unmatched_players.csv` |

**输出文件**：
- `display/unmatched_players.csv` - 未匹配玩家列表（按出现频率降序）

---

## Quick Start

完整运行流程：

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入数据库信息

# 3. 获取数据
uv run python preprocess.py

# 4. 数据转换（在 VS Code/Jupyter 中运行 notebook）
# 或: uv run jupyter execute data_transform.ipynb

# 5. 运行分析
uv run python card_analysis.py
uv run python game_analysis.py
uv run python user_analysis.py -p 4
uv run python user_analysis.py -p 2

# 6. 用户聚合查询示例
uv run python user_aggregate.py -u "your_user_id" -p 4
```

## Notes

- 远程数据库为**只读**，禁止任何写操作
- 分析脚本默认筛选 `breakthrough=true` 的对局（user_analysis 除外）
- 贝叶斯平均使用 `prior_n=5, prior_mean=2.5` 进行平滑
- 用户分析要求玩家至少有 2 场对局
