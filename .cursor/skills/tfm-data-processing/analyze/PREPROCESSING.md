# TFM 数据预处理指南

本文档说明如何从远程 PostgreSQL 获取数据并存储到本地，为后续分析做准备。

---

## 快速开始

### 运行预处理脚本

```bash
# 1. 确保已配置 .env 文件（参考 .env.example）
cp .env.example .env
# 编辑 .env 填入数据库连接信息

# 2. 安装依赖（使用 uv）
uv sync

# 3. 运行预处理脚本（任选其一）
uv run preprocess           # 使用 pyproject.toml 定义的脚本入口
uv run python preprocess.py # 直接运行 Python 文件
```

### 输出文件

运行完成后，数据将存储在 `./local_data/` 目录：

| 文件 | 说明 |
|------|------|
| `tfm.db` | SQLite 数据库，包含 games, game_results, user_game_results 表 |
| `user_rank.csv` | 用户排名数据 |
| `users.csv` | 用户信息数据 |

### 在 Jupyter 中使用本地数据

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

---

## ⚠️ 数据获取限制

1. **时间范围**：只获取 2025 年及以后的数据（`createtime >= '2025-01-01'`）
2. **远程数据库只读**：严禁任何写操作！
3. **games 表分片获取**：由于数据量大，需要分页获取

---

## 存储方案选择

### SQLite vs CSV 对比

| 特性 | SQLite | CSV |
|------|--------|-----|
| 查询性能 | ✅ 支持 SQL 查询，有索引 | ❌ 需要全部加载到内存 |
| 存储效率 | ✅ 压缩存储 | ❌ 文本格式，体积较大 |
| 数据完整性 | ✅ 支持事务、约束 | ❌ 无约束 |
| 增量更新 | ✅ 方便追加和更新 | ❌ 需要重写整个文件 |
| 通用性 | ⚠️ 需要 sqlite3 库 | ✅ 任何工具都能读取 |
| JSON 字段 | ⚠️ 存为 TEXT，查询需解析 | ⚠️ 存为字符串 |

### 推荐方案：混合使用

| 表 | 存储格式 | 原因 |
|----|----------|------|
| `games` | **SQLite** | 数据量大，需要按 game_id 查询和去重 |
| `game_results` | SQLite | 需要与 games 关联查询 |
| `user_game_results` | SQLite | 关联查询频繁 |
| `user_rank` | CSV | 数据量小，便于查看 |
| `users` | CSV | 数据量小，便于查看 |

**本地数据库路径**：`./local_data/tfm.db`
**CSV 存储路径**：`./local_data/`

---

## 需要获取的表

| 表名 | 时间过滤字段 | 说明 |
|------|--------------|------|
| `games` | `createtime` | 主表 |
| `games_2024.12_2025.02` | `createtime` | 分表（包含 2025.01-02 数据） |
| `games_2025.03_2025.06` | `createtime` | 分表 |
| `game_results` | `createtime` | 游戏结果 |
| `user_game_results` | `createtime` | 用户游戏记录 |
| `user_rank` | 无（全量） | 用户排名 |
| `users` | 无（全量） | 用户信息 |

---

## 预处理代码

### 1. 环境准备

```python
import os
import json
import sqlite3
import pandas as pd
from pgOperation import PgOperation
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
load_dotenv()

# 初始化远程 PG 连接（只读！）
pg = PgOperation(
    ip=os.getenv('PG_HOST'),
    port=os.getenv('PG_PORT'),
    user=os.getenv('PG_USER'),
    pwd=os.getenv('PG_PASSWORD'),
    db=os.getenv('PG_DATABASE')
)

# 本地存储路径
LOCAL_DATA_DIR = './local_data'
SQLITE_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'tfm.db')

# 创建本地存储目录
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# 时间过滤条件
DATE_FILTER = '2025-01-01'
```

### 2. games 表分片获取（核心）

#### 去重策略说明

**目标**：同一个 `game_id` 只保留 `save_id` 最大的那条记录（代表游戏最终状态）。

**性能优化**：
- ❌ 不在数据库端做排序/去重（ORDER BY + 子查询会增加数据库负载）
- ✅ 先分片获取所有数据到本地，再在本地去重（减轻数据库压力）

```
远程 PostgreSQL                    本地
┌─────────────┐                 ┌─────────────────┐
│ games 表    │  分片获取       │ 内存/DataFrame  │
│ (不排序)    │ ────────────►  │                 │
│             │  chunk 1       │  合并所有 chunk │
│             │  chunk 2       │        ↓        │
│             │  chunk N       │  按 game_id     │
└─────────────┘                │  本地去重       │
                                │  (取 max save_id)│
                                └─────────────────┘
```

```python
def fetch_games_in_chunks(pg: PgOperation, table: str, chunk_size: int = 1000):
    """
    分片获取 games 表数据（不在数据库端排序，减轻数据库负载）

    策略：
    1. 分片获取所有满足时间条件的数据（不 ORDER BY，减轻 DB 负担）
    2. 本地合并后再去重（按 game_id 取 max save_id）

    Args:
        pg: PgOperation 实例
        table: 表名（如 'games', 'games_2024.12_2025.02'）
        chunk_size: 每次获取的行数

    Returns:
        DataFrame: 表的原始数据（未去重）
    """
    all_chunks = []
    offset = 0

    while True:
        # 注意：不加 ORDER BY，减轻数据库排序压力
        sql = f'''
            SELECT game_id, save_id, game, status, createtime, prop
            FROM "{table}"
            WHERE createtime >= '{DATE_FILTER}'
            LIMIT {chunk_size} OFFSET {offset}
        '''

        chunk = pg.readSql(sql)

        if chunk is None or len(chunk) == 0:
            break

        all_chunks.append(chunk)
        print(f"[{table}] Fetched {len(chunk)} rows (offset: {offset})")

        if len(chunk) < chunk_size:
            break

        offset += chunk_size

    if not all_chunks:
        return pd.DataFrame()

    result = pd.concat(all_chunks, ignore_index=True)
    print(f"[{table}] Total: {len(result)} rows (before dedup)")
    return result


def fetch_all_games_tables(pg: PgOperation):
    """
    获取所有指定的 games 分表并合并去重

    处理流程：
    1. 依次分片获取每个表的数据（不排序）
    2. 合并所有表的数据
    3. 按 game_id 分组，取 save_id 最大值（本地去重）

    只获取以下表：
    - games
    - games_2024.12_2025.02
    - games_2025.03_2025.06
    """
    GAMES_TABLES = [
        'games',
        'games_2024.12_2025.02',
        'games_2025.03_2025.06'
    ]

    all_games = []

    for table in GAMES_TABLES:
        print(f"\n=== Fetching {table} ===")
        try:
            df = fetch_games_in_chunks(pg, table)
            if len(df) > 0:
                df['source_table'] = table
                all_games.append(df)
        except Exception as e:
            print(f"Error fetching {table}: {e}")

    if not all_games:
        return pd.DataFrame()

    # Step 1: 合并所有表
    combined = pd.concat(all_games, ignore_index=True)
    print(f"\n=== Combined: {len(combined)} rows (all tables) ===")

    # Step 2: 本地去重 - 按 game_id 分组取 save_id 最大值
    # 这是在本地内存中执行，不会增加数据库负载
    deduped = combined.loc[combined.groupby('game_id')['save_id'].idxmax()]
    print(f"=== After dedup: {len(deduped)} rows (unique game_id with max save_id) ===")

    return deduped
```

### 3. 其他表获取

```python
def fetch_game_results(pg: PgOperation):
    """获取 game_results 表（2025年数据）"""
    sql = f'''
        SELECT *
        FROM game_results
        WHERE createtime >= '{DATE_FILTER}'
    '''
    df = pg.readSql(sql)
    print(f"[game_results] Fetched {len(df)} rows")
    return df


def fetch_user_game_results(pg: PgOperation):
    """获取 user_game_results 表（2025年数据）"""
    sql = f'''
        SELECT *
        FROM user_game_results
        WHERE createtime >= '{DATE_FILTER}'
    '''
    df = pg.readSql(sql)
    print(f"[user_game_results] Fetched {len(df)} rows")
    return df


def fetch_user_rank(pg: PgOperation):
    """获取 user_rank 表（全量）"""
    df = pg.readTable('user_rank')
    print(f"[user_rank] Fetched {len(df)} rows")
    return df


def fetch_users(pg: PgOperation):
    """获取 users 表（全量）"""
    df = pg.readTable('users')
    print(f"[users] Fetched {len(df)} rows")
    return df
```

### 4. 存储到本地

```python
def save_to_sqlite(df: pd.DataFrame, table_name: str, db_path: str = SQLITE_DB_PATH):
    """
    保存 DataFrame 到 SQLite

    Args:
        df: 要保存的数据
        table_name: 表名
        db_path: SQLite 数据库路径
    """
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Saved {len(df)} rows to SQLite: {table_name}")
    finally:
        conn.close()


def save_to_csv(df: pd.DataFrame, filename: str, data_dir: str = LOCAL_DATA_DIR):
    """
    保存 DataFrame 到 CSV

    Args:
        df: 要保存的数据
        filename: 文件名（不含路径）
        data_dir: 存储目录
    """
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to CSV: {filepath}")
```

### 5. 完整预处理流程

```python
def run_preprocessing():
    """
    运行完整的预处理流程

    1. 从远程 PG 获取数据（只读！）
    2. 存储到本地（SQLite + CSV）
    """
    print("=" * 50)
    print("TFM Data Preprocessing")
    print(f"Date filter: >= {DATE_FILTER}")
    print("=" * 50)

    # 1. 获取 games 表（分片 + 去重）
    print("\n[Step 1/5] Fetching games tables...")
    games_df = fetch_all_games_tables(pg)
    if len(games_df) > 0:
        save_to_sqlite(games_df, 'games')

    # 2. 获取 game_results
    print("\n[Step 2/5] Fetching game_results...")
    game_results_df = fetch_game_results(pg)
    if len(game_results_df) > 0:
        save_to_sqlite(game_results_df, 'game_results')

    # 3. 获取 user_game_results
    print("\n[Step 3/5] Fetching user_game_results...")
    user_game_results_df = fetch_user_game_results(pg)
    if len(user_game_results_df) > 0:
        save_to_sqlite(user_game_results_df, 'user_game_results')

    # 4. 获取 user_rank（全量，存 CSV）
    print("\n[Step 4/5] Fetching user_rank...")
    user_rank_df = fetch_user_rank(pg)
    if len(user_rank_df) > 0:
        save_to_csv(user_rank_df, 'user_rank.csv')

    # 5. 获取 users（全量，存 CSV）
    print("\n[Step 5/5] Fetching users...")
    users_df = fetch_users(pg)
    if len(users_df) > 0:
        save_to_csv(users_df, 'users.csv')

    print("\n" + "=" * 50)
    print("Preprocessing completed!")
    print(f"SQLite database: {SQLITE_DB_PATH}")
    print(f"CSV files: {LOCAL_DATA_DIR}/")
    print("=" * 50)


# 运行预处理
if __name__ == '__main__':
    run_preprocessing()
```

---

## 本地数据读取

### 从 SQLite 读取

```python
def read_from_sqlite(table_name: str, db_path: str = SQLITE_DB_PATH):
    """从 SQLite 读取表数据"""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        return df
    finally:
        conn.close()


# 使用示例
games_df = read_from_sqlite('games')
game_results_df = read_from_sqlite('game_results')
user_game_results_df = read_from_sqlite('user_game_results')
```

### 从 CSV 读取

```python
# 使用示例
user_rank_df = pd.read_csv(os.path.join(LOCAL_DATA_DIR, 'user_rank.csv'))
users_df = pd.read_csv(os.path.join(LOCAL_DATA_DIR, 'users.csv'))
```

### 解析 games.game JSON

```python
def parse_game_json(game_str):
    """解析 games.game 字段的 JSON"""
    if pd.isna(game_str):
        return None
    if isinstance(game_str, str):
        try:
            return json.loads(game_str)
        except json.JSONDecodeError:
            return None
    return game_str


# 使用示例
games_df = read_from_sqlite('games')
games_df['game_parsed'] = games_df['game'].apply(parse_game_json)
```

---

## 数据验证

预处理完成后，建议执行以下验证：

```python
def validate_local_data():
    """验证本地数据完整性"""
    conn = sqlite3.connect(SQLITE_DB_PATH)

    # 检查表是否存在
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )['name'].tolist()

    print("SQLite tables:", tables)

    # 检查各表行数
    for table in ['games', 'game_results', 'user_game_results']:
        if table in tables:
            count = pd.read_sql(f'SELECT COUNT(*) as cnt FROM {table}', conn)['cnt'][0]
            print(f"  {table}: {count} rows")

    conn.close()

    # 检查 CSV 文件
    csv_files = ['user_rank.csv', 'users.csv']
    for f in csv_files:
        filepath = os.path.join(LOCAL_DATA_DIR, f)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  {f}: {len(df)} rows")
        else:
            print(f"  {f}: NOT FOUND")


validate_local_data()
```

---

## 增量更新（可选）

如果需要增量更新，可以记录上次同步时间：

```python
def incremental_update(last_sync_time: str):
    """
    增量更新本地数据

    Args:
        last_sync_time: 上次同步时间，格式 'YYYY-MM-DD HH:MM:SS'
    """
    # 只获取 last_sync_time 之后的新数据
    sql = f'''
        SELECT *
        FROM user_game_results
        WHERE createtime > '{last_sync_time}'
    '''
    new_data = pg.readSql(sql)

    if len(new_data) > 0:
        # 追加到 SQLite
        conn = sqlite3.connect(SQLITE_DB_PATH)
        new_data.to_sql('user_game_results', conn, if_exists='append', index=False)
        conn.close()
        print(f"Appended {len(new_data)} new rows")
```

---

## 注意事项

1. **远程数据库只读**：所有操作都是 SELECT，禁止任何写入！
2. **分片大小**：默认 `chunk_size=1000`，可根据内存情况调整
3. **JSON 字段**：`games.game` 存储为 TEXT，查询时需要手动解析
4. **时间过滤**：所有带 `createtime` 的表都过滤 `>= 2025-01-01`
5. **去重逻辑**：`games` 表按 `game_id` 分组取 `save_id` 最大值
