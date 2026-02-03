#!/usr/bin/env python
# coding: utf-8
"""
TFM Data Preprocessing Script

从远程 PostgreSQL 获取 2025 年数据并存储到本地。

使用方法:
    python preprocess.py                    # 运行所有步骤
    python preprocess.py --step 1           # 只运行第 1 步（获取 games）
    python preprocess.py --step 2           # 只运行第 2 步（获取 game_results）
    python preprocess.py --step 1,2,3       # 运行指定步骤
    python preprocess.py --force            # 强制重新获取（忽略已有数据）

注意:
    - 远程数据库只读，禁止任何写操作！
    - 需要先配置 .env 文件中的数据库连接信息
"""

import os
import sys
import json
import sqlite3
import argparse
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from pgOperation import PgOperation


# ============================================================
# 配置
# ============================================================

# 加载环境变量
load_dotenv()

# 本地存储路径
LOCAL_DATA_DIR = './local_data'
SQLITE_DB_PATH = os.path.join(LOCAL_DATA_DIR, 'tfm.db')

# 时间过滤条件：只获取 2025 年及以后的数据
DATE_FILTER = '2025-01-01'

# games 分表列表（只获取这些表）
GAMES_TABLES = [
    'games',
    'games_2024.12_2025.02',
    'games_2025.03_2025.06'
]

# 分片大小
CHUNK_SIZE = 1000


# ============================================================
# 数据库连接
# ============================================================

def get_pg_connection():
    """获取远程 PostgreSQL 连接（只读！）"""
    return PgOperation(
        ip=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT'),
        user=os.getenv('PG_USER'),
        pwd=os.getenv('PG_PASSWORD'),
        db=os.getenv('PG_DATABASE'),
        schema=os.getenv('PG_SCHEMA', 'public')
    )


# ============================================================
# games 表分片获取
# ============================================================

def fetch_games_in_chunks_with_save(pg: PgOperation, table: str, chunk_size: int = CHUNK_SIZE):
    """
    分片获取 games 表数据，每个分片完成后保存到临时表

    策略：
    1. 分片获取所有满足时间条件的数据（不 ORDER BY，减轻 DB 负担）
    2. 每获取一个分片，立即保存到 SQLite 临时表（避免中断丢失）
    3. 最后本地去重（按 game_id 取 max save_id）

    Args:
        pg: PgOperation 实例
        table: 表名（如 'games', 'games_2024.12_2025.02'）
        chunk_size: 每次获取的行数

    Returns:
        int: 获取的总行数
    """
    # 临时表名（用于分片保存）
    temp_table = f'games_raw_{table.replace(".", "_").replace("-", "_")}'

    offset = 0
    total_rows = 0
    is_first_chunk = True

    while True:
        # 注意：不加 ORDER BY，减轻数据库排序压力
        sql = f'''
            SELECT game_id, save_id, game, status, createtime, prop
            FROM "{table}"
            WHERE createtime >= '{DATE_FILTER}'
            LIMIT {chunk_size} OFFSET {offset}
        '''

        try:
            chunk = pg.readSql(sql)
        except Exception as e:
            print(f"  Error at offset {offset}: {e}")
            break

        if chunk is None or len(chunk) == 0:
            break

        # 添加来源表标识
        chunk['source_table'] = table

        # 保存到临时表（第一个 chunk 用 replace，后续用 append）
        save_to_sqlite(
            chunk,
            temp_table,
            if_exists='replace' if is_first_chunk else 'append'
        )
        is_first_chunk = False

        total_rows += len(chunk)
        print(f"  [{table}] Fetched and saved {len(chunk)} rows (offset: {offset}, total: {total_rows})")

        if len(chunk) < chunk_size:
            break

        offset += chunk_size

    print(f"  [{table}] Completed: {total_rows} rows saved to temp table '{temp_table}'")
    return total_rows


def merge_and_dedup_games():
    """
    合并所有 games 临时表并去重

    从 SQLite 中读取所有临时表，合并后按 game_id 取 max save_id
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)

    try:
        # 获取所有 games_raw_* 临时表
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'games_raw_%'",
            conn
        )['name'].tolist()

        if not tables:
            print("No games temp tables found!")
            return pd.DataFrame()

        print(f"\n=== Merging {len(tables)} temp tables: {tables} ===")

        # 合并所有临时表
        all_games = []
        for table in tables:
            df = pd.read_sql(f'SELECT * FROM "{table}"', conn)
            all_games.append(df)
            print(f"  Read {len(df)} rows from {table}")

        combined = pd.concat(all_games, ignore_index=True)
        print(f"=== Combined: {len(combined)} rows (all tables) ===")

        # 按 game_id 去重，保留最大 save_id
        deduped = combined.loc[combined.groupby('game_id')['save_id'].idxmax()]
        print(f"=== After dedup: {len(deduped)} rows (unique game_id with max save_id) ===")

        # 保存最终结果
        # 转换 dict 列为 JSON 字符串
        deduped = convert_dict_columns_to_json(deduped)
        deduped.to_sql('games', conn, if_exists='replace', index=False)
        print(f"=== Saved {len(deduped)} rows to final 'games' table ===")

        # 清理临时表
        for table in tables:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        conn.commit()
        print(f"=== Cleaned up {len(tables)} temp tables ===")

        return deduped

    finally:
        conn.close()


def get_existing_temp_tables():
    """获取已存在的 games 临时表"""
    if not os.path.exists(SQLITE_DB_PATH):
        return []

    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'games_raw_%'",
            conn
        )['name'].tolist()
        return tables
    finally:
        conn.close()


def fetch_all_games_tables(pg: PgOperation, force_refetch: bool = False):
    """
    获取所有指定的 games 分表并合并去重

    处理流程：
    1. 依次分片获取每个表的数据（每个分片保存到临时表）
    2. 合并所有临时表
    3. 按 game_id 分组，取 save_id 最大值（本地去重）
    4. 清理临时表

    Args:
        force_refetch: 是否强制重新获取（忽略已存在的临时表）
    """
    # 检查已存在的临时表（断点续传）
    existing_temps = get_existing_temp_tables()
    if existing_temps and not force_refetch:
        print(f"\n=== Found existing temp tables: {existing_temps} ===")
        print("  Skipping fetch, using existing data. Use force_refetch=True to refetch.")
    else:
        for table in GAMES_TABLES:
            temp_table = f'games_raw_{table.replace(".", "_").replace("-", "_")}'

            # 检查是否已有该表的临时数据
            if temp_table in existing_temps and not force_refetch:
                print(f"\n=== Skipping {table} (temp table exists) ===")
                continue

            print(f"\n=== Fetching {table} ===")
            try:
                fetch_games_in_chunks_with_save(pg, table)
            except Exception as e:
                print(f"  Error fetching {table}: {e}")

    # 合并并去重
    return merge_and_dedup_games()


# ============================================================
# 其他表获取
# ============================================================

def fetch_game_results(pg: PgOperation):
    """获取 game_results 表（2025年数据）"""
    sql = f'''
        SELECT *
        FROM game_results
        WHERE createtime >= '{DATE_FILTER}'
    '''
    df = pg.readSql(sql)
    print(f"[game_results] Fetched {len(df) if df is not None else 0} rows")
    return df if df is not None else pd.DataFrame()


def fetch_user_game_results(pg: PgOperation):
    """获取 user_game_results 表（2025年数据）"""
    sql = f'''
        SELECT *
        FROM user_game_results
        WHERE createtime >= '{DATE_FILTER}'
    '''
    df = pg.readSql(sql)
    print(f"[user_game_results] Fetched {len(df) if df is not None else 0} rows")
    return df if df is not None else pd.DataFrame()


def fetch_user_rank(pg: PgOperation):
    """获取 user_rank 表（全量）"""
    df = pg.readTable('user_rank')
    print(f"[user_rank] Fetched {len(df) if df is not None else 0} rows")
    return df if df is not None else pd.DataFrame()


def fetch_users(pg: PgOperation):
    """获取 users 表（全量）"""
    df = pg.readTable('users')
    print(f"[users] Fetched {len(df) if df is not None else 0} rows")
    return df if df is not None else pd.DataFrame()


# ============================================================
# 存储功能
# ============================================================

def convert_dict_columns_to_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中的 dict 类型列转换为 JSON 字符串

    SQLite 不支持直接存储 dict 类型，需要转换为 JSON 字符串
    """
    df = df.copy()
    for col in df.columns:
        # 检查是否有 dict 类型的值
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
            print(f"    Converted column '{col}' from dict to JSON string")
    return df


def save_to_sqlite(df: pd.DataFrame, table_name: str, db_path: str = SQLITE_DB_PATH, if_exists: str = 'replace'):
    """
    保存 DataFrame 到 SQLite

    Args:
        df: 要保存的数据
        table_name: 表名
        db_path: SQLite 数据库路径
        if_exists: 'replace' 或 'append'
    """
    if df is None or len(df) == 0:
        print(f"  Skipped {table_name} (no data)")
        return

    # 转换 dict 列为 JSON 字符串
    df = convert_dict_columns_to_json(df)

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        print(f"  Saved {len(df)} rows to SQLite: {table_name} (mode: {if_exists})")
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
    if df is None or len(df) == 0:
        print(f"  Skipped {filename} (no data)")
        return

    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df)} rows to CSV: {filepath}")


# ============================================================
# 数据验证
# ============================================================

def validate_local_data():
    """验证本地数据完整性"""
    print("\n" + "=" * 50)
    print("Data Validation")
    print("=" * 50)

    # 检查 SQLite 表
    if os.path.exists(SQLITE_DB_PATH):
        conn = sqlite3.connect(SQLITE_DB_PATH)
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )['name'].tolist()

        print(f"\nSQLite tables in {SQLITE_DB_PATH}:")
        for table in ['games', 'game_results', 'user_game_results']:
            if table in tables:
                count = pd.read_sql(f'SELECT COUNT(*) as cnt FROM {table}', conn)['cnt'][0]
                print(f"  {table}: {count} rows")
            else:
                print(f"  {table}: NOT FOUND")
        conn.close()
    else:
        print(f"\nSQLite database not found: {SQLITE_DB_PATH}")

    # 检查 CSV 文件
    print(f"\nCSV files in {LOCAL_DATA_DIR}:")
    csv_files = ['user_rank.csv', 'users.csv']
    for f in csv_files:
        filepath = os.path.join(LOCAL_DATA_DIR, f)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  {f}: {len(df)} rows")
        else:
            print(f"  {f}: NOT FOUND")


# ============================================================
# 主流程
# ============================================================

def run_preprocessing(steps: list = None, force_refetch: bool = False):
    """
    运行预处理流程

    Args:
        steps: 要执行的步骤列表，如 [1, 2, 3]。None 表示执行所有步骤
        force_refetch: 是否强制重新获取（忽略已有数据）

    步骤说明:
        1. 获取 games 表（分片保存 + 合并去重）
        2. 获取 game_results
        3. 获取 user_game_results
        4. 获取 user_rank
        5. 获取 users
    """
    # 默认执行所有步骤
    if steps is None:
        steps = [1, 2, 3, 4, 5]

    start_time = datetime.now()

    print("=" * 60)
    print("TFM Data Preprocessing")
    print("=" * 60)
    print(f"Start time: {start_time}")
    print(f"Date filter: >= {DATE_FILTER}")
    print(f"Steps to run: {steps}")
    print(f"Force refetch: {force_refetch}")
    print(f"Local data dir: {LOCAL_DATA_DIR}")
    print(f"SQLite database: {SQLITE_DB_PATH}")
    print("=" * 60)

    # 创建本地存储目录
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    # 获取数据库连接
    pg = get_pg_connection()
    print(f"\nConnected to: {os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}")

    # 1. 获取 games 表（分片保存 + 合并去重）
    if 1 in steps:
        print("\n" + "-" * 50)
        print("[Step 1/5] Fetching games tables (with incremental save)...")
        print("-" * 50)
        games_df = fetch_all_games_tables(pg, force_refetch=force_refetch)
        # games_df 已在 merge_and_dedup_games() 中保存到 SQLite

    # 2. 获取 game_results
    if 2 in steps:
        print("\n" + "-" * 50)
        print("[Step 2/5] Fetching game_results...")
        print("-" * 50)
        game_results_df = fetch_game_results(pg)
        if len(game_results_df) > 0:
            save_to_sqlite(game_results_df, 'game_results')

    # 3. 获取 user_game_results
    if 3 in steps:
        print("\n" + "-" * 50)
        print("[Step 3/5] Fetching user_game_results...")
        print("-" * 50)
        user_game_results_df = fetch_user_game_results(pg)
        if len(user_game_results_df) > 0:
            save_to_sqlite(user_game_results_df, 'user_game_results')

    # 4. 获取 user_rank（全量，存 CSV）
    if 4 in steps:
        print("\n" + "-" * 50)
        print("[Step 4/5] Fetching user_rank...")
        print("-" * 50)
        user_rank_df = fetch_user_rank(pg)
        if len(user_rank_df) > 0:
            save_to_csv(user_rank_df, 'user_rank.csv')

    # 5. 获取 users（全量，存 CSV）
    if 5 in steps:
        print("\n" + "-" * 50)
        print("[Step 5/5] Fetching users...")
        print("-" * 50)
        users_df = fetch_users(pg)
        if len(users_df) > 0:
            save_to_csv(users_df, 'users.csv')

    # 验证数据
    validate_local_data()

    # 完成
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)
    print(f"End time: {end_time}")
    print(f"Duration: {duration}")
    print(f"SQLite database: {SQLITE_DB_PATH}")
    print(f"CSV files: {LOCAL_DATA_DIR}/")
    print("=" * 60)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='TFM Data Preprocessing Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
步骤说明:
  1 - 获取 games 表（分片保存 + 合并去重）
  2 - 获取 game_results
  3 - 获取 user_game_results
  4 - 获取 user_rank
  5 - 获取 users

示例:
  python preprocess.py                # 运行所有步骤
  python preprocess.py --step 1       # 只运行第 1 步
  python preprocess.py --step 1,2,3   # 运行步骤 1, 2, 3
  python preprocess.py --force        # 强制重新获取
        '''
    )
    parser.add_argument(
        '--step', '-s',
        type=str,
        default=None,
        help='要执行的步骤，用逗号分隔。如: 1 或 1,2,3'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='强制重新获取（忽略已有的临时数据）'
    )
    return parser.parse_args()


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    args = parse_args()

    # 解析步骤参数
    steps = None
    if args.step:
        steps = [int(s.strip()) for s in args.step.split(',')]

    run_preprocessing(steps=steps, force_refetch=args.force)
