#!/usr/bin/env python
# coding: utf-8
"""
只读检查：tfm_db 中 games 数据在指定日期范围内的覆盖情况。

仅使用 pgOperation 的 listTables / readSql，不对数据库做任何写操作。

用法:
    python check_games_coverage.py
    python check_games_coverage.py --start 2025-01-01 --end 2026-02-03
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# 默认检查范围：20250101 - 20260203
DEFAULT_START = '2025-01-01'
DEFAULT_END = '2026-02-03'


def main():
    parser = argparse.ArgumentParser(description='Check games coverage in tfm_db (read-only)')
    parser.add_argument('--start', default=DEFAULT_START, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=DEFAULT_END, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    from pgOperation import PgOperation

    pg = PgOperation(
        ip=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT'),
        user=os.getenv('PG_USER'),
        pwd=os.getenv('PG_PASSWORD'),
        db=os.getenv('PG_DATABASE'),
        schema=os.getenv('PG_SCHEMA', 'public')
    )

    schema = pg.schema
    start = args.start
    end = args.end

    print(f"Schema: {schema}")
    print(f"Date range: {start} ~ {end}")
    print()

    # 1. 列出所有表，筛选 games 相关
    tables = pg.listTables()
    if not tables:
        print("No tables found.")
        return

    games_tables = [t for t in tables if t.startswith('games') and not t.startswith('games_raw')]
    games_tables.sort()
    print(f"Games-related tables: {games_tables}")
    print()

    if not games_tables:
        print("No games* tables found. Coverage: NOT COVERED.")
        return

    # 2. 对每个表查询范围内行数、最小/最大 createtime
    total_in_range = 0
    all_min = None
    all_max = None

    for table in games_tables:
        # 表名在 SQL 里要双引号（可能有小数点）；readSql 仅接受 sql 字符串
        sql = f'''
            SELECT
                count(*) AS cnt,
                min(createtime) AS min_createtime,
                max(createtime) AS max_createtime
            FROM "{schema}"."{table}"
            WHERE createtime >= '{start}' AND createtime <= '{end}'
        '''
        df = pg.readSql(sql)

        if df is None or len(df) == 0:
            print(f"  {table}: (query failed or no rows)")
            continue

        row = df.iloc[0]
        cnt = int(row['cnt'])
        min_ct = row['min_createtime']
        max_ct = row['max_createtime']

        total_in_range += cnt
        if min_ct is not None and (all_min is None or min_ct < all_min):
            all_min = min_ct
        if max_ct is not None and (all_max is None or max_ct > all_max):
            all_max = max_ct

        print(f"  {table}: count={cnt}, min={min_ct}, max={max_ct}")

    print()
    print(f"Total rows in range: {total_in_range}")
    print(f"Overall min createtime in range: {all_min}")
    print(f"Overall max createtime in range: {all_max}")

    # 3. 简单结论：是否有数据且是否覆盖到 end 日
    if total_in_range == 0:
        print("\n结论: 该日期范围内无 games 数据，不完整。")
    else:
        end_ok = all_max is not None and str(all_max)[:10] >= end
        if end_ok:
            print(f"\n结论: 范围内共 {total_in_range} 条；已覆盖到 {end}，数据完整。")
        else:
            print(f"\n结论: 范围内共 {total_in_range} 条；最大日期为 {all_max}，未覆盖到 {end}（差约 1 天或更久）。")


if __name__ == '__main__':
    main()
