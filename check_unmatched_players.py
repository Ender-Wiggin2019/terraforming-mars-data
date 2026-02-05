#!/usr/bin/env python3
"""
Check Unmatched Players Script

从 game_results 表提取所有玩家名字，检查是否能与 users.csv 匹配。
输出无法匹配且出现次数 >= 2 的名字列表，按出现频率排序。

支持:
- 大小写不敏感匹配
- 基于 matched_players.csv 的别名映射
- 生成 final_matched.csv 和 final_unmatched.csv

使用方法:
    uv run python check_unmatched_players.py
    uv run python check_unmatched_players.py -o ./display/unmatched_players.csv
"""

import pandas as pd
import sqlite3
import json
import argparse
import re
from collections import Counter
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
LOCAL_DB_PATH = './local_data/tfm.db'
USERS_CSV_PATH = './local_data/users.csv'
MATCHED_PLAYERS_CSV = './display/matched_players.csv'
DISPLAY_DIR = './display'

MIN_OCCURRENCES = 2  # 最小出现次数


def load_users() -> tuple[set, dict]:
    """
    Load user names from users.csv
    
    Returns:
        tuple: (原始用户名集合, 小写->原始名字的映射字典)
    """
    users_df = pd.read_csv(USERS_CSV_PATH)
    # 原始用户名集合
    user_names = set(users_df['name'].dropna().unique())
    # 创建小写到原始名字的映射（用于大小写不敏感匹配）
    user_names_lower_map = {name.lower(): name for name in user_names}
    print(f"Loaded {len(user_names)} unique user names from users.csv")
    return user_names, user_names_lower_map


def load_matched_players_mapping() -> dict:
    """
    Load alias->real_user mapping from matched_players.csv
    
    CSV format:
    - Top section: alias1，alias2，...，real_user (Chinese comma separated)
    - Bottom section (after "## 未匹配"): occurrences,alias,,real_user
    
    Returns:
        dict: {alias_lower: real_user} mapping
    """
    alias_map = {}
    
    if not Path(MATCHED_PLAYERS_CSV).exists():
        print(f"Warning: {MATCHED_PLAYERS_CSV} not found, skipping alias mapping")
        return alias_map
    
    with open(MATCHED_PLAYERS_CSV, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    in_unmatched_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检测 "## 未匹配" 分隔符
        if '未匹配' in line:
            in_unmatched_section = True
            continue
        
        if not in_unmatched_section:
            # 顶部区域：alias1，alias2，...，real_user (使用中文逗号)
            # 也可能使用英文逗号
            parts = re.split(r'[，,]', line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 2:
                real_user = parts[-1]  # 最后一个是真实用户名
                aliases = parts[:-1]   # 其他都是别名
                
                for alias in aliases:
                    if alias:
                        alias_map[alias.lower()] = real_user
        else:
            # 底部区域：occurrences,alias,,real_user
            parts = line.split(',')
            if len(parts) >= 4:
                # parts[0] = occurrences, parts[1] = alias, parts[2] = empty, parts[3] = real_user
                alias = parts[1].strip()
                real_user = parts[3].strip() if len(parts) > 3 else ''
                
                # 处理可能带有 tab 的情况
                real_user = real_user.replace('\t', '').strip()
                
                if alias and real_user:
                    alias_map[alias.lower()] = real_user
    
    print(f"Loaded {len(alias_map)} alias mappings from matched_players.csv")
    return alias_map


def extract_player_names_from_scores(scores_json: str) -> list:
    """Extract player names from scores JSON string"""
    if pd.isna(scores_json) or not scores_json:
        return []
    
    try:
        scores = json.loads(scores_json)
        if isinstance(scores, list):
            return [s.get('player', '') for s in scores if s.get('player')]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def load_game_results_players() -> Counter:
    """Load all player names from game_results and count occurrences"""
    conn = sqlite3.connect(LOCAL_DB_PATH)
    
    # 只需要 scores 列
    df = pd.read_sql('SELECT scores FROM game_results', conn)
    conn.close()
    
    print(f"Loaded {len(df)} game_results records")
    
    # 提取所有玩家名字
    player_counter = Counter()
    
    for scores_json in df['scores']:
        names = extract_player_names_from_scores(scores_json)
        for name in names:
            if name:  # 跳过空名字
                player_counter[name] += 1
    
    print(f"Extracted {len(player_counter)} unique player names")
    print(f"Total player occurrences: {sum(player_counter.values())}")
    
    return player_counter


def find_matched_and_unmatched_players(
    player_counter: Counter, 
    user_names: set, 
    user_names_lower_map: dict,
    alias_map: dict,
    min_occurrences: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find matched and unmatched players using:
    1. Exact match with users.csv
    2. Case-insensitive match with users.csv
    3. Alias mapping from matched_players.csv
    
    Returns:
        tuple: (matched_df, unmatched_df)
    """
    matched = []
    unmatched = []
    
    for name, count in player_counter.items():
        matched_user = None
        match_type = None
        
        # 1. 精确匹配
        if name in user_names:
            matched_user = name
            match_type = 'exact'
        # 2. 大小写不敏感匹配
        elif name.lower() in user_names_lower_map:
            matched_user = user_names_lower_map[name.lower()]
            match_type = 'case_insensitive'
        # 3. 别名映射匹配
        elif name.lower() in alias_map:
            matched_user = alias_map[name.lower()]
            match_type = 'alias'
        
        if matched_user:
            matched.append({
                'player_name': name,
                'matched_user': matched_user,
                'match_type': match_type,
                'occurrences': count
            })
        else:
            if count >= min_occurrences:
                unmatched.append({
                    'player_name': name,
                    'occurrences': count
                })
    
    # 统计信息
    exact_count = sum(1 for m in matched if m['match_type'] == 'exact')
    case_insensitive_count = sum(1 for m in matched if m['match_type'] == 'case_insensitive')
    alias_count = sum(1 for m in matched if m['match_type'] == 'alias')
    total_unmatched = len(player_counter) - len(matched)
    
    print(f"\nMatching results:")
    print(f"  - Total unique players: {len(player_counter)}")
    print(f"  - Matched (exact): {exact_count}")
    print(f"  - Matched (case-insensitive): {case_insensitive_count}")
    print(f"  - Matched (alias): {alias_count}")
    print(f"  - Total matched: {len(matched)}")
    print(f"  - Unmatched (total): {total_unmatched}")
    print(f"  - Unmatched with >= {min_occurrences} occurrences: {len(unmatched)}")
    
    # 创建 matched DataFrame
    matched_df = pd.DataFrame(matched)
    if len(matched_df) > 0:
        matched_df = matched_df.sort_values('occurrences', ascending=False)
        matched_df = matched_df.reset_index(drop=True)
        matched_df.index = matched_df.index + 1
        matched_df.index.name = 'rank'
    
    # 创建 unmatched DataFrame
    unmatched_df = pd.DataFrame(unmatched)
    if len(unmatched_df) > 0:
        unmatched_df = unmatched_df.sort_values('occurrences', ascending=False)
        unmatched_df = unmatched_df.reset_index(drop=True)
        unmatched_df.index = unmatched_df.index + 1
        unmatched_df.index.name = 'rank'
    
    return matched_df, unmatched_df


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Check unmatched players')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory path (default: ./display)')
    parser.add_argument('-m', '--min-occurrences', type=int, default=MIN_OCCURRENCES,
                        help=f'Minimum occurrences to include (default: {MIN_OCCURRENCES})')
    args = parser.parse_args()
    
    min_occ = args.min_occurrences
    output_dir = args.output or DISPLAY_DIR
    
    print("=" * 60)
    print("Player Matching Analysis")
    print("=" * 60)
    
    # 加载数据
    user_names, user_names_lower_map = load_users()
    alias_map = load_matched_players_mapping()
    player_counter = load_game_results_players()
    
    # 查找匹配和未匹配的玩家
    matched_df, unmatched_df = find_matched_and_unmatched_players(
        player_counter, user_names, user_names_lower_map, alias_map, min_occ
    )
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 输出 final_matched.csv
    # =========================================================================
    print("\n" + "=" * 60)
    print("Matched Players Summary")
    print("=" * 60)
    
    if len(matched_df) > 0:
        # 按匹配类型统计
        match_type_stats = matched_df.groupby('match_type').agg({
            'player_name': 'count',
            'occurrences': 'sum'
        }).rename(columns={'player_name': 'count'})
        print("\nMatch type statistics:")
        print(match_type_stats.to_string())
        
        # 显示前20个别名匹配
        alias_matched = matched_df[matched_df['match_type'] == 'alias']
        if len(alias_matched) > 0:
            print(f"\nTop 20 alias matches:")
            print(alias_matched.head(20)[['player_name', 'matched_user', 'occurrences']].to_string())
        
        # 保存 final_matched.csv
        matched_path = f'{output_dir}/final_matched.csv'
        matched_df.to_csv(matched_path)
        print(f"\nMatched players saved to: {matched_path}")
        print(f"  - Total matched players: {len(matched_df)}")
        print(f"  - Total matched occurrences: {matched_df['occurrences'].sum()}")
    else:
        print("\nNo matched players found.")
    
    # =========================================================================
    # 输出 final_unmatched.csv
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Unmatched Players (>= {min_occ} occurrences)")
    print("=" * 60)
    
    if len(unmatched_df) > 0:
        # 显示前20个
        print("\nTop 20 unmatched players:")
        print(unmatched_df.head(20).to_string())
        
        # 统计信息
        print(f"\n--- Unmatched Statistics ---")
        print(f"Total unmatched players: {len(unmatched_df)}")
        print(f"Total occurrences of unmatched: {unmatched_df['occurrences'].sum()}")
        print(f"Max occurrences: {unmatched_df['occurrences'].max()}")
        print(f"Min occurrences: {unmatched_df['occurrences'].min()}")
        print(f"Mean occurrences: {unmatched_df['occurrences'].mean():.2f}")
        
        # 保存 final_unmatched.csv
        unmatched_path = f'{output_dir}/final_unmatched.csv'
        unmatched_df.to_csv(unmatched_path)
        print(f"\nUnmatched players saved to: {unmatched_path}")
    else:
        print("\nNo unmatched players found with the specified criteria.")
    
    # =========================================================================
    # 总结
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_players = len(player_counter)
    total_matched = len(matched_df) if len(matched_df) > 0 else 0
    total_unmatched = len(unmatched_df) if len(unmatched_df) > 0 else 0
    match_rate = total_matched / total_players * 100 if total_players > 0 else 0
    
    print(f"Total unique players in games: {total_players}")
    print(f"Total matched: {total_matched} ({match_rate:.1f}%)")
    print(f"Total unmatched (>= {min_occ} occurrences): {total_unmatched}")


if __name__ == '__main__':
    main()
