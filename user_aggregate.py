#!/usr/bin/env python3
"""
TFM User Aggregation Script
Aggregates player data from existing CSV files and outputs JSON.

Input: A list of player IDs (user_id) OR usernames
Output: JSON with aggregated player statistics

Usage:
    # By user IDs
    uv run python user_aggregate.py -u "user_id_1,user_id_2,user_id_3"
    
    # By usernames (supports emoji!)
    uv run python user_aggregate.py -n "player1,🐴,小麦"
    
    # With custom player count
    uv run python user_aggregate.py -u "user_id_1" -p 2
    
    # Batch mode: process multiple groups in one call
    uv run python user_aggregate.py --batch '[["name1","name2"],["name3"]]'
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================
DISPLAY_DIR = './display'
CN_MAP_FILE = './data/cn_merged.json'
USERS_CSV_PATH = './local_data/users.csv'

# =============================================================================
# Username to User ID Mapping
# =============================================================================
def load_users_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load users.csv and create name->user_id mappings.
    
    Returns:
        Tuple: (exact_name_to_id, lowercase_name_to_id)
    """
    if not Path(USERS_CSV_PATH).exists():
        print(f"Warning: {USERS_CSV_PATH} not found")
        return {}, {}
    
    users_df = pd.read_csv(USERS_CSV_PATH)
    
    # Exact name to user_id mapping
    exact_map = {}
    # Lowercase name to user_id mapping (for case-insensitive lookup)
    lower_map = {}
    
    for _, row in users_df.iterrows():
        user_id = row['id']
        name = row['name']
        if pd.notna(name) and pd.notna(user_id):
            exact_map[name] = user_id
            lower_map[name.lower()] = user_id
    
    print(f"Loaded {len(exact_map)} user name mappings")
    return exact_map, lower_map


def usernames_to_user_ids(
    usernames: List[str],
    exact_map: Dict[str, str],
    lower_map: Dict[str, str]
) -> Tuple[List[str], List[str]]:
    """
    Convert usernames to user IDs.
    
    Args:
        usernames: List of usernames (may include emoji)
        exact_map: Exact name to user_id mapping
        lower_map: Lowercase name to user_id mapping
    
    Returns:
        Tuple: (found_user_ids, not_found_usernames)
    """
    found_ids = []
    not_found = []
    
    for name in usernames:
        if not name:
            continue
        
        # Try exact match first
        if name in exact_map:
            found_ids.append(exact_map[name])
        # Try case-insensitive match
        elif name.lower() in lower_map:
            found_ids.append(lower_map[name.lower()])
        else:
            not_found.append(name)
    
    return found_ids, not_found


# =============================================================================
# Data Loading
# =============================================================================
def load_csv_data(players_filter: int) -> Dict[str, pd.DataFrame]:
    """
    Load all relevant CSV files for the given player count.
    
    Returns dict with DataFrames for:
    - player_stats
    - records_by_generation
    - corporation_stats
    - corp_top100_players
    """
    data = {}
    
    # Player stats
    player_stats_path = f'{DISPLAY_DIR}/user_player_stats_{players_filter}p.csv'
    if Path(player_stats_path).exists():
        data['player_stats'] = pd.read_csv(player_stats_path)
        print(f"Loaded: {player_stats_path} ({len(data['player_stats'])} rows)")
    else:
        print(f"Warning: {player_stats_path} not found")
        data['player_stats'] = pd.DataFrame()
    
    # Records by generation
    records_path = f'{DISPLAY_DIR}/user_records_by_generation_{players_filter}p.csv'
    if Path(records_path).exists():
        data['records_by_generation'] = pd.read_csv(records_path)
        print(f"Loaded: {records_path} ({len(data['records_by_generation'])} rows)")
    else:
        print(f"Warning: {records_path} not found")
        data['records_by_generation'] = pd.DataFrame()
    
    # Corporation stats
    corp_stats_path = f'{DISPLAY_DIR}/user_corporation_stats_{players_filter}p.csv'
    if Path(corp_stats_path).exists():
        data['corporation_stats'] = pd.read_csv(corp_stats_path)
        print(f"Loaded: {corp_stats_path} ({len(data['corporation_stats'])} rows)")
    else:
        print(f"Warning: {corp_stats_path} not found")
        data['corporation_stats'] = pd.DataFrame()
    
    # Corporation top 100 players
    corp_top_path = f'{DISPLAY_DIR}/user_corp_top100_players_{players_filter}p.csv'
    if Path(corp_top_path).exists():
        data['corp_top100_players'] = pd.read_csv(corp_top_path)
        print(f"Loaded: {corp_top_path} ({len(data['corp_top100_players'])} rows)")
    else:
        print(f"Warning: {corp_top_path} not found")
        data['corp_top100_players'] = pd.DataFrame()
    
    # Time stats
    time_stats_path = f'{DISPLAY_DIR}/user_time_stats_{players_filter}p.csv'
    if Path(time_stats_path).exists():
        data['time_stats'] = pd.read_csv(time_stats_path)
        print(f"Loaded: {time_stats_path} ({len(data['time_stats'])} rows)")
    else:
        print(f"Warning: {time_stats_path} not found")
        data['time_stats'] = pd.DataFrame()
    
    return data

def load_cn_map() -> dict:
    """Load Chinese name mapping"""
    try:
        with open(CN_MAP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# =============================================================================
# Aggregation Functions
# =============================================================================
def aggregate_player_stats(
    player_stats_df: pd.DataFrame,
    user_ids: List[str]
) -> Dict:
    """
    Aggregate player stats for multiple user IDs.
    
    Uses weighted average based on game counts from each player.
    """
    if player_stats_df is None or len(player_stats_df) == 0:
        return {}
    
    # Filter to specified user IDs
    filtered = player_stats_df[player_stats_df['user_id'].isin(user_ids)]
    
    if len(filtered) == 0:
        print(f"Warning: No player stats found for user IDs: {user_ids}")
        return {}
    
    # Use total columns for weighted aggregation
    total_games = filtered['total_games'].sum()
    total_wins = filtered['total_wins'].sum()
    total_position_sum = filtered['total_position_sum'].sum()
    total_score_sum = filtered['total_score_sum'].sum()
    total_generations_sum = filtered['total_generations_sum'].sum()
    total_tr_sum = filtered['total_tr_sum'].sum()
    total_cards_played_sum = filtered['total_cards_played_sum'].sum()
    tr_games = filtered['tr_games'].sum()
    cards_games = filtered['cards_games'].sum()
    
    # Calculate weighted averages
    result = {
        'total_games': int(total_games),
        'total_wins': int(total_wins),
        'win_rate': round(total_wins / total_games * 100, 2) if total_games > 0 else 0,
        'avg_position': round(total_position_sum / total_games, 3) if total_games > 0 else 0,
        'avg_score': round(total_score_sum / total_games, 2) if total_games > 0 else 0,
        'avg_generations': round(total_generations_sum / total_games, 2) if total_games > 0 else 0,
        'avg_tr': round(total_tr_sum / tr_games, 2) if tr_games > 0 else 0,
        'avg_cards_played': round(total_cards_played_sum / cards_games, 2) if cards_games > 0 else 0,
        # Include totals for reference
        'total_position_sum': int(total_position_sum),
        'total_score_sum': int(total_score_sum),
        'total_generations_sum': int(total_generations_sum),
        'total_tr_sum': float(total_tr_sum),
        'total_cards_played_sum': float(total_cards_played_sum),
        'tr_games': int(tr_games),
        'cards_games': int(cards_games),
        # Players included
        'players_count': len(filtered),
        'user_ids': filtered['user_id'].tolist(),
        'user_names': filtered['user_name'].tolist(),
    }
    
    return result

def aggregate_records_by_generation(
    records_df: pd.DataFrame,
    user_ids: List[str]
) -> Dict:
    """
    Aggregate records by generation for multiple user IDs.
    
    For each generation, return the max score and max cards played
    across all specified users.
    """
    if records_df is None or len(records_df) == 0:
        return {}
    
    # Filter to specified user IDs
    filtered = records_df[records_df['user_id'].isin(user_ids)]
    
    if len(filtered) == 0:
        print(f"Warning: No records by generation found for user IDs: {user_ids}")
        return {}
    
    # Aggregate by generation (take max across all users)
    aggregated = filtered.groupby('generations').agg(
        total_game_count=('game_count', 'sum'),
        max_score=('max_score', 'max'),
        max_cards_played=('max_cards_played', 'max'),
    ).reset_index()
    
    # Convert to dict format
    result = {}
    for _, row in aggregated.iterrows():
        gen = int(row['generations'])
        result[gen] = {
            'generation': gen,
            'total_game_count': int(row['total_game_count']),
            'max_score': int(row['max_score']) if pd.notna(row['max_score']) else None,
            'max_cards_played': int(row['max_cards_played']) if pd.notna(row['max_cards_played']) else None,
        }
    
    return result

def aggregate_time_stats(
    time_stats_df: pd.DataFrame,
    user_ids: List[str]
) -> Dict:
    """
    Aggregate time statistics for multiple user IDs.
    
    Combines time stats from all users.
    """
    if time_stats_df is None or len(time_stats_df) == 0:
        return {}
    
    # Filter to specified user IDs
    filtered = time_stats_df[time_stats_df['user_id'].isin(user_ids)]
    
    if len(filtered) == 0:
        return {}
    
    # Aggregate time stats
    result = {
        'total_games': int(filtered['total_games'].sum()),
        'total_active_days': int(filtered['active_days'].sum()),
        'first_game': filtered['first_game'].min() if 'first_game' in filtered.columns else None,
        'last_game': filtered['last_game'].max() if 'last_game' in filtered.columns else None,
        # Per-user time patterns
        'user_time_patterns': []
    }
    
    for _, row in filtered.iterrows():
        result['user_time_patterns'].append({
            'user_id': row['user_id'],
            'user_name': row.get('user_name', row['user_id']),
            'most_common_hour': int(row['most_common_hour']) if pd.notna(row.get('most_common_hour')) else None,
            'most_common_weekday_name': row.get('most_common_weekday_name', ''),
            'busiest_day': row.get('busiest_day'),
            'busiest_day_count': int(row['busiest_day_count']) if pd.notna(row.get('busiest_day_count')) else 0,
            'busiest_month': row.get('busiest_month'),
            'busiest_month_count': int(row['busiest_month_count']) if pd.notna(row.get('busiest_month_count')) else 0,
        })
    
    return result


def get_top100_corporations(
    corp_top_df: pd.DataFrame,
    corp_stats_df: pd.DataFrame,
    user_ids: List[str],
    cn_map: dict
) -> List[Dict]:
    """
    Get list of corporations where any of the user IDs is in top 100.
    
    Returns list of corporations with rank, avg_score, avg_position.
    """
    if corp_top_df is None or len(corp_top_df) == 0:
        return []
    
    # Filter to specified user IDs
    filtered = corp_top_df[corp_top_df['user_id'].isin(user_ids)]
    
    if len(filtered) == 0:
        print(f"Warning: No top 100 corporation rankings found for user IDs: {user_ids}")
        return []
    
    # For each corporation, take the best rank among all specified users
    # Group by corporation and take the minimum rank (best)
    best_ranks = filtered.groupby('corporation').agg(
        best_rank=('corp_rank', 'min'),
        best_user_id=('user_id', 'first'),  # User with best rank
        best_user_name=('user_name', 'first'),
        usage_count=('usage_count', 'sum'),  # Total usage across users
        avg_position=('avg_position', 'mean'),  # Average of averages (approximation)
        bayesian_avg_position=('bayesian_avg_position', 'min'),  # Best bayesian
        avg_score=('avg_score', 'mean'),  # Average of averages
        win_rate=('win_rate', 'mean'),
    ).reset_index()
    
    # Sort by best rank
    best_ranks = best_ranks.sort_values('best_rank')
    
    # Convert to list of dicts
    result = []
    for _, row in best_ranks.iterrows():
        corp_name = row['corporation']
        cn_name = cn_map.get(corp_name, corp_name)
        
        result.append({
            'corporation': corp_name,
            'cn_name': cn_name,
            'rank': int(row['best_rank']),
            'best_user_id': row['best_user_id'],
            'best_user_name': row['best_user_name'],
            'usage_count': int(row['usage_count']),
            'avg_score': round(row['avg_score'], 2),
            'avg_position': round(row['avg_position'], 3),
            'bayesian_avg_position': round(row['bayesian_avg_position'], 4),
            'win_rate': round(row['win_rate'], 2),
        })
    
    return result

# =============================================================================
# Main Entry Point
# =============================================================================
def aggregate_user_data(
    user_ids: List[str],
    players_filter: int = 4,
    output_file: str = None
) -> Dict:
    """
    Main function to aggregate user data.
    
    Args:
        user_ids: List of user IDs to aggregate
        players_filter: Player count (default 4)
        output_file: Optional output JSON file path
    
    Returns:
        Dict with aggregated data
    """
    print(f"\n{'='*60}")
    print(f"Aggregating data for {len(user_ids)} user(s)")
    print(f"Player filter: {players_filter}P games")
    print(f"{'='*60}")
    
    # Load data
    data = load_csv_data(players_filter)
    cn_map = load_cn_map()
    
    # Check if data is available
    if len(data['player_stats']) == 0:
        print(f"Error: No player stats data available for {players_filter}P games")
        print(f"Please run: uv run python user_analysis.py -p {players_filter}")
        return {}
    
    # Aggregate
    result = {
        'metadata': {
            'user_ids': user_ids,
            'players_filter': players_filter,
            'user_count': len(user_ids),
        },
        'player_stats': aggregate_player_stats(data['player_stats'], user_ids),
        'records_by_generation': aggregate_records_by_generation(
            data['records_by_generation'], user_ids
        ),
        'top100_corporations': get_top100_corporations(
            data['corp_top100_players'],
            data['corporation_stats'],
            user_ids,
            cn_map
        ),
        'time_stats': aggregate_time_stats(data.get('time_stats', pd.DataFrame()), user_ids),
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("Aggregation Summary")
    print(f"{'='*60}")
    
    if result['player_stats']:
        ps = result['player_stats']
        print(f"\n--- 玩家综合统计 ---")
        print(f"  总对局数: {ps['total_games']}")
        print(f"  总胜利数: {ps['total_wins']}")
        print(f"  胜率: {ps['win_rate']}%")
        print(f"  平均顺位: {ps['avg_position']}")
        print(f"  平均分数: {ps['avg_score']}")
        print(f"  平均时代数: {ps['avg_generations']}")
        print(f"  平均TR: {ps['avg_tr']}")
        print(f"  平均打牌数: {ps['avg_cards_played']}")
    
    if result['records_by_generation']:
        print(f"\n--- 各时代最高纪录 ---")
        for gen, rec in sorted(result['records_by_generation'].items()):
            print(f"  时代{gen}: 最高分{rec['max_score']}, 最多打牌{rec['max_cards_played']}")
    
    if result['top100_corporations']:
        print(f"\n--- Top 100 公司排名 (共{len(result['top100_corporations'])}个) ---")
        for corp in result['top100_corporations'][:10]:  # Show top 10
            print(f"  #{corp['rank']:3d} {corp['cn_name'][:20]:20s} 平均分:{corp['avg_score']:6.1f} 平均顺位:{corp['avg_position']:.3f}")
        if len(result['top100_corporations']) > 10:
            print(f"  ... 和其他 {len(result['top100_corporations']) - 10} 个公司")
    
    # Output to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {output_file}")
    
    return result

def batch_aggregate_user_data(
    name_groups: List[List[str]],
    players_filter: int = 4,
    output_file: str = None,
    quiet: bool = False
) -> Dict:
    """
    Batch aggregate user data for multiple groups of usernames.
    
    Args:
        name_groups: List of lists, each inner list contains usernames to aggregate
        players_filter: Player count filter (default 4)
        output_file: Optional output JSON file path
        quiet: If True, suppress detailed output
    
    Returns:
        Dict with aggregated data for all groups
    """
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Batch Aggregation for {len(name_groups)} groups")
        print(f"Player filter: {players_filter}P games")
        print(f"{'='*60}")
    
    # Load mappings and data once
    exact_map, lower_map = load_users_mapping()
    data = load_csv_data(players_filter)
    cn_map = load_cn_map()
    
    if len(data['player_stats']) == 0:
        print(f"Error: No player stats data available for {players_filter}P games")
        return {}
    
    results = {}
    
    for idx, names in enumerate(name_groups):
        if not names:
            continue
        
        # Convert usernames to user IDs
        user_ids, not_found = usernames_to_user_ids(names, exact_map, lower_map)
        
        if not quiet and not_found:
            print(f"  Group {idx}: {len(not_found)} names not found: {not_found[:5]}...")
        
        if not user_ids:
            if not quiet:
                print(f"  Group {idx}: No valid user IDs found for names: {names}")
            continue
        
        # Create a unique key for this group
        group_key = ','.join(sorted(set(n.lower() for n in names)))
        
        # Aggregate for this group
        group_result = {
            'metadata': {
                'input_names': names,
                'user_ids': user_ids,
                'not_found_names': not_found,
                'players_filter': players_filter,
            },
            'player_stats': aggregate_player_stats(data['player_stats'], user_ids),
            'records_by_generation': aggregate_records_by_generation(
                data['records_by_generation'], user_ids
            ),
            'top100_corporations': get_top100_corporations(
                data['corp_top100_players'],
                data['corporation_stats'],
                user_ids,
                cn_map
            ),
            'time_stats': aggregate_time_stats(data.get('time_stats', pd.DataFrame()), user_ids),
        }
        
        results[group_key] = group_result
        
        if not quiet and group_result['player_stats']:
            ps = group_result['player_stats']
            print(f"  Group {idx}: {ps['total_games']} games, {ps['win_rate']}% WR")
    
    if not quiet:
        print(f"\nProcessed {len(results)} groups successfully")
    
    # Output to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_file}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='TFM User Aggregation Script')
    parser.add_argument('-u', '--users', type=str, default=None,
                        help='Comma-separated list of user IDs')
    parser.add_argument('-n', '--names', type=str, default=None,
                        help='Comma-separated list of usernames (supports emoji)')
    parser.add_argument('--batch', type=str, default=None,
                        help='JSON array of name groups: [["name1","name2"],["name3"]]')
    parser.add_argument('-p', '--players', type=int, default=4,
                        help='Filter by player count (default: 4)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress detailed output (for batch mode)')
    args = parser.parse_args()
    
    # Validate arguments
    if not args.users and not args.names and not args.batch:
        print("Error: Must provide either -u (user IDs), -n (names), or --batch")
        return
    
    # Batch mode
    if args.batch:
        try:
            name_groups = json.loads(args.batch)
            if not isinstance(name_groups, list):
                print("Error: --batch must be a JSON array of arrays")
                return
        except json.JSONDecodeError as e:
            print(f"Error parsing batch JSON: {e}")
            return
        
        output_file = args.output or f'{DISPLAY_DIR}/user_batch_aggregate_result.json'
        
        results = batch_aggregate_user_data(
            name_groups=name_groups,
            players_filter=args.players,
            output_file=output_file,
            quiet=args.quiet
        )
        
        if not args.quiet:
            print(f"\n{'='*60}")
            print("Batch Aggregation Complete")
            print(f"{'='*60}")
            print(f"Total groups: {len(results)}")
        
        return
    
    # Single/multi user mode
    user_ids = []
    
    if args.users:
        # Direct user IDs
        user_ids = [uid.strip() for uid in args.users.split(',') if uid.strip()]
    
    if args.names:
        # Convert names to user IDs
        exact_map, lower_map = load_users_mapping()
        names = [n.strip() for n in args.names.split(',') if n.strip()]
        found_ids, not_found = usernames_to_user_ids(names, exact_map, lower_map)
        
        if not_found:
            print(f"Warning: Names not found: {not_found}")
        
        user_ids.extend(found_ids)
    
    if not user_ids:
        print("Error: No valid user IDs found")
        return
    
    # Remove duplicates while preserving order
    seen = set()
    unique_ids = []
    for uid in user_ids:
        if uid not in seen:
            seen.add(uid)
            unique_ids.append(uid)
    user_ids = unique_ids
    
    # Default output file
    output_file = args.output
    if output_file is None:
        output_file = f'{DISPLAY_DIR}/user_aggregate_result.json'
    
    # Run aggregation
    result = aggregate_user_data(
        user_ids=user_ids,
        players_filter=args.players,
        output_file=output_file
    )
    
    # Print JSON to stdout as well
    print(f"\n{'='*60}")
    print("JSON Output")
    print(f"{'='*60}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
