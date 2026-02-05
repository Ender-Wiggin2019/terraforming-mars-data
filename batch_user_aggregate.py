#!/usr/bin/env python3
"""
Batch User Aggregate Script

根据 final_matched.csv 中的 matched_user (user_id) 维度，
将所有对应的 player_name 作为一组进行聚合。

流程：
1. 读取 final_matched.csv，按 matched_user 分组
2. 过滤 matched_user 总对局数 >= MIN_GAMES
3. 将每个 matched_user 对应的所有 player_name 作为一组
4. 调用 user_aggregate 的批量聚合功能
5. 输出完整 JSON

使用方法:
    uv run python batch_user_aggregate.py
    uv run python batch_user_aggregate.py -p 4 -m 3
    uv run python batch_user_aggregate.py -o ./display/all_users_aggregated.json
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Import from user_aggregate
from user_aggregate import (
    load_csv_data,
    load_cn_map,
    load_users_mapping,
    aggregate_player_stats,
    aggregate_records_by_generation,
    aggregate_time_stats,
    get_top100_corporations,
    usernames_to_user_ids,
)

# =============================================================================
# Configuration
# =============================================================================
DISPLAY_DIR = './display'
FINAL_MATCHED_CSV = './display/final_matched.csv'
MIN_GAMES_FOR_AGGREGATE = 3  # 最小对局数

# =============================================================================
# Main Functions
# =============================================================================

def load_matched_players() -> pd.DataFrame:
    """
    Load final_matched.csv and return DataFrame.
    
    Columns: rank, player_name, matched_user, match_type, occurrences
    """
    if not Path(FINAL_MATCHED_CSV).exists():
        print(f"Error: {FINAL_MATCHED_CSV} not found")
        print("Please run: uv run python check_unmatched_players.py")
        return pd.DataFrame()
    
    df = pd.read_csv(FINAL_MATCHED_CSV)
    print(f"Loaded {len(df)} matched player records from {FINAL_MATCHED_CSV}")
    return df


def group_players_by_user(
    matched_df: pd.DataFrame,
    min_games: int = MIN_GAMES_FOR_AGGREGATE
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Group player names by matched_user (case-insensitive).
    
    Args:
        matched_df: DataFrame from final_matched.csv
        min_games: Minimum total occurrences to include
    
    Returns:
        Tuple: (user_to_names_map, user_to_total_games_map)
    """
    # Group by matched_user (lowercase for consistency)
    user_groups = {}
    user_totals = {}
    
    for _, row in matched_df.iterrows():
        matched_user = row['matched_user']
        player_name = row['player_name']
        occurrences = row['occurrences']
        
        # Use lowercase matched_user as key
        user_key = matched_user.lower() if pd.notna(matched_user) else None
        if not user_key:
            continue
        
        if user_key not in user_groups:
            user_groups[user_key] = {
                'original_user': matched_user,
                'names': set(),
                'total_games': 0
            }
        
        user_groups[user_key]['names'].add(player_name)
        user_groups[user_key]['total_games'] += occurrences
    
    # Filter by minimum games and convert to final format
    filtered_user_to_names = {}
    filtered_user_to_games = {}
    
    for user_key, data in user_groups.items():
        if data['total_games'] >= min_games:
            filtered_user_to_names[user_key] = list(data['names'])
            filtered_user_to_games[user_key] = data['total_games']
    
    print(f"\nGrouped players by matched_user:")
    print(f"  - Total unique matched_users: {len(user_groups)}")
    print(f"  - Filtered (>= {min_games} games): {len(filtered_user_to_names)}")
    
    return filtered_user_to_names, filtered_user_to_games


def batch_aggregate_all_users(
    user_to_names: Dict[str, List[str]],
    user_to_games: Dict[str, int],
    players_filter: int = 4,
    quiet: bool = True
) -> Dict:
    """
    Batch aggregate all users.
    
    Args:
        user_to_names: Dict mapping user_key to list of player names
        user_to_games: Dict mapping user_key to total game count
        players_filter: Player count filter
        quiet: Suppress detailed output
    
    Returns:
        Dict with aggregated data for all users
    """
    print(f"\n{'='*60}")
    print(f"Batch Aggregation for {len(user_to_names)} users")
    print(f"Player filter: {players_filter}P games")
    print(f"{'='*60}")
    
    # Load data once
    exact_map, lower_map = load_users_mapping()
    data = load_csv_data(players_filter)
    cn_map = load_cn_map()
    
    if len(data['player_stats']) == 0:
        print(f"Error: No player stats data available for {players_filter}P games")
        return {}
    
    results = {}
    processed = 0
    skipped = 0
    
    # Sort by total games descending
    sorted_users = sorted(
        user_to_names.keys(),
        key=lambda x: user_to_games.get(x, 0),
        reverse=True
    )
    
    for user_key in sorted_users:
        names = user_to_names[user_key]
        total_games_raw = user_to_games[user_key]
        
        # Convert usernames to user IDs
        user_ids, not_found = usernames_to_user_ids(names, exact_map, lower_map)
        
        if not user_ids:
            skipped += 1
            continue
        
        # Aggregate for this user
        player_stats = aggregate_player_stats(data['player_stats'], user_ids)
        
        if not player_stats or player_stats.get('total_games', 0) == 0:
            skipped += 1
            continue
        
        group_result = {
            'metadata': {
                'user_key': user_key,
                'input_names': names,
                'user_ids': user_ids,
                'not_found_names': not_found,
                'players_filter': players_filter,
                'raw_game_count': total_games_raw,
            },
            'player_stats': player_stats,
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
        
        results[user_key] = group_result
        processed += 1
        
        if not quiet and processed <= 20:
            ps = group_result['player_stats']
            print(f"  {user_key}: {ps['total_games']} games, {ps['win_rate']}% WR, names={len(names)}")
    
    print(f"\nProcessed: {processed} users")
    print(f"Skipped (no data): {skipped} users")
    
    return results


# =============================================================================
# Global Rankings
# =============================================================================
TOP_RANKING_SIZE = 100

def calculate_global_rankings(results: Dict) -> Dict:
    """
    Calculate global rankings based on aggregated data.
    
    Rankings:
    1. total_games - 游玩次数 Top 100 (higher is better)
    2. win_rate - 胜率 Top 100 (higher is better)
    3. avg_position - 平均顺位 Top 100 (lower is better)
    4. total_cards_played - 总打牌数 Top 100 (higher is better)
    5. avg_generations_short - 最短平均时代数 Top 100 (lower is better)
    6. avg_generations_long - 最长平均时代数 Top 100 (higher is better)
    7. max_score - 最高分 Top 100 (higher is better)
    
    Returns:
        Dict with rankings for each category
    """
    print(f"\n{'='*60}")
    print("Calculating Global Rankings")
    print(f"{'='*60}")
    
    # Prepare data for ranking
    ranking_data = []
    
    for user_key, data in results.items():
        ps = data.get('player_stats', {})
        records = data.get('records_by_generation', {})
        
        if not ps or ps.get('total_games', 0) == 0:
            continue
        
        # Calculate max score from records_by_generation
        max_score = 0
        if records:
            for gen, rec in records.items():
                if rec.get('max_score') and rec['max_score'] > max_score:
                    max_score = rec['max_score']
        
        # Calculate total cards played (sum across all games)
        total_cards_played = ps.get('total_cards_played_sum', 0) or 0
        
        ranking_data.append({
            'user_key': user_key,
            'total_games': ps.get('total_games', 0),
            'win_rate': ps.get('win_rate', 0),
            'avg_position': ps.get('avg_position', 999),
            'total_cards_played': total_cards_played,
            'avg_generations': ps.get('avg_generations', 0),
            'max_score': max_score,
            'total_wins': ps.get('total_wins', 0),
            'avg_score': ps.get('avg_score', 0),
        })
    
    if not ranking_data:
        return {}
    
    df = pd.DataFrame(ranking_data)
    
    # Calculate rankings
    rankings = {}
    
    # 1. Total games (higher is better)
    df_sorted = df.sort_values(['total_games', 'win_rate'], ascending=[False, False])
    rankings['total_games_top100'] = df_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'total_games', 'win_rate', 'avg_position'
    ]].to_dict('records')
    
    # Add rank to each record
    for i, rec in enumerate(rankings['total_games_top100']):
        rec['rank'] = i + 1
    
    # 2. Win rate (higher is better) - require minimum games
    min_games_for_winrate = 10
    df_wr = df[df['total_games'] >= min_games_for_winrate].copy()
    df_wr_sorted = df_wr.sort_values(['win_rate', 'total_games'], ascending=[False, False])
    rankings['win_rate_top100'] = df_wr_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'win_rate', 'total_games', 'total_wins'
    ]].to_dict('records')
    for i, rec in enumerate(rankings['win_rate_top100']):
        rec['rank'] = i + 1
    
    # 3. Average position (lower is better) - require minimum games
    df_pos = df[df['total_games'] >= min_games_for_winrate].copy()
    df_pos_sorted = df_pos.sort_values(['avg_position', 'total_games'], ascending=[True, False])
    rankings['avg_position_top100'] = df_pos_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'avg_position', 'total_games', 'win_rate'
    ]].to_dict('records')
    for i, rec in enumerate(rankings['avg_position_top100']):
        rec['rank'] = i + 1
    
    # 4. Total cards played (higher is better)
    df_sorted = df.sort_values(['total_cards_played', 'total_games'], ascending=[False, False])
    rankings['total_cards_top100'] = df_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'total_cards_played', 'total_games', 'avg_score'
    ]].to_dict('records')
    for i, rec in enumerate(rankings['total_cards_top100']):
        rec['rank'] = i + 1
    
    # 5. Shortest average generations (lower is better) - require minimum games
    df_gen = df[(df['total_games'] >= min_games_for_winrate) & (df['avg_generations'] > 0)].copy()
    df_gen_sorted = df_gen.sort_values(['avg_generations', 'avg_position'], ascending=[True, True])
    rankings['shortest_generations_top100'] = df_gen_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'avg_generations', 'total_games', 'avg_score'
    ]].to_dict('records')
    for i, rec in enumerate(rankings['shortest_generations_top100']):
        rec['rank'] = i + 1
    
    # 6. Longest average generations (higher is better) - require minimum games
    df_gen_sorted = df_gen.sort_values(['avg_generations', 'avg_score'], ascending=[False, False])
    rankings['longest_generations_top100'] = df_gen_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'avg_generations', 'total_games', 'avg_score'
    ]].to_dict('records')
    for i, rec in enumerate(rankings['longest_generations_top100']):
        rec['rank'] = i + 1
    
    # 7. Max score (higher is better)
    df_score = df[df['max_score'] > 0].copy()
    df_score_sorted = df_score.sort_values(['max_score', 'avg_score'], ascending=[False, False])
    rankings['max_score_top100'] = df_score_sorted.head(TOP_RANKING_SIZE)[[
        'user_key', 'max_score', 'total_games', 'avg_score'
    ]].to_dict('records')
    for i, rec in enumerate(rankings['max_score_top100']):
        rec['rank'] = i + 1
    
    # Print summary
    print(f"\n--- Top 10 by Total Games ---")
    for rec in rankings['total_games_top100'][:10]:
        print(f"  #{rec['rank']:3d} {rec['user_key']}: {rec['total_games']} games")
    
    print(f"\n--- Top 10 by Win Rate (>={min_games_for_winrate} games) ---")
    for rec in rankings['win_rate_top100'][:10]:
        print(f"  #{rec['rank']:3d} {rec['user_key']}: {rec['win_rate']}% ({rec['total_games']} games)")
    
    print(f"\n--- Top 10 by Avg Position (>={min_games_for_winrate} games) ---")
    for rec in rankings['avg_position_top100'][:10]:
        print(f"  #{rec['rank']:3d} {rec['user_key']}: {rec['avg_position']:.3f} ({rec['total_games']} games)")
    
    print(f"\n--- Top 10 by Max Score ---")
    for rec in rankings['max_score_top100'][:10]:
        print(f"  #{rec['rank']:3d} {rec['user_key']}: {rec['max_score']} pts")
    
    return rankings


def add_rankings_to_users(results: Dict, rankings: Dict) -> None:
    """
    Add ranking information to each user's data.
    
    Modifies results in-place.
    """
    # Create lookup maps for each ranking
    ranking_maps = {}
    
    for ranking_name, ranking_list in rankings.items():
        ranking_maps[ranking_name] = {
            rec['user_key']: rec['rank'] for rec in ranking_list
        }
    
    # Add rankings to each user
    for user_key, data in results.items():
        user_rankings = {}
        
        # Check each ranking
        for ranking_name, rank_map in ranking_maps.items():
            if user_key in rank_map:
                user_rankings[ranking_name] = rank_map[user_key]
            else:
                user_rankings[ranking_name] = None  # Not in top 100
        
        # Add to user data
        data['global_rankings'] = user_rankings
    
    # Count users with at least one ranking
    users_with_ranking = sum(
        1 for data in results.values()
        if any(r is not None for r in data.get('global_rankings', {}).values())
    )
    print(f"\nUsers with at least one Top 100 ranking: {users_with_ranking}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Batch User Aggregate Script')
    parser.add_argument('-p', '--players', type=int, default=4,
                        help='Filter by player count (default: 4)')
    parser.add_argument('-m', '--min-games', type=int, default=MIN_GAMES_FOR_AGGREGATE,
                        help=f'Minimum games to include (default: {MIN_GAMES_FOR_AGGREGATE})')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress detailed output')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of users to process (for testing)')
    args = parser.parse_args()
    
    # Load matched players
    matched_df = load_matched_players()
    if len(matched_df) == 0:
        return
    
    # Group by matched_user
    user_to_names, user_to_games = group_players_by_user(matched_df, args.min_games)
    
    if len(user_to_names) == 0:
        print("No users meet the minimum games requirement")
        return
    
    # Limit for testing
    if args.limit:
        limited_keys = list(user_to_names.keys())[:args.limit]
        user_to_names = {k: user_to_names[k] for k in limited_keys}
        user_to_games = {k: user_to_games[k] for k in limited_keys}
        print(f"Limited to {args.limit} users for processing")
    
    # Batch aggregate
    results = batch_aggregate_all_users(
        user_to_names=user_to_names,
        user_to_games=user_to_games,
        players_filter=args.players,
        quiet=args.quiet
    )
    
    if len(results) == 0:
        print("No results generated")
        return
    
    # Calculate global rankings
    rankings = calculate_global_rankings(results)
    
    # Add rankings to each user's data
    add_rankings_to_users(results, rankings)
    
    # Generate summary statistics
    summary = {
        'total_users': len(results),
        'total_games': sum(r['player_stats'].get('total_games', 0) for r in results.values()),
        'total_wins': sum(r['player_stats'].get('total_wins', 0) for r in results.values()),
        'players_filter': args.players,
        'min_games': args.min_games,
    }
    
    output_data = {
        'summary': summary,
        'rankings': rankings,
        'users': results
    }
    
    # Output to file
    output_path = args.output or f'{DISPLAY_DIR}/batch_user_aggregate_{args.players}p.json'
    Path(DISPLAY_DIR).mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total users aggregated: {summary['total_users']}")
    print(f"Total games: {summary['total_games']}")
    print(f"Total wins: {summary['total_wins']}")
    print(f"Output saved to: {output_path}")
    
    # Show top 10 users by game count
    top_users = sorted(
        results.items(),
        key=lambda x: x[1]['player_stats'].get('total_games', 0),
        reverse=True
    )[:10]
    
    print(f"\n--- Top 10 Users by Game Count ---")
    for user_key, data in top_users:
        ps = data['player_stats']
        names_count = len(data['metadata']['input_names'])
        print(f"  {user_key}: {ps['total_games']} games, {ps['win_rate']}% WR, {names_count} aliases")


if __name__ == '__main__':
    main()
