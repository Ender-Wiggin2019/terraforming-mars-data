#!/usr/bin/env python3
"""
TFM User Analysis Script
Analyzes player statistics from user dimension:
1. Player average stats: win rate, position, score, generations, TR, cards played
2. Player records by generation: highest score, highest cards played
3. Corporation/Prelude/Card usage per player: count and win rate

Key differences from card_analysis.py / game_analysis.py:
- NO breakthrough filter (all data allowed)
- Only players with user_id and >= 2 games
- Configurable player count filter (default 4)
"""

import argparse
import pandas as pd
import numpy as np
import os
import json
import sqlite3
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
import re
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

# Load environment variables
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
LOCAL_DB_PATH = './local_data/tfm.db'
USERS_CSV_PATH = './local_data/users.csv'
CN_MAP_FILE = './data/cn_merged.json'
DISPLAY_DIR = './display'
FONT_PATH = './fonts/MapleMono-NF-CN-Regular.ttf'

# Analysis parameters
MIN_GAMES_PER_PLAYER = 2  # Players with only 1 game are excluded
MIN_ITEM_USAGE = 2  # Ignore corp/prelude/card if used only once

# Bayesian average parameters for corporation ranking
CORP_RANKING_PRIOR_N = 5  # Prior games for Bayesian smoothing
CORP_RANKING_PRIOR_MEAN = 2.5  # Prior mean position (for 4P)
TOP_PLAYERS_PER_CORP = 100  # Top N players per corporation

# =============================================================================
# Setup matplotlib
# =============================================================================
def setup_matplotlib():
    """Configure matplotlib for Chinese character support"""
    if Path(FONT_PATH).exists():
        font_manager.fontManager.addfont(FONT_PATH)
        prop = font_manager.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)

# =============================================================================
# Chinese Name Mapping
# =============================================================================
def load_cn_map(filepath: str) -> dict:
    """Load Chinese name mapping from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            cn_map = json.load(f)
        print(f"Loaded {len(cn_map)} Chinese name mappings from {filepath}")
        return cn_map
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return {}

def strip_emoji(text: str) -> str:
    """Remove emoji characters for matplotlib display"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002600-\U000026FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text).strip()

def get_cn_name(name: str, cn_map: dict) -> str:
    """Get Chinese name, return original if not found"""
    return cn_map.get(name, name)

def get_cn_name_for_plot(name: str, cn_map: dict) -> str:
    """Get Chinese name for plotting (emoji removed)"""
    cn_name = cn_map.get(name, name)
    return strip_emoji(cn_name)

# =============================================================================
# Bayesian Average Functions
# =============================================================================
def bayesian_avg_position(
    total_sum: float,
    count: int,
    prior_mean: float = CORP_RANKING_PRIOR_MEAN,
    prior_n: int = CORP_RANKING_PRIOR_N
) -> float:
    """
    Calculate Bayesian smoothed average position.
    
    Args:
        total_sum: Sum of positions
        count: Number of games
        prior_mean: Prior mean position (default 2.5 for 4P)
        prior_n: Prior number of games for smoothing (default 5)
    
    Returns:
        Bayesian smoothed average position
    """
    if count is None or (isinstance(count, (int, float)) and count == 0):
        return prior_mean
    return (prior_n * prior_mean + total_sum) / (prior_n + count)

def add_cn_name_column(df: pd.DataFrame, name_column: str, cn_map: dict) -> pd.DataFrame:
    """Add cn_name and cn_name_plot columns to DataFrame"""
    df = df.copy()
    df['cn_name'] = df[name_column].apply(lambda x: get_cn_name(x, cn_map))
    df['cn_name_plot'] = df[name_column].apply(lambda x: get_cn_name_for_plot(x, cn_map))
    return df

# =============================================================================
# Data Loading
# =============================================================================
def load_users() -> pd.DataFrame:
    """Load users data from CSV"""
    if not Path(USERS_CSV_PATH).exists():
        print(f"Error: {USERS_CSV_PATH} not found")
        return pd.DataFrame()
    
    users_df = pd.read_csv(USERS_CSV_PATH)
    print(f"Loaded {len(users_df)} users from {USERS_CSV_PATH}")
    return users_df[['id', 'name']].rename(columns={'id': 'user_id', 'name': 'user_name'})

def load_data_from_local(players_filter: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from local SQLite database.
    
    NOTE: Unlike card_analysis.py, this does NOT filter breakthrough games.
    All data is included.
    
    Args:
        players_filter: Filter to specific player count (e.g., 2 or 4), None for all
    """
    if not Path(LOCAL_DB_PATH).exists():
        print(f"Error: {LOCAL_DB_PATH} not found")
        return None, None, None

    conn = sqlite3.connect(LOCAL_DB_PATH)
    games_df = pd.read_sql('SELECT * FROM games', conn)
    game_results_df = pd.read_sql('SELECT * FROM game_results', conn)
    user_game_results_df = pd.read_sql('SELECT * FROM user_game_results', conn)
    conn.close()
    
    print(f"Loaded raw data:")
    print(f"  - games: {len(games_df)} rows")
    print(f"  - game_results: {len(game_results_df)} rows")
    print(f"  - user_game_results: {len(user_game_results_df)} rows")
    
    # NO breakthrough filter - all data allowed
    
    # Data cleaning: remove games with generation=1 or 2
    if 'generations' in game_results_df.columns:
        initial_count = len(game_results_df)
        valid_games = game_results_df[game_results_df['generations'] > 2]['game_id'].tolist()
        game_results_df = game_results_df[game_results_df['game_id'].isin(valid_games)]
        user_game_results_df = user_game_results_df[user_game_results_df['game_id'].isin(valid_games)]
        games_df = games_df[games_df['game_id'].isin(valid_games)]
        print(f"Removed games with generation<=2: {initial_count - len(game_results_df)} games removed")
    
    # Data cleaning: remove records with negative scores
    if 'score' in user_game_results_df.columns:
        negative_score_games = user_game_results_df[user_game_results_df['score'] < 0]['game_id'].unique().tolist()
        if len(negative_score_games) > 0:
            game_results_df = game_results_df[~game_results_df['game_id'].isin(negative_score_games)]
            user_game_results_df = user_game_results_df[~user_game_results_df['game_id'].isin(negative_score_games)]
            games_df = games_df[~games_df['game_id'].isin(negative_score_games)]
            print(f"Removed games with negative scores: {len(negative_score_games)} games removed")
    
    # Filter by player count if specified
    if players_filter is not None:
        initial_game_count = game_results_df['game_id'].nunique()
        game_results_df = game_results_df[game_results_df['players'] == players_filter]
        valid_game_ids = game_results_df['game_id'].tolist()
        user_game_results_df = user_game_results_df[user_game_results_df['game_id'].isin(valid_game_ids)]
        games_df = games_df[games_df['game_id'].isin(valid_game_ids)]
        print(f"Filtered to {players_filter}-player games: {initial_game_count} -> {len(game_results_df)} games")
    
    # Data cleaning: only players with user_id
    initial_ugr_count = len(user_game_results_df)
    user_game_results_df = user_game_results_df[user_game_results_df['user_id'].notna()]
    user_game_results_df = user_game_results_df[user_game_results_df['user_id'] != '']
    print(f"Removed records without user_id: {initial_ugr_count - len(user_game_results_df)} removed")
    
    print(f"\nFinal dataset:")
    print(f"  - game_results: {len(game_results_df)} rows")
    print(f"  - user_game_results: {len(user_game_results_df)} rows")
    print(f"  - games: {len(games_df)} rows")

    return games_df, game_results_df, user_game_results_df

def parse_json_safe(json_str):
    """Safely parse JSON string"""
    if pd.isna(json_str):
        return None
    if isinstance(json_str, dict):
        return json_str
    if isinstance(json_str, str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return eval(json_str)
            except:
                return None
    return None

# =============================================================================
# Helper Functions
# =============================================================================
def split_corporations(corp_str: str) -> list:
    """Split corporation string into list"""
    if pd.isna(corp_str) or corp_str == '':
        return []
    return [c.strip() for c in str(corp_str).split('|') if c.strip()]

def is_prelude(card_name: str) -> bool:
    """Check if a card is a prelude"""
    prelude_names = [
        'Allied Bank', 'Aquifer Turbines', 'Biosphere Support', 'Business Empire',
        'Donation', 'Double Down', 'Early Settlement', 'Ecology Experts',
        'Experimental Forest', 'Galilean Mining', 'Huge Asteroid', 'Io Research Outpost',
        'Loan', 'Martian Industries', 'Metal-Rich Asteroid', 'Mining Operations',
        'Mohole', 'Nitrogen Shipment', 'Orbital Construction Yard', 'Power Generation',
        'Research Network', 'Self-Sufficient Settlement', 'SF Memorial', 'Smelting Plant',
        'Society Support', 'Supplier', 'Supply Drop', 'UNMI Contractor',
        'Acquired Space Agency', 'Biofuels', 'Early Colonization', 'Excentric Sponsor',
        'New Partner', 'Research Grant', 'Science Award', 'Terraforming Deal',
        'Mining Complex', 'Merger', 'Soil Bacteria', 'Project Eden',
        'WG Partnership', 'World Government Advisor', 'Suitable Infrastructure',
        'Nitrate Reducers', 'Space Lanes', 'Venus Contract', 'Inherited Fortune',
        'Lunar Planning Office'
    ]
    return card_name in prelude_names

def is_corporation(card_name: str) -> bool:
    """Check if a card is a corporation"""
    return '(breakthrough)' in card_name.lower() or card_name.startswith('🌸')

def is_ceo(card_name: str) -> bool:
    """Check if a card is a CEO"""
    return card_name.endswith(' CEO') or card_name.startswith('CEO')

# =============================================================================
# Data Extraction Functions
# =============================================================================
def extract_player_game_stats(games_df: pd.DataFrame, user_game_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract per-game player stats including TR and cards played from games JSON.
    
    Returns DataFrame with game_id, user_id, terraform_rating, cards_played_count
    """
    if games_df is None or len(games_df) == 0:
        print("No games data available")
        return pd.DataFrame()
    
    records = []
    
    for _, row in games_df.iterrows():
        game_data = parse_json_safe(row['game'])
        if not game_data:
            continue
            
        game_id = row['game_id']
        
        for player in game_data.get('players', []):
            user_id = player.get('userId')
            if not user_id:
                continue
                
            tr = player.get('terraformRating', 0)
            played_cards = player.get('playedCards', [])
            cards_played_count = len(played_cards) if played_cards else 0
            
            records.append({
                'game_id': game_id,
                'user_id': user_id,
                'terraform_rating': tr,
                'cards_played_count': cards_played_count,
            })
    
    stats_df = pd.DataFrame(records)
    print(f"Extracted {len(stats_df)} player game stats records")
    
    return stats_df

def extract_played_cards_data(games_df: pd.DataFrame, user_game_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all played cards from games JSON.
    Similar to card_analysis.py but returns user-level data.
    """
    if games_df is None or len(games_df) == 0:
        return pd.DataFrame()

    card_records = []
    
    for _, row in games_df.iterrows():
        game_data = parse_json_safe(row['game'])
        if not game_data:
            continue
            
        game_id = row['game_id']
        
        for player in game_data.get('players', []):
            user_id = player.get('userId')
            if not user_id:
                continue
                
            played_cards = player.get('playedCards', [])
            for card in played_cards:
                card_name = card.get('name', '')
                if not card_name:
                    continue
                
                card_records.append({
                    'game_id': game_id,
                    'user_id': user_id,
                    'card_name': card_name,
                    'is_prelude': is_prelude(card_name),
                    'is_corporation': is_corporation(card_name),
                    'is_ceo': is_ceo(card_name)
                })
    
    card_df = pd.DataFrame(card_records)
    print(f"Extracted {len(card_df)} card usage records")
    
    return card_df

# =============================================================================
# Part 1: Player Average Statistics
# =============================================================================
def analyze_player_average_stats(
    user_game_results_df: pd.DataFrame,
    player_game_stats_df: pd.DataFrame,
    users_df: pd.DataFrame,
    min_games: int = MIN_GAMES_PER_PLAYER
) -> pd.DataFrame:
    """
    Analyze player average statistics:
    - Win rate, average position, average score, average generations, average TR, average cards played
    - Include total columns for future aggregation
    """
    print("\n" + "=" * 60)
    print("玩家平均统计")
    print("=" * 60)
    
    # Filter to completed games only
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    if len(ugr_end) == 0:
        print("No completed game records")
        return pd.DataFrame()
    
    # Merge with player game stats (TR and cards played)
    if player_game_stats_df is not None and len(player_game_stats_df) > 0:
        ugr_end = ugr_end.merge(
            player_game_stats_df[['game_id', 'user_id', 'terraform_rating', 'cards_played_count']],
            on=['game_id', 'user_id'],
            how='left'
        )
    else:
        ugr_end['terraform_rating'] = np.nan
        ugr_end['cards_played_count'] = np.nan
    
    # Aggregate by user
    player_stats = ugr_end.groupby('user_id').agg(
        # Counts
        total_games=('game_id', 'count'),
        total_wins=('position', lambda x: (x == 1).sum()),
        # Totals for aggregation
        total_position_sum=('position', 'sum'),
        total_score_sum=('player_score', 'sum'),
        total_generations_sum=('generations', 'sum'),
        total_tr_sum=('terraform_rating', 'sum'),
        total_cards_played_sum=('cards_played_count', 'sum'),
        # Counts for TR and cards (may have nulls)
        tr_games=('terraform_rating', 'count'),
        cards_games=('cards_played_count', 'count'),
        # Averages
        avg_position=('position', 'mean'),
        avg_score=('player_score', 'mean'),
        avg_generations=('generations', 'mean'),
        avg_tr=('terraform_rating', 'mean'),
        avg_cards_played=('cards_played_count', 'mean'),
        # Min/Max for reference
        min_score=('player_score', 'min'),
        max_score=('player_score', 'max'),
        min_position=('position', 'min'),
    ).reset_index()
    
    # Calculate win rate
    player_stats['win_rate'] = (player_stats['total_wins'] / player_stats['total_games'] * 100).round(2)
    
    # Round averages
    player_stats['avg_position'] = player_stats['avg_position'].round(3)
    player_stats['avg_score'] = player_stats['avg_score'].round(2)
    player_stats['avg_generations'] = player_stats['avg_generations'].round(2)
    player_stats['avg_tr'] = player_stats['avg_tr'].round(2)
    player_stats['avg_cards_played'] = player_stats['avg_cards_played'].round(2)
    
    # Filter by minimum games
    player_stats = player_stats[player_stats['total_games'] >= min_games]
    print(f"Players with >= {min_games} games: {len(player_stats)}")
    
    # Merge with user names
    if users_df is not None and len(users_df) > 0:
        player_stats = player_stats.merge(users_df, on='user_id', how='left')
        player_stats['user_name'] = player_stats['user_name'].fillna(player_stats['user_id'])
    else:
        player_stats['user_name'] = player_stats['user_id']
    
    # Sort by total games descending
    player_stats = player_stats.sort_values('total_games', ascending=False)
    
    # Display summary
    print(f"\n总玩家数 (>=2局): {len(player_stats)}")
    print(f"总对局记录数: {player_stats['total_games'].sum()}")
    print(f"\n--- Top 20 玩家 (按对局数) ---")
    display_cols = ['user_name', 'total_games', 'win_rate', 'avg_position', 'avg_score', 
                    'avg_generations', 'avg_tr', 'avg_cards_played']
    print(player_stats[display_cols].head(20).to_string(index=False))
    
    return player_stats

# =============================================================================
# Part 2: Player Records by Generation
# =============================================================================
def analyze_player_records_by_generation(
    user_game_results_df: pd.DataFrame,
    player_game_stats_df: pd.DataFrame,
    users_df: pd.DataFrame,
    min_games: int = MIN_GAMES_PER_PLAYER
) -> pd.DataFrame:
    """
    Analyze player's highest score and highest cards played by generation.
    """
    print("\n" + "=" * 60)
    print("玩家按时代数的最高纪录")
    print("=" * 60)
    
    # Filter to completed games only
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    if len(ugr_end) == 0:
        print("No completed game records")
        return pd.DataFrame()
    
    # Merge with player game stats
    if player_game_stats_df is not None and len(player_game_stats_df) > 0:
        ugr_end = ugr_end.merge(
            player_game_stats_df[['game_id', 'user_id', 'cards_played_count']],
            on=['game_id', 'user_id'],
            how='left'
        )
    else:
        ugr_end['cards_played_count'] = np.nan
    
    # Filter players with minimum games
    player_game_counts = ugr_end.groupby('user_id').size()
    valid_players = player_game_counts[player_game_counts >= min_games].index.tolist()
    ugr_filtered = ugr_end[ugr_end['user_id'].isin(valid_players)]
    
    # Aggregate by user and generation
    records_by_gen = ugr_filtered.groupby(['user_id', 'generations']).agg(
        game_count=('game_id', 'count'),
        max_score=('player_score', 'max'),
        max_cards_played=('cards_played_count', 'max'),
        avg_score=('player_score', 'mean'),
        avg_cards_played=('cards_played_count', 'mean'),
    ).reset_index()
    
    # Merge with user names
    if users_df is not None and len(users_df) > 0:
        records_by_gen = records_by_gen.merge(users_df, on='user_id', how='left')
        records_by_gen['user_name'] = records_by_gen['user_name'].fillna(records_by_gen['user_id'])
    else:
        records_by_gen['user_name'] = records_by_gen['user_id']
    
    records_by_gen = records_by_gen.round(2)
    
    print(f"\n玩家时代记录数: {len(records_by_gen)}")
    print(f"\n--- 样例数据 (前20行) ---")
    display_cols = ['user_name', 'generations', 'game_count', 'max_score', 'max_cards_played']
    print(records_by_gen[display_cols].head(20).to_string(index=False))
    
    return records_by_gen

# =============================================================================
# Part 3: Corporation/Prelude/Card Usage Per Player
# =============================================================================
def analyze_player_corporation_usage(
    user_game_results_df: pd.DataFrame,
    users_df: pd.DataFrame,
    cn_map: dict,
    min_games: int = MIN_GAMES_PER_PLAYER,
    min_usage: int = MIN_ITEM_USAGE
) -> pd.DataFrame:
    """
    Analyze corporation usage per player: count and win rate.
    Ignore if usage count <= 1.
    """
    print("\n" + "=" * 60)
    print("玩家公司使用统计")
    print("=" * 60)
    
    # Filter to completed games
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    if len(ugr_end) == 0:
        print("No completed game records")
        return pd.DataFrame()
    
    # Filter players with minimum games
    player_game_counts = ugr_end.groupby('user_id').size()
    valid_players = player_game_counts[player_game_counts >= min_games].index.tolist()
    ugr_filtered = ugr_end[ugr_end['user_id'].isin(valid_players)]
    
    # Expand corporations (handle multiple corps separated by |)
    corp_records = []
    for _, row in ugr_filtered.iterrows():
        corps = split_corporations(row['corporation'])
        for corp in corps:
            corp_records.append({
                'user_id': row['user_id'],
                'game_id': row['game_id'],
                'corporation': corp,
                'position': row['position'],
                'player_score': row['player_score'],
                'is_win': 1 if row['position'] == 1 else 0
            })
    
    corp_df = pd.DataFrame(corp_records)
    
    if len(corp_df) == 0:
        print("No corporation data")
        return pd.DataFrame()
    
    # Aggregate by user and corporation
    corp_stats = corp_df.groupby(['user_id', 'corporation']).agg(
        usage_count=('game_id', 'count'),
        total_wins=('is_win', 'sum'),
        total_position_sum=('position', 'sum'),
        total_score_sum=('player_score', 'sum'),
        avg_score=('player_score', 'mean'),
    ).reset_index()
    
    # Filter by minimum usage
    corp_stats = corp_stats[corp_stats['usage_count'] >= min_usage]
    
    # Calculate win rate and average position
    corp_stats['win_rate'] = (corp_stats['total_wins'] / corp_stats['usage_count'] * 100).round(2)
    corp_stats['avg_position'] = (corp_stats['total_position_sum'] / corp_stats['usage_count']).round(3)
    corp_stats['avg_score'] = corp_stats['avg_score'].round(2)
    
    # Calculate Bayesian weighted average position (prior: 5 games at 2.5)
    corp_stats['bayesian_avg_position'] = corp_stats.apply(
        lambda r: bayesian_avg_position(
            r['total_position_sum'], 
            r['usage_count'],
            CORP_RANKING_PRIOR_MEAN,
            CORP_RANKING_PRIOR_N
        ),
        axis=1
    ).round(4)
    
    # Merge with user names
    if users_df is not None and len(users_df) > 0:
        corp_stats = corp_stats.merge(users_df, on='user_id', how='left')
        corp_stats['user_name'] = corp_stats['user_name'].fillna(corp_stats['user_id'])
    else:
        corp_stats['user_name'] = corp_stats['user_id']
    
    # Add Chinese names
    corp_stats = add_cn_name_column(corp_stats, 'corporation', cn_map)
    
    print(f"\n玩家-公司组合数 (使用次数>={min_usage}): {len(corp_stats)}")
    print(f"\n--- 样例数据 (前20行, 按使用次数排序) ---")
    display_cols = ['user_name', 'corporation', 'cn_name', 'usage_count', 'win_rate', 'avg_position', 'bayesian_avg_position']
    print(corp_stats.sort_values('usage_count', ascending=False)[display_cols].head(20).to_string(index=False))
    
    return corp_stats


def generate_top_players_per_corporation(
    corp_stats: pd.DataFrame,
    cn_map: dict,
    top_n: int = TOP_PLAYERS_PER_CORP
) -> pd.DataFrame:
    """
    Generate top N players per corporation ranked by Bayesian average position.
    
    Args:
        corp_stats: DataFrame from analyze_player_corporation_usage
        top_n: Number of top players per corporation (default 100)
    
    Returns:
        DataFrame with top players per corporation with rank
    """
    print("\n" + "=" * 60)
    print(f"每个公司的 Top {top_n} 玩家 (按贝叶斯平均顺位)")
    print("=" * 60)
    
    if corp_stats is None or len(corp_stats) == 0:
        print("No corporation stats data")
        return pd.DataFrame()
    
    # For each corporation, rank players by bayesian_avg_position (lower is better)
    corp_stats_sorted = corp_stats.sort_values(
        ['corporation', 'bayesian_avg_position', 'usage_count'],
        ascending=[True, True, False]
    )
    
    # Assign rank within each corporation
    corp_stats_sorted['corp_rank'] = corp_stats_sorted.groupby('corporation').cumcount() + 1
    
    # Filter to top N per corporation
    top_players = corp_stats_sorted[corp_stats_sorted['corp_rank'] <= top_n].copy()
    
    # Calculate average score for each player-corp combination
    # (We need to merge with game results to get avg_score)
    # For now, we'll just include the existing columns
    
    print(f"\n公司总数: {corp_stats['corporation'].nunique()}")
    print(f"Top {top_n} 玩家记录数: {len(top_players)}")
    
    # Show sample
    print(f"\n--- 样例: 某公司的 Top 10 玩家 ---")
    sample_corp = top_players['corporation'].iloc[0] if len(top_players) > 0 else None
    if sample_corp:
        sample_data = top_players[top_players['corporation'] == sample_corp].head(10)
        display_cols = ['corp_rank', 'user_name', 'corporation', 'cn_name', 'usage_count', 
                        'win_rate', 'avg_position', 'bayesian_avg_position']
        print(sample_data[display_cols].to_string(index=False))
    
    return top_players

def analyze_player_prelude_usage(
    cards_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame,
    users_df: pd.DataFrame,
    cn_map: dict,
    min_games: int = MIN_GAMES_PER_PLAYER,
    min_usage: int = MIN_ITEM_USAGE
) -> pd.DataFrame:
    """
    Analyze prelude usage per player: count and win rate.
    Ignore if usage count <= 1.
    """
    print("\n" + "=" * 60)
    print("玩家前序卡使用统计")
    print("=" * 60)
    
    if cards_df is None or len(cards_df) == 0:
        print("No card data")
        return pd.DataFrame()
    
    # Filter to preludes only
    prelude_df = cards_df[cards_df['is_prelude']].copy()
    
    if len(prelude_df) == 0:
        print("No prelude data")
        return pd.DataFrame()
    
    # Filter to completed games
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    # Filter players with minimum games
    player_game_counts = ugr_end.groupby('user_id').size()
    valid_players = player_game_counts[player_game_counts >= min_games].index.tolist()
    
    # Merge prelude with game results
    prelude_merged = prelude_df.merge(
        ugr_end[['game_id', 'user_id', 'position']],
        on=['game_id', 'user_id'],
        how='inner'
    )
    prelude_merged = prelude_merged[prelude_merged['user_id'].isin(valid_players)]
    prelude_merged['is_win'] = (prelude_merged['position'] == 1).astype(int)
    
    # Aggregate by user and prelude
    prelude_stats = prelude_merged.groupby(['user_id', 'card_name']).agg(
        usage_count=('game_id', 'count'),
        total_wins=('is_win', 'sum'),
        total_position_sum=('position', 'sum'),
    ).reset_index()
    
    prelude_stats = prelude_stats.rename(columns={'card_name': 'prelude_name'})
    
    # Filter by minimum usage
    prelude_stats = prelude_stats[prelude_stats['usage_count'] >= min_usage]
    
    # Calculate win rate and average position
    prelude_stats['win_rate'] = (prelude_stats['total_wins'] / prelude_stats['usage_count'] * 100).round(2)
    prelude_stats['avg_position'] = (prelude_stats['total_position_sum'] / prelude_stats['usage_count']).round(3)
    
    # Merge with user names
    if users_df is not None and len(users_df) > 0:
        prelude_stats = prelude_stats.merge(users_df, on='user_id', how='left')
        prelude_stats['user_name'] = prelude_stats['user_name'].fillna(prelude_stats['user_id'])
    else:
        prelude_stats['user_name'] = prelude_stats['user_id']
    
    # Add Chinese names
    prelude_stats = add_cn_name_column(prelude_stats, 'prelude_name', cn_map)
    
    print(f"\n玩家-前序组合数 (使用次数>={min_usage}): {len(prelude_stats)}")
    print(f"\n--- 样例数据 (前20行, 按使用次数排序) ---")
    display_cols = ['user_name', 'prelude_name', 'cn_name', 'usage_count', 'win_rate', 'avg_position']
    print(prelude_stats.sort_values('usage_count', ascending=False)[display_cols].head(20).to_string(index=False))
    
    return prelude_stats

def analyze_player_card_usage(
    cards_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame,
    users_df: pd.DataFrame,
    cn_map: dict,
    min_games: int = MIN_GAMES_PER_PLAYER,
    min_usage: int = MIN_ITEM_USAGE
) -> pd.DataFrame:
    """
    Analyze card (project cards only) usage per player: count and win rate.
    Ignore if usage count <= 1.
    """
    print("\n" + "=" * 60)
    print("玩家项目卡使用统计")
    print("=" * 60)
    
    if cards_df is None or len(cards_df) == 0:
        print("No card data")
        return pd.DataFrame()
    
    # Filter to project cards only (exclude preludes, corporations, CEOs)
    project_df = cards_df[
        (~cards_df['is_prelude']) &
        (~cards_df['is_corporation']) &
        (~cards_df['is_ceo'])
    ].copy()
    
    if len(project_df) == 0:
        print("No project card data")
        return pd.DataFrame()
    
    # Filter to completed games
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    # Filter players with minimum games
    player_game_counts = ugr_end.groupby('user_id').size()
    valid_players = player_game_counts[player_game_counts >= min_games].index.tolist()
    
    # Merge cards with game results
    cards_merged = project_df.merge(
        ugr_end[['game_id', 'user_id', 'position']],
        on=['game_id', 'user_id'],
        how='inner'
    )
    cards_merged = cards_merged[cards_merged['user_id'].isin(valid_players)]
    cards_merged['is_win'] = (cards_merged['position'] == 1).astype(int)
    
    # Aggregate by user and card
    card_stats = cards_merged.groupby(['user_id', 'card_name']).agg(
        usage_count=('game_id', 'count'),
        total_wins=('is_win', 'sum'),
        total_position_sum=('position', 'sum'),
    ).reset_index()
    
    # Filter by minimum usage
    card_stats = card_stats[card_stats['usage_count'] >= min_usage]
    
    # Calculate win rate and average position
    card_stats['win_rate'] = (card_stats['total_wins'] / card_stats['usage_count'] * 100).round(2)
    card_stats['avg_position'] = (card_stats['total_position_sum'] / card_stats['usage_count']).round(3)
    
    # Merge with user names
    if users_df is not None and len(users_df) > 0:
        card_stats = card_stats.merge(users_df, on='user_id', how='left')
        card_stats['user_name'] = card_stats['user_name'].fillna(card_stats['user_id'])
    else:
        card_stats['user_name'] = card_stats['user_id']
    
    # Add Chinese names
    card_stats = add_cn_name_column(card_stats, 'card_name', cn_map)
    
    print(f"\n玩家-卡牌组合数 (使用次数>={min_usage}): {len(card_stats)}")
    print(f"\n--- 样例数据 (前20行, 按使用次数排序) ---")
    display_cols = ['user_name', 'card_name', 'cn_name', 'usage_count', 'win_rate', 'avg_position']
    print(card_stats.sort_values('usage_count', ascending=False)[display_cols].head(20).to_string(index=False))
    
    return card_stats

# =============================================================================
# Part 4: Player Time Analytics
# =============================================================================
def analyze_player_time_stats(
    user_game_results_df: pd.DataFrame,
    games_df: pd.DataFrame,
    users_df: pd.DataFrame,
    min_games: int = MIN_GAMES_PER_PLAYER
) -> pd.DataFrame:
    """
    Analyze player time statistics:
    - Most common game hour (0-23)
    - Day with most games played (YYYY-MM-DD)
    - Month with most games played (YYYY-MM)
    
    Based on create_time from games table.
    """
    print("\n" + "=" * 60)
    print("玩家时间统计")
    print("=" * 60)
    
    # Filter to completed games
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    if len(ugr_end) == 0:
        print("No completed game records")
        return pd.DataFrame()
    
    # Get createtime from games_df (note: column is 'createtime' not 'create_time')
    if games_df is None or len(games_df) == 0:
        print("No games data available")
        return pd.DataFrame()
    
    # Check for createtime column
    time_col = None
    if 'createtime' in games_df.columns:
        time_col = 'createtime'
    elif 'create_time' in games_df.columns:
        time_col = 'create_time'
    
    if time_col is None:
        print(f"No createtime column found. Available columns: {games_df.columns.tolist()}")
        return pd.DataFrame()
    
    print(f"Using time column: {time_col}")
    
    # Merge with games to get createtime (use suffix to avoid conflict with existing createtime column)
    games_time_df = games_df[['game_id', time_col]].copy()
    games_time_df = games_time_df.rename(columns={time_col: 'game_createtime'})
    
    # Drop existing createtime column from ugr_end if it exists
    ugr_for_merge = ugr_end.copy()
    if 'createtime' in ugr_for_merge.columns:
        ugr_for_merge = ugr_for_merge.drop(columns=['createtime'])
    
    ugr_with_time = ugr_for_merge.merge(
        games_time_df,
        on='game_id',
        how='left'
    )
    
    # Parse createtime
    ugr_with_time['createtime'] = pd.to_datetime(ugr_with_time['game_createtime'], errors='coerce')
    ugr_with_time = ugr_with_time[ugr_with_time['createtime'].notna()]
    
    if len(ugr_with_time) == 0:
        print("No valid createtime data")
        return pd.DataFrame()
    
    # Extract time components
    ugr_with_time['hour'] = ugr_with_time['createtime'].dt.hour
    ugr_with_time['date'] = ugr_with_time['createtime'].dt.strftime('%Y-%m-%d')
    ugr_with_time['month'] = ugr_with_time['createtime'].dt.strftime('%Y-%m')
    ugr_with_time['weekday'] = ugr_with_time['createtime'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Filter players with minimum games
    player_game_counts = ugr_with_time.groupby('user_id').size()
    valid_players = player_game_counts[player_game_counts >= min_games].index.tolist()
    ugr_filtered = ugr_with_time[ugr_with_time['user_id'].isin(valid_players)]
    
    # Calculate stats for each player
    time_stats_list = []
    
    for user_id, group in ugr_filtered.groupby('user_id'):
        # Most common hour
        hour_counts = group['hour'].value_counts()
        most_common_hour = hour_counts.index[0] if len(hour_counts) > 0 else None
        most_common_hour_count = hour_counts.iloc[0] if len(hour_counts) > 0 else 0
        
        # Most common weekday
        weekday_counts = group['weekday'].value_counts()
        most_common_weekday = weekday_counts.index[0] if len(weekday_counts) > 0 else None
        most_common_weekday_count = weekday_counts.iloc[0] if len(weekday_counts) > 0 else 0
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        most_common_weekday_name = weekday_names[most_common_weekday] if most_common_weekday is not None else ''
        
        # Day with most games
        day_counts = group['date'].value_counts()
        busiest_day = day_counts.index[0] if len(day_counts) > 0 else None
        busiest_day_count = day_counts.iloc[0] if len(day_counts) > 0 else 0
        
        # Month with most games
        month_counts = group['month'].value_counts()
        busiest_month = month_counts.index[0] if len(month_counts) > 0 else None
        busiest_month_count = month_counts.iloc[0] if len(month_counts) > 0 else 0
        
        # First and last game time
        first_game = group['createtime'].min()
        last_game = group['createtime'].max()
        
        # Average games per active day
        active_days = group['date'].nunique()
        avg_games_per_day = len(group) / active_days if active_days > 0 else 0
        
        time_stats_list.append({
            'user_id': user_id,
            'total_games': len(group),
            'most_common_hour': int(most_common_hour) if most_common_hour is not None else None,
            'most_common_hour_count': int(most_common_hour_count),
            'most_common_weekday': int(most_common_weekday) if most_common_weekday is not None else None,
            'most_common_weekday_name': most_common_weekday_name,
            'most_common_weekday_count': int(most_common_weekday_count),
            'busiest_day': busiest_day,
            'busiest_day_count': int(busiest_day_count),
            'busiest_month': busiest_month,
            'busiest_month_count': int(busiest_month_count),
            'first_game': first_game.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(first_game) else None,
            'last_game': last_game.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(last_game) else None,
            'active_days': int(active_days),
            'avg_games_per_day': round(avg_games_per_day, 2),
        })
    
    time_stats_df = pd.DataFrame(time_stats_list)
    
    # Merge with user names
    if users_df is not None and len(users_df) > 0:
        time_stats_df = time_stats_df.merge(users_df, on='user_id', how='left')
        time_stats_df['user_name'] = time_stats_df['user_name'].fillna(time_stats_df['user_id'])
    else:
        time_stats_df['user_name'] = time_stats_df['user_id']
    
    # Sort by total games
    time_stats_df = time_stats_df.sort_values('total_games', ascending=False)
    
    # Display summary
    print(f"\n玩家时间统计数 (>=2局): {len(time_stats_df)}")
    print(f"\n--- Top 20 玩家时间统计 ---")
    display_cols = ['user_name', 'total_games', 'most_common_hour', 'most_common_weekday_name',
                    'busiest_day', 'busiest_day_count', 'busiest_month', 'busiest_month_count']
    print(time_stats_df[display_cols].head(20).to_string(index=False))
    
    # Hour distribution summary
    print(f"\n--- 整体小时分布 ---")
    overall_hours = ugr_filtered['hour'].value_counts().sort_index()
    for hour, count in overall_hours.items():
        bar = '█' * (count // 100)
        print(f"  {hour:02d}:00 - {hour:02d}:59: {count:5d} {bar}")
    
    return time_stats_df


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_player_stats_distribution(player_stats: pd.DataFrame, players_filter: int):
    """Plot player statistics distributions"""
    if player_stats is None or len(player_stats) == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'玩家统计分布 ({players_filter}人局)', fontsize=14, fontweight='bold')
    
    # 1. Win rate distribution
    ax = axes[0, 0]
    ax.hist(player_stats['win_rate'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=100/players_filter, color='red', linestyle='--', label=f'期望 {100/players_filter:.1f}%')
    ax.set_xlabel('胜率 (%)')
    ax.set_ylabel('玩家数')
    ax.set_title('胜率分布')
    ax.legend()
    
    # 2. Average position distribution
    ax = axes[0, 1]
    ax.hist(player_stats['avg_position'], bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(x=(players_filter + 1) / 2, color='red', linestyle='--', label=f'期望 {(players_filter + 1) / 2:.1f}')
    ax.set_xlabel('平均顺位')
    ax.set_ylabel('玩家数')
    ax.set_title('平均顺位分布')
    ax.legend()
    
    # 3. Average score distribution
    ax = axes[0, 2]
    ax.hist(player_stats['avg_score'].dropna(), bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('平均分数')
    ax.set_ylabel('玩家数')
    ax.set_title('平均分数分布')
    
    # 4. Average generations distribution
    ax = axes[1, 0]
    ax.hist(player_stats['avg_generations'].dropna(), bins=30, color='purple', edgecolor='black', alpha=0.7)
    ax.set_xlabel('平均时代数')
    ax.set_ylabel('玩家数')
    ax.set_title('平均时代数分布')
    
    # 5. Average TR distribution
    ax = axes[1, 1]
    ax.hist(player_stats['avg_tr'].dropna(), bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax.set_xlabel('平均TR')
    ax.set_ylabel('玩家数')
    ax.set_title('平均TR分布')
    
    # 6. Average cards played distribution
    ax = axes[1, 2]
    ax.hist(player_stats['avg_cards_played'].dropna(), bins=30, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel('平均打牌数')
    ax.set_ylabel('玩家数')
    ax.set_title('平均打牌数分布')
    
    plt.tight_layout()
    plt.savefig(f'{DISPLAY_DIR}/user_stats_distribution_{players_filter}p.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {DISPLAY_DIR}/user_stats_distribution_{players_filter}p.png")

def plot_top_players(player_stats: pd.DataFrame, players_filter: int, top_n: int = 20):
    """Plot top players by various metrics"""
    if player_stats is None or len(player_stats) == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Top {top_n} 玩家排行 ({players_filter}人局, 至少{MIN_GAMES_PER_PLAYER}局)', fontsize=14, fontweight='bold')
    
    # 1. Top by total games
    ax = axes[0, 0]
    top_games = player_stats.nlargest(top_n, 'total_games')
    ax.barh(range(len(top_games)), top_games['total_games'], color='steelblue')
    ax.set_yticks(range(len(top_games)))
    ax.set_yticklabels(top_games['user_name'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('对局数')
    ax.set_title('对局数 Top 20')
    
    # 2. Top by win rate (min 10 games for this chart)
    ax = axes[0, 1]
    top_wr = player_stats[player_stats['total_games'] >= 10].nlargest(top_n, 'win_rate')
    colors = plt.cm.RdYlGn(np.clip(top_wr['win_rate'] / 100, 0, 1))
    ax.barh(range(len(top_wr)), top_wr['win_rate'], color=colors)
    ax.set_yticks(range(len(top_wr)))
    ax.set_yticklabels(top_wr['user_name'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('胜率 (%)')
    ax.set_title('胜率 Top 20 (>=10局)')
    ax.axvline(x=100/players_filter, color='red', linestyle='--', alpha=0.5)
    
    # 3. Top by average score (min 10 games)
    ax = axes[1, 0]
    top_score = player_stats[player_stats['total_games'] >= 10].nlargest(top_n, 'avg_score')
    ax.barh(range(len(top_score)), top_score['avg_score'], color='coral')
    ax.set_yticks(range(len(top_score)))
    ax.set_yticklabels(top_score['user_name'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('平均分数')
    ax.set_title('平均分数 Top 20 (>=10局)')
    
    # 4. Top by average TR (min 10 games)
    ax = axes[1, 1]
    top_tr = player_stats[player_stats['total_games'] >= 10].nlargest(top_n, 'avg_tr')
    ax.barh(range(len(top_tr)), top_tr['avg_tr'], color='orange')
    ax.set_yticks(range(len(top_tr)))
    ax.set_yticklabels(top_tr['user_name'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('平均TR')
    ax.set_title('平均TR Top 20 (>=10局)')
    
    plt.tight_layout()
    plt.savefig(f'{DISPLAY_DIR}/user_top_players_{players_filter}p.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {DISPLAY_DIR}/user_top_players_{players_filter}p.png")

# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TFM User Analysis Script')
    parser.add_argument('-p', '--players', type=int, default=4,
                        help='Filter by player count (default: 4)')
    args = parser.parse_args()
    
    players_filter = args.players
    print(f"Running user analysis for {players_filter}-player games")
    
    # Setup
    setup_matplotlib()
    Path(DISPLAY_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load Chinese name mapping
    cn_map = load_cn_map(CN_MAP_FILE)
    
    # Load users
    users_df = load_users()
    
    # Load data with player filter
    print("\nLoading data from local SQLite...")
    games_df, game_results_df, user_game_results_df = load_data_from_local(players_filter)
    
    if game_results_df is None:
        print("Error: Local data not found. Please run preprocess.py first.")
        return
    
    # Extract player game stats (TR and cards played)
    player_game_stats_df = extract_player_game_stats(games_df, user_game_results_df)
    
    # Extract played cards data
    cards_df = extract_played_cards_data(games_df, user_game_results_df)
    
    # Collect all results
    all_results = {}
    
    # Part 1: Player Average Statistics
    player_stats = analyze_player_average_stats(
        user_game_results_df, player_game_stats_df, users_df
    )
    all_results['player_stats'] = player_stats
    
    if len(player_stats) > 0:
        player_stats.to_csv(f'{DISPLAY_DIR}/user_player_stats_{players_filter}p.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/user_player_stats_{players_filter}p.csv")
    
    # Part 2: Player Records by Generation
    records_by_gen = analyze_player_records_by_generation(
        user_game_results_df, player_game_stats_df, users_df
    )
    all_results['records_by_generation'] = records_by_gen
    
    if len(records_by_gen) > 0:
        records_by_gen.to_csv(f'{DISPLAY_DIR}/user_records_by_generation_{players_filter}p.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/user_records_by_generation_{players_filter}p.csv")
    
    # Part 3: Corporation/Prelude/Card Usage Per Player
    corp_stats = analyze_player_corporation_usage(
        user_game_results_df, users_df, cn_map
    )
    all_results['player_corp_stats'] = corp_stats
    
    if len(corp_stats) > 0:
        corp_stats.to_csv(f'{DISPLAY_DIR}/user_corporation_stats_{players_filter}p.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/user_corporation_stats_{players_filter}p.csv")
        
        # Generate top 100 players per corporation
        top_players_corp = generate_top_players_per_corporation(corp_stats, cn_map)
        all_results['top_players_per_corp'] = top_players_corp
        
        if len(top_players_corp) > 0:
            top_players_corp.to_csv(f'{DISPLAY_DIR}/user_corp_top100_players_{players_filter}p.csv', index=False)
            print(f"Saved: {DISPLAY_DIR}/user_corp_top100_players_{players_filter}p.csv")
    
    prelude_stats = analyze_player_prelude_usage(
        cards_df, user_game_results_df, users_df, cn_map
    )
    all_results['player_prelude_stats'] = prelude_stats
    
    if len(prelude_stats) > 0:
        prelude_stats.to_csv(f'{DISPLAY_DIR}/user_prelude_stats_{players_filter}p.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/user_prelude_stats_{players_filter}p.csv")
    
    card_stats = analyze_player_card_usage(
        cards_df, user_game_results_df, users_df, cn_map
    )
    all_results['player_card_stats'] = card_stats
    
    if len(card_stats) > 0:
        card_stats.to_csv(f'{DISPLAY_DIR}/user_card_stats_{players_filter}p.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/user_card_stats_{players_filter}p.csv")
    
    # Part 4: Player Time Analytics
    time_stats = analyze_player_time_stats(
        user_game_results_df, games_df, users_df
    )
    all_results['time_stats'] = time_stats
    
    if len(time_stats) > 0:
        time_stats.to_csv(f'{DISPLAY_DIR}/user_time_stats_{players_filter}p.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/user_time_stats_{players_filter}p.csv")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    if len(player_stats) > 0:
        plot_player_stats_distribution(player_stats, players_filter)
        plot_top_players(player_stats, players_filter)
    
    print("\n" + "=" * 60)
    print(f"分析完成! ({players_filter}人局)")
    print("=" * 60)
    print(f"输出文件保存至: {DISPLAY_DIR}/")
    
    return all_results

if __name__ == '__main__':
    main()
