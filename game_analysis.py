#!/usr/bin/env python3
"""
TFM Game Analysis Script
Analyzes:
1. Game basic statistics (2P and 4P breakthrough games)
2. Milestones and Awards - win rate and claim frequency
3. Colonies - settlement frequency and win rate

Based on card_analysis.py patterns.
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
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
ANALYSIS_DB_PATH = './local_data/tfm_analysis.db'
CN_MAP_FILE = './data/cn_merged.json'
DISPLAY_DIR = './display'
FONT_PATH = './fonts/MapleMono-NF-CN-Regular.ttf'

# Analysis parameters
DEFAULT_PRIOR_N = 30
MIN_GAMES_MILESTONE = 30
MIN_GAMES_AWARD = 30
MIN_GAMES_COLONY = 30

# Prior means for different player counts
PRIOR_MEANS = {
    2: 1.5,
    3: 2.0,
    4: 2.5,
    5: 3.0
}
DEFAULT_PRIOR_MEAN = 2.5

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

def add_cn_name_column(df: pd.DataFrame, name_column: str, cn_map: dict) -> pd.DataFrame:
    """Add cn_name and cn_name_plot columns to DataFrame"""
    df = df.copy()
    df['cn_name'] = df[name_column].apply(lambda x: get_cn_name(x, cn_map))
    df['cn_name_plot'] = df[name_column].apply(lambda x: get_cn_name_for_plot(x, cn_map))
    return df

# =============================================================================
# Bayesian Average Functions
# =============================================================================
def get_prior_mean(players: int) -> float:
    """Get prior mean position based on player count"""
    return PRIOR_MEANS.get(players, DEFAULT_PRIOR_MEAN)

def bayesian_avg_position(
    total_sum: float,
    count: int,
    prior_mean: float = None,
    prior_n: int = DEFAULT_PRIOR_N
) -> float:
    """Calculate Bayesian smoothed average position."""
    if prior_mean is None:
        prior_mean = DEFAULT_PRIOR_MEAN
    if count is None or (isinstance(count, (int, float)) and count == 0):
        return prior_mean
    return (prior_n * prior_mean + total_sum) / (prior_n + count)

# =============================================================================
# Data Loading
# =============================================================================
def load_data_from_local() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data from local SQLite database"""
    if not Path(LOCAL_DB_PATH).exists():
        print(f"Error: {LOCAL_DB_PATH} not found")
        return None, None, None

    conn = sqlite3.connect(LOCAL_DB_PATH)
    games_df = pd.read_sql('SELECT * FROM games', conn)
    game_results_df = pd.read_sql('SELECT * FROM game_results', conn)
    user_game_results_df = pd.read_sql('SELECT * FROM user_game_results', conn)
    conn.close()

    # Filter breakthrough=true games
    if 'game_options' in game_results_df.columns:
        breakthrough_games = game_results_df[
            game_results_df['game_options'].str.contains('"breakthrough":true', na=False)
        ]['game_id'].tolist()
        game_results_df = game_results_df[game_results_df['game_id'].isin(breakthrough_games)]
        user_game_results_df = user_game_results_df[user_game_results_df['game_id'].isin(breakthrough_games)]
        games_df = games_df[games_df['game_id'].isin(breakthrough_games)]
        print(f"Filtered breakthrough=true games: {len(breakthrough_games)}")

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
        initial_count = len(user_game_results_df)
        negative_score_games = user_game_results_df[user_game_results_df['score'] < 0]['game_id'].unique().tolist()
        if len(negative_score_games) > 0:
            game_results_df = game_results_df[~game_results_df['game_id'].isin(negative_score_games)]
            user_game_results_df = user_game_results_df[~user_game_results_df['game_id'].isin(negative_score_games)]
            games_df = games_df[~games_df['game_id'].isin(negative_score_games)]
            print(f"Removed games with negative scores: {len(negative_score_games)} games removed")

    print(f"Final dataset: {len(game_results_df)} games")

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
# Part 1: Game Basic Statistics
# =============================================================================
def analyze_game_basic_stats(
    game_results_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame,
    games_df: pd.DataFrame
) -> Dict:
    """
    Analyze basic game statistics for 2P and 4P breakthrough games.
    
    Returns dict with statistics and DataFrames.
    """
    print("\n" + "=" * 60)
    print("游戏基础统计 (突破环境)")
    print("=" * 60)
    
    # Filter to completed games
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    results = {}
    
    # Overall statistics
    total_games = game_results_df['game_id'].nunique()
    total_player_records = len(ugr_end)
    
    print(f"\n总游戏数 (突破环境): {total_games}")
    print(f"总玩家记录数: {total_player_records}")
    
    # Statistics by player count
    stats_by_players = ugr_end.groupby('players').agg(
        game_count=('game_id', 'nunique'),
        player_records=('user_id', 'count'),
        avg_generations=('generations', 'mean'),
        std_generations=('generations', 'std'),
        min_generations=('generations', 'min'),
        max_generations=('generations', 'max'),
        avg_score=('player_score', 'mean'),
        std_score=('player_score', 'std'),
        min_score=('player_score', 'min'),
        max_score=('player_score', 'max'),
    ).reset_index()
    
    stats_by_players = stats_by_players.round(2)
    results['stats_by_players'] = stats_by_players
    
    print("\n--- 按玩家人数统计 (突破环境) ---")
    print(stats_by_players.to_string(index=False))
    
    # Detailed 2P and 4P analysis
    for player_count in [2, 4]:
        player_data = ugr_end[ugr_end['players'] == player_count]
        if len(player_data) == 0:
            continue
            
        print(f"\n--- {player_count}人局详情 (突破环境) ---")
        print(f"  总游戏数: {player_data['game_id'].nunique()}")
        print(f"  平均时代数: {player_data['generations'].mean():.2f} ± {player_data['generations'].std():.2f}")
        print(f"  时代数范围: {player_data['generations'].min()} - {player_data['generations'].max()}")
        print(f"  平均分数: {player_data['player_score'].mean():.1f} ± {player_data['player_score'].std():.1f}")
        
        # Score distribution by position
        score_by_position = player_data.groupby('position').agg(
            count=('player_score', 'count'),
            avg_score=('player_score', 'mean'),
            std_score=('player_score', 'std')
        ).reset_index()
        score_by_position = score_by_position.round(2)
        print(f"\n  按名次统计分数:")
        print(score_by_position.to_string(index=False))
        
        results[f'{player_count}p_score_by_position'] = score_by_position
        
        # NEW: Generation distribution
        print(f"\n  时代数分布 ({player_count}人局):")
        gen_distribution = player_data.groupby('generations').agg(
            game_count=('game_id', 'nunique'),
        ).reset_index()
        gen_distribution['percentage'] = (gen_distribution['game_count'] / gen_distribution['game_count'].sum() * 100).round(2)
        print(gen_distribution.to_string(index=False))
        results[f'{player_count}p_generation_distribution'] = gen_distribution
        
        # NEW: Average and max score by generation
        print(f"\n  按时代数统计分数 ({player_count}人局):")
        score_by_gen = player_data.groupby('generations').agg(
            game_count=('game_id', 'nunique'),
            avg_score=('player_score', 'mean'),
            max_score=('player_score', 'max'),
            min_score=('player_score', 'min'),
            winner_avg_score=('player_score', lambda x: player_data.loc[x.index][player_data.loc[x.index]['position'] == 1]['player_score'].mean())
        ).reset_index()
        score_by_gen = score_by_gen.round(2)
        print(score_by_gen.to_string(index=False))
        results[f'{player_count}p_score_by_generation'] = score_by_gen
        
        # NEW: Additional interesting metrics
        print(f"\n  额外统计指标 ({player_count}人局):")
        
        # Score gap between winner and 2nd place
        game_scores = player_data.pivot_table(index='game_id', columns='position', values='player_score', aggfunc='first')
        if 1 in game_scores.columns and 2 in game_scores.columns:
            score_gap = (game_scores[1] - game_scores[2]).dropna()
            print(f"    第一名与第二名平均分差: {score_gap.mean():.1f} ± {score_gap.std():.1f}")
            print(f"    最大分差: {score_gap.max():.0f}")
            print(f"    最小分差: {score_gap.min():.0f}")
            results[f'{player_count}p_score_gap_stats'] = {
                'avg_gap': round(score_gap.mean(), 1),
                'std_gap': round(score_gap.std(), 1),
                'max_gap': int(score_gap.max()),
                'min_gap': int(score_gap.min())
            }
        
        # Winner score statistics
        winner_data = player_data[player_data['position'] == 1]
        print(f"    第一名平均分: {winner_data['player_score'].mean():.1f}")
        print(f"    第一名最高分: {winner_data['player_score'].max():.0f}")
        print(f"    第一名最低分: {winner_data['player_score'].min():.0f}")
    
    # Extract global parameters from games.game JSON for terraforming speed analysis
    if games_df is not None and len(games_df) > 0:
        print("\n--- 地球化速度分析 (突破环境) ---")
        terraforming_stats = analyze_terraforming_speed(games_df, ugr_end)
        if terraforming_stats is not None:
            results['terraforming_stats'] = terraforming_stats
    
    return results

def analyze_terraforming_speed(games_df: pd.DataFrame, ugr_end: pd.DataFrame) -> pd.DataFrame:
    """Analyze terraforming speed by player count"""
    records = []
    
    for _, row in games_df.iterrows():
        game_data = parse_json_safe(row['game'])
        if not game_data:
            continue
            
        game_id = row['game_id']
        generation = game_data.get('generation', 0)
        if generation == 0:
            continue
            
        # Get player count from game data
        player_count = len(game_data.get('players', []))
        if player_count == 0:
            continue
        
        # Get final global parameters
        temp = game_data.get('temperature', -30)
        oxygen = game_data.get('oxygenLevel', 0)
        venus = game_data.get('venusScaleLevel', 0)
        
        # Calculate terraforming speed (parameters raised per generation)
        temp_raised = temp - (-30)  # Temperature starts at -30
        oxygen_raised = oxygen  # Oxygen starts at 0
        
        records.append({
            'game_id': game_id,
            'players': player_count,
            'generations': generation,
            'final_temperature': temp,
            'final_oxygen': oxygen,
            'final_venus': venus,
            'temp_per_gen': temp_raised / generation if generation > 0 else 0,
            'oxygen_per_gen': oxygen_raised / generation if generation > 0 else 0,
        })
    
    if not records:
        return None
        
    tf_df = pd.DataFrame(records)
    
    # Aggregate by player count
    tf_stats = tf_df.groupby('players').agg(
        game_count=('game_id', 'count'),
        avg_generations=('generations', 'mean'),
        avg_temp_per_gen=('temp_per_gen', 'mean'),
        avg_oxygen_per_gen=('oxygen_per_gen', 'mean'),
        avg_final_temp=('final_temperature', 'mean'),
        avg_final_oxygen=('final_oxygen', 'mean'),
    ).reset_index().round(2)
    
    print(tf_stats.to_string(index=False))
    
    return tf_stats

# =============================================================================
# Part 2: Milestones and Awards Analysis
# =============================================================================
def extract_milestones_awards(games_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract milestones and awards data from games.game JSON.
    
    Returns:
        milestones_df: DataFrame with milestone claims
        awards_df: DataFrame with award fundings
    """
    milestone_records = []
    award_records = []
    available_milestones = []
    available_awards = []
    
    for _, row in games_df.iterrows():
        game_data = parse_json_safe(row['game'])
        if not game_data:
            continue
            
        game_id = row['game_id']
        createtime = row['createtime']
        player_count = len(game_data.get('players', []))
        
        # Build player_id to user_id mapping
        player_map = {}
        for player in game_data.get('players', []):
            player_map[player.get('id')] = player.get('userId')
        
        # Extract available milestones for this game
        for m in game_data.get('milestones', []):
            available_milestones.append({
                'game_id': game_id,
                'milestone_name': m.get('name'),
                'players': player_count,
            })
        
        # Extract claimed milestones
        for order, m in enumerate(game_data.get('claimedMilestones', []), start=1):
            player_id = m.get('player', {}).get('id')
            milestone_records.append({
                'game_id': game_id,
                'createtime': createtime,
                'milestone_name': m.get('milestone', {}).get('name'),
                'player_id': player_id,
                'user_id': player_map.get(player_id),
                'claim_order': order,
                'players': player_count,
            })
        
        # Extract available awards for this game
        for a in game_data.get('awards', []):
            available_awards.append({
                'game_id': game_id,
                'award_name': a.get('name'),
                'players': player_count,
            })
        
        # Extract funded awards
        for order, a in enumerate(game_data.get('fundedAwards', []), start=1):
            player_id = a.get('player', {}).get('id')
            award_records.append({
                'game_id': game_id,
                'createtime': createtime,
                'award_name': a.get('award', {}).get('name'),
                'player_id': player_id,
                'user_id': player_map.get(player_id),
                'fund_order': order,
                'players': player_count,
            })
    
    milestones_df = pd.DataFrame(milestone_records)
    awards_df = pd.DataFrame(award_records)
    available_milestones_df = pd.DataFrame(available_milestones)
    available_awards_df = pd.DataFrame(available_awards)
    
    print(f"Extracted {len(milestones_df)} milestone claims")
    print(f"Extracted {len(awards_df)} award fundings")
    
    return milestones_df, awards_df, available_milestones_df, available_awards_df

def analyze_milestones(
    milestones_df: pd.DataFrame,
    available_milestones_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame,
    cn_map: dict,
    min_games: int = MIN_GAMES_MILESTONE
) -> Dict:
    """
    Analyze milestones - frequency and win rate.
    """
    print("\n" + "=" * 60)
    print("里程碑分析 (突破环境)")
    print("=" * 60)
    
    if milestones_df is None or len(milestones_df) == 0:
        print("No milestone data")
        return {}
    
    results = {}
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    # Merge with user_game_results to get position and score
    milestones_merged = milestones_df.merge(
        ugr_end[['game_id', 'user_id', 'position', 'player_score']],
        on=['game_id', 'user_id'],
        how='left'
    )
    milestones_merged = milestones_merged.dropna(subset=['position'])
    
    # Analyze overall milestone frequency
    print("\n--- 里程碑宣称频率 (突破环境 - 全部游戏) ---")
    
    # Calculate availability counts
    available_counts = available_milestones_df.groupby('milestone_name').size().reset_index(name='available_count')
    
    # Calculate claim stats
    milestone_stats = milestones_merged.groupby('milestone_name').agg(
        claim_count=('game_id', 'count'),
        wins=('position', lambda x: (x == 1).sum()),
        avg_position=('position', 'mean'),
        position_sum=('position', 'sum'),
        avg_score=('player_score', 'mean'),
    ).reset_index()
    
    # Merge with availability
    milestone_stats = milestone_stats.merge(available_counts, on='milestone_name', how='left')
    milestone_stats['claim_rate'] = (milestone_stats['claim_count'] / milestone_stats['available_count'] * 100).round(2)
    
    # Calculate win rate and Bayesian weighted avg
    milestone_stats['win_rate'] = (milestone_stats['wins'] / milestone_stats['claim_count'] * 100).round(2)
    
    # Bayesian weighted average position (prior = 2.5 for 4P assumed)
    milestone_stats['weighted_avg_position'] = milestone_stats.apply(
        lambda r: bayesian_avg_position(r['position_sum'], r['claim_count'], DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_N),
        axis=1
    )
    
    # Filter by minimum games
    milestone_stats_filtered = milestone_stats[milestone_stats['claim_count'] >= min_games].copy()
    milestone_stats_filtered = add_cn_name_column(milestone_stats_filtered, 'milestone_name', cn_map)
    milestone_stats_filtered = milestone_stats_filtered.sort_values('weighted_avg_position', ascending=True)
    
    results['milestone_stats'] = milestone_stats_filtered
    
    print(milestone_stats_filtered[['milestone_name', 'cn_name', 'claim_count', 'claim_rate', 'win_rate', 'weighted_avg_position', 'avg_score']].to_string(index=False))
    
    # Save to CSV
    milestone_stats_filtered.to_csv(f'{DISPLAY_DIR}/milestone_stats.csv', index=False)
    print(f"\nSaved: {DISPLAY_DIR}/milestone_stats.csv")
    
    # Analysis by player count (2P and 4P)
    for player_count in [2, 4]:
        player_milestones = milestones_merged[milestones_merged['players'] == player_count]
        if len(player_milestones) < min_games:
            continue
            
        print(f"\n--- {player_count}人局里程碑统计 (突破环境) ---")
        
        prior_mean = get_prior_mean(player_count)
        
        ms_player_stats = player_milestones.groupby('milestone_name').agg(
            claim_count=('game_id', 'count'),
            wins=('position', lambda x: (x == 1).sum()),
            avg_position=('position', 'mean'),
            position_sum=('position', 'sum'),
        ).reset_index()
        
        ms_player_stats['win_rate'] = (ms_player_stats['wins'] / ms_player_stats['claim_count'] * 100).round(2)
        ms_player_stats['weighted_avg_position'] = ms_player_stats.apply(
            lambda r: bayesian_avg_position(r['position_sum'], r['claim_count'], prior_mean, DEFAULT_PRIOR_N),
            axis=1
        )
        
        ms_player_stats = ms_player_stats[ms_player_stats['claim_count'] >= min_games]
        ms_player_stats = add_cn_name_column(ms_player_stats, 'milestone_name', cn_map)
        ms_player_stats = ms_player_stats.sort_values('weighted_avg_position', ascending=True)
        
        results[f'milestone_{player_count}p_stats'] = ms_player_stats
        
        if len(ms_player_stats) > 0:
            print(ms_player_stats[['milestone_name', 'cn_name', 'claim_count', 'win_rate', 'weighted_avg_position']].to_string(index=False))
            ms_player_stats.to_csv(f'{DISPLAY_DIR}/milestone_{player_count}p_stats.csv', index=False)
    
    # Analysis by claim order
    print("\n--- 按里程碑宣称顺序统计胜率 (突破环境) ---")
    order_stats = milestones_merged.groupby('claim_order').agg(
        count=('game_id', 'count'),
        wins=('position', lambda x: (x == 1).sum()),
        avg_position=('position', 'mean'),
    ).reset_index()
    order_stats['win_rate'] = (order_stats['wins'] / order_stats['count'] * 100).round(2)
    print(order_stats.to_string(index=False))
    results['milestone_order_stats'] = order_stats
    
    # Analysis by number of milestones claimed per player
    print("\n--- 宣称里程碑数量与胜率关系 (突破环境) ---")
    milestones_per_player = milestones_merged.groupby(['game_id', 'user_id', 'position', 'players']).size().reset_index(name='milestone_count')
    
    for player_count in [2, 4]:
        player_ms = milestones_per_player[milestones_per_player['players'] == player_count]
        if len(player_ms) == 0:
            continue
        
        print(f"\n  {player_count}人局 - 里程碑数量与胜率:")
        ms_count_stats = player_ms.groupby('milestone_count').agg(
            player_count_stat=('user_id', 'count'),
            wins=('position', lambda x: (x == 1).sum()),
            avg_position=('position', 'mean'),
        ).reset_index()
        ms_count_stats['win_rate'] = (ms_count_stats['wins'] / ms_count_stats['player_count_stat'] * 100).round(2)
        print(f"    里程碑数  玩家数   胜利数  胜率%   平均顺位")
        for _, row in ms_count_stats.iterrows():
            print(f"    {int(row['milestone_count']):^8}  {int(row['player_count_stat']):^6}  {int(row['wins']):^6}  {row['win_rate']:>5.1f}%  {row['avg_position']:.2f}")
        results[f'milestone_count_{player_count}p'] = ms_count_stats
    
    return results

def analyze_awards(
    awards_df: pd.DataFrame,
    available_awards_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame,
    cn_map: dict,
    min_games: int = MIN_GAMES_AWARD
) -> Dict:
    """
    Analyze awards - frequency and win rate for funders.
    """
    print("\n" + "=" * 60)
    print("奖励分析 (突破环境)")
    print("=" * 60)
    
    if awards_df is None or len(awards_df) == 0:
        print("No award data")
        return {}
    
    results = {}
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    # Merge with user_game_results to get position and score
    awards_merged = awards_df.merge(
        ugr_end[['game_id', 'user_id', 'position', 'player_score']],
        on=['game_id', 'user_id'],
        how='left'
    )
    awards_merged = awards_merged.dropna(subset=['position'])
    
    print("\n--- 奖励资助频率 (突破环境 - 全部游戏) ---")
    
    # Calculate availability counts
    available_counts = available_awards_df.groupby('award_name').size().reset_index(name='available_count')
    
    # Calculate fund stats
    award_stats = awards_merged.groupby('award_name').agg(
        fund_count=('game_id', 'count'),
        wins=('position', lambda x: (x == 1).sum()),
        avg_position=('position', 'mean'),
        position_sum=('position', 'sum'),
        avg_score=('player_score', 'mean'),
    ).reset_index()
    
    # Merge with availability
    award_stats = award_stats.merge(available_counts, on='award_name', how='left')
    award_stats['fund_rate'] = (award_stats['fund_count'] / award_stats['available_count'] * 100).round(2)
    
    # Calculate win rate and Bayesian weighted avg
    award_stats['win_rate'] = (award_stats['wins'] / award_stats['fund_count'] * 100).round(2)
    award_stats['weighted_avg_position'] = award_stats.apply(
        lambda r: bayesian_avg_position(r['position_sum'], r['fund_count'], DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_N),
        axis=1
    )
    
    # Filter by minimum games
    award_stats_filtered = award_stats[award_stats['fund_count'] >= min_games].copy()
    award_stats_filtered = add_cn_name_column(award_stats_filtered, 'award_name', cn_map)
    award_stats_filtered = award_stats_filtered.sort_values('weighted_avg_position', ascending=True)
    
    results['award_stats'] = award_stats_filtered
    
    print(award_stats_filtered[['award_name', 'cn_name', 'fund_count', 'fund_rate', 'win_rate', 'weighted_avg_position', 'avg_score']].to_string(index=False))
    
    # Save to CSV
    award_stats_filtered.to_csv(f'{DISPLAY_DIR}/award_stats.csv', index=False)
    print(f"\nSaved: {DISPLAY_DIR}/award_stats.csv")
    
    # Analysis by player count (2P and 4P)
    for player_count in [2, 4]:
        player_awards = awards_merged[awards_merged['players'] == player_count]
        if len(player_awards) < min_games:
            continue
            
        print(f"\n--- {player_count}人局奖励统计 (突破环境) ---")
        
        prior_mean = get_prior_mean(player_count)
        
        aw_player_stats = player_awards.groupby('award_name').agg(
            fund_count=('game_id', 'count'),
            wins=('position', lambda x: (x == 1).sum()),
            avg_position=('position', 'mean'),
            position_sum=('position', 'sum'),
        ).reset_index()
        
        aw_player_stats['win_rate'] = (aw_player_stats['wins'] / aw_player_stats['fund_count'] * 100).round(2)
        aw_player_stats['weighted_avg_position'] = aw_player_stats.apply(
            lambda r: bayesian_avg_position(r['position_sum'], r['fund_count'], prior_mean, DEFAULT_PRIOR_N),
            axis=1
        )
        
        aw_player_stats = aw_player_stats[aw_player_stats['fund_count'] >= min_games]
        aw_player_stats = add_cn_name_column(aw_player_stats, 'award_name', cn_map)
        aw_player_stats = aw_player_stats.sort_values('weighted_avg_position', ascending=True)
        
        results[f'award_{player_count}p_stats'] = aw_player_stats
        
        if len(aw_player_stats) > 0:
            print(aw_player_stats[['award_name', 'cn_name', 'fund_count', 'win_rate', 'weighted_avg_position']].to_string(index=False))
            aw_player_stats.to_csv(f'{DISPLAY_DIR}/award_{player_count}p_stats.csv', index=False)
    
    # Analysis by fund order
    print("\n--- 按奖励资助顺序统计胜率 (突破环境) ---")
    order_stats = awards_merged.groupby('fund_order').agg(
        count=('game_id', 'count'),
        wins=('position', lambda x: (x == 1).sum()),
        avg_position=('position', 'mean'),
    ).reset_index()
    order_stats['win_rate'] = (order_stats['wins'] / order_stats['count'] * 100).round(2)
    print(order_stats.to_string(index=False))
    results['award_order_stats'] = order_stats
    
    return results

# =============================================================================
# Part 3: Colony Analysis
# =============================================================================
def extract_tr_data(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract terraform rating data from games.game JSON.
    
    Returns DataFrame with TR, position, players, generations for each player.
    """
    tr_records = []
    
    for _, row in games_df.iterrows():
        game_data = parse_json_safe(row['game'])
        if not game_data:
            continue
            
        game_id = row['game_id']
        createtime = row['createtime']
        generation = game_data.get('generation', 0)
        
        for player in game_data.get('players', []):
            user_id = player.get('userId')
            tr = player.get('terraformRating', 0)
            
            tr_records.append({
                'game_id': game_id,
                'createtime': createtime,
                'user_id': user_id,
                'terraform_rating': tr,
                'generations': generation,
                'players': len(game_data.get('players', [])),
            })
    
    tr_df = pd.DataFrame(tr_records)
    print(f"Extracted {len(tr_df)} TR records")
    
    return tr_df

def analyze_tr_data(
    tr_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame
) -> Dict:
    """
    Analyze terraform rating vs win rate and TR distribution by generation.
    """
    print("\n" + "=" * 60)
    print("改造度(TR)分析 (突破环境)")
    print("=" * 60)
    
    if tr_df is None or len(tr_df) == 0:
        print("No TR data")
        return {}
    
    results = {}
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    # Merge with user_game_results to get position
    tr_merged = tr_df.merge(
        ugr_end[['game_id', 'user_id', 'position', 'player_score']],
        on=['game_id', 'user_id'],
        how='left'
    )
    tr_merged = tr_merged.dropna(subset=['position'])
    
    # TR vs Win Rate analysis
    print("\n--- TR与胜率关系 (突破环境) ---")
    
    for player_count in [2, 4]:
        player_tr = tr_merged[tr_merged['players'] == player_count]
        if len(player_tr) == 0:
            continue
        
        print(f"\n  {player_count}人局 - TR分布与胜率:")
        
        # Create TR bins
        if player_count == 2:
            bins = [0, 30, 35, 40, 45, 50, 55, 60, 100]
            labels = ['<30', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60+']
        else:
            bins = [0, 25, 30, 35, 40, 45, 50, 100]
            labels = ['<25', '25-29', '30-34', '35-39', '40-44', '45-49', '50+']
        
        player_tr['tr_bin'] = pd.cut(player_tr['terraform_rating'], bins=bins, labels=labels, right=False)
        
        tr_stats = player_tr.groupby('tr_bin', observed=True).agg(
            count=('user_id', 'count'),
            wins=('position', lambda x: (x == 1).sum()),
            avg_position=('position', 'mean'),
            avg_score=('player_score', 'mean'),
        ).reset_index()
        tr_stats['win_rate'] = (tr_stats['wins'] / tr_stats['count'] * 100).round(2)
        
        print(f"    TR范围    玩家数   胜利数   胜率%   平均顺位  平均分数")
        for _, row in tr_stats.iterrows():
            print(f"    {str(row['tr_bin']):^8}  {int(row['count']):^6}  {int(row['wins']):^6}  {row['win_rate']:>5.1f}%  {row['avg_position']:.2f}    {row['avg_score']:.1f}")
        results[f'tr_stats_{player_count}p'] = tr_stats
    
    # TR distribution by generation
    print("\n--- 不同时代结束游戏的TR分布 (突破环境) ---")
    
    for player_count in [2, 4]:
        player_tr = tr_merged[tr_merged['players'] == player_count]
        if len(player_tr) == 0:
            continue
        
        print(f"\n  {player_count}人局 - 各时代TR统计:")
        
        gen_tr_stats = player_tr.groupby('generations').agg(
            game_count=('game_id', 'nunique'),
            player_count_stat=('user_id', 'count'),
            avg_tr=('terraform_rating', 'mean'),
            min_tr=('terraform_rating', 'min'),
            max_tr=('terraform_rating', 'max'),
            std_tr=('terraform_rating', 'std'),
            winner_avg_tr=('terraform_rating', lambda x: player_tr.loc[x.index][player_tr.loc[x.index]['position'] == 1]['terraform_rating'].mean())
        ).reset_index()
        gen_tr_stats = gen_tr_stats.round(2)
        
        # Only show generations with significant data
        gen_tr_stats = gen_tr_stats[gen_tr_stats['game_count'] >= 5]
        
        print(f"    时代  游戏数  平均TR  最小TR  最大TR  标准差  冠军平均TR")
        for _, row in gen_tr_stats.iterrows():
            print(f"    {int(row['generations']):^4}  {int(row['game_count']):^6}  {row['avg_tr']:>6.1f}  {int(row['min_tr']):^6}  {int(row['max_tr']):^6}  {row['std_tr']:>5.1f}  {row['winner_avg_tr']:.1f}")
        
        results[f'gen_tr_stats_{player_count}p'] = gen_tr_stats
        
        # Save to CSV
        gen_tr_stats.to_csv(f'{DISPLAY_DIR}/tr_by_generation_{player_count}p.csv', index=False)
        print(f"    Saved: {DISPLAY_DIR}/tr_by_generation_{player_count}p.csv")
    
    return results

def extract_colony_data(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract colony settlement data from games.game JSON.
    
    Returns DataFrame with one row per player-colony settlement.
    """
    colony_records = []
    
    for _, row in games_df.iterrows():
        game_data = parse_json_safe(row['game'])
        if not game_data:
            continue
            
        game_id = row['game_id']
        createtime = row['createtime']
        player_count = len(game_data.get('players', []))
        
        # Skip games without colonies
        colonies = game_data.get('colonies', [])
        if not colonies:
            continue
        
        # Build player_id to user_id mapping
        player_map = {}
        for player in game_data.get('players', []):
            player_map[player.get('id')] = player.get('userId')
        
        # Extract colony settlements
        for colony in colonies:
            colony_name = colony.get('name')
            settlers = colony.get('colonies', [])  # List of player objects who settled
            
            for settler in settlers:
                player_id = settler.get('id')
                colony_records.append({
                    'game_id': game_id,
                    'createtime': createtime,
                    'colony_name': colony_name,
                    'player_id': player_id,
                    'user_id': player_map.get(player_id),
                    'players': player_count,
                })
    
    colony_df = pd.DataFrame(colony_records)
    print(f"Extracted {len(colony_df)} colony settlement records")
    
    return colony_df

def analyze_colonies(
    colony_df: pd.DataFrame,
    user_game_results_df: pd.DataFrame,
    cn_map: dict,
    min_games: int = MIN_GAMES_COLONY
) -> Dict:
    """
    Analyze colonies - settlement frequency and win rate.
    """
    print("\n" + "=" * 60)
    print("殖民地分析 (突破环境)")
    print("=" * 60)
    
    if colony_df is None or len(colony_df) == 0:
        print("No colony data")
        return {}
    
    results = {}
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    
    # Merge with user_game_results to get position and score
    colonies_merged = colony_df.merge(
        ugr_end[['game_id', 'user_id', 'position', 'player_score']],
        on=['game_id', 'user_id'],
        how='left'
    )
    colonies_merged = colonies_merged.dropna(subset=['position'])
    
    # Count games with colonies enabled
    games_with_colonies = colony_df['game_id'].nunique()
    print(f"\n包含殖民地扩展的游戏数: {games_with_colonies}")
    
    # Overall colony statistics
    print("\n--- 殖民地定居统计 (突破环境 - 全部游戏) ---")
    
    colony_stats = colonies_merged.groupby('colony_name').agg(
        settlement_count=('game_id', 'count'),
        unique_games=('game_id', 'nunique'),
        wins=('position', lambda x: (x == 1).sum()),
        avg_position=('position', 'mean'),
        position_sum=('position', 'sum'),
        avg_score=('player_score', 'mean'),
    ).reset_index()
    
    # Calculate average settlements per game
    colony_stats['avg_settlements_per_game'] = (colony_stats['settlement_count'] / colony_stats['unique_games']).round(2)
    
    # Calculate win rate and Bayesian weighted avg
    colony_stats['win_rate'] = (colony_stats['wins'] / colony_stats['settlement_count'] * 100).round(2)
    colony_stats['weighted_avg_position'] = colony_stats.apply(
        lambda r: bayesian_avg_position(r['position_sum'], r['settlement_count'], DEFAULT_PRIOR_MEAN, DEFAULT_PRIOR_N),
        axis=1
    )
    
    # Filter by minimum games
    colony_stats_filtered = colony_stats[colony_stats['settlement_count'] >= min_games].copy()
    colony_stats_filtered = add_cn_name_column(colony_stats_filtered, 'colony_name', cn_map)
    colony_stats_filtered = colony_stats_filtered.sort_values('weighted_avg_position', ascending=True)
    
    results['colony_stats'] = colony_stats_filtered
    
    print(colony_stats_filtered[['colony_name', 'cn_name', 'settlement_count', 'avg_settlements_per_game', 'win_rate', 'weighted_avg_position', 'avg_score']].to_string(index=False))
    
    # Save to CSV
    colony_stats_filtered.to_csv(f'{DISPLAY_DIR}/colony_stats.csv', index=False)
    print(f"\nSaved: {DISPLAY_DIR}/colony_stats.csv")
    
    # Analysis by player count (2P and 4P)
    for player_count in [2, 4]:
        player_colonies = colonies_merged[colonies_merged['players'] == player_count]
        if len(player_colonies) < min_games:
            continue
            
        print(f"\n--- {player_count}人局殖民地统计 (突破环境) ---")
        
        prior_mean = get_prior_mean(player_count)
        
        col_player_stats = player_colonies.groupby('colony_name').agg(
            settlement_count=('game_id', 'count'),
            unique_games=('game_id', 'nunique'),
            wins=('position', lambda x: (x == 1).sum()),
            avg_position=('position', 'mean'),
            position_sum=('position', 'sum'),
        ).reset_index()
        
        col_player_stats['avg_settlements_per_game'] = (col_player_stats['settlement_count'] / col_player_stats['unique_games']).round(2)
        col_player_stats['win_rate'] = (col_player_stats['wins'] / col_player_stats['settlement_count'] * 100).round(2)
        col_player_stats['weighted_avg_position'] = col_player_stats.apply(
            lambda r: bayesian_avg_position(r['position_sum'], r['settlement_count'], prior_mean, DEFAULT_PRIOR_N),
            axis=1
        )
        
        col_player_stats = col_player_stats[col_player_stats['settlement_count'] >= min_games]
        col_player_stats = add_cn_name_column(col_player_stats, 'colony_name', cn_map)
        col_player_stats = col_player_stats.sort_values('weighted_avg_position', ascending=True)
        
        results[f'colony_{player_count}p_stats'] = col_player_stats
        
        if len(col_player_stats) > 0:
            print(col_player_stats[['colony_name', 'cn_name', 'settlement_count', 'avg_settlements_per_game', 'win_rate', 'weighted_avg_position']].to_string(index=False))
            col_player_stats.to_csv(f'{DISPLAY_DIR}/colony_{player_count}p_stats.csv', index=False)
    
    return results

# =============================================================================
# Visualization Functions
# =============================================================================
def plot_milestone_award_ranking(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    name_col: str = 'cn_name_plot',
    score_col: str = 'weighted_avg_position',
    count_col: str = 'claim_count',
    expected_line: float = None,
    figsize: tuple = (12, 8)
):
    """Plot ranking bar chart for milestones/awards"""
    if df is None or len(df) == 0:
        print(f"No data to plot for {title}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color based on score (green for low position, red for high)
    max_score = df[score_col].max()
    min_score = df[score_col].min()
    score_range = max_score - min_score if max_score > min_score else 1
    colors = plt.cm.RdYlGn(np.clip((max_score - df[score_col]) / score_range, 0, 1))
    
    bars = ax.barh(range(len(df)), df[score_col], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df[name_col], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('平均顺位 (Weighted)')
    
    if expected_line is not None:
        ax.axvline(x=expected_line, color='red', linestyle='--', alpha=0.5,
                   label=f'期望 {expected_line}')
        ax.legend()
    
    ax.set_title(title)
    
    # Add labels
    for bar, score, count in zip(bars, df[score_col], df[count_col]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f} (n={count})', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_generation_distribution(gen_df: pd.DataFrame, player_count: int):
    """Plot generation distribution for a specific player count"""
    if gen_df is None or len(gen_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(gen_df['generations'].astype(str), gen_df['game_count'], color='steelblue')
    ax.set_xlabel('时代数')
    ax.set_ylabel('游戏数量')
    ax.set_title(f'{player_count}人局时代数分布 (突破环境)')
    
    # Add percentage labels
    for bar, pct in zip(bars, gen_df['percentage']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 5, f'{pct:.1f}%', 
                ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{DISPLAY_DIR}/generation_distribution_{player_count}p.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {DISPLAY_DIR}/generation_distribution_{player_count}p.png")

def plot_score_by_generation(score_df: pd.DataFrame, player_count: int):
    """Plot score statistics by generation"""
    if score_df is None or len(score_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(score_df))
    width = 0.35
    
    bars_avg = ax.bar([i - width/2 for i in x], score_df['avg_score'], width, label='平均分数', color='steelblue')
    bars_max = ax.bar([i + width/2 for i in x], score_df['max_score'], width, label='最高分数', color='coral', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(score_df['generations'].astype(str))
    ax.set_xlabel('时代数')
    ax.set_ylabel('分数')
    ax.set_title(f'{player_count}人局按时代数统计分数 (突破环境)')
    ax.legend()
    
    # Add value labels
    for i, (avg, max_val) in enumerate(zip(score_df['avg_score'], score_df['max_score'])):
        ax.text(i - width/2, avg + 1, f'{avg:.1f}', ha='center', fontsize=7)
        ax.text(i + width/2, max_val + 1, f'{int(max_val)}', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{DISPLAY_DIR}/score_by_generation_{player_count}p.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {DISPLAY_DIR}/score_by_generation_{player_count}p.png")

def plot_tr_by_generation(tr_stats: pd.DataFrame, player_count: int):
    """Plot TR statistics by generation"""
    if tr_stats is None or len(tr_stats) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(tr_stats))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], tr_stats['avg_tr'], width, label='平均TR', color='steelblue')
    ax.bar([i + width/2 for i in x], tr_stats['winner_avg_tr'], width, label='冠军平均TR', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(tr_stats['generations'].astype(int).astype(str))
    ax.set_xlabel('时代数')
    ax.set_ylabel('TR (改造度)')
    ax.set_title(f'{player_count}人局各时代TR分布 (突破环境)')
    ax.legend()
    
    # Add value labels
    for i, (avg, winner) in enumerate(zip(tr_stats['avg_tr'], tr_stats['winner_avg_tr'])):
        ax.text(i - width/2, avg + 0.5, f'{avg:.1f}', ha='center', fontsize=7)
        ax.text(i + width/2, winner + 0.5, f'{winner:.1f}', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{DISPLAY_DIR}/tr_by_generation_{player_count}p.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {DISPLAY_DIR}/tr_by_generation_{player_count}p.png")

def plot_game_stats_charts(stats_by_players: pd.DataFrame):
    """Plot game statistics charts"""
    if stats_by_players is None or len(stats_by_players) == 0:
        print("No game stats to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('游戏基础统计 (突破环境)', fontsize=14, fontweight='bold')
    
    # 1. Game count by player count
    ax = axes[0, 0]
    ax.bar(stats_by_players['players'].astype(str), stats_by_players['game_count'], color='steelblue')
    ax.set_xlabel('玩家人数')
    ax.set_ylabel('游戏数量')
    ax.set_title('游戏数量分布 (突破环境)')
    for i, v in enumerate(stats_by_players['game_count']):
        ax.text(i, v + 50, str(v), ha='center', fontsize=9)
    
    # 2. Average generations by player count
    ax = axes[0, 1]
    ax.bar(stats_by_players['players'].astype(str), stats_by_players['avg_generations'], 
           color='forestgreen', yerr=stats_by_players['std_generations'], capsize=5)
    ax.set_xlabel('玩家人数')
    ax.set_ylabel('平均时代数')
    ax.set_title('平均时代数 (突破环境)')
    for i, (avg, std) in enumerate(zip(stats_by_players['avg_generations'], stats_by_players['std_generations'])):
        ax.text(i, avg + std + 0.2, f'{avg:.1f}', ha='center', fontsize=9)
    
    # 3. Average score by player count
    ax = axes[1, 0]
    ax.bar(stats_by_players['players'].astype(str), stats_by_players['avg_score'], 
           color='coral', yerr=stats_by_players['std_score'], capsize=5)
    ax.set_xlabel('玩家人数')
    ax.set_ylabel('平均分数')
    ax.set_title('平均玩家分数 (突破环境)')
    for i, (avg, std) in enumerate(zip(stats_by_players['avg_score'], stats_by_players['std_score'])):
        ax.text(i, avg + std + 2, f'{avg:.0f}', ha='center', fontsize=9)
    
    # 4. Score range by player count
    ax = axes[1, 1]
    x = range(len(stats_by_players))
    width = 0.35
    ax.bar([i - width/2 for i in x], stats_by_players['min_score'], width, label='最低分', color='lightcoral')
    ax.bar([i + width/2 for i in x], stats_by_players['max_score'], width, label='最高分', color='lightgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_by_players['players'].astype(str))
    ax.set_xlabel('玩家人数')
    ax.set_ylabel('分数')
    ax.set_title('分数范围 (突破环境)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{DISPLAY_DIR}/game_basic_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {DISPLAY_DIR}/game_basic_stats.png")

def create_visualizations(results: Dict, cn_map: dict):
    """Create all visualizations"""
    print("\n" + "=" * 60)
    print("生成可视化图表 (突破环境)")
    print("=" * 60)
    
    # Game stats charts
    if 'stats_by_players' in results:
        plot_game_stats_charts(results['stats_by_players'])
    
    # Generation distribution charts for 2P and 4P
    for player_count in [2, 4]:
        key = f'{player_count}p_generation_distribution'
        if key in results:
            plot_generation_distribution(results[key], player_count)
        
        score_key = f'{player_count}p_score_by_generation'
        if score_key in results:
            plot_score_by_generation(results[score_key], player_count)
    
    # Milestone charts
    if 'milestone_stats' in results:
        milestone_df = results['milestone_stats']
        if len(milestone_df) > 0:
            plot_milestone_award_ranking(
                milestone_df,
                '里程碑平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/milestone_weighted_ranking.png',
                count_col='claim_count',
                expected_line=DEFAULT_PRIOR_MEAN
            )
    
    if 'milestone_4p_stats' in results:
        ms_4p = results['milestone_4p_stats']
        if len(ms_4p) > 0:
            plot_milestone_award_ranking(
                ms_4p,
                '4人局里程碑平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/milestone_4p_weighted_ranking.png',
                count_col='claim_count',
                expected_line=2.5
            )
    
    if 'milestone_2p_stats' in results:
        ms_2p = results['milestone_2p_stats']
        if len(ms_2p) > 0:
            plot_milestone_award_ranking(
                ms_2p,
                '2人局里程碑平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/milestone_2p_weighted_ranking.png',
                count_col='claim_count',
                expected_line=1.5
            )
    
    # Award charts
    if 'award_stats' in results:
        award_df = results['award_stats']
        if len(award_df) > 0:
            plot_milestone_award_ranking(
                award_df,
                '奖励平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/award_weighted_ranking.png',
                count_col='fund_count',
                expected_line=DEFAULT_PRIOR_MEAN
            )
    
    if 'award_4p_stats' in results:
        aw_4p = results['award_4p_stats']
        if len(aw_4p) > 0:
            plot_milestone_award_ranking(
                aw_4p,
                '4人局奖励平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/award_4p_weighted_ranking.png',
                count_col='fund_count',
                expected_line=2.5
            )
    
    if 'award_2p_stats' in results:
        aw_2p = results['award_2p_stats']
        if len(aw_2p) > 0:
            plot_milestone_award_ranking(
                aw_2p,
                '2人局奖励平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/award_2p_weighted_ranking.png',
                count_col='fund_count',
                expected_line=1.5
            )
    
    # Colony charts
    if 'colony_stats' in results:
        colony_df = results['colony_stats']
        if len(colony_df) > 0:
            plot_milestone_award_ranking(
                colony_df,
                '殖民地平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/colony_weighted_ranking.png',
                count_col='settlement_count',
                expected_line=DEFAULT_PRIOR_MEAN
            )
    
    if 'colony_4p_stats' in results:
        col_4p = results['colony_4p_stats']
        if len(col_4p) > 0:
            plot_milestone_award_ranking(
                col_4p,
                '4人局殖民地平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/colony_4p_weighted_ranking.png',
                count_col='settlement_count',
                expected_line=2.5
            )
    
    if 'colony_2p_stats' in results:
        col_2p = results['colony_2p_stats']
        if len(col_2p) > 0:
            plot_milestone_award_ranking(
                col_2p,
                '2人局殖民地平均顺位排行榜 (突破环境)',
                f'{DISPLAY_DIR}/colony_2p_weighted_ranking.png',
                count_col='settlement_count',
                expected_line=1.5
            )
    
    # TR by generation charts
    for player_count in [2, 4]:
        key = f'gen_tr_stats_{player_count}p'
        if key in results:
            plot_tr_by_generation(results[key], player_count)

# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point"""
    # Setup
    setup_matplotlib()
    Path(DISPLAY_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load Chinese name mapping
    cn_map = load_cn_map(CN_MAP_FILE)
    
    # Load data
    print("Loading data from local SQLite...")
    games_df, game_results_df, user_game_results_df = load_data_from_local()
    
    if game_results_df is None:
        print("Error: Local data not found. Please run preprocess.py first.")
        return
    
    print(f"\nLoaded data successfully:")
    print(f"  - game_results: {len(game_results_df)} rows")
    print(f"  - user_game_results: {len(user_game_results_df)} rows")
    if games_df is not None:
        print(f"  - games: {len(games_df)} rows")
    
    # Collect all results
    all_results = {}
    
    # Part 1: Game Basic Statistics
    game_stats = analyze_game_basic_stats(game_results_df, user_game_results_df, games_df)
    all_results.update(game_stats)
    
    # Save basic stats to CSV
    if 'stats_by_players' in game_stats:
        game_stats['stats_by_players'].to_csv(f'{DISPLAY_DIR}/game_stats_by_players.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/game_stats_by_players.csv")
    
    # Part 2: Milestones and Awards
    if games_df is not None and len(games_df) > 0:
        milestones_df, awards_df, available_milestones_df, available_awards_df = extract_milestones_awards(games_df)
        
        milestone_results = analyze_milestones(
            milestones_df, available_milestones_df, user_game_results_df, cn_map
        )
        all_results.update(milestone_results)
        
        award_results = analyze_awards(
            awards_df, available_awards_df, user_game_results_df, cn_map
        )
        all_results.update(award_results)
        
        # Part 3: Colonies
        colony_df = extract_colony_data(games_df)
        colony_results = analyze_colonies(colony_df, user_game_results_df, cn_map)
        all_results.update(colony_results)
        
        # Part 4: TR Analysis
        tr_df = extract_tr_data(games_df)
        tr_results = analyze_tr_data(tr_df, user_game_results_df)
        all_results.update(tr_results)
    
    # Create visualizations
    create_visualizations(all_results, cn_map)
    
    print("\n" + "=" * 60)
    print("分析完成! (突破环境)")
    print("=" * 60)
    print(f"输出文件保存至: {DISPLAY_DIR}/")

if __name__ == '__main__':
    main()
