#!/usr/bin/env python3
"""
TFM Card Data Analysis Script
Converted from card.ipynb with improvements:
1. bayesian_avg_position now accepts configurable prior_mean and prior_n parameters
2. Card analysis separated into 2P and 4P with configurable min_games (100 for cards)
3. Code simplified with reusable functions
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

# Load environment variables
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
LOCAL_DB_PATH = './local_data/tfm.db'
CN_MAP_FILE = './data/cn_merged.json'
DISPLAY_DIR = './display'
FONT_PATH = './fonts/MapleMono-NF-CN-Regular.ttf'

# Analysis parameters - now configurable
DEFAULT_PRIOR_N = 30  # Default prior games for Bayesian smoothing
CARD_MIN_GAMES = 100  # Minimum games for card analysis (can be higher)
CORP_MIN_GAMES = 30   # Minimum games for corporation analysis
PRELUDE_MIN_GAMES = 30  # Minimum games for prelude analysis

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
# Bayesian Average Functions - Now with configurable parameters
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
    """
    Calculate Bayesian smoothed average position.

    Args:
        total_sum: Sum of positions
        count: Number of games
        prior_mean: Prior mean (default uses DEFAULT_PRIOR_MEAN)
        prior_n: Prior number of games for smoothing (default 30)

    Returns:
        Bayesian smoothed average position
    """
    if prior_mean is None:
        prior_mean = DEFAULT_PRIOR_MEAN

    if count is None or (isinstance(count, (int, float)) and count == 0):
        return prior_mean

    return (prior_n * prior_mean + total_sum) / (prior_n + count)

def add_bayesian_weighted_avg(
    df: pd.DataFrame,
    total_col: str = 'total_weighted',
    count_col: str = 'count_weighted',
    players_col: str = 'players',
    prior_mean: float = None,
    prior_n: int = DEFAULT_PRIOR_N
) -> pd.DataFrame:
    """
    Add weighted_avg_score column with Bayesian smoothing.

    Args:
        df: DataFrame with aggregated data
        total_col: Column name for sum of weighted scores
        count_col: Column name for count
        players_col: Column name for player count (used to determine prior_mean if not specified)
        prior_mean: Override prior mean (if None, uses player count based prior)
        prior_n: Prior number of games for smoothing
    """
    df = df.copy()

    if prior_mean is not None:
        # Use fixed prior_mean
        df['weighted_avg_score'] = df.apply(
            lambda r: bayesian_avg_position(r[total_col], r[count_col], prior_mean, prior_n),
            axis=1
        )
    elif players_col in df.columns:
        # Use player count based prior_mean
        df['weighted_avg_score'] = df.apply(
            lambda r: bayesian_avg_position(
                r[total_col], r[count_col],
                get_prior_mean(r[players_col]), prior_n
            ),
            axis=1
        )
    else:
        # Use default prior_mean
        df['weighted_avg_score'] = df.apply(
            lambda r: bayesian_avg_position(r[total_col], r[count_col], DEFAULT_PRIOR_MEAN, prior_n),
            axis=1
        )

    return df

def calc_weighted_score(position: int, players: int) -> float:
    """Calculate weighted score (simply returns position for aggregation)"""
    return float(position)

# =============================================================================
# Data Loading
# =============================================================================
def load_data_from_local():
    """Load data from local SQLite database"""
    if not Path(LOCAL_DB_PATH).exists():
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
        print(f"Filtered breakthrough=true games: {len(breakthrough_games)}")

    return games_df, game_results_df, user_game_results_df

# =============================================================================
# Common Analysis Functions
# =============================================================================
def split_corporations(corp_str: str) -> list:
    """Split corporation string into list"""
    if pd.isna(corp_str) or corp_str == '':
        return []
    return [c.strip() for c in str(corp_str).split('|') if c.strip()]

def is_prelude(card_name: str) -> bool:
    """Check if a card is a prelude (simplified check)"""
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
# Generic Stats Analysis Function
# =============================================================================
def analyze_stats_by_players(
    df: pd.DataFrame,
    name_col: str,
    min_games: int,
    players_filter: int = None,
    prior_mean: float = None,
    prior_n: int = DEFAULT_PRIOR_N,
    additional_agg: dict = None
) -> pd.DataFrame:
    """
    Generic function to analyze stats, optionally filtered by player count.

    Args:
        df: DataFrame with game data
        name_col: Column name for item (card_name, corporation, prelude_name)
        min_games: Minimum games threshold
        players_filter: Filter to specific player count (2 or 4), None for all
        prior_mean: Override prior mean for Bayesian smoothing
        prior_n: Prior games for Bayesian smoothing
        additional_agg: Additional aggregation columns
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    work_df = df.copy()

    # Filter by player count if specified
    if players_filter is not None:
        work_df = work_df[work_df['players'] == players_filter]
        if prior_mean is None:
            prior_mean = get_prior_mean(players_filter)

    if len(work_df) == 0:
        return pd.DataFrame()

    # Base aggregation
    agg_dict = {
        'total_games': ('game_id', 'count'),
        'wins': ('position', lambda x: (x == 1).sum()),
        'avg_position': ('position', 'mean'),
        'avg_score': ('player_score', 'mean'),
        'total_weighted': ('weighted_score', 'sum'),
        'count_weighted': ('weighted_score', 'count')
    }

    # Add additional aggregations if provided
    if additional_agg:
        agg_dict.update(additional_agg)

    stats = work_df.groupby(name_col).agg(**agg_dict).reset_index()
    stats = add_bayesian_weighted_avg(stats, prior_mean=prior_mean, prior_n=prior_n)

    # Calculate win rate
    stats['win_rate'] = (stats['wins'] / stats['total_games'] * 100).round(2)

    # Filter by minimum games
    stats = stats[stats['total_games'] >= min_games]

    return stats.sort_values('weighted_avg_score', ascending=True)

# =============================================================================
# Visualization Functions
# =============================================================================
def plot_weighted_ranking(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    top_n: int = 20,
    name_col: str = 'cn_name_plot',
    score_col: str = 'weighted_avg_score',
    count_col: str = 'total_games',
    expected_line: float = None,
    figsize: tuple = (12, 8)
):
    """
    Generic function to plot weighted ranking bar chart.

    Args:
        df: DataFrame with stats
        title: Chart title
        output_path: Path to save the figure
        top_n: Number of items to show
        name_col: Column for item names
        score_col: Column for scores
        count_col: Column for counts
        expected_line: Draw a vertical line at this position
        figsize: Figure size
    """
    if len(df) == 0:
        print(f"No data to plot for {title}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    top_df = df.head(top_n)

    # Color based on score (green for low, red for high)
    colors = plt.cm.RdYlGn(np.clip((4 - top_df[score_col]) / 3, 0, 1))

    bars = ax.barh(range(len(top_df)), top_df[score_col], color=colors)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df[name_col], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('平均顺位')
    ax.set_xlim(1, 4)

    if expected_line is not None:
        ax.axvline(x=expected_line, color='red', linestyle='--', alpha=0.5,
                   label=f'期望 {expected_line}')
        ax.legend()

    ax.set_title(title)

    # Add labels
    for bar, score, count in zip(bars, top_df[score_col], top_df[count_col]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{score:.2f} (n={count})', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_2p_vs_4p_scatter(
    df_2p: pd.DataFrame,
    df_4p: pd.DataFrame,
    name_col: str,
    title: str,
    output_path: str,
    cn_map: dict,
    min_games: int = 30
):
    """Plot scatter comparing 2P vs 4P weighted scores"""
    if len(df_2p) == 0 or len(df_4p) == 0:
        print(f"No data for 2P vs 4P scatter: {title}")
        return

    # Prepare 2P stats
    stats_2p = df_2p[[name_col, 'total_games', 'weighted_avg_score']].copy()
    stats_2p.columns = [name_col, 'games_2p', 'weighted_2p']

    # Prepare 4P stats
    stats_4p = df_4p[[name_col, 'total_games', 'weighted_avg_score']].copy()
    stats_4p.columns = [name_col, 'games_4p', 'weighted_4p']

    # Merge
    comparison = stats_2p.merge(stats_4p, on=name_col, how='inner')
    comparison = comparison[
        (comparison['games_2p'] >= min_games) &
        (comparison['games_4p'] >= min_games)
    ]
    comparison = add_cn_name_column(comparison, name_col, cn_map)

    if len(comparison) == 0:
        print(f"No overlapping data for 2P vs 4P scatter: {title}")
        return

    # Save comparison CSV
    csv_path = output_path.replace('.png', '.csv')
    comparison.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        comparison['weighted_2p'],
        comparison['weighted_4p'],
        c=comparison['games_2p'] + comparison['games_4p'],
        cmap='viridis',
        s=80,
        alpha=0.6
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label('Total Games (2P + 4P)')

    # Reference lines
    ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.3, label='4P Expected (2.5)')
    ax.axvline(x=1.5, color='blue', linestyle='--', alpha=0.3, label='2P Expected (1.5)')

    ax.set_xlabel('2P Weighted Avg Position')
    ax.set_ylabel('4P Weighted Avg Position')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
# Data Extraction Functions
# =============================================================================
def extract_corporation_data(user_game_results_df: pd.DataFrame) -> pd.DataFrame:
    """Extract corporation usage data from user_game_results"""
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'].copy()
    print(f"Completed game records: {len(ugr_end)}")

    corp_records = []
    for _, row in ugr_end.iterrows():
        corps = split_corporations(row['corporation'])
        for corp in corps:
            corp_records.append({
                'game_id': row['game_id'],
                'user_id': row['user_id'],
                'corporation': corp,
                'position': row['position'],
                'player_score': row['player_score'],
                'players': row['players'],
                'generations': row['generations'],
                'is_rank': row.get('is_rank', 0),
                'createtime': row['createtime']
            })

    corp_df = pd.DataFrame(corp_records)
    corp_df['weighted_score'] = corp_df.apply(
        lambda row: calc_weighted_score(row['position'], row['players']), axis=1
    )
    print(f"Corporation usage records: {len(corp_df)}")
    return corp_df

def extract_prelude_data(games_df: pd.DataFrame, user_game_results_df: pd.DataFrame) -> pd.DataFrame:
    """Extract prelude data from games JSON"""
    if games_df is None or len(games_df) == 0:
        print("games data not available, cannot extract prelude data")
        return pd.DataFrame()

    prelude_records = []
    for _, row in games_df.iterrows():
        try:
            game_data = row['game']
            if isinstance(game_data, str):
                game_data = json.loads(game_data)

            game_id = row['game_id']

            for player in game_data.get('players', []):
                user_id = player.get('userId', '')
                player_name = player.get('name', '')

                played_cards = player.get('playedCards', [])
                for card in played_cards:
                    card_name = card.get('name', '')
                    if is_prelude(card_name):
                        prelude_records.append({
                            'game_id': game_id,
                            'user_id': user_id,
                            'player_name': player_name,
                            'prelude_name': card_name
                        })
        except Exception:
            continue

    if not prelude_records:
        return pd.DataFrame()

    prelude_df = pd.DataFrame(prelude_records)

    # Merge with user_game_results
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'][
        ['game_id', 'user_id', 'position', 'player_score', 'players', 'generations']
    ]

    prelude_df = prelude_df.merge(ugr_end, on=['game_id', 'user_id'], how='left')
    prelude_df['weighted_score'] = prelude_df.apply(
        lambda row: calc_weighted_score(row['position'], row['players'])
        if pd.notna(row['position']) else np.nan, axis=1
    )

    # Drop rows with NaN position
    prelude_df = prelude_df.dropna(subset=['position'])

    print(f"Extracted prelude usage records: {len(prelude_df)}")
    return prelude_df

def extract_played_cards(games_df: pd.DataFrame, user_game_results_df: pd.DataFrame) -> pd.DataFrame:
    """Extract all played cards from games JSON"""
    if games_df is None or len(games_df) == 0:
        print("games data not available, cannot extract card data")
        return pd.DataFrame()

    card_records = []
    error_count = 0
    processed_count = 0

    for idx, row in games_df.iterrows():
        try:
            game_data = row['game']
            if game_data is None:
                continue
            if isinstance(game_data, str):
                game_data = json.loads(game_data)

            game_id = row['game_id']
            players = game_data.get('players', [])

            if not players:
                continue

            processed_count += 1

            for player in players:
                user_id = player.get('userId', '')
                player_name = player.get('name', '')
                corporations = player.get('corporations', [])
                # Handle corporations as list of dicts or list of strings
                corp_names = []
                for corp in corporations:
                    if isinstance(corp, dict):
                        corp_names.append(corp.get('name', ''))
                    else:
                        corp_names.append(str(corp))
                corporation = '|'.join(corp_names) if corp_names else ''
                terraform_rating = player.get('terraformRating', 0)

                played_cards = player.get('playedCards', [])
                for card in played_cards:
                    card_name = card.get('name', '')
                    if not card_name:
                        continue
                    resource_count = card.get('resourceCount', 0) or 0

                    card_records.append({
                        'game_id': game_id,
                        'user_id': user_id,
                        'player_name': player_name,
                        'card_name': card_name,
                        'resource_count': resource_count,
                        'terraform_rating': terraform_rating,
                        'corporation': corporation,
                        'is_prelude': is_prelude(card_name),
                        'is_corporation': is_corporation(card_name),
                        'is_ceo': is_ceo(card_name)
                    })
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Print first 5 errors
                print(f"Error processing game {idx}: {e}")
            continue

    print(f"Processed {processed_count} games, {error_count} errors, {len(card_records)} card records")

    if not card_records:
        return pd.DataFrame()

    card_df = pd.DataFrame(card_records)

    # Merge with user_game_results
    ugr_end = user_game_results_df[user_game_results_df['phase'] == 'end'][
        ['game_id', 'user_id', 'position', 'player_score', 'players', 'generations']
    ]

    card_df = card_df.merge(ugr_end, on=['game_id', 'user_id'], how='left')
    card_df['weighted_score'] = card_df.apply(
        lambda row: calc_weighted_score(row['position'], row['players'])
        if pd.notna(row['position']) else np.nan, axis=1
    )

    # Drop rows with NaN position
    card_df = card_df.dropna(subset=['position'])

    print(f"Extracted card usage records: {len(card_df)}")
    print(f"Unique cards: {card_df['card_name'].nunique()}")
    return card_df

# =============================================================================
# Main Analysis Functions
# =============================================================================
def analyze_corporations(corp_df: pd.DataFrame, cn_map: dict, min_games: int = CORP_MIN_GAMES):
    """Analyze corporation data and generate outputs"""
    if corp_df is None or len(corp_df) == 0:
        print("No corporation data to analyze")
        return

    print("\n" + "=" * 60)
    print("Corporation Analysis")
    print("=" * 60)

    # 4P analysis
    corp_4p_stats = analyze_stats_by_players(
        corp_df, 'corporation', min_games, players_filter=4,
        prior_mean=2.5,
        additional_agg={'avg_generations': ('generations', 'mean')}
    )
    if len(corp_4p_stats) > 0:
        corp_4p_stats = add_cn_name_column(corp_4p_stats, 'corporation', cn_map)
        corp_4p_stats.to_csv(f'{DISPLAY_DIR}/corporation_4p_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/corporation_4p_stats.csv ({len(corp_4p_stats)} rows)")

        plot_weighted_ranking(
            corp_4p_stats, '4人局公司平均顺位排行榜 Top 20',
            f'{DISPLAY_DIR}/corporation_4p_weighted_top20.png',
            expected_line=2.5
        )

    # 2P analysis
    corp_2p_stats = analyze_stats_by_players(
        corp_df, 'corporation', min_games, players_filter=2,
        prior_mean=1.5,
        additional_agg={'avg_generations': ('generations', 'mean')}
    )
    if len(corp_2p_stats) > 0:
        corp_2p_stats = add_cn_name_column(corp_2p_stats, 'corporation', cn_map)
        corp_2p_stats.to_csv(f'{DISPLAY_DIR}/corporation_2p_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/corporation_2p_stats.csv ({len(corp_2p_stats)} rows)")

        plot_weighted_ranking(
            corp_2p_stats, '2人局公司平均顺位排行榜 Top 20',
            f'{DISPLAY_DIR}/corporation_2p_weighted_top20.png',
            expected_line=1.5
        )

    # 2P vs 4P scatter
    if len(corp_2p_stats) > 0 and len(corp_4p_stats) > 0:
        plot_2p_vs_4p_scatter(
            corp_2p_stats, corp_4p_stats, 'corporation',
            'Corporation 2P vs 4P Weighted Score Comparison',
            f'{DISPLAY_DIR}/corporation_2p_vs_4p_scatter.png',
            cn_map, min_games
        )

def analyze_preludes(prelude_df: pd.DataFrame, cn_map: dict, min_games: int = PRELUDE_MIN_GAMES):
    """Analyze prelude data and generate outputs"""
    if prelude_df is None or len(prelude_df) == 0:
        print("No prelude data to analyze")
        return

    print("\n" + "=" * 60)
    print("Prelude Analysis")
    print("=" * 60)

    # 4P analysis
    prelude_4p_stats = analyze_stats_by_players(
        prelude_df, 'prelude_name', min_games, players_filter=4,
        prior_mean=2.5
    )
    if len(prelude_4p_stats) > 0:
        prelude_4p_stats = add_cn_name_column(prelude_4p_stats, 'prelude_name', cn_map)
        prelude_4p_stats.to_csv(f'{DISPLAY_DIR}/prelude_4p_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/prelude_4p_stats.csv ({len(prelude_4p_stats)} rows)")

        plot_weighted_ranking(
            prelude_4p_stats, '4人局前序卡平均顺位排行榜 Top 20',
            f'{DISPLAY_DIR}/prelude_4p_weighted_top20.png',
            expected_line=2.5
        )

    # 2P analysis
    prelude_2p_stats = analyze_stats_by_players(
        prelude_df, 'prelude_name', min_games, players_filter=2,
        prior_mean=1.5
    )
    if len(prelude_2p_stats) > 0:
        prelude_2p_stats = add_cn_name_column(prelude_2p_stats, 'prelude_name', cn_map)
        prelude_2p_stats.to_csv(f'{DISPLAY_DIR}/prelude_2p_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/prelude_2p_stats.csv ({len(prelude_2p_stats)} rows)")

        plot_weighted_ranking(
            prelude_2p_stats, '2人局前序卡平均顺位排行榜 Top 20',
            f'{DISPLAY_DIR}/prelude_2p_weighted_top20.png',
            expected_line=1.5
        )

    # 2P vs 4P scatter
    if len(prelude_2p_stats) > 0 and len(prelude_4p_stats) > 0:
        plot_2p_vs_4p_scatter(
            prelude_2p_stats, prelude_4p_stats, 'prelude_name',
            'Prelude 2P vs 4P Weighted Score Comparison',
            f'{DISPLAY_DIR}/prelude_2p_vs_4p_scatter.png',
            cn_map, min_games
        )

def analyze_cards(card_df: pd.DataFrame, cn_map: dict, min_games: int = CARD_MIN_GAMES):
    """Analyze card data and generate outputs"""
    if card_df is None or len(card_df) == 0:
        print("No card data to analyze")
        return

    print("\n" + "=" * 60)
    print(f"Card Analysis (min_games={min_games})")
    print("=" * 60)

    # Filter to project cards only (exclude prelude, corporation, CEO)
    project_cards_df = card_df[
        (~card_df['is_prelude']) &
        (~card_df['is_corporation']) &
        (~card_df['is_ceo'])
    ].copy()

    print(f"Project cards records: {len(project_cards_df)}")

    # All players analysis (use default prior_mean=2.5 for mixed player counts)
    card_all_stats = analyze_stats_by_players(
        project_cards_df, 'card_name', min_games, players_filter=None,
        prior_mean=DEFAULT_PRIOR_MEAN,
        additional_agg={'avg_resources': ('resource_count', 'mean')}
    )
    if len(card_all_stats) > 0:
        card_all_stats = add_cn_name_column(card_all_stats, 'card_name', cn_map)
        card_all_stats.to_csv(f'{DISPLAY_DIR}/card_all_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/card_all_stats.csv ({len(card_all_stats)} rows)")

        plot_weighted_ranking(
            card_all_stats, f'项目卡平均顺位排行榜 Top 30 (All Players, n>={min_games})',
            f'{DISPLAY_DIR}/card_weighted_all_top30.png',
            top_n=30,
            figsize=(14, 10)
        )

    # 4P analysis
    card_4p_stats = analyze_stats_by_players(
        project_cards_df, 'card_name', min_games, players_filter=4,
        prior_mean=2.5,
        additional_agg={'avg_resources': ('resource_count', 'mean')}
    )
    if len(card_4p_stats) > 0:
        card_4p_stats = add_cn_name_column(card_4p_stats, 'card_name', cn_map)
        card_4p_stats.to_csv(f'{DISPLAY_DIR}/card_4p_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/card_4p_stats.csv ({len(card_4p_stats)} rows)")

        plot_weighted_ranking(
            card_4p_stats, f'4人局项目卡平均顺位排行榜 Top 30 (n>={min_games})',
            f'{DISPLAY_DIR}/card_4p_weighted_top30.png',
            top_n=30,
            expected_line=2.5,
            figsize=(14, 10)
        )

    # 2P analysis
    card_2p_stats = analyze_stats_by_players(
        project_cards_df, 'card_name', min_games, players_filter=2,
        prior_mean=1.5,
        additional_agg={'avg_resources': ('resource_count', 'mean')}
    )
    if len(card_2p_stats) > 0:
        card_2p_stats = add_cn_name_column(card_2p_stats, 'card_name', cn_map)
        card_2p_stats.to_csv(f'{DISPLAY_DIR}/card_2p_stats.csv', index=False)
        print(f"Saved: {DISPLAY_DIR}/card_2p_stats.csv ({len(card_2p_stats)} rows)")

        plot_weighted_ranking(
            card_2p_stats, f'2人局项目卡平均顺位排行榜 Top 30 (n>={min_games})',
            f'{DISPLAY_DIR}/card_2p_weighted_top30.png',
            top_n=30,
            expected_line=1.5,
            figsize=(14, 10)
        )

    # 2P vs 4P scatter
    if len(card_2p_stats) > 0 and len(card_4p_stats) > 0:
        plot_2p_vs_4p_scatter(
            card_2p_stats, card_4p_stats, 'card_name',
            f'Card 2P vs 4P Weighted Score Comparison (n>={min_games})',
            f'{DISPLAY_DIR}/card_2p_vs_4p_scatter.png',
            cn_map, min_games
        )

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

    print(f"Loaded data successfully:")
    print(f"  - game_results: {len(game_results_df)} rows")
    print(f"  - user_game_results: {len(user_game_results_df)} rows")
    if games_df is not None:
        print(f"  - games: {len(games_df)} rows")

    # Add weighted_score to user_game_results
    user_game_results_df['weighted_score'] = user_game_results_df.apply(
        lambda row: calc_weighted_score(row['position'], row['players']), axis=1
    )

    # Extract and analyze corporations
    corp_df = extract_corporation_data(user_game_results_df)
    analyze_corporations(corp_df, cn_map, min_games=CORP_MIN_GAMES)

    # Extract and analyze preludes
    prelude_df = extract_prelude_data(games_df, user_game_results_df)
    analyze_preludes(prelude_df, cn_map, min_games=PRELUDE_MIN_GAMES)

    # Extract and analyze cards (with min_games=100)
    card_df = extract_played_cards(games_df, user_game_results_df)
    analyze_cards(card_df, cn_map, min_games=CARD_MIN_GAMES)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Outputs saved to: {DISPLAY_DIR}/")

if __name__ == '__main__':
    main()
