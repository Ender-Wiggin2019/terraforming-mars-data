#!/usr/bin/env python3
"""
Test script to verify user_aggregate.py calculations are correct.

Tests:
1. Individual user stats match CSV data
2. Combined stats are correctly weighted averages
3. Records by generation are correct max values
"""

import json
import pandas as pd
from pathlib import Path
import math

DISPLAY_DIR = './display'

def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_data(players_filter: int = 4) -> dict:
    """Load CSV data"""
    return {
        'player_stats': pd.read_csv(f'{DISPLAY_DIR}/user_player_stats_{players_filter}p.csv'),
        'records_by_generation': pd.read_csv(f'{DISPLAY_DIR}/user_records_by_generation_{players_filter}p.csv'),
        'corp_top100': pd.read_csv(f'{DISPLAY_DIR}/user_corp_top100_players_{players_filter}p.csv'),
    }

def test_individual_user(user_id: str, json_data: dict, csv_data: dict) -> bool:
    """Test that individual user stats match CSV"""
    print(f"\n{'='*60}")
    print(f"Testing individual user: {user_id}")
    print(f"{'='*60}")
    
    # Get user from CSV
    user_csv = csv_data['player_stats'][csv_data['player_stats']['user_id'] == user_id]
    
    if len(user_csv) == 0:
        print(f"ERROR: User {user_id} not found in CSV")
        return False
    
    user_csv = user_csv.iloc[0]
    json_stats = json_data['player_stats']
    
    errors = []
    
    # Test each field
    tests = [
        ('total_games', user_csv['total_games'], json_stats['total_games']),
        ('total_wins', user_csv['total_wins'], json_stats['total_wins']),
        ('win_rate', user_csv['win_rate'], json_stats['win_rate']),
        ('avg_position', user_csv['avg_position'], json_stats['avg_position']),
        ('avg_score', user_csv['avg_score'], json_stats['avg_score']),
        ('avg_generations', user_csv['avg_generations'], json_stats['avg_generations']),
        ('avg_tr', user_csv['avg_tr'], json_stats['avg_tr']),
        ('avg_cards_played', user_csv['avg_cards_played'], json_stats['avg_cards_played']),
    ]
    
    all_passed = True
    for field, csv_val, json_val in tests:
        # Allow small floating point differences
        if isinstance(csv_val, float) or isinstance(json_val, float):
            match = abs(float(csv_val) - float(json_val)) < 0.01
        else:
            match = csv_val == json_val
        
        status = "✓" if match else "✗"
        print(f"  {status} {field}: CSV={csv_val}, JSON={json_val}")
        
        if not match:
            all_passed = False
            errors.append(f"{field}: expected {csv_val}, got {json_val}")
    
    if all_passed:
        print(f"\n  ✓ All individual tests passed for {user_id}")
    else:
        print(f"\n  ✗ Some tests failed: {errors}")
    
    return all_passed

def test_combined_users(user_ids: list, json_combined: dict, json_individual: list, csv_data: dict) -> bool:
    """Test that combined stats are correctly weighted averages"""
    print(f"\n{'='*60}")
    print(f"Testing combined users: {user_ids}")
    print(f"{'='*60}")
    
    # Calculate expected values from individual data
    total_games = sum(j['player_stats']['total_games'] for j in json_individual)
    total_wins = sum(j['player_stats']['total_wins'] for j in json_individual)
    total_position_sum = sum(j['player_stats']['total_position_sum'] for j in json_individual)
    total_score_sum = sum(j['player_stats']['total_score_sum'] for j in json_individual)
    total_generations_sum = sum(j['player_stats']['total_generations_sum'] for j in json_individual)
    total_tr_sum = sum(j['player_stats']['total_tr_sum'] for j in json_individual)
    total_cards_sum = sum(j['player_stats']['total_cards_played_sum'] for j in json_individual)
    tr_games = sum(j['player_stats']['tr_games'] for j in json_individual)
    cards_games = sum(j['player_stats']['cards_games'] for j in json_individual)
    
    # Expected averages
    expected_win_rate = round(total_wins / total_games * 100, 2)
    expected_avg_position = round(total_position_sum / total_games, 3)
    expected_avg_score = round(total_score_sum / total_games, 2)
    expected_avg_generations = round(total_generations_sum / total_games, 2)
    expected_avg_tr = round(total_tr_sum / tr_games, 2)
    expected_avg_cards = round(total_cards_sum / cards_games, 2)
    
    combined_stats = json_combined['player_stats']
    
    tests = [
        ('total_games', total_games, combined_stats['total_games']),
        ('total_wins', total_wins, combined_stats['total_wins']),
        ('win_rate', expected_win_rate, combined_stats['win_rate']),
        ('avg_position', expected_avg_position, combined_stats['avg_position']),
        ('avg_score', expected_avg_score, combined_stats['avg_score']),
        ('avg_generations', expected_avg_generations, combined_stats['avg_generations']),
        ('avg_tr', expected_avg_tr, combined_stats['avg_tr']),
        ('avg_cards_played', expected_avg_cards, combined_stats['avg_cards_played']),
    ]
    
    all_passed = True
    for field, expected, actual in tests:
        # Allow small floating point differences
        if isinstance(expected, float) or isinstance(actual, float):
            match = abs(float(expected) - float(actual)) < 0.01
        else:
            match = expected == actual
        
        status = "✓" if match else "✗"
        print(f"  {status} {field}: expected={expected}, actual={actual}")
        
        if not match:
            all_passed = False
    
    if all_passed:
        print(f"\n  ✓ All combined aggregation tests passed")
    else:
        print(f"\n  ✗ Some combined tests failed")
    
    return all_passed

def test_records_by_generation(user_ids: list, json_combined: dict, json_individual: list) -> bool:
    """Test that records by generation are correct max values"""
    print(f"\n{'='*60}")
    print(f"Testing records by generation")
    print(f"{'='*60}")
    
    combined_records = json_combined['records_by_generation']
    
    all_passed = True
    
    for gen_str, combined_rec in combined_records.items():
        gen = int(gen_str)
        
        # Find max across all individual users for this generation
        max_score = None
        max_cards = None
        
        for j in json_individual:
            if gen_str in j['records_by_generation']:
                rec = j['records_by_generation'][gen_str]
                if rec['max_score'] is not None:
                    if max_score is None or rec['max_score'] > max_score:
                        max_score = rec['max_score']
                if rec['max_cards_played'] is not None:
                    if max_cards is None or rec['max_cards_played'] > max_cards:
                        max_cards = rec['max_cards_played']
        
        # Compare
        score_match = combined_rec['max_score'] == max_score
        cards_match = combined_rec['max_cards_played'] == max_cards
        
        status = "✓" if (score_match and cards_match) else "✗"
        print(f"  {status} Generation {gen}:")
        print(f"      max_score: expected={max_score}, actual={combined_rec['max_score']} {'✓' if score_match else '✗'}")
        print(f"      max_cards: expected={max_cards}, actual={combined_rec['max_cards_played']} {'✓' if cards_match else '✗'}")
        
        if not (score_match and cards_match):
            all_passed = False
    
    if all_passed:
        print(f"\n  ✓ All records by generation tests passed")
    else:
        print(f"\n  ✗ Some records tests failed")
    
    return all_passed

def main():
    """Main test function"""
    print("=" * 60)
    print("User Aggregate Test Suite")
    print("=" * 60)
    
    # Test user IDs
    user1_id = "69215eb418a5"
    user2_id = "9007426e7f53"
    
    # Load CSV data
    csv_data = load_csv_data(4)
    print(f"Loaded CSV data: {len(csv_data['player_stats'])} players")
    
    # Load JSON files
    json_user1 = load_json(f'{DISPLAY_DIR}/test_user1.json')
    json_user2 = load_json(f'{DISPLAY_DIR}/test_user2.json')
    json_combined = load_json(f'{DISPLAY_DIR}/test_combined.json')
    
    # Run tests
    results = []
    
    # Test individual users
    results.append(("Individual User 1", test_individual_user(user1_id, json_user1, csv_data)))
    results.append(("Individual User 2", test_individual_user(user2_id, json_user2, csv_data)))
    
    # Test combined aggregation
    results.append(("Combined Aggregation", test_combined_users(
        [user1_id, user2_id], 
        json_combined, 
        [json_user1, json_user2],
        csv_data
    )))
    
    # Test records by generation
    results.append(("Records by Generation", test_records_by_generation(
        [user1_id, user2_id],
        json_combined,
        [json_user1, json_user2]
    )))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
