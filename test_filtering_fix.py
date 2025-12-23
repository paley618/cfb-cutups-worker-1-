#!/usr/bin/env python3
"""
Unit test to verify the CFBD filtering fix works correctly.
Tests that _play_belongs_to_game handles all field name variations.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.cfbd_client import _play_belongs_to_game


def test_play_belongs_to_game():
    """Test that _play_belongs_to_game handles all field name variations."""

    game_id = 401636921

    # Test 1: Play with gameId (camelCase) - likely what CFBD returns
    play_with_gameId = {
        "gameId": 401636921,
        "id": 12345,
        "play_text": "Test play"
    }
    assert _play_belongs_to_game(play_with_gameId, game_id), \
        "Should match play with 'gameId' field"
    print("✅ Test 1 passed: gameId (camelCase) field works")

    # Test 2: Play with game_id (snake_case) - old format
    play_with_game_id = {
        "game_id": 401636921,
        "id": 12345,
        "play_text": "Test play"
    }
    assert _play_belongs_to_game(play_with_game_id, game_id), \
        "Should match play with 'game_id' field"
    print("✅ Test 2 passed: game_id (snake_case) field works")

    # Test 3: Play with game field
    play_with_game = {
        "game": 401636921,
        "id": 12345,
        "play_text": "Test play"
    }
    assert _play_belongs_to_game(play_with_game, game_id), \
        "Should match play with 'game' field"
    print("✅ Test 3 passed: game field works")

    # Test 4: Play from different game (should NOT match)
    play_from_other_game = {
        "gameId": 999999999,
        "id": 12345,
        "play_text": "Test play"
    }
    assert not _play_belongs_to_game(play_from_other_game, game_id), \
        "Should NOT match play from different game"
    print("✅ Test 4 passed: Correctly rejects play from different game")

    # Test 5: Play with no game identifier (should NOT match)
    play_no_game = {
        "id": 12345,
        "play_text": "Test play"
    }
    assert not _play_belongs_to_game(play_no_game, game_id), \
        "Should NOT match play with no game identifier"
    print("✅ Test 5 passed: Correctly rejects play with no game identifier")

    # Test 6: Play with string game_id (should still match)
    play_string_id = {
        "gameId": "401636921",  # String instead of int
        "id": 12345,
        "play_text": "Test play"
    }
    assert _play_belongs_to_game(play_string_id, game_id), \
        "Should match play with string gameId"
    print("✅ Test 6 passed: Handles string gameId correctly")

    # Test 7: Priority test - gameId should take priority over game_id
    play_multiple_fields = {
        "gameId": 401636921,
        "game_id": 999999999,  # Different value
        "id": 12345,
        "play_text": "Test play"
    }
    assert _play_belongs_to_game(play_multiple_fields, game_id), \
        "Should prioritize 'gameId' over 'game_id'"
    print("✅ Test 7 passed: Correctly prioritizes gameId field")

    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe filtering fix correctly handles all field name variations:")
    print("  1. gameId (camelCase) - likely CFBD API format")
    print("  2. game_id (snake_case) - fallback format")
    print("  3. game - alternative format")
    print("\nThis fix should resolve the issue where 0 CSV files were created.")


if __name__ == "__main__":
    try:
        test_play_belongs_to_game()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
