"""CFBD play types lookup module"""

import json
from pathlib import Path
from typing import Dict, Optional

PLAY_TYPES_PATH = Path(__file__).parent / "data" / "play_types.json"

def load_play_types() -> Dict[int, str]:
    """Load play types from JSON file and create id -> name mapping"""

    if not PLAY_TYPES_PATH.exists():
        print(f"Warning: {PLAY_TYPES_PATH} not found")
        return {}

    with open(PLAY_TYPES_PATH) as f:
        play_types_list = json.load(f)

    # Create mapping: play_type_id -> play_type_text
    mapping = {}
    for item in play_types_list:
        play_id = item.get("id")
        play_text = item.get("text")
        if play_id is not None and play_text:
            mapping[play_id] = play_text

    return mapping

# Load on import
_PLAY_TYPES = load_play_types()

def get_play_type_name(play_type_id: int) -> Optional[str]:
    """Get the name of a play type by ID"""
    return _PLAY_TYPES.get(play_type_id)

def get_all_play_types() -> Dict[int, str]:
    """Get all play types"""
    return _PLAY_TYPES

# Debug: print available play types
if __name__ == "__main__":
    print(f"Loaded {len(_PLAY_TYPES)} play types:")
    for play_id, play_name in sorted(_PLAY_TYPES.items()):
        print(f"  {play_id}: {play_name}")
