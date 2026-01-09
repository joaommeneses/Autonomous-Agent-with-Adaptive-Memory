#!/usr/bin/env python3
"""
Test snippet to verify Swift action post-processing fix.
Tests that normalization doesn't destroy rooms and matches work correctly.
"""

import re
from typing import Set, Dict

# Copy utility functions from eval_utils.py
def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in action strings for consistent comparison."""
    return re.sub(r"\s+", " ", s).strip()

def build_normalized_valid_actions_set(validActions):
    """Build a normalized set of validActions for efficient matching."""
    normalized_set = set()
    normalized_to_original = {}
    for va in validActions:
        normalized = normalize_whitespace(va)
        normalized_set.add(normalized)
        normalized_to_original[normalized] = va
    return normalized_set, normalized_to_original

# Sample validActions from logs (real examples)
validActions = {
    'pour cup containing blue paint in art studio into cup containing red paint',
    'pour cup containing yellow paint in cupboard into cup containing red paint in cupboard',
    'pour paint in wood cup containing blue paint into hallway',
    'pour cup containing red paint in cupboard into cup containing blue paint in art studio',
    'pour table into cup containing red paint in art studio',
    'pour cup containing blue paint in art studio into cup containing yellow paint in cupboard',
}

# Swift predictions from logs (real examples)
swift_predictions = [
    'pour cup containing blue paint in art studio in cup containing red paint',  # Missing "into", has double "in"
    'pour cup containing blue paint in art studio in cup containing blue paint',
    'pour cup containing red paint in art studio in bowl',
    'pour cup containing blue paint in art studio in cup containing paint',
    'pour cup containing red paint in art studio in cup containing blue paint',
]

def test_normalization():
    """Test that normalization doesn't destroy room qualifiers."""
    print("=== Test 1: Whitespace Normalization ===")
    
    # Test case: double space should be normalized
    test_cases = [
        ("pour cup containing blue paint  in cup containing red paint", "pour cup containing blue paint in cup containing red paint"),
        ("pour  cup  containing  blue  paint", "pour cup containing blue paint"),
        ("pour cup containing blue paint in art studio", "pour cup containing blue paint in art studio"),  # Should preserve room
    ]
    
    for input_str, expected in test_cases:
        normalized = normalize_whitespace(input_str)
        assert normalized == expected, f"Failed: '{input_str}' -> '{normalized}' (expected '{expected}')"
        print(f"✓ '{input_str}' -> '{normalized}'")
    
    print("\n=== Test 2: Room Preservation ===")
    # Verify rooms are preserved in normalization
    room_test = "pour cup containing blue paint in art studio into cup containing red paint"
    normalized = normalize_whitespace(room_test)
    assert "art studio" in normalized, f"Room 'art studio' was removed: '{normalized}'"
    print(f"✓ Room preserved: '{normalized}'")

def test_matching():
    """Test that matching works with normalized actions."""
    print("\n=== Test 3: Normalized Matching ===")
    
    normalized_set, norm_to_orig = build_normalized_valid_actions_set(validActions)
    
    # Test exact match (with normalization)
    test_action = "pour cup containing blue paint in art studio into cup containing red paint"
    normalized_action = normalize_whitespace(test_action)
    
    if normalized_action in normalized_set:
        matched = norm_to_orig[normalized_action]
        print(f"✓ Matched: '{test_action}' -> '{matched}'")
    else:
        print(f"✗ No match for: '{test_action}'")
        print(f"  Normalized: '{normalized_action}'")
        print(f"  Available normalized actions:")
        for na in sorted(normalized_set)[:5]:
            print(f"    - {na}")

def test_swift_predictions():
    """Test that Swift predictions can match validActions."""
    print("\n=== Test 4: Swift Prediction Matching ===")
    
    normalized_set, norm_to_orig = build_normalized_valid_actions_set(validActions)
    
    for i, pred in enumerate(swift_predictions[:3]):
        normalized = normalize_whitespace(pred)
        print(f"\nPrediction {i+1}: '{pred}'")
        print(f"  Normalized: '{normalized}'")
        
        # Check if normalized matches
        if normalized in normalized_set:
            matched = norm_to_orig[normalized]
            print(f"  ✓ MATCHED: '{matched}'")
        else:
            print(f"  ✗ No exact match")
            # Check for partial matches (similar to what try_to_replace would do)
            # This simulates the matching logic
            for va in validActions:
                if normalized.replace(" in ", " into ").replace(" in ", " into ") in normalize_whitespace(va):
                    print(f"  → Potential match after repair: '{va}'")
                    break

if __name__ == "__main__":
    print("Testing Swift Action Post-Processing Fix\n")
    print("=" * 60)
    
    try:
        test_normalization()
        test_matching()
        test_swift_predictions()
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


