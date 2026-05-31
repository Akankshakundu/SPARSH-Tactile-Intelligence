"""
Braille Mapper - converts detected dot patterns to English text.
Handles Grade 1 Braille: letters, numbers, punctuation, capitals.
"""

import json
import os

# Load braille cell definitions
_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'braille_cells.json')

with open(_DATA_PATH, 'r') as f:
    _BRAILLE_DATA = json.load(f)

# Merge all pattern lookups into one flat dict
_PATTERN_TO_CHAR: dict[str, str] = {}
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["letters"])
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["punctuation"])
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["contractions"])
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["numbers"])

CAPITAL_INDICATOR = "000001"
NUMBER_INDICATOR  = "010011"
SPACE_PATTERN     = "000000"


def pattern_to_char(pattern: str, capital_mode: bool = False, number_mode: bool = False) -> str:
    """
    Convert a 6-bit binary pattern string to a character.
    Returns '?' for unknown patterns.
    """
    if pattern == SPACE_PATTERN:
        return " "

    if pattern == CAPITAL_INDICATOR:
        return "__CAPITAL__"

    if pattern == NUMBER_INDICATOR:
        return "__NUMBER__"

    # Number mode: map same patterns to digits
    if number_mode and pattern in _BRAILLE_DATA["numbers"]:
        return _BRAILLE_DATA["numbers"][pattern]

    # Letter mode
    if pattern in _BRAILLE_DATA["letters"]:
        ch = _BRAILLE_DATA["letters"][pattern]
        return ch.upper() if capital_mode else ch

    # Punctuation / contractions fallback
    if pattern in _PATTERN_TO_CHAR:
        return _PATTERN_TO_CHAR[pattern]

    return "?"


def decode_cell_sequence(patterns: list[str]) -> str:
    """
    Convert a list of 6-bit pattern strings (one per Braille cell) to English text.
    Handles capital and number indicator cells automatically.
    """
    result = []
    capital_mode = False
    number_mode = False

    for pattern in patterns:
        if pattern == CAPITAL_INDICATOR:
            capital_mode = True
            continue

        if pattern == NUMBER_INDICATOR:
            number_mode = True
            continue

        if pattern == SPACE_PATTERN:
            result.append(" ")
            number_mode = False   # number mode resets at space
            capital_mode = False
            continue

        char = pattern_to_char(pattern, capital_mode=capital_mode, number_mode=number_mode)

        if char not in ("__CAPITAL__", "__NUMBER__"):
            result.append(char)
            capital_mode = False  # capital indicator applies to one char only

    return "".join(result)


def decode_lines(line_cell_patterns: list[list[str]]) -> list[str]:
    """
    Decode multiple lines of Braille cells.
    Each line is a list of 6-bit pattern strings.
    Returns a list of decoded English strings, one per line.
    """
    return [decode_cell_sequence(line) for line in line_cell_patterns]


def decode_from_cell_chars(cells) -> str:
    """Build text from classified cells; prefer pattern lookup for letters."""
    parts: list[str] = []
    for cell in cells:
        if cell.pattern == "000000":
            if parts and parts[-1] != " ":
                parts.append(" ")
            continue

        ch = pattern_to_char(cell.pattern)
        if ch in ("?", "·", "__CAPITAL__", "__NUMBER__"):
            ch = getattr(cell, "char", None) or ch
        if ch in ("?", "·"):
            continue
        if ch == " ":
            if parts and parts[-1] != " ":
                parts.append(" ")
            continue
        parts.append(ch)

    text = "".join(parts)
    return " ".join(text.split())


def get_all_patterns() -> dict:
    """Return all known patterns for debugging/visualization."""
    return _PATTERN_TO_CHAR.copy()
