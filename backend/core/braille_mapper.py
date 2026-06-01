"""
Braille pattern lookup and text translation.
"""

from __future__ import annotations

import json
import os


_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "braille_cells.json")

with open(_DATA_PATH, "r", encoding="utf-8") as handle:
    _BRAILLE_DATA = json.load(handle)

_PATTERN_TO_CHAR: dict[str, str] = {}
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["letters"])
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["punctuation"])
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["contractions"])
_PATTERN_TO_CHAR.update(_BRAILLE_DATA["numbers"])

CAPITAL_INDICATOR = "000001"
NUMBER_INDICATOR = "010011"
SPACE_PATTERN = "000000"


def pattern_to_char(pattern: str, capital_mode: bool = False, number_mode: bool = False) -> str:
    if pattern == SPACE_PATTERN:
        return " "
    if pattern == CAPITAL_INDICATOR:
        return "__CAPITAL__"
    if pattern == NUMBER_INDICATOR:
        return "__NUMBER__"

    if number_mode and pattern in _BRAILLE_DATA["numbers"]:
        return _BRAILLE_DATA["numbers"][pattern]

    if pattern in _BRAILLE_DATA["letters"]:
        char = _BRAILLE_DATA["letters"][pattern]
        return char.upper() if capital_mode else char

    return _PATTERN_TO_CHAR.get(pattern, "?")


def decode_cell_sequence(patterns: list[str]) -> str:
    text, _ = decode_cell_sequence_with_metadata(patterns)
    return text


def decode_cell_sequence_with_metadata(patterns: list[str]) -> tuple[str, list[str]]:
    output: list[str] = []
    cell_chars: list[str] = []
    capital_mode = False
    number_mode = False

    for pattern in patterns:
        if pattern == CAPITAL_INDICATOR:
            capital_mode = True
            cell_chars.append("^")
            continue

        if pattern == NUMBER_INDICATOR:
            number_mode = True
            cell_chars.append("#")
            continue

        if pattern == SPACE_PATTERN:
            output.append(" ")
            cell_chars.append(" ")
            capital_mode = False
            number_mode = False
            continue

        char = pattern_to_char(pattern, capital_mode=capital_mode, number_mode=number_mode)
        cell_chars.append(char if char not in {"__CAPITAL__", "__NUMBER__"} else "?")

        if char not in {"__CAPITAL__", "__NUMBER__"}:
            output.append(char)
            capital_mode = False
            if not char.isdigit():
                number_mode = False

    text = "".join(output)
    return " ".join(text.split()), cell_chars


def decode_lines(line_cell_patterns: list[list[str]]) -> list[str]:
    return [decode_cell_sequence(line) for line in line_cell_patterns]


def decode_lines_with_metadata(line_cell_patterns: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    lines: list[str] = []
    cell_chars: list[list[str]] = []
    for patterns in line_cell_patterns:
        text, chars = decode_cell_sequence_with_metadata(patterns)
        lines.append(text)
        cell_chars.append(chars)
    return lines, cell_chars


def decode_from_cell_chars(cells) -> str:
    patterns = [cell.pattern for cell in cells]
    return decode_cell_sequence(patterns)


def get_all_patterns() -> dict[str, str]:
    return _PATTERN_TO_CHAR.copy()
