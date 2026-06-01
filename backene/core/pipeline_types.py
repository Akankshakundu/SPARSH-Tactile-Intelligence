from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dot:
    x: int
    y: int
    radius: float
    area: float
    bbox: tuple[int, int, int, int]


@dataclass
class BrailleCell:
    col: int
    row: int
    pattern: str
    char: str
    bbox: tuple[int, int, int, int]
    dot_centers: list[tuple[int, int]] = field(default_factory=list)
    intensities: list[float] = field(default_factory=list)
    confidence: float = 0.0
    matrix: list[list[int]] = field(default_factory=list)


@dataclass
class BrailleLine:
    row: int
    dots: list[Dot]
    cells: list[BrailleCell]
    row_centers: tuple[float, float, float]
    column_centers: list[float]
    within_cell_gap: float
    word_gap: float


@dataclass
class DetectionResult:
    cells: list[BrailleCell]
    lines: list[list[BrailleCell]]
    patterns_by_line: list[list[str]]
    dot_radius_est: float
    cell_width_est: float
    cell_height_est: float
    annotated_image: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)
