"""
Validation layer for Braille OCR pipeline.
Provides pattern validation, error checking, and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .braille_mapper import get_all_patterns
from .pipeline_types import BrailleCell, Dot


LOGGER = logging.getLogger(__name__)
KNOWN_PATTERNS = set(get_all_patterns().keys())
KNOWN_PATTERNS.update({"000001", "010011", "000000"})  # Indicators + space


@dataclass
class ValidationResult:
    """Result of cell pattern validation."""
    is_valid: bool
    pattern: str
    issues: list[str]
    warnings: list[str]
    confidence_adjustment: float = 0.0


def validate_cell_pattern(
    pattern: str,
    dot_count: int,
    dot_centers: list[tuple[int, int]],
) -> ValidationResult:
    """
    Validate a Braille cell pattern for logical consistency.
    
    Args:
        pattern: 6-bit binary pattern string (e.g., "100101")
        dot_count: Number of detected dots in cell
        dot_centers: Pixel coordinates of detected dots
        
    Returns:
        ValidationResult with validity status and issues/warnings
    """
    issues = []
    warnings = []
    confidence_adj = 0.0
    
    # Check pattern format
    if len(pattern) != 6 or not all(c in "01" for c in pattern):
        issues.append(f"Invalid pattern format: {pattern}")
        return ValidationResult(
            is_valid=False,
            pattern=pattern,
            issues=issues,
            warnings=warnings,
            confidence_adjustment=confidence_adj,
        )
    
    # Count expected dots in pattern
    expected_dot_count = pattern.count("1")
    
    # Check dot count consistency
    if dot_count != expected_dot_count:
        warning = f"Dot count mismatch: pattern expects {expected_dot_count}, found {dot_count}"
        if abs(dot_count - expected_dot_count) > 2:
            issues.append(warning)
        else:
            warnings.append(warning)
            confidence_adj -= 0.05
    
    # Check for impossible configurations
    if dot_count > 6:
        issues.append(f"Too many dots: {dot_count} (max 6 per cell)")
    
    if dot_count < 0:
        issues.append(f"Negative dot count: {dot_count}")
    
    # Check pattern validity
    if pattern != "000000" and pattern not in KNOWN_PATTERNS:
        warnings.append(f"Pattern {pattern} not in known Braille dictionary")
        # Don't fail; might be valid variant or error in data
    
    # All zeros but non-zero dots
    if pattern == "000000" and dot_count > 1:
        issues.append(f"Pattern is space (000000) but {dot_count} dots detected")
    
    # Non-zero pattern but no dots
    if pattern != "000000" and dot_count == 0:
        issues.append(f"Pattern {pattern} expects dots but none detected")
    
    is_valid = len(issues) == 0
    return ValidationResult(
        is_valid=is_valid,
        pattern=pattern,
        issues=issues,
        warnings=warnings,
        confidence_adjustment=confidence_adj,
    )


def validate_cell_geometry(
    cell: BrailleCell,
    median_radius: float,
) -> ValidationResult:
    """
    Validate a cell's geometric properties.
    
    Args:
        cell: BrailleCell to validate
        median_radius: Reference dot radius for Braille image
        
    Returns:
        ValidationResult with geometric validity
    """
    issues = []
    warnings = []
    confidence_adj = 0.0
    
    x, y, w, h = cell.bbox
    
    # Check bounding box dimensions
    min_width = median_radius * 2.0
    max_width = median_radius * 8.0
    min_height = median_radius * 3.0
    max_height = median_radius * 10.0
    
    if w < min_width:
        warnings.append(f"Cell width too small: {w:.1f}px (expected ≥ {min_width:.1f}px)")
        confidence_adj -= 0.03
    
    if w > max_width:
        warnings.append(f"Cell width too large: {w:.1f}px (expected ≤ {max_width:.1f}px)")
        confidence_adj -= 0.03
    
    if h < min_height:
        warnings.append(f"Cell height too small: {h:.1f}px (expected ≥ {min_height:.1f}px)")
        confidence_adj -= 0.03
    
    if h > max_height:
        warnings.append(f"Cell height too large: {h:.1f}px (expected ≤ {max_height:.1f}px)")
        confidence_adj -= 0.03
    
    # Check matrix consistency
    if cell.matrix:
        matrix_dots = sum(1 for col in cell.matrix for row in col if row == 1)
        if matrix_dots != cell.pattern.count("1"):
            issues.append(
                f"Matrix/pattern mismatch: {matrix_dots} matrix dots vs {cell.pattern.count('1')} pattern dots"
            )
    
    is_valid = len(issues) == 0
    return ValidationResult(
        is_valid=is_valid,
        pattern=cell.pattern,
        issues=issues,
        warnings=warnings,
        confidence_adjustment=confidence_adj,
    )


def diagnose_cell(
    cell: BrailleCell,
    median_radius: float,
) -> dict[str, any]:
    """
    Comprehensive diagnostic information for a cell.
    
    Returns:
        Dictionary with validation results and diagnostic info
    """
    pattern_validation = validate_cell_pattern(
        cell.pattern,
        len(cell.dot_centers),
        cell.dot_centers,
    )
    
    geometry_validation = validate_cell_geometry(cell, median_radius)
    
    return {
        "cell": {
            "pattern": cell.pattern,
            "char": cell.char,
            "confidence": cell.confidence,
            "dot_count": len(cell.dot_centers),
            "bbox": cell.bbox,
        },
        "pattern_validation": {
            "is_valid": pattern_validation.is_valid,
            "issues": pattern_validation.issues,
            "warnings": pattern_validation.warnings,
        },
        "geometry_validation": {
            "is_valid": geometry_validation.is_valid,
            "issues": geometry_validation.issues,
            "warnings": geometry_validation.warnings,
        },
        "confidence_adjustments": {
            "pattern": pattern_validation.confidence_adjustment,
            "geometry": geometry_validation.confidence_adjustment,
        },
    }


def log_segmentation_diagnostics(
    line_idx: int,
    line_dots: list[Dot],
    columns: int,
    cells: list[BrailleCell],
    row_centers: tuple[float, float, float],
    within_gap: float,
    word_gap: float,
) -> None:
    """
    Log detailed diagnostics for a line segmentation.
    
    Args:
        line_idx: Line index
        line_dots: Detected dots in line
        columns: Number of columns detected
        cells: BrailleCells generated
        row_centers: Detected row center positions
        within_gap: Gap threshold for within-cell spacing
        word_gap: Gap threshold for word spacing
    """
    median_radius = float(__import__("numpy").median([dot.radius for dot in line_dots])) if line_dots else 5.0
    
    # Log overview
    LOGGER.info(
        "Line %d segmentation: %d dots → %d columns → %d cells",
        line_idx,
        len(line_dots),
        columns,
        len(cells),
    )
    
    # Log row detection
    row_info = " ".join(f"{y:.1f}" for y in row_centers)
    LOGGER.debug(
        "  Row centers: %s (gap %.1f)",
        row_info,
        row_centers[1] - row_centers[0] if len(row_centers) > 1 else 0,
    )
    
    # Log gap analysis
    if word_gap != float("inf"):
        LOGGER.debug(
            "  Gap thresholds: within-cell=%.1f, word=%.1f",
            within_gap,
            word_gap,
        )
    
    # Log cell details
    for idx, cell in enumerate(cells):
        if cell.pattern == "000000":
            LOGGER.debug(f"  Cell {idx}: SPACE")
        else:
            issues = []
            if len(cell.dot_centers) != cell.pattern.count("1"):
                issues.append(f"dot_count_mismatch")
            
            issues_str = f" ({', '.join(issues)})" if issues else ""
            LOGGER.debug(
                f"  Cell {idx}: pattern={cell.pattern} char={cell.char!r} confidence={cell.confidence:.3f}{issues_str}"
            )
