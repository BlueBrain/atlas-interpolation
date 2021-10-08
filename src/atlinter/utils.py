"""Module for utility functions."""
from __future__ import annotations


def find_closest(x: float, values: list[int]) -> tuple[int, float]:
    """Find closest element to a given element in a list and its position.

    Parameters
    ----------
    x
        Value to compare with values list.
    values
        List of values where to find closest element to x and its position.

    Returns
    -------
    int
        Indice of the closest element to x in the values list.
    float
        Closest value to x in the values list.
    """
    idx = min(range(len(values)), key=lambda idx: abs(values[idx] - x))

    return idx, values[idx]
