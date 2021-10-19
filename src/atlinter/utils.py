# Copyright 2021, Blue Brain Project, EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
