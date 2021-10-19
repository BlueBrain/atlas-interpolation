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
import numpy as np
import pytest

from atlinter.utils import find_closest


@pytest.mark.parametrize(
    ("x", "values", "expected_index", "expected_value"),
    [
        (6.9, np.arange(10), 7, 7),
        (3, np.arange(10), 3, 3),
        (3, np.arange(1, 10), 2, 3),
        (9.4, [0.1, 4.3, 9.5], 2, 9.5),
    ],
)
def test_find_closest(x, values, expected_index, expected_value):
    """Test find closest function."""
    assert find_closest(x, values) == (expected_index, expected_value)
