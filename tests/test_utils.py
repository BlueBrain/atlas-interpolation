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
