"""Type aliases."""

from typing import Tuple, Dict, Any, Union

import numpy as np

Dataset = Dict[str, Tuple[Any, np.ndarray]]
"""A dictionary where the key is string representing the name of the split, and the value is tuple (x, y)."""

Mask = Union[float, np.ndarray]
"""Either floating point value to mask a single value or an explicit masking vector."""


def is_numeric(obj: Any) -> bool:
    """Checks whether an object has a numeric type or not.

    :param obj:
        The object to test.

    :return:
        True if the object has a numeric type, False otherwise.
    """
    return isinstance(obj, (int, float, np.number))
