"""Type aliases."""

from typing import Tuple, Dict, Any

import numpy as np

Dataset = Dict[str, Tuple[Any, np.ndarray]]
"""A dictionary where the key is string representing the name of the split, and the value is tuple (x, y)."""
