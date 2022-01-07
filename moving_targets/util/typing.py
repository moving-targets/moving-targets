"""Type aliases."""

from typing import Union, Tuple, Dict, Any

Number = Union[int, float]
"""A scalar."""

Dataset = Dict[str, Tuple[Any, Any]]
"""A dictionary where the key is string representing the name of the split, and the value is tuple (x, y)."""
