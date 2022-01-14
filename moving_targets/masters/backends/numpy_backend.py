from typing import Any, Union, List, Optional

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.typing import Number


class NumpyBackend(Backend):
    """`Backend` implementation for Numpy.

    Numpy cannot be used as a solver, still, this backend may come in handy when a loss object is needed to compute the
    loss between two numeric vectors, instead of a numeric vector and a vector of model variables.
    """

    _ERROR_MESSAGE: str = 'Numpy is not a solver, thus '
    """Error message for unsupported operations."""

    def _build_model(self) -> Any:
        raise AssertionError(self._ERROR_MESSAGE + 'no model can be built')

    def _solve_model(self) -> Optional:
        raise AssertionError(self._ERROR_MESSAGE + 'no solution can be retrieved')

    def minimize(self, cost) -> Any:
        raise AssertionError(self._ERROR_MESSAGE + 'no solution can be minimized')

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        raise AssertionError(self._ERROR_MESSAGE + 'no constraint can be added')

    def add_variables(self, *keys: int, vtype: str, lb: Number, ub: Number, name: Optional[str] = None) -> np.ndarray:
        raise AssertionError(self._ERROR_MESSAGE + 'no variable can be added')

    def get_objective(self) -> Number:
        raise AssertionError(self._ERROR_MESSAGE + 'no objective can be retrieved')

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return expressions

    def sum_expr(self, vector: np.ndarray) -> Any:
        return np.sum(vector)

    def sum(self, vector: np.ndarray) -> Any:
        # since the super method calls 'add_continuous_variable' which raises an error, override it
        return np.sum(vector)

    def sqr(self, vector: np.ndarray) -> np.ndarray:
        return vector ** 2

    def abs(self, vector: np.ndarray) -> np.ndarray:
        return np.abs(vector)

    def log(self, vector: np.ndarray) -> np.ndarray:
        return np.log(vector)
