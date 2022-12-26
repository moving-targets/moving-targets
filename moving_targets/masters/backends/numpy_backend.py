from typing import Any, Optional

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import BackendError


class NumpyBackend(Backend):
    """`Backend` implementation for Numpy.

    Numpy cannot be used as a solver, still, this backend may come in handy when a loss object is needed to compute the
    loss between two numeric vectors, instead of a numeric vector and a vector of model variables.
    """

    _ERROR_MESSAGE: str = 'Numpy is not a solver, thus '
    """Error message for unsupported operations."""

    def __init__(self, clip: Optional[float] = 1e-15):
        """
        :param clip:
            The clipping value to use when calling the 'log()' function to avoid numeric errors. If None, does not clip.
        """
        super(NumpyBackend, self).__init__()

        self.clip: float = clip
        """The clipping value to use when calling the 'log()' function to avoid numeric errors."""

    def _build_model(self) -> Any:
        return np

    def _solve_model(self) -> Optional:
        return np

    def minimize(self, cost) -> Any:
        raise BackendError(unsupported='no solution can be minimized', message=self._ERROR_MESSAGE)

    def maximize(self, cost) -> Any:
        raise BackendError(unsupported='no solution can be maximized', message=self._ERROR_MESSAGE)

    def get_objective(self) -> float:
        raise BackendError(unsupported='no objective can be retrieved', message=self._ERROR_MESSAGE)

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return expressions

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        raise BackendError(unsupported='no variable can be added', message=self._ERROR_MESSAGE)

    def add_constant(self, val: Any, vtype: str = 'continuous', name: Optional[str] = None) -> Any:
        return val

    def add_constants(self, val: np.ndarray, vtype: str = 'continuous', name: Optional[str] = None) -> np.ndarray:
        return val

    def aux(self,
            expressions: Any,
            aux_vtype: Optional[str] = None,
            aux_lb: float = -float('inf'),
            aux_ub: float = float('inf'),
            aux_name: Optional[str] = None) -> Any:
        return expressions

    def add_constraint(self, constraint, name: Optional[str] = None) -> Any:
        raise BackendError(unsupported='no constraint can be added', message=self._ERROR_MESSAGE)

    def add_indicator_constraint(self,
                                 indicator: Any,
                                 expression: Any,
                                 value: int = 1,
                                 name: Optional[str] = None) -> Any:
        raise BackendError(unsupported='no constraint can be added', message=self._ERROR_MESSAGE)

    def is_greater(self, a, b) -> np.ndarray:
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        return (a > b).astype(int)

    def is_less(self, a, b) -> np.ndarray:
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        return (a < b).astype(int)

    def sqrt(self, a) -> np.ndarray:
        return np.sqrt(np.atleast_1d(a))

    def abs(self, a) -> np.ndarray:
        return np.abs(np.atleast_1d(a))

    def log(self, a) -> np.ndarray:
        if self.clip is not None:
            a_min = np.min(a)
            if a_min < 0:
                self._LOGGER.warning(f'Trying to compute logarithm of negative number, clipping to {self.clip}')
            elif a_min < self.clip:
                self._LOGGER.info(f'Values in the interval [0, {self.clip}] will be clipped to {self.clip}')
            a = np.clip(a, a_min=self.clip, a_max=float('inf'))
        return np.log(a)

    def square(self, a) -> np.ndarray:
        return np.atleast_1d(a) ** 2

    def min(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        return np.min(a, axis=axis)

    def max(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        return np.max(a, axis=axis)

    def var(self,
            a: np.ndarray,
            axis: Optional[int] = None,
            definition: bool = False,
            asarray: bool = False,
            aux: bool = False) -> Any:
        return np.var(a, axis=axis)

    def cov(self, a: np.ndarray, b: np.ndarray, definition: bool = True, asarray: bool = False, aux: bool = False):
        return np.cov(a, b, bias=True)[0, 1]

    def norm_0(self, a: np.ndarray, axis: Optional[int] = None, asarray: bool = False, aux: bool = False) -> Any:
        return np.count_nonzero(a, axis=axis)
