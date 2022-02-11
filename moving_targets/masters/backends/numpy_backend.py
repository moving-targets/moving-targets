from typing import Any, Union, List, Optional

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
        raise BackendError(unsupported='no model can be built', message=self._ERROR_MESSAGE)

    def _solve_model(self) -> Optional:
        raise BackendError(unsupported='no solution can be retrieved', message=self._ERROR_MESSAGE)

    def minimize(self, cost) -> Any:
        raise BackendError(unsupported='no solution can be minimized', message=self._ERROR_MESSAGE)

    def add_constraints(self, constraints: Union[List, np.ndarray], name: Optional[str] = None) -> Any:
        raise BackendError(unsupported='no constraint can be added')

    def add_variable(self, vtype: str, lb: float, ub: float, name: Optional[str] = None) -> Any:
        raise BackendError(unsupported='no variable can be added', message=self._ERROR_MESSAGE)

    def get_objective(self) -> float:
        raise BackendError(unsupported='no objective can be retrieved', message=self._ERROR_MESSAGE)

    def get_values(self, expressions: np.ndarray) -> np.ndarray:
        return expressions

    def sum(self, a: np.ndarray, aux: Optional[str] = None) -> Any:
        # since the super method calls 'add_continuous_variable' which raises an error, override it
        return np.sum(a)

    def square(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        return a ** 2

    def abs(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        return np.abs(a)

    def log(self, a: np.ndarray, aux: Optional[str] = None) -> np.ndarray:
        if self.clip is not None:
            a_min = np.min(a)
            if a_min < 0:
                self._LOGGER.warning(f'Trying to compute logarithm of negative number, clipping to {self.clip}')
            elif a_min < self.clip:
                self._LOGGER.info(f'Values in the interval [0, {self.clip}] will be clipped to {self.clip}')
            a = a.clip(min=self.clip)
        return np.log(a)
