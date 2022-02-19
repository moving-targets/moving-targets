"""Moving Targets Optimizers."""
from typing import Optional, List

import numpy as np

from moving_targets.util.errors import not_implemented_message


class Optimizer:
    """Basic interface for a Moving Targets Master Optimizer."""

    def __init__(self, base: float = 1):
        """
        :param base:
            A floating point number representing the base (initial) value.
        """

        self.base: float = base
        """The base value of the optimizer."""

        self.value: Optional[float] = None
        """The last value computed from the optimizer."""

        self.history: List[float] = []
        """The history of past values yielded by the optimizer since the last initialization."""

    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> float:
        """Runs the optimization strategy then returns the computed value.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            The newly computed value.
        """
        assert self.base is not None, "There is not base value for the optimizer, please call the 'initialize' method"
        self.value = self._next(x=x, y=y, p=p)
        self.history.append(self.value)
        return self.value

    def _next(self, x, y: np.ndarray, p: np.ndarray) -> float:
        """Implements the optimization strategy by returning the next computed value.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            The computed value.
        """
        raise NotImplementedError(not_implemented_message(name='_next'))


class Constant(Optimizer):
    """Optimizer which returns a constant value."""

    def _next(self, x, y: np.ndarray, p: np.ndarray):
        return self.base


class Harmonic(Optimizer):
    """Optimizer which updates the values according to the harmonic series, i.e., by dividing the base value for the
    iteration number. E.g., starting from 10, the sequence will be: 10 -> 5 -> 3.33 -> 2.5 -> 2 -> ..."""

    def _next(self, x, y: np.ndarray, p: np.ndarray):
        return self.base / (len(self.history) + 1)


class Geometric(Optimizer):
    """Optimizer which updates the values according to the harmonic series, i.e., by dividing the last value for the
    given slope. E.g., starting from 10 and with a slope of 2, the sequence will be: 10 -> 5 -> 2.5 -> 1.25 -> ..."""

    def __init__(self, base: float = 1, slope: float = 2):
        """
        :param base:
            A floating point number representing the base (initial) value.

        :param slope:
            The base of the slope used to decrease the hyper-parameter value.
        """
        super(Geometric, self).__init__(base=base)

        self.slope: float = slope
        """The slope used to decrease the hyper-parameter value."""

    def _next(self, x, y: np.ndarray, p: np.ndarray) -> float:
        return self.base if self.value is None else self.value / self.slope


aliases: dict = {
    'constant': Constant,
    'harmonic': Harmonic,
    'geometric': Geometric,
}
"""Dictionary which associates to each optimizer alias the respective optimizer type."""
