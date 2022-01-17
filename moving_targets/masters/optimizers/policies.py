from typing import Union

import numpy as np

from moving_targets.masters.optimizers.optimizer import Optimizer


class ConstantSlope(Optimizer):
    """Optimizer which reduces the hyper-parameter value by the iteration number."""

    def __init__(self, base: Union[float, Optimizer]):
        """
        :param base:
            Either a fixed floating point value representing the initial value for the hyper-parameter to optimize, or
            a wrapped custom optimizer which returns a dynamic value to reduce by the given factor after each iteration.
        """
        super(ConstantSlope, self).__init__(base=base)

    def __call__(self, macs, x, y: np.ndarray, p: np.ndarray) -> float:
        value = super(ConstantSlope, self).__call__(macs, x, y, p)
        return value / macs.iteration


class ExponentialSlope(Optimizer):
    """Optimizer which reduces the hyper-parameter value by the same factor at each iteration."""

    def __init__(self, base: Union[float, Optimizer], slope: float = 2):
        """
        :param base:
            Either a fixed floating point value representing the initial value for the hyper-parameter to optimize, or
            a wrapped custom optimizer which returns a dynamic value to reduce by the given factor after each iteration.

        :param slope:
            The base of the exponential slope used to decrease the hyper-parameter value (e.g., starting from a value
            of 10 and using a slope of 2, the updates will lead to the sequence 10 -> 5 -> 2.5 -> 1.25 -> ...).
        """
        super(ExponentialSlope, self).__init__(base=base)

        self.slope: float = slope
        """The slope used to decrease the hyper-parameter value"""

    def __call__(self, macs, x, y: np.ndarray, p: np.ndarray) -> float:
        value = super(ExponentialSlope, self).__call__(macs, x, y, p)
        return value / (self.slope ** macs.iteration)
