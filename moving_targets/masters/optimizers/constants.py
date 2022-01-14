import numpy as np

from moving_targets.masters.optimizers.optimizer import Optimizer


class ConstantValue(Optimizer):
    """Constant optimizer which can be used for both alpha and beta."""

    def __call__(self, macs, x, y: np.ndarray, p: np.ndarray) -> float:
        return self.initial_value


class ConstantSlope(Optimizer):
    """Optimizer which reduces the hyper-parameter value by the same factor after each iteration."""

    def __init__(self, initial_value: float, slope: float = 2):
        """
        :param initial_value:
            The initial value for the hyper-parameter to optimize.

        :param slope:
            The slope used to decrease the hyper-parameter value (e.g., starting from a value of 10 and using a slope
             of 2, the updates will lead to the sequence 10 -> 5 -> 2.5 -> 1.25 -> ...).
        """

        super(ConstantSlope, self).__init__(initial_value=initial_value)

        self.slope = slope
        """The slope used to decrease the hyper-parameter value"""

    def __call__(self, macs, x, y: np.ndarray, p: np.ndarray) -> float:
        return self.initial_value / (self.slope ** macs.iteration)
