from moving_targets.masters.optimizers.optimizer import Optimizer
from moving_targets.util.typing import Number


class ConstantValue(Optimizer):
    """Constant optimizer which can be used for both alpha and beta."""

    def _strategy(self, value, macs, x, y, p) -> Number:
        return value


class ConstantSlope(Optimizer):
    """Optimizer which reduces the hyper-parameter value by the same factor after each update."""

    def __init__(self, initial_value: Number, slope: Number = 2):
        """
        :param initial_value:
            The initial value for the hyper-parameter to optimize.

        :param slope:
            The slope used to decrease the hyper-parameter value (e.g., starting from a value of 10 and using a slope
             of 2, the updates will lead to the sequence 10 -> 5 -> 2.5 -> 1.25 -> ...).
        """

        super(ConstantSlope, self).__init__(initial_value=initial_value)

        self.factor = slope
        """The slope used to decrease the hyper-parameter value"""

    def _strategy(self, value, macs, x, y, p) -> Number:
        return value / self.factor
