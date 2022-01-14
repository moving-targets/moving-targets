"""Basic Optimizer Interface."""
from typing import List

from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Number


class Optimizer:
    """Basic interface for a Moving Targets Master Optimizer."""

    def __init__(self, initial_value: Number):
        """
        :param initial_value:
            The initial value for the hyper-parameter to optimize.
        """

        self.values: List[Number] = [initial_value]
        """The history of updated values."""

    def __call__(self) -> Number:
        """Gets the current value of the hyper-parameter to optimize.

        :return:
            The value of the hyper-parameter.
        """
        return self.values[-1]

    def update(self, macs, x, y, p):
        """Updates the value of the hyper-parameter to optimize according to the given strategy.

        :param macs:
            The `MACS` instance encapsulating the master.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.
        """
        new_value = self._strategy(value=self.__call__(), macs=macs, x=x, y=y, p=p)
        self.values.append(new_value)

    def _strategy(self, value, macs, x, y, p) -> Number:
        """Defines the optimization strategy.

        :param value:
            The current value of the hyper-parameter to optimize.

        :param macs:
            The `MACS` instance encapsulating the master.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            The next value of the hyper-parameter to optimize.
        """
        raise NotImplementedError(not_implemented_message(name='_strategy'))
