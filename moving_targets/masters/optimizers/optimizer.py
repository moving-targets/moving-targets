"""Basic Optimizer Interface."""

import numpy as np

from moving_targets.util.errors import not_implemented_message


class Optimizer:
    """Basic interface for a Moving Targets Master Optimizer."""

    def __init__(self, initial_value: float):
        """
        :param initial_value:
            The initial value for the hyper-parameter to optimize.
        """

        self.initial_value: float = initial_value
        """The initial value for the hyper-parameter to optimize."""

    def __call__(self, macs, x, y: np.ndarray, p: np.ndarray) -> float:
        """Returns the current value for the hyper-parameter, computed via the custom optimization strategy.

        :param macs:
            The `MACS` instance encapsulating the master.

        :param x:
            The training samples.

        :param y:
            The training labels.

        :param p:
            The `Learner` predictions.

        :return:
            The current value of the hyper-parameter to optimize.
        """
        raise NotImplementedError(not_implemented_message(name='__call__'))
