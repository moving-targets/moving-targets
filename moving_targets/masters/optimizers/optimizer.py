"""Basic Optimizer Interface."""
from typing import Callable

import numpy as np


class Optimizer:
    """Basic interface for a Moving Targets Master Optimizer."""

    def __init__(self, base):
        """
        :param base:
            Either a fixed floating point value representing the initial value for the hyper-parameter to optimize, or
            a wrapped custom optimizer which returns a dynamic value to reduce by the given factor after each iteration.
        """

        self.base: Callable = base if isinstance(base, Optimizer) else (lambda macs, x, y, p: base)
        """The base value of the hyper-parameter to optimize or a base optimizer wrapper."""

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
        return self.base(macs=macs, x=x, y=y, p=p)
