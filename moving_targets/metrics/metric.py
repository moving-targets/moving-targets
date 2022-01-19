"""Basic Metric Interface."""
from typing import Dict, Union

import numpy as np

from moving_targets.util.errors import not_implemented_message


class Metric:
    """Basic interface for a Moving Targets Metric."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the metric.
        """
        super(Metric, self).__init__()

        self.__name__: str = name
        """The name of the metric."""

    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> Union[float, Dict[str, float]]:
        """Core method used to compute the metric value.

        :param x:
            The input matrix.

        :param y:
            The vector of ground truths.

        :param p:
            The vector of predictions.

        :return:
            Either a single value representing the metric, or a dictionary of multiple values for multi-metrics.
        """
        raise NotImplementedError(not_implemented_message(name='__call__'))
