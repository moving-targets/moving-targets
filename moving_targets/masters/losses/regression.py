from typing import Callable

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.masters.losses.loss import Loss


class RegressionLoss(Loss):
    """Basic interface for a Moving Targets Regression Loss."""

    def __init__(self, norm: int, aggregation: str, name: str = 'absolute_errors'):
        """
        :param norm:
            Either '1' for absolute errors or '2' for squared errors.

        :param aggregation:
            The kind of aggregation needed, either 'sum' or 'mean'.

        :param name:
            The name of the loss.
        """
        assert norm in [1, 2], f"norm must be either '1' for absolute errors or '2' for squared errors, but is {norm}"
        super(RegressionLoss, self).__init__(aggregation=aggregation, name=name)

        self._norm: Callable = (lambda b, v: b.abs(v)) if norm == 1 else (lambda b, v: b.square(v))
        """The norm strategy."""

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        return self._norm(b=backend, v=model_variables - numeric_variables)


class SAE(RegressionLoss):
    """Sum of Absolute Errors Loss."""

    def __init__(self, name: str = 'sum_of_absolute_errors'):
        """
        :param name:
            The name of the loss.
        """
        super(SAE, self).__init__(norm=1, aggregation='sum', name=name)


class SSE(RegressionLoss):
    """Sum of Squared Errors Loss."""

    def __init__(self, name: str = 'sum_of_squared_errors'):
        """
        :param name:
            The name of the loss.
        """
        super(SSE, self).__init__(norm=2, aggregation='sum', name=name)


class MAE(RegressionLoss):
    """Mean Absolute Error Loss."""

    def __init__(self, name: str = 'mean_absolute_error'):
        """
        :param name:
            The name of the loss.
        """
        super(MAE, self).__init__(norm=1, aggregation='mean', name=name)


class MSE(RegressionLoss):
    """Mean Squared Error Loss."""

    def __init__(self, name: str = 'mean_squared_error'):
        """
        :param name:
            The name of the loss.
        """
        super(MSE, self).__init__(norm=2, aggregation='mean', name=name)
