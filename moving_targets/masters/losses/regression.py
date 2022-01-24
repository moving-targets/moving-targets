from typing import Callable

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.masters.losses.loss import WeightedLoss


class RegressionLoss(WeightedLoss):
    """Basic interface for a Moving Targets Regression Loss."""

    def __init__(self, norm: int, sum_samples: bool, sum_features: bool, binary: bool, name: str = 'absolute_errors'):
        """
        :param norm:
            Either '1' for absolute errors or '2' for squared errors.

        :param sum_samples:
            Whether to aggregate the partial losses via sum or via mean.

        :param sum_features:
            Whether to sum the results of each feature or to get their mean value, in case of multiple features.

        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param name:
            The name of the loss.
        """
        assert norm in [1, 2], f"'norm' should be either '1' for absolute errors or '2' for squared errors, got {norm}"
        # handle aggregation by considering strategy to aggregate both samples and features
        if sum_samples:
            aggregation = 'sum' if sum_features else 'mean_of_sums'
        else:
            aggregation = 'sum_of_means' if sum_features else 'mean'
        super(RegressionLoss, self).__init__(aggregation=aggregation, name=name)

        if binary:
            # given the model variables <m> and the numeric variables <n>, in this scenario we have that:
            #
            # > abs(m, n) = m * (1 - n) + (1 - m) * n =
            #             = m - 2mn + n
            #
            # > sqr(m, n) = [m * (1 - n) + (1 - m) * n]^2 =
            #             = [m * (1 - n)]^2        + [(1 - m) * n]^2	    - 2 * [m * (1 - n) * (1 - m) * n] =
            #             = [m^2 * (1 - 2n + n^2)] + [n^2 * (1 - 2m + m^2)] - 2 * [m * (1 - m) * (1 - n) * n] =
            #             = [m - 2mn + mn^2] 	   + [n^2 - 2mn^2 + mn^2]   - 2 * [(m - m) * (n - n^2)] = --> as m^2 = m
            #             = [m - 2mn + mn^2]       + [n^2 - mn^2]           - 2 * [0] =
            #             = m - 2mn + n^2
            #
            # thus, since the absolute values represent norm 1 and the squared values represent norm 2, we can rewrite
            # the formulae as "m - 2mn + n^{norm}", which is a linear expressions that can be handled in a faster way
            strategy = lambda b, n, m: m - 2 * m * n + n ** norm
        elif norm == 1:
            # if we have continuous targets with norm one, we apply the absolute value of the differences
            strategy = lambda b, n, m: b.abs(m - n)
        else:
            # if we have continuous targets with norm one, we square the differences
            strategy = lambda b, n, m: b.square(m - n)

        self._strategy: Callable = strategy
        """The norm strategy."""

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        return self._strategy(b=backend, n=numeric_variables, m=model_variables)


class SAE(RegressionLoss):
    """Sum of Absolute Errors Loss."""

    def __init__(self, binary: bool = False, sum_features: bool = True, name: str = 'sum_of_absolute_errors'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param sum_features:
            Whether to sum the results of each feature or to get their mean value, in case of multiple features.

        :param name:
            The name of the loss.
        """
        super(SAE, self).__init__(norm=1, sum_samples=True, sum_features=sum_features, binary=binary, name=name)


class SSE(RegressionLoss):
    """Sum of Squared Errors Loss."""

    def __init__(self, binary: bool = False, sum_features: bool = True, name: str = 'sum_of_squared_errors'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param sum_features:
            Whether to sum the results of each feature or to get their mean value, in case of multiple features.

        :param name:
            The name of the loss.
        """
        super(SSE, self).__init__(norm=2, sum_samples=True, sum_features=sum_features, binary=binary, name=name)


class MAE(RegressionLoss):
    """Mean Absolute Error Loss."""

    def __init__(self, binary: bool = False, sum_features: bool = True, name: str = 'mean_absolute_error'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param sum_features:
            Whether to sum the results of each feature or to get their mean value, in case of multiple features.

        :param name:
            The name of the loss.
        """
        super(MAE, self).__init__(norm=1, sum_samples=False, sum_features=sum_features, binary=binary, name=name)


class MSE(RegressionLoss):
    """Mean Squared Error Loss."""

    def __init__(self, binary: bool = False, sum_features: bool = True, name: str = 'mean_squared_error'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            Knowing that the model variables are binary allows to deal with error computation in a faster way since it
            will lead to a simplified scenario in which the model variables can only have an integer value in {0, 1},
            while the numeric variables can only have a continuous value in [0, 1].

        :param sum_features:
            Whether to sum the results of each feature or to get their mean value, in case of multiple features.

        :param name:
            The name of the loss.
        """
        super(MSE, self).__init__(norm=2, sum_samples=False, sum_features=sum_features, binary=binary, name=name)
