from abc import ABC
from typing import Callable

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.masters.losses.loss import Loss
from moving_targets.util import probabilities


class ClassificationLoss(Loss, ABC):
    """Basic interface for a Moving Targets Classification Loss."""

    def __init__(self, loss_fn: Callable, aggregation: str, use_continuous_targets: bool, use_prob: bool, name: str):
        """
        :param loss_fn:
            Callable function of kind f(<backend>, <model_variables>, <numeric_variables>) -> <partial_losses> that is
            used as strategy inside the _losses() abstract method.

            This is useful since the majority of classification losses deal with probabilities and, therefore, with 2D
            variables that cannot be handled properly by the backends primitives which expect a 1D array. In order to
            deal with this, the default implementation of the _losses() abstract method is in charge of flattening the
            input data, calling the strategy on the flattened vectors and, finally, reshaping the variables in order to
            be consistent for the multiplication with the sample weights. When it comes to classification losses that
            use class values instead of probabilities (e.g., hamming distance), however, this cannot be done, thus an
            explicit _losses() implementation must be passed.

        :param aggregation:
            The kind of aggregation needed, either 'sum' or 'mean'.

        :param use_prob:
            Whether to use class probabilities or class targets.

        :param use_continuous_targets:
            Whether the model variables are expected to be continuous or binary.

        :param name:
            The name of the loss.
        """
        super(ClassificationLoss, self).__init__(aggregation=aggregation, name=name)

        self.use_continuous_targets: bool = use_continuous_targets
        """Whether the model variables are expected to be continuous or binary."""

        self.use_prob: bool = use_prob
        """Whether to use class probabilities or class targets."""

        self._loss_fn: Callable = loss_fn
        """Callable function of kind used as strategy inside the _losses() abstract method."""

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        if model_variables.ndim == 1:
            # for binary classification tasks the model variables are in the form of a 1d array, thus in this case we
            # can compute the loss by calling the strategy function two times, one per each binary value
            first_term = self._loss_fn(backend, 1 - numeric_variables, 1 - model_variables)
            second_term = self._loss_fn(backend, numeric_variables, model_variables)
            return first_term + second_term
        else:
            # for multiclass classification tasks, we need to acknowledge that MT losses must work with both original
            # targets and predictions thus, since predictions usually come as 2d floating point array of probabilities
            # while class targets usually come as an integer vector, we must onehot encode the latter for compatibility
            if numeric_variables.ndim == 1:
                numeric_variables = probabilities.get_onehot(numeric_variables, classes=model_variables.shape[1])
            return self._loss_fn(backend, numeric_variables, model_variables)


class BinaryRegressionLoss(ClassificationLoss):
    """Basic interface for a Moving Targets Regression Loss for Classification Tasks with Binary Variables.

    This allows to deal with both absolute and squared errors in a faster way in this simplified scenario in which the
    model variables can only have an integer value in {0, 1}, while the numeric variables can only have a continuous
    value in [0, 1]."""

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

        # noinspection PyUnusedLocal
        def loss_fn(backend, numeric_variables, model_variables):
            # Given the model variables <m> and the numeric variables <n>, in both scenarios we have that:
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
            # the formulae as "m - 2mn + n^{norm}", which is a linear expressions that can be handled in a faster way.
            return model_variables - 2 * model_variables * numeric_variables + numeric_variables ** norm

        super(BinaryRegressionLoss, self).__init__(loss_fn=loss_fn,
                                                   aggregation=aggregation,
                                                   use_continuous_targets=False,
                                                   use_prob=True,
                                                   name=name)


class BinarySAE(BinaryRegressionLoss):
    """Sum of Absolute Errors Loss for Binary Variables."""

    def __init__(self, name: str = 'sum_of_absolute_errors'):
        """
        :param name:
            The name of the metric.
        """
        super(BinarySAE, self).__init__(norm=1, aggregation='sum', name=name)


class BinarySSE(BinaryRegressionLoss):
    """Sum of Squared Errors Loss for Binary Variables."""

    def __init__(self, name: str = 'sum_of_squared_errors'):
        """
        :param name:
            The name of the metric.
        """
        super(BinarySSE, self).__init__(norm=2, aggregation='sum', name=name)


class BinaryMAE(BinaryRegressionLoss):
    """Mean Absolute Error Loss for Binary Variables."""

    def __init__(self, name: str = 'mean_absolute_error'):
        """
        :param name:
            The name of the metric.
        """
        super(BinaryMAE, self).__init__(norm=1, aggregation='mean', name=name)


class BinaryMSE(BinaryRegressionLoss):
    """Mean Squared Error Loss for Binary Variables."""

    def __init__(self, name: str = 'mean_squared_error'):
        """
        :param name:
            The name of the metric.
        """
        super(BinaryMSE, self).__init__(norm=2, aggregation='mean', name=name)


class HammingDistance(ClassificationLoss):
    """Hamming Distance."""

    def __init__(self, multi_label: bool = False, name: str = 'hamming_distance'):
        """
        :param multi_label:
            Whether the classification task must handle multiple labels or not.

        :param name:
            The name of the metric.
        """

        # noinspection PyUnusedLocal
        def loss_fn(backend, numeric_variables, model_variables):
            # it should be 1 - sum([mv[i] * nv[i] for i in range(num_classes)]), but this creates problems due to the
            # "backend.sum()" call, therefore the defined strategy simply returns raw the losses which are then handled
            # in the _losses() function
            return model_variables * numeric_variables

        super(HammingDistance, self).__init__(loss_fn=loss_fn,
                                              aggregation='mean',
                                              use_continuous_targets=False,
                                              use_prob=False,
                                              name=name)

        self.multi_label: bool = multi_label
        """Whether the classification task must handle multiple labels or not."""

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        # the hamming distance is computed on class values, not on probabilities
        numeric_variables = probabilities.get_classes(numeric_variables, multi_label=self.multi_label)
        losses = super(HammingDistance, self)._losses(backend, numeric_variables, model_variables)
        # as said before, the "1 - ..." part is moved here due to computational problems, in particular, for binary
        # tasks we just need to do (1 - losses) while for multiclass tasks it must be (1 - sum(losses, axis=1))
        return 1 - (losses if losses.ndim == 1 else np.array([backend.sum(row) for row in losses]))


class CrossEntropy(ClassificationLoss):
    """Negative Log-Likelihood Loss."""

    def __init__(self, clip_value: float = 1e-15, name: str = 'crossentropy'):
        """
        :param clip_value:
            The clipping value to be used to avoid numerical errors with the log.

        :param name:
            The name of the metric.
        """

        # noinspection PyUnusedLocal
        def loss_fn(backend, numeric_variables, model_variables):
            numeric_variables = numeric_variables.clip(min=clip_value, max=1 - clip_value)
            return -model_variables * np.log(numeric_variables)

        super(CrossEntropy, self).__init__(loss_fn=loss_fn,
                                           aggregation='mean',
                                           use_continuous_targets=False,
                                           use_prob=True,
                                           name=name)


class ReversedCrossEntropy(ClassificationLoss):
    """Reversed Negative Log-Likelihood Loss."""

    def __init__(self, name: str = 'reversed_crossentropy'):
        """
        :param name:
            The name of the metric.
        """

        def loss_fn(backend, numeric_variables, model_variables):
            return -numeric_variables * backend.log(model_variables)

        super(ReversedCrossEntropy, self).__init__(loss_fn=loss_fn,
                                                   aggregation='mean',
                                                   use_continuous_targets=True,
                                                   use_prob=True,
                                                   name=name)


class SymmetricCrossEntropy(ClassificationLoss):
    """Reversed Negative Log-Likelihood Loss."""

    def __init__(self, clip_value: float = 1e-15, name: str = 'symmetric_crossentropy'):
        """
        :param clip_value:
            The clipping value to be used to avoid numerical errors with the log.

        :param name:
            The name of the metric.
        """

        def loss_fn(backend, numeric_variables, model_variables):
            standard_ce = -model_variables * np.log(numeric_variables.clip(min=clip_value, max=1 - clip_value))
            reversed_ce = -numeric_variables * backend.log(model_variables)
            return standard_ce + reversed_ce

        super(SymmetricCrossEntropy, self).__init__(loss_fn=loss_fn,
                                                    aggregation='mean',
                                                    use_continuous_targets=True,
                                                    use_prob=True,
                                                    name=name)
