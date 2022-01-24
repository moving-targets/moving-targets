from abc import ABC
from typing import Callable

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.masters.losses.loss import WeightedLoss
from moving_targets.util import probabilities
from moving_targets.util.errors import not_implemented_message


class ClassificationLoss(WeightedLoss, ABC):
    """Basic interface for a Moving Targets Classification Loss."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the loss.
        """
        super(ClassificationLoss, self).__init__(aggregation='sum_of_means', name=name)

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        if model_variables.ndim == 1:
            # for binary classification tasks the model variables are in the form of a 1d array, thus in this case we
            # can compute the loss by calling the strategy function two times, one per each binary value
            first_term = self._strategy(backend, 1 - numeric_variables, 1 - model_variables)
            second_term = self._strategy(backend, numeric_variables, model_variables)
            return first_term + second_term
        else:
            return self._strategy(backend, numeric_variables, model_variables)

    def _strategy(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        """Template method used as the loss strategy inside the _losses() method.

        This is useful since the majority of classification losses deal with probabilities and, therefore, with 2D
        variables that cannot be handled properly by the backends primitives which expect a 1D array. In order to deal
        with this, the default implementation of the _losses() abstract method is in charge of flattening the input
        data, calling the strategy on the flattened vectors and, finally, reshaping the variables in order to be
        consistent for the multiplication with the sample weights. When it comes to classification losses that use
        class values instead of probabilities (e.g., hamming distance), however, this cannot be done, thus an explicit
        _losses() implementation must be passed.
        """
        raise NotImplementedError(not_implemented_message(name='_strategy'))


class HammingDistance(ClassificationLoss):
    """Hamming Distance."""

    def __init__(self, task: str = 'auto', name: str = 'hamming_distance'):
        """
        :param task:
            The kind of classification task, either 'classification', 'labelling' or 'auto' for automatic task
            inference depending on the given numeric variables.

        :param name:
            The name of the metric.
        """
        super(HammingDistance, self).__init__(name=name)

        self.task: str = task
        """The kind of classification task."""

    def _strategy(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        # it should be 1 - sum([mv[i] * nv[i] for i in range(num_classes)]), but this creates problems due to the
        # "backend.sum()" call, therefore the defined strategy simply returns raw the losses which are then handled
        # in the _losses() function
        return model_variables * numeric_variables

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        # the hamming distance is computed on class values, not on probabilities, thus if we do not get integer values
        # we first obtain classes/labels then we bring them back to a onehot encoded version for compatibility (this is
        # necessary only for the multiclass task, since classes are given in a 1d vector but model variables are 2d)
        if not np.issubdtype(numeric_variables.dtype, np.integer):
            numeric_variables = probabilities.get_discrete(numeric_variables, task=self.task)
            if numeric_variables.ndim == 1 and model_variables.ndim == 2:
                numeric_variables = probabilities.get_onehot(vector=numeric_variables,
                                                             classes=model_variables.shape[1],
                                                             onehot_binary=True)
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
        super(CrossEntropy, self).__init__(name=name)

        self.clip_value: float = clip_value
        """The clipping value to be used to avoid numerical errors with the log."""

    def _strategy(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        numeric_variables = numeric_variables.clip(min=self.clip_value, max=1 - self.clip_value)
        return -model_variables * np.log(numeric_variables)


class ReversedCrossEntropy(ClassificationLoss):
    """Reversed Negative Log-Likelihood Loss."""

    def __init__(self, binary: bool = False, clip_value: float = 1e-15, name: str = 'crossentropy'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            If the model variables are binary, their value will be clipped, otherwise for continuous variables there
            will be no explicit clipping in order to let the backend take care of choosing the most ideal values.

        :param clip_value:
            The clipping value to be used to avoid numerical errors with the log.

            This will be used only for binary model variables.

        :param name:
            The name of the metric.
        """
        if binary:
            # if we expect binary variables, we clip them into [<clip_value>, 1 - <clip_value>] relying on the formula:
            #   (1 - 2 * <clip_value>) * <m> + <clip_value>
            # since this formula will evaluate to:
            #   > (1 - 2 * <clip_value>) * 0 + <clip_value> = <clip_value>, if <m> = 0
            #   > (1 - 2 * <clip_value>) * 1 + <clip_value> = 1 - <clip_value>, if <m> = 1
            clip = lambda m: (1.0 - 2 * clip_value) * m + clip_value
        else:
            # otherwise we simply return the continuous variables
            clip = lambda m: m

        super(ReversedCrossEntropy, self).__init__(name=name)

        self._clip: Callable = clip
        """The clipping function to be called before the logarithm."""

    def _strategy(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        clipped_variables = self._clip(model_variables)
        return -numeric_variables * backend.log(clipped_variables)


class SymmetricCrossEntropy(ClassificationLoss):
    """Reversed Negative Log-Likelihood Loss."""

    def __init__(self, binary: bool = False, clip_value: float = 1e-15, name: str = 'crossentropy'):
        """
        :param binary:
            Whether the model variables are expected to be binary or not.

            If the model variables are binary, their value will be clipped, otherwise for continuous variables there
            will be no explicit clipping in order to let the backend take care of choosing the most ideal values.

        :param clip_value:
            The clipping value to be used to avoid numerical errors with the log.

        :param name:
            The name of the metric.
        """

        if binary:
            # if we expect binary variables, we clip them into [<clip_value>, 1 - <clip_value>] relying on the formula:
            #   (1 - 2 * <clip_value>) * <m> + <clip_value>
            # since this formula will evaluate to:
            #   > (1 - 2 * <clip_value>) * 0 + <clip_value> = <clip_value>, if <m> = 0
            #   > (1 - 2 * <clip_value>) * 1 + <clip_value> = 1 - <clip_value>, if <m> = 1
            clip = lambda m: (1.0 - 2 * clip_value) * m + clip_value
        else:
            # otherwise we simply return the continuous variables
            clip = lambda m: m

        super(SymmetricCrossEntropy, self).__init__(name=name)

        self.clip_value: float = clip_value
        """The clipping value to be used to avoid numerical errors with the log."""

        self._clip: Callable = clip
        """The clipping function to be called before the logarithm."""

    def _strategy(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        clipped_variables = self._clip(model_variables)
        standard_ce = -model_variables * np.log(numeric_variables.clip(min=self.clip_value, max=1 - self.clip_value))
        reversed_ce = -numeric_variables * backend.log(clipped_variables)
        return standard_ce + reversed_ce
