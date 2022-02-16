"""Basic Loss Interface."""
from typing import Optional, Any, Callable, Union

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import not_implemented_message


class Loss:
    """Basic interface for a Moving Targets Master Loss."""

    def __init__(self, name: str):
        """
        :param name:
            The name of the loss.
        """

        self.__name__: str = name
        """The name of the loss."""

    def __call__(self,
                 backend: Backend,
                 numeric_variables: np.ndarray,
                 model_variables: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None) -> Any:
        """Computes the loss value.

        :param backend:
            The `Backend` instance used to compute the loss.

        :param numeric_variables:
            The array of numeric ground truths.

        :param model_variables:
            The array of backend variables.

        :param sample_weight:
            The (optional) array of sample weights.

        :return:
            Either a single value/expression or an array of values/expressions.
        """
        raise NotImplementedError(not_implemented_message(name='__call__'))


class WeightedLoss(Loss):
    """Abstract class for a Moving Targets Master Loss which already handles Sample Weights."""

    def __init__(self, aggregation: Union[str, Callable], name: str):
        """
        :param aggregation:
            The kind of aggregation needed or a `Callable` function of type f(<backend>, <losses>) -> <result>.

            If the aggregation is 'sum', computes the sum of all losses over all dimensions.
            If the aggregation is 'mean', computes the mean of all losses over all dimensions.
            If the aggregation is 'mean_of_sums', aggregates each column (feature) via sum, then computes the mean.
            If the aggregation is 'sum_of_means', aggregates each column (feature) via mean, then computes the sum.

        :param name:
            The name of the loss.
        """
        if aggregation == 'sum':
            aggregation = lambda backend, losses: backend.sum(losses)
        elif aggregation == 'mean':
            aggregation = lambda backend, losses: backend.mean(losses)
        elif aggregation == 'mean_of_sums':
            # reshape the losses to (len(losses), -1) in order to always have bi-dimensional arrays
            aggregation = lambda backend, losses: backend.mean(backend.sum(losses.reshape((len(losses), -1)), axis=-1))
        elif aggregation == 'sum_of_means':
            aggregation = lambda backend, losses: backend.sum(backend.mean(losses.reshape((len(losses), -1)), axis=-1))
        elif not isinstance(aggregation, Callable):
            raise AssertionError(f"Unsupported aggregation strategy '{aggregation}'")

        super(WeightedLoss, self).__init__(name=name)

        self.aggregation: Callable = aggregation
        """The aggregation strategy."""

    def __call__(self,
                 backend: Backend,
                 numeric_variables: np.ndarray,
                 model_variables: np.ndarray,
                 sample_weight: Optional[np.ndarray] = None) -> Any:
        """Core method used to compute the master loss.

        :param backend:
            The `Backend` instance.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weight:
            The sample weights associated to each sample.

        :return:
            The loss expression.
        """
        losses = self._losses(backend=backend, numeric_variables=numeric_variables, model_variables=model_variables)
        if sample_weight is not None:
            # normalize weights
            sample_weight = len(sample_weight) * np.array(sample_weight) / np.sum(sample_weight)
            # handle bi-dimensional outputs
            if losses.ndim == 2:
                sample_weight = np.repeat(sample_weight.reshape((-1, 1)), repeats=losses.shape[1], axis=1)
            # multiply partial loss per respective weight
            losses = sample_weight * losses
        return self.aggregation(backend, losses)

    def _losses(self, backend: Backend, numeric_variables: np.ndarray, model_variables: np.ndarray) -> np.ndarray:
        """Computes the partial losses computed over the pairs of variables.

        :param backend:
            The `Backend` instance.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :return:
            The array of partial losses computed over the pairs of variables.
        """
        raise NotImplementedError(not_implemented_message(name='_losses'))
