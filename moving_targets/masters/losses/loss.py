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
                 sample_weights: Optional[np.ndarray] = None) -> Any:
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
            aggregation = lambda backend, losses: backend.sum(losses) / losses.size
        elif aggregation == 'mean_of_sums':
            # if losses is a one-dimensional vector, this corresponds to losses.sum()
            # if losses is a two-dimensional vector, this corresponds to losses.mean(axis=0).sum()
            # in both cases, this can be achieved by computing losses.sum() / losses.shape[1]
            aggregation = lambda backend, losses: backend.sum(losses) / (1 if losses.ndim == 1 else losses.shape[1])
        elif aggregation == 'sum_of_means':
            # if losses is a one-dimensional vector, this corresponds to losses.mean()
            # if losses is a two-dimensional vector, this corresponds to losses.sum(axis=1).mean()
            # in both cases, this can be achieved by computing losses.sum() / losses.shape[0]
            aggregation = lambda backend, losses: backend.sum(losses) / losses.shape[0]
        elif not isinstance(aggregation, Callable):
            raise AssertionError(f"Unsupported aggregation strategy '{aggregation}'")

        super(WeightedLoss, self).__init__(name=name)

        self.aggregation: Callable = aggregation
        """The aggregation strategy."""

    def __call__(self,
                 backend: Backend,
                 numeric_variables: np.ndarray,
                 model_variables: np.ndarray,
                 sample_weights: Optional[np.ndarray] = None) -> Any:
        """Core method used to compute the master loss.

        :param backend:
            The `Backend` instance.

        :param numeric_variables:
            The real-number variables (i.e., the ground truths).

        :param model_variables:
            The model variables (i.e., the predictions).

        :param sample_weights:
            The sample weights associated to each sample.

        :return:
            The loss expression.
        """
        losses = self._losses(backend=backend, numeric_variables=numeric_variables, model_variables=model_variables)
        if sample_weights is not None:
            # normalize weights
            sample_weights = len(sample_weights) * np.array(sample_weights) / np.sum(sample_weights)
            # handle bi-dimensional outputs
            if losses.ndim == 2:
                sample_weights = np.repeat(sample_weights.reshape((-1, 1)), repeats=losses.shape[1], axis=1)
            # multiply partial loss per respective weight
            losses = sample_weights * losses
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
