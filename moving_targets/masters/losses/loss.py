"""Basic Loss Interface."""
from typing import Optional, Any, Callable

import numpy as np

from moving_targets.masters.backends import Backend
from moving_targets.util.errors import not_implemented_message


class Loss:
    """Basic interface for a Moving Targets Master Loss."""

    def __init__(self, aggregation: str, name: str):
        """
        :param aggregation:
            The kind of aggregation needed, either 'sum' or 'mean'.

        :param name:
            The name of the loss.
        """
        assert aggregation in ['sum', 'mean'], f"aggregation must be either 'sum' or 'mean', but is {aggregation}"
        super(Loss, self).__init__()

        self.__name__: str = name
        """The name of the loss."""

        self.aggregation: Callable = (lambda agg, num: agg) if aggregation == 'sum' else (lambda agg, num: agg / num)
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
        return self.aggregation(agg=backend.sum(losses), num=len(numeric_variables))

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
