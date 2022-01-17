from typing import Optional, Any, Tuple, List, Union, Dict

import numpy as np

from moving_targets.learners import Learner
from moving_targets.util.errors import MissingDependencyError


class TensorflowLearner(Learner):
    """Wrapper for a custom Tensorflow/Keras model."""

    def __init__(self, model, stats: Union[bool, List[str]] = False, **fit_kwargs):
        """
        :param model:
            The Tensorflow/Keras model which should have been already compiled.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param fit_kwargs:
            Custom arguments to be passed to the fit function (they may be overwritten when calling `self.fit()`).
        """
        super(TensorflowLearner, self).__init__(stats=stats)

        self.model = model
        """The Tensorflow/Keras model."""

        self.fit_kwargs = fit_kwargs
        """Custom arguments to be passed to the fit function."""

    def fit(self, x, y, **additional_kwargs):
        fit_args = self.fit_kwargs.copy()
        fit_args.update(additional_kwargs)
        self.model.fit(x, y, **fit_args)

    def predict(self, x) -> Any:
        return self.model.predict(x)


class MultiLayerPerceptron(TensorflowLearner):
    """Tensorflow/Keras Dense Neural Network Wrapper"""

    def __init__(self,
                 loss: str,
                 output_units: int,
                 output_activation: Optional[str],
                 hidden_units: Tuple[int] = (128,),
                 hidden_activation: Optional[str] = 'relu',
                 optimizer: str = 'adam',
                 metrics: Optional[List] = None,
                 loss_weights: Optional[Union[List, Dict]] = None,
                 weighted_metrics: Optional[List] = None,
                 run_eagerly: bool = False,
                 epochs: int = 1,
                 validation_split: float = 0.,
                 batch_size: Optional[int] = None,
                 class_weight: Optional[Dict] = None,
                 sample_weight: Optional[np.ndarray] = None,
                 callbacks: Optional[List] = None,
                 verbose: Union[bool, str] = 'auto',
                 stats: Union[bool, List[str]] = False):
        """
        :param loss:
            The neural network loss function.

        :param output_units:
            The neural network number of output units.

        :param output_activation:
            The neural network output activation.

        :param hidden_units:
            The tuple of neural network hidden units.

        :param hidden_activation:
            The neural network hidden activations.

        :param optimizer:
            The neural network optimizer.

        :param metrics:
            The list of keras metrics for the evaluation phase.

        :param loss_weights:
            Optional list or dictionary specifying scalar coefficients to weight the loss contributions.

        :param weighted_metrics:
            List of metrics to be evaluated and weighted by `sample_weight` or `class_weight`.

        :param run_eagerly:
            Whether or not to run tensorflow in eager mode.

        :param epochs:
            The number of training epochs.

        :param validation_split:
            The validation split for neural network training.

        :param batch_size:
            The batch size for neural network training.

        :param class_weight:
            Optional dictionary mapping class indices to a weight, used for weighting the loss function during training.

        :param sample_weight:
            Optional array of weights for the training samples, used for weighting the loss function during training.

        :param callbacks:
            The list of keras callbacks for the training phase.

        :param verbose:
            Whether or not to print information during the neural network training.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.
        """
        try:
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.models import Sequential
        except ModuleNotFoundError:
            raise MissingDependencyError(package='tensorflow')

        network = Sequential()
        for units in hidden_units:
            network.add(layer=Dense(units=units, activation=hidden_activation))
        network.add(layer=Dense(units=output_units, activation=output_activation))
        network.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        weighted_metrics=weighted_metrics,
                        run_eagerly=run_eagerly)

        super(MultiLayerPerceptron, self).__init__(model=network,
                                                   epochs=epochs,
                                                   batch_size=batch_size,
                                                   validation_split=validation_split,
                                                   callbacks=callbacks,
                                                   class_weight=class_weight,
                                                   sample_weight=sample_weight,
                                                   verbose=verbose,
                                                   stats=stats)
