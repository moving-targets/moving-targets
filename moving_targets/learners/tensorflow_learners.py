from typing import Optional, List, Union, Dict

import numpy as np

from moving_targets.learners import Learner
from moving_targets.util.errors import MissingDependencyError
from moving_targets.util.scalers import Scaler


class TensorflowLearner(Learner):
    """Wrapper for a custom Tensorflow/Keras model."""

    def __init__(self,
                 model,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **fit_kwargs):
        """
        :param model:
            The Tensorflow/Keras model which should have been already compiled.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param fit_kwargs:
            Custom arguments to be passed to the model '.fit()' method.
        """
        super(TensorflowLearner, self).__init__(x_scaler=x_scaler, y_scaler=y_scaler, stats=stats)

        self.model = model
        """The Tensorflow/Keras model."""

        self.fit_kwargs = fit_kwargs
        """Custom arguments to be passed to the model '.fit()' method."""

    def _fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model.fit(x, y, sample_weight=sample_weight, **self.fit_kwargs)

    def _predict(self, x) -> np.ndarray:
        return self.model.predict(x).squeeze()


class MultiLayerPerceptron(TensorflowLearner):
    """Tensorflow/Keras Dense Neural Network Wrapper"""

    def __init__(self,
                 loss: str,
                 output_units: int = 1,
                 output_activation: Optional[str] = None,
                 hidden_units: List[int] = (128,),
                 hidden_activation: Optional[str] = 'relu',
                 optimizer: str = 'adam',
                 metrics: Optional[List] = None,
                 loss_weights: Union[None, List, Dict] = None,
                 weighted_metrics: Optional[List] = None,
                 run_eagerly: bool = False,
                 epochs: int = 1,
                 validation_split: float = 0.,
                 batch_size: Optional[int] = None,
                 class_weight: Optional[Dict] = None,
                 callbacks: Optional[List] = None,
                 verbose: Union[bool, str] = 'auto',
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
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

        :param callbacks:
            The list of keras callbacks for the training phase.

        :param verbose:
            Whether or not to print information during the neural network training.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

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
                                                   verbose=verbose,
                                                   x_scaler=x_scaler,
                                                   y_scaler=y_scaler,
                                                   stats=stats)
