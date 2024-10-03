from typing import Optional, List, Union, Dict, Any

import numpy as np

from moving_targets.learners import Learner
from moving_targets.util.errors import MissingDependencyError
from moving_targets.util.scalers import Scaler


class TensorflowLearner(Learner):
    """Wrapper for a custom Tensorflow/Keras model."""

    def __init__(self,
                 model,
                 loss: Any,
                 optimizer: Any = 'adam',
                 metrics: Optional[List] = None,
                 loss_weights: Union[None, List, Dict] = None,
                 weighted_metrics: Optional[List] = None,
                 run_eagerly: bool = False,
                 warm_start: Union[bool, int] = False,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **fit_kwargs):
        """
        :param model:
            The tensorflow/keras model structure.

        :param loss:
            The neural network loss function (either a string or a loss object).

        :param optimizer:
            The neural network optimizer (either a string or an optimizer object).

        :param metrics:
            The list of tensorflow/keras metrics for the evaluation phase.

        :param loss_weights:
            Optional list or dictionary specifying scalar coefficients to weight the loss contributions.

        :param weighted_metrics:
            List of metrics to be evaluated and weighted by `sample_weight` or `class_weight`.

        :param run_eagerly:
            Whether to run tensorflow in eager mode.

        :param warm_start:
            Handles the warming start policy after each iteration. If 0 (or False), the weights are reinitialized and
            the model is trained from scratch. If 1 (or True), only the optimized is reinitialized, while the weights
            are kept from the previous iteration. If 2, neither the weights nor the optimizer are reinitialized.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether to log statistics, or a list of parameters whose
            statistics must be logged.

        :param fit_kwargs:
            Custom arguments to be passed to the model '.fit()' method.
        """
        super(TensorflowLearner, self).__init__(mask=mask, x_scaler=x_scaler, y_scaler=y_scaler, stats=stats)

        self.model = model
        """The tensorflow/keras model structure."""

        assert warm_start in [0, 1, 2, True, False], "'warm_start' must be either a boolean or an integer in {0, 1, 2}"
        self.warm_start: int = warm_start if isinstance(warm_start, int) else int(warm_start)
        """The warm start level."""

        self.compile_kwargs: Dict[str, Any] = dict(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly
        )
        """Custom arguments to be passed to the '.compile()' method."""

        self.fit_kwargs: Dict[str, Any] = fit_kwargs
        """Custom arguments to be passed to the model '.fit()' method."""

        # if the warm start involves the optimizer as well, pre-compile the model
        if self.warm_start == 2:
            self.model.compile(**self.compile_kwargs)

    def _fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        if self.warm_start == 0:
            # leverage the 'clone_model' utility to create a copy of the model structure with uninitialized weights
            from tensorflow.keras.models import clone_model
            self.model = clone_model(self.model)
            self.model.compile(**self.compile_kwargs)
        elif self.warm_start == 1:
            self.model.compile(**self.compile_kwargs)
        self.model.fit(x, y, sample_weight=sample_weight, **self.fit_kwargs)

    def _predict(self, x) -> np.ndarray:
        return self.model.predict(x).squeeze()


class TensorflowMLP(TensorflowLearner):
    """Tensorflow/Keras Dense Neural Network Wrapper"""

    def __init__(self,
                 loss: Any,
                 output_units: int = 1,
                 output_activation: Optional[str] = None,
                 hidden_units: List[int] = (128,),
                 hidden_activation: Optional[str] = 'relu',
                 optimizer: Any = 'adam',
                 metrics: Optional[List] = None,
                 loss_weights: Union[None, List, Dict] = None,
                 weighted_metrics: Optional[List] = None,
                 run_eagerly: bool = False,
                 epochs: int = 1,
                 shuffle: bool = True,
                 validation_split: float = 0.,
                 batch_size: Optional[int] = None,
                 class_weight: Optional[Dict] = None,
                 callbacks: Optional[List] = None,
                 verbose: Union[bool, str] = 'auto',
                 warm_start: Union[bool, int] = False,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **layer_kwargs: Any):
        """
        :param loss:
            The neural network loss function (either a string or a loss object).

        :param output_units:
            The neural network number of output units.

        :param output_activation:
            The neural network output activation.

        :param hidden_units:
            The tuple of neural network hidden units.

        :param hidden_activation:
            The neural network hidden activations.

        :param optimizer:
            The neural network optimizer (either a string or an optimizer object).

        :param metrics:
            The list of tensorflow/keras metrics for the evaluation phase.

        :param loss_weights:
            Optional list or dictionary specifying scalar coefficients to weight the loss contributions.

        :param weighted_metrics:
            List of metrics to be evaluated and weighted by `sample_weight` or `class_weight`.

        :param run_eagerly:
            Whether to run tensorflow in eager mode.

        :param epochs:
            The number of training epochs.

        :param shuffle:
            Whether to shuffle the dataset when training.

        :param validation_split:
            The validation split for neural network training.

        :param batch_size:
            The batch size for neural network training.

        :param class_weight:
            Optional dictionary mapping class indices to a weight, used for weighting the loss function during training.

        :param callbacks:
            The list of tensorflow/keras callbacks for the training phase.

        :param verbose:
            Whether to print information during the neural network training.

        :param warm_start:
            Handles the warming start policy after each iteration. If 0 (or False), the weights are reinitialized and
            the model is trained from scratch. If 1 (or True), only the optimized is reinitialized, while the weights
            are kept from the previous iteration. If 2, neither the weights nor the optimizer are reinitialized.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether to log statistics, or a list of parameters whose
            statistics must be logged.

        :param layer_kwargs:
            Additional arguments to be passed to the layer instances, both hidden and output. Examples are: use_bias,
            kernel_initializer, kernel_regularizer, kernel_constraint, etc.
        """
        try:
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.models import Sequential
        except ModuleNotFoundError:
            raise MissingDependencyError(package='tensorflow')

        network = Sequential()
        for units in hidden_units:
            network.add(layer=Dense(units=units, activation=hidden_activation, **layer_kwargs))
        network.add(layer=Dense(units=output_units, activation=output_activation, **layer_kwargs))

        super(TensorflowMLP, self).__init__(model=network,
                                            loss=loss,
                                            optimizer=optimizer,
                                            metrics=metrics,
                                            loss_weights=loss_weights,
                                            weighted_metrics=weighted_metrics,
                                            run_eagerly=run_eagerly,
                                            epochs=epochs,
                                            shuffle=shuffle,
                                            batch_size=batch_size,
                                            validation_split=validation_split,
                                            callbacks=callbacks,
                                            class_weight=class_weight,
                                            verbose=verbose,
                                            warm_start=warm_start,
                                            mask=mask,
                                            x_scaler=x_scaler,
                                            y_scaler=y_scaler,
                                            stats=stats)
