"""Basic Learner Interface."""
import time
from typing import Any, Optional, Set, Union, List

import numpy as np

from moving_targets.callbacks import StatsLogger
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.masking import mask_data, get_mask
from moving_targets.util.scalers import Scaler
from moving_targets.util.typing import Dataset


class Learner(StatsLogger):
    """Basic interface for a Moving Targets Learner."""

    @staticmethod
    def _parameters() -> Set[str]:
        return {'elapsed_time'}

    def __init__(self,
                 mask: Optional[float],
                 x_scaler: Union[None, Scaler, str],
                 y_scaler: Union[None, Scaler, str],
                 stats: Union[bool, List[str]]):
        """
        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether to log statistics, or a list of parameters whose statistics
            must be logged.
        """
        super(Learner, self).__init__(stats=stats, name='Learner')

        self.mask: Optional[float] = mask
        """The (optional) masking value used to mask the original targets."""

        self.x_scaler: Optional[Scaler] = Scaler(default_method=x_scaler) if isinstance(x_scaler, str) else x_scaler
        """The (optional) scaler for the input data."""

        self.y_scaler: Optional[Scaler] = Scaler(default_method=y_scaler) if isinstance(y_scaler, str) else y_scaler
        """The (optional) scaler for the output data."""

        self._macs: Optional = None
        """Reference to the MACS object encapsulating the `Learner`."""

        self._time: Optional[float] = None
        """An auxiliary variable to keep track of the elapsed time between iterations."""

    def log(self, **cache):
        if self._macs is not None:
            self._macs.log(**cache)

    def on_training_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._macs = macs
        self._time = time.time()

    def on_training_end(self, macs, x, y: np.ndarray, p: np.ndarray, val_data: Optional[Dataset]):
        self._log_stats(elapsed_time=time.time() - self._time)
        self._time = None
        self._macs = None

    def fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Any:
        """Scales the (x, y) data and use it to fit the `Learner`.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param sample_weight:
            The (optional) array of sample weights.

        :return:
            The `Learner` itself.
        """
        x, y = mask_data(x, y, mask=get_mask(y, self.mask))
        x = x if self.x_scaler is None else self.x_scaler.fit_transform(data=x)
        y = y if self.y_scaler is None else self.y_scaler.fit_transform(data=y)
        self._fit(x=x, y=y, sample_weight=sample_weight)
        return self

    def predict(self, x) -> np.ndarray:
        """Uses the fitted `Learner` configuration to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        x = x if self.x_scaler is None else self.x_scaler.transform(data=x)
        p = self._predict(x)
        return p if self.y_scaler is None else self.y_scaler.inverse_transform(data=p)

    def _fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """Implements the fitting strategy based on the kind of learner.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param sample_weight:
            The (optional) array of sample weights.
        """
        raise NotImplementedError(not_implemented_message(name='_fit'))

    def _predict(self, x) -> np.ndarray:
        """Implements the predictive strategy based on the kind of learner.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        raise NotImplementedError(not_implemented_message(name='_predict'))
