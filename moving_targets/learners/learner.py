"""Basic Learner Interface."""
import time
from typing import Any, Optional, Set, Union, List

import numpy as np

from moving_targets.callbacks import StatsLogger
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Dataset


class Learner(StatsLogger):
    """Basic interface for a Moving Targets Learner."""

    @staticmethod
    def _parameters() -> Set[str]:
        return {'elapsed_time'}

    def __init__(self, stats: Union[bool, List[str]]):
        """
        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.
        """
        super(Learner, self).__init__(stats=stats, name='Learner')

        self._macs: Optional = None
        """Reference to the MACS object encapsulating the `Learner`."""

        self._time: Optional[float] = None
        """An auxiliary variable to keep track of the elapsed time between iterations."""

    def log(self, **cache):
        self._macs.log(**cache)

    def on_training_start(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._macs = macs
        self._time = time.time()

    def on_training_end(self, macs, x, y: np.ndarray, val_data: Optional[Dataset]):
        self._log_stats(elapsed_time=time.time() - self._time)
        self._time = None
        self._macs = None

    def fit(self, x, y: np.ndarray) -> Any:
        """Fits the `Learner` according to the implemented procedure using (x, y) as training data.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :return:
            The `Learner` itself.
        """
        raise NotImplementedError(not_implemented_message(name='fit'))

    def predict(self, x) -> np.ndarray:
        """Uses the fitted `Learner` configuration to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        raise NotImplementedError(not_implemented_message(name='predict'))
