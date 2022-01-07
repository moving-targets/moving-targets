"""Basic Learner Interface."""
from typing import Any, Optional, Set, Union, List

from moving_targets.callbacks import StatsLogger
from moving_targets.util.errors import not_implemented_message
from moving_targets.util.typing import Dataset


class Learner(StatsLogger):
    """Basic interface for a Moving Targets Learner."""

    @staticmethod
    def _parameters() -> Set[str]:
        return set()

    def __init__(self, stats: Union[bool, List[str]] = False):
        """
        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.
        """
        super(Learner, self).__init__(stats=stats, logger='Learner')

        self._macs: Optional = None
        """Reference to the MACS object encapsulating the `Learner`."""

    def log(self, **cache):
        self._macs.log(**cache)

    def on_process_start(self, macs, x, y, val_data: Optional[Dataset], **additional_kwargs):
        self._macs = macs

    def on_process_end(self, macs, val_data: Optional[Dataset], **additional_kwargs):
        self._macs = None

    def fit(self, x, y, **additional_kwargs):
        """Fits the `Learner` according to the implemented procedure using (x, y) as training data.

        :param x:
            The matrix/dataframe of training samples.

        :param y:
            The vector of training labels.

        :param additional_kwargs:
            Any other useful parameter.
        """
        raise NotImplementedError(not_implemented_message(name='fit'))

    def predict(self, x) -> Any:
        """Uses the fitted `Learner` configuration to predict labels from input samples.

        :param x:
            The matrix/dataframe of input samples.

        :return:
            The vector of predicted labels.
        """
        raise NotImplementedError(not_implemented_message(name='predict'))
