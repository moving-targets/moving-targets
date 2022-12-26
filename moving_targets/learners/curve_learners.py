"""Learners to fit custom curves."""
from typing import Optional, Union, List, Callable

import numpy as np
from scipy.optimize import curve_fit

from moving_targets.learners.learner import Learner
from moving_targets.util.scalers import Scaler


class CurveLearner(Learner):
    """Template class for a learner which fits a custom curve."""

    def __init__(self,
                 curve: Callable,
                 method: Callable,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False):
        """
        :param curve:
            A function f(x; p) -> y which is fed with an input vector (x) and the learnable parameters (p).

        :param method:
            An optimization routine m(f; x, y) -> p* which is fed with the curve function (f) and the training data
            (x, y), and eventually returns the optimal configuration for the parameters (p*).

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.
        """
        super(CurveLearner, self).__init__(mask=mask, x_scaler=x_scaler, y_scaler=y_scaler, stats=stats)

        self.curve: Callable = curve
        """The function to be learned."""

        self.method: Callable = method
        """The optimization routine."""

        self.configuration: Optional = None
        """The optimal configuration of the curve parameters."""

    def _fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        if sample_weight is not None:
            self.logger.warning("CurveLearner does not support sample weights, please pass 'sample_weight'=None")
        self.configuration = self.method(self.curve, x, y)

    def _predict(self, x) -> np.ndarray:
        return self.curve(x, *self.configuration)


class ScipyCurveFit(CurveLearner):
    def __init__(self,
                 curve: Callable,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **method_kwargs):
        """
        :param curve:
            A function f(x; p) -> y which is fed with an input vector (x) and the learnable parameters (p).

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param method_kwargs:
            Additional arguments to be passed to `scipy.optimize.curve_fit`.
        """

        def method(f, x, y):
            return curve_fit(f=f, xdata=x, ydata=y, **method_kwargs)[0]

        super(ScipyCurveFit, self).__init__(curve=curve,
                                            method=method,
                                            mask=mask,
                                            x_scaler=x_scaler,
                                            y_scaler=y_scaler,
                                            stats=stats)
