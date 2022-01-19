"""Regression Metrics."""

from typing import Callable, Dict, Any

import numpy as np

from moving_targets.metrics.metric import Metric


class RegressionMetric(Metric):
    """Basic interface for a Moving Targets Regression Metric."""

    def __init__(self, metric_function: Callable, name: str, **metric_kwargs):
        """
        :param metric_function:
            Callable function that computes the metric given the true targets and the predictions.

        :param name:
            The name of the metric.

        :param metric_kwargs:
            Custom kwargs to be passed to the metric function.
        """
        super(RegressionMetric, self).__init__(name=name)

        self.metric_function: Callable = metric_function
        """The callable function used to compute the metric."""

        self.metric_kwargs: Dict[str, Any] = metric_kwargs
        """Custom arguments to be passed to the metric function."""

    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> float:
        return self.metric_function(y, p, **self.metric_kwargs)


class MSE(RegressionMetric):
    """Wrapper for scikit-learn 'mean_squared_error' function."""

    def __init__(self, name: str = 'mse', **scikit_kwargs):
        """
        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'mean_squared_error' function.
        """
        from sklearn.metrics import mean_squared_error
        super(MSE, self).__init__(metric_function=mean_squared_error, name=name, **scikit_kwargs)


class MAE(RegressionMetric):
    """Wrapper for scikit-learn 'mean_absolute_error' function."""

    def __init__(self, name: str = 'mae', **scikit_kwargs):
        """
        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'mean_absolute_error' function.
        """
        from sklearn.metrics import mean_absolute_error
        super(MAE, self).__init__(metric_function=mean_absolute_error, name=name, **scikit_kwargs)


class R2(RegressionMetric):
    """Wrapper for scikit-learn 'r2_score' function."""

    def __init__(self, name: str = 'r2', **scikit_kwargs):
        """
        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'r2_score' function.
        """
        from sklearn.metrics import r2_score
        super(R2, self).__init__(metric_function=r2_score, name=name, **scikit_kwargs)
