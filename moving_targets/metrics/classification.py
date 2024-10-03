"""Classification Metrics."""

from typing import Callable, Optional, Dict, Any

import numpy as np

from moving_targets.metrics.metric import Metric
from moving_targets.util import probabilities


class ClassificationMetric(Metric):
    """Basic interface for a Moving Targets Classification Metric."""

    def __init__(self,
                 metric_function: Callable,
                 classes: Optional[int],
                 name: str,
                 use_prob: bool = False,
                 **metric_kwargs):
        """
        :param metric_function:
            Callable function that computes the metric given the true targets and the predictions.

        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.

        :param use_prob:
            Whether to use the output labels or the output probabilities.

        :param metric_kwargs:
            Custom kwargs to be passed to the metric function.
        """
        super(ClassificationMetric, self).__init__(name=name)

        self.metric_function: Callable = metric_function
        """Callable function that computes the metric given the true targets and the predictions."""

        self.classes: Optional[int] = classes
        """The number of classes or None for automatic class inference."""

        self.use_prob: bool = use_prob
        """Whether to use the output labels or the output probabilities."""

        self.metric_kwargs: Dict[str, Any] = metric_kwargs
        """Custom arguments to be passed to the metric function."""

    def __call__(self, x, y: np.ndarray, p: np.ndarray) -> float:
        p = p.astype(float)
        if not self.use_prob:
            # if the metric does not use probabilities, convert the probabilities into class targets
            p = probabilities.get_classes(prob=p)
        elif p.ndim == 1 and np.issubdtype(p.dtype, np.integer):
            # otherwise, if we use class probabilities we expect a bi-dimensional array for predictions, thus if a
            # one-dimensional array is passed instead because the master directly returns adjusted class targets,
            # we onehot encode them for compatibility
            c = self.classes or probabilities.count_classes(vector=y)
            p = probabilities.get_onehot(vector=p, classes=c)
        return self.metric_function(y, p, **self.metric_kwargs)


class CrossEntropy(ClassificationMetric):
    """Wrapper for scikit-learn 'log_loss' function."""

    def __init__(self, classes: Optional[int] = None, name: str = 'crossentropy', **scikit_kwargs):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'log_loss' function.
        """
        from sklearn.metrics import log_loss
        super(CrossEntropy, self).__init__(
            metric_function=log_loss,
            classes=classes,
            name=name,
            use_prob=True,
            **scikit_kwargs
        )


class Precision(ClassificationMetric):
    """Wrapper for scikit-learn 'precision_score' function."""

    def __init__(self, classes: Optional[int] = None, name: str = 'precision', **scikit_kwargs):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'precision_score' function.
        """
        from sklearn.metrics import precision_score
        super(Precision, self).__init__(
            metric_function=precision_score,
            classes=classes,
            name=name,
            use_prob=False,
            **scikit_kwargs
        )


class Recall(ClassificationMetric):
    """Wrapper for scikit-learn 'recall_score' function."""

    def __init__(self, classes: Optional[int] = None, name: str = 'recall', **scikit_kwargs):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'recall_score' function.
        """
        from sklearn.metrics import recall_score
        super(Recall, self).__init__(
            metric_function=recall_score,
            classes=classes,
            name=name,
            use_prob=False,
            **scikit_kwargs
        )


class F1(ClassificationMetric):
    """Wrapper for scikit-learn 'f1_score' function."""

    def __init__(self, classes: Optional[int] = None, name: str = 'f1', **scikit_kwargs):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'f1_score' function.
        """
        from sklearn.metrics import f1_score
        super(F1, self).__init__(
            metric_function=f1_score,
            classes=classes,
            name=name,
            use_prob=False,
            **scikit_kwargs
        )


class Accuracy(ClassificationMetric):
    """Wrapper for scikit-learn 'accuracy_score' function."""

    def __init__(self, classes: Optional[int] = None, name: str = 'accuracy', **scikit_kwargs):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Custom arguments to be passed to scikit-learn 'accuracy_score' function.
        """
        from sklearn.metrics import accuracy_score
        super(Accuracy, self).__init__(
            metric_function=accuracy_score,
            classes=classes,
            name=name,
            use_prob=False,
            **scikit_kwargs
        )


class AUC(ClassificationMetric):
    """Wrapper for scikit-learn 'roc_auc_score' function."""

    def __init__(self,
                 classes: Optional[int] = None,
                 multi_class: str = 'ovo',
                 name: str = 'auc',
                 **scikit_kwargs):
        """
        :param classes:
            The number of classes or None for automatic class inference.

        :param multi_class:
            Determines the type of configuration to use when dealing with multiclass targets, either 'ovr' or 'ovo'.

        :param name:
            The name of the metric.

        :param scikit_kwargs:
            Additional custom arguments to be passed to scikit-learn 'roc_auc_score' function.
        """
        from sklearn.metrics import roc_auc_score
        super(AUC, self).__init__(
            metric_function=roc_auc_score,
            classes=classes,
            name=name,
            use_prob=True,
            multi_class=multi_class,
            **scikit_kwargs
        )
