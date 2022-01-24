"""Sklearn-based Learners."""
from typing import Any, Optional, Union, List

import numpy as np

from moving_targets.learners.learner import Learner


class ScikitLearner(Learner):
    """Wrapper for a custom Scikit-Learn model."""

    def __init__(self, model, stats: Union[bool, List[str]] = False, **fit_kwargs):
        """
        :param model:
            The Scikit-Learn model.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param fit_kwargs:
            Custom arguments to be passed to the model '.fit()' method.
        """
        super(ScikitLearner, self).__init__(stats=stats)

        self.model = model
        """The Scikit-Learn model."""

        self.fit_kwargs = fit_kwargs
        """Custom arguments to be passed to the model '.fit()' method."""

    def fit(self, x, y: np.ndarray) -> Any:
        self.model.fit(x, y, **self.fit_kwargs)
        return self

    def predict(self, x) -> np.ndarray:
        return self.model.predict(x)


class ScikitClassifier(ScikitLearner):
    """Wrapper for a custom Scikit-Learn classification model."""

    def __init__(self, model, task: str = 'auto', stats: Union[bool, List[str]] = False, **fit_kwargs):
        """
        :param model:
            The Scikit-Learn model.

        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param fit_kwargs:
            Custom arguments to be passed to the model '.fit()' method.
        """
        super(ScikitClassifier, self).__init__(model=model, stats=stats, **fit_kwargs)

        assert task in ['binary', 'multiclass', 'auto'], f"'task' should be either 'binary' or 'multiclass', got {task}"

        self.task: str = task
        """The kind of classification task."""

    def predict(self, x) -> np.ndarray:
        probabilities = self.model.predict_proba(x)
        # handle automatic task inference and return a 1d vector in case of binary task, otherwise return the matrix
        task = ('binary' if probabilities.shape[1] == 2 else 'multiclass') if self.task == 'auto' else self.task
        return probabilities[:, 1].squeeze() if task == 'binary' else probabilities


class LinearRegression(ScikitLearner):
    """Scikit-Learn Linear Regression wrapper."""

    def __init__(self, sample_weight: Optional = None, stats: Union[bool, List[str]] = False, **model_kwargs):
        """
        :param sample_weight:
            Array-like of shape (n_samples,) containing individual weights for each sample during the training.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        import sklearn.linear_model as lm
        m = lm.LinearRegression(**model_kwargs)
        super(LinearRegression, self).__init__(model=m, stats=stats, sample_weight=sample_weight)


class LogisticRegression(ScikitClassifier):
    """Scikit-Learn Logistic Regression wrapper."""

    def __init__(self,
                 sample_weight: Optional = None,
                 task: str = 'auto',
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param sample_weight:
            Array-like of shape (n_samples,) containing individual weights for each sample during the training.

        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        import sklearn.linear_model as lm
        m = lm.LogisticRegression(**model_kwargs)
        super(LogisticRegression, self).__init__(model=m, task=task, stats=stats, sample_weight=sample_weight)


class RandomForestRegressor(ScikitLearner):
    """Scikit-Learn Random Forest Regressor wrapper."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 sample_weight: Optional = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param sample_weight:
            Array-like of shape (n_samples,) containing individual weights for each sample during the training.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestRegressor instance.
        """
        import sklearn.ensemble as ens
        m = ens.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, **model_kwargs)
        super(RandomForestRegressor, self).__init__(model=m, stats=stats, sample_weight=sample_weight)


class RandomForestClassifier(ScikitClassifier):
    """Scikit-Learn Random Forest Classifier wrapper."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 sample_weight: Optional = None,
                 task: str = 'auto',
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param sample_weight:
            Array-like of shape (n_samples,) containing individual weights for each sample during the training.

        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestClassifier instance.
        """
        import sklearn.ensemble as ens
        m = ens.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **model_kwargs)
        super(RandomForestClassifier, self).__init__(model=m, task=task, stats=stats, sample_weight=sample_weight)


class GradientBoostingRegressor(ScikitLearner):
    """Scikit-Learn Gradient Boosting Regressor wrapper."""

    def __init__(self,
                 n_estimators: int = 100,
                 min_samples_leaf: Union[int, float] = 1,
                 sample_weight: Optional = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param n_estimators:
            The number of boosting stages to perform.

        :param min_samples_leaf:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param sample_weight:
            Array-like of shape (n_samples,) containing individual weights for each sample during the training.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.GradientBoostingRegressor instance.
        """
        import sklearn.ensemble as ens
        m = ens.GradientBoostingRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **model_kwargs)
        super(GradientBoostingRegressor, self).__init__(model=m, stats=stats, sample_weight=sample_weight)


class GradientBoostingClassifier(ScikitClassifier):
    """Scikit-Learn Gradient Boosting Classifier wrapper."""

    def __init__(self,
                 n_estimators: int = 100,
                 min_samples_leaf: Union[int, float] = 1,
                 sample_weight: Optional = None,
                 task: str = 'auto',
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param n_estimators:
            The number of boosting stages to perform.

        :param min_samples_leaf:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param sample_weight:
            Array-like of shape (n_samples,) containing individual weights for each sample during the training.

        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.GradientBoostingClassifier instance.
        """
        import sklearn.ensemble as ens
        m = ens.GradientBoostingClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **model_kwargs)
        super(GradientBoostingClassifier, self).__init__(model=m, task=task, stats=stats, sample_weight=sample_weight)
