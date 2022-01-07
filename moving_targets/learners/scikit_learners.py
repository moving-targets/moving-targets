"""Sklearn-based Learners."""
from typing import Any, Optional, Union

from moving_targets.learners.learner import Learner


class ScikitLearner(Learner):
    """Wrapper for a custom Scikit-Learn model."""

    def __init__(self, model):
        """
        :param model:
            The Scikit-Learn model.
        """
        super(ScikitLearner, self).__init__()

        self.model = model
        """The Scikit-Learn model."""

    def fit(self, x, y, **additional_kwargs):
        self.model.fit(x, y)

    def predict(self, x) -> Any:
        return self.model.predict(x)


class ScikitClassifier(ScikitLearner):
    """Wrapper for a custom Scikit-Learn classification model."""

    def predict(self, x) -> Any:
        probabilities = self.model.predict_proba(x)
        return probabilities[:, 1].squeeze() if probabilities.shape[1] == 2 else probabilities


class LinearRegression(ScikitLearner):
    """Scikit-Learn Linear Regression wrapper."""

    def __init__(self, **model_kwargs):
        """
        :param model_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        import sklearn.linear_model as lm
        super(LinearRegression, self).__init__(model=lm.LinearRegression(**model_kwargs))


class LogisticRegression(ScikitClassifier):
    """Scikit-Learn Logistic Regression wrapper."""

    def __init__(self, **model_kwargs):
        """
        :param model_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        import sklearn.linear_model as lm
        super(LogisticRegression, self).__init__(model=lm.LogisticRegression(**model_kwargs))


class RandomForestRegressor(ScikitLearner):
    """Scikit-Learn Random Forest Regressor wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **model_kwargs):
        """
        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestRegressor instance.
        """
        import sklearn.ensemble as ens
        m = ens.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, **model_kwargs)
        super(RandomForestRegressor, self).__init__(model=m)


class RandomForestClassifier(ScikitClassifier):
    """Scikit-Learn Random Forest Classifier wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, **model_kwargs):
        """
        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestClassifier instance.
        """
        import sklearn.ensemble as ens
        m = ens.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **model_kwargs)
        super(RandomForestClassifier, self).__init__(model=m)


class GradientBoostingRegressor(ScikitLearner):
    """Scikit-Learn Gradient Boosting Regressor wrapper."""

    def __init__(self, n_estimators: int = 100, min_samples_leaf: Union[int, float] = 1, **model_kwargs):
        """
        :param n_estimators:
            The number of boosting stages to perform.

        :param max_depth:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.GradientBoostingRegressor instance.
        """
        import sklearn.ensemble as ens
        m = ens.GradientBoostingRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **model_kwargs)
        super(GradientBoostingRegressor, self).__init__(model=m)


class GradientBoostingClassifier(ScikitClassifier):
    """Scikit-Learn Gradient Boosting Classifier wrapper."""

    def __init__(self, n_estimators: int = 100, min_samples_leaf: Union[int, float] = 1, **model_kwargs):
        """
        :param n_estimators:
            The number of boosting stages to perform.

        :param max_depth:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.GradientBoostingClassifier instance.
        """
        import sklearn.ensemble as ens
        m = ens.GradientBoostingClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **model_kwargs)
        super(GradientBoostingClassifier, self).__init__(model=m)
