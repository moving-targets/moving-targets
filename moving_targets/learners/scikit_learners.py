"""Sklearn-based Learners."""
from typing import Optional, Union, List, Any

import numpy as np
import sklearn.ensemble as ens
import sklearn.linear_model as lm
from sklearn.preprocessing import PolynomialFeatures

from moving_targets.learners.learner import Learner
from moving_targets.util.scalers import Scaler


class ScikitLearner(Learner):
    """Wrapper for a custom Scikit-Learn model."""

    def __init__(self,
                 model,
                 polynomial: Union[None, int, PolynomialFeatures] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **fit_kwargs):
        """
        :param model:
            The Scikit-Learn model.

        :param polynomial:
            The polynomial feature preprocessor. It can be either None for no preprocessing, an actual scikit learn
            `PolynomialFeatures` instance, or an integer representing the degree used to generate polynomial features
             which will be used to create a `PolynomialFeatures` object with no bias and all interactions.

        :param mask:
            The (optional) masking value used to mask the original targets.

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
        super(ScikitLearner, self).__init__(mask=mask, x_scaler=x_scaler, y_scaler=y_scaler, stats=stats)

        if isinstance(polynomial, int):
            polynomial = PolynomialFeatures(degree=polynomial, interaction_only=False, include_bias=False)

        self.model = model
        """The Scikit-Learn model."""

        self.polynomial: Optional[PolynomialFeatures] = polynomial
        """The (optional) polynomial features preprocessor."""

        self.fit_kwargs = fit_kwargs
        """Custom arguments to be passed to the model '.fit()' method."""

    def fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Any:
        x = x if self.polynomial is None else self.polynomial.fit_transform(x)
        return super(ScikitLearner, self).fit(x=x, y=y, sample_weight=sample_weight)

    def predict(self, x) -> np.ndarray:
        x = x if self.polynomial is None else self.polynomial.transform(x)
        return super(ScikitLearner, self).predict(x=x)

    def _fit(self, x, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model.fit(x, y, sample_weight=sample_weight)

    def _predict(self, x) -> np.ndarray:
        return self.model.predict(x)


class ScikitClassifier(ScikitLearner):
    """Wrapper for a custom Scikit-Learn classification model."""

    def __init__(self,
                 model,
                 task: str = 'auto',
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **fit_kwargs):
        """
        :param model:
            The Scikit-Learn model.

        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param mask:
            The (optional) masking value used to mask the original targets.

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
        super(ScikitClassifier, self).__init__(model=model,
                                               polynomial=polynomial,
                                               mask=mask,
                                               x_scaler=x_scaler,
                                               y_scaler=y_scaler,
                                               stats=stats,
                                               **fit_kwargs)

        assert task in ['binary', 'multiclass', 'auto'], f"'task' should be either 'binary' or 'multiclass', got {task}"

        self.task: str = task
        """The kind of classification task."""

    def _predict(self, x) -> np.ndarray:
        probabilities = self.model.predict_proba(x)
        if self.task == 'multiclass' or probabilities.shape[1] > 2:
            # return probabilities matrix if the task is explicitly multiclass or if the number of classes is > 2
            return probabilities
        else:
            # return probabilities vector otherwise (i.e., explicitly binary task or automatic with classes == 2)
            return probabilities[:, 1].squeeze()


class LinearRegression(ScikitLearner):
    """Scikit-Learn Linear Regression wrapper."""

    def __init__(self,
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param model_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.
        """
        super(LinearRegression, self).__init__(model=lm.LinearRegression(**model_kwargs),
                                               polynomial=polynomial,
                                               mask=mask,
                                               x_scaler=x_scaler,
                                               y_scaler=y_scaler,
                                               stats=stats)


class LogisticRegression(ScikitClassifier):
    """Scikit-Learn Logistic Regression wrapper."""

    def __init__(self,
                 task: str = 'auto',
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Custom arguments to be passed to a sklearn.linear_model.LogisticRegression instance.
        """
        super(LogisticRegression, self).__init__(model=lm.LogisticRegression(**model_kwargs),
                                                 task=task,
                                                 polynomial=polynomial,
                                                 mask=mask,
                                                 x_scaler=x_scaler,
                                                 y_scaler=y_scaler,
                                                 stats=stats)


class RandomForestRegressor(ScikitLearner):
    """Scikit-Learn Random Forest Regressor wrapper."""

    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestRegressor instance.
        """
        m = ens.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, **model_kwargs)
        super(RandomForestRegressor, self).__init__(model=m,
                                                    polynomial=polynomial,
                                                    mask=mask,
                                                    x_scaler=x_scaler,
                                                    y_scaler=y_scaler,
                                                    stats=stats)


class RandomForestClassifier(ScikitClassifier):
    """Scikit-Learn Random Forest Classifier wrapper."""

    def __init__(self,
                 task: str = 'auto',
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param n_estimators:
            The number of trees in the forest.

        :param max_depth:
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all
            leaves contain less than min_samples_split samples.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.RandomForestClassifier instance.
        """
        m = ens.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, **model_kwargs)
        super(RandomForestClassifier, self).__init__(model=m,
                                                     task=task,
                                                     polynomial=polynomial,
                                                     mask=mask,
                                                     x_scaler=x_scaler,
                                                     y_scaler=y_scaler,
                                                     stats=stats)


class GradientBoostingRegressor(ScikitLearner):
    """Scikit-Learn Gradient Boosting Regressor wrapper."""

    def __init__(self,
                 n_estimators: int = 100,
                 min_samples_leaf: Union[int, float] = 1,
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param n_estimators:
            The number of boosting stages to perform.

        :param min_samples_leaf:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.GradientBoostingRegressor instance.
        """
        m = ens.GradientBoostingRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **model_kwargs)
        super(GradientBoostingRegressor, self).__init__(model=m,
                                                        polynomial=polynomial,
                                                        mask=mask,
                                                        x_scaler=x_scaler,
                                                        y_scaler=y_scaler,
                                                        stats=stats)


class GradientBoostingClassifier(ScikitClassifier):
    """Scikit-Learn Gradient Boosting Classifier wrapper."""

    def __init__(self,
                 task: str = 'auto',
                 n_estimators: int = 100,
                 min_samples_leaf: Union[int, float] = 1,
                 polynomial: Optional[int] = None,
                 mask: Optional[float] = None,
                 x_scaler: Union[None, Scaler, str] = None,
                 y_scaler: Union[None, Scaler, str] = None,
                 stats: Union[bool, List[str]] = False,
                 **model_kwargs):
        """
        :param task:
            The kind of classification task, which can be either 'binary' (i.e., probabilities will be returned as a
            one-dimensional array), 'multiclass' (i.e., probabilities will be returned in a bi-dimensional array), or
            'auto' for automatic task detection.

        :param n_estimators:
            The number of boosting stages to perform.

        :param min_samples_leaf:
            The minimum number of samples required to be at a leaf node. If int, then consider `min_samples_leaf` as
            the minimum number, otherwise `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are
            the minimum number of samples for each node.

        :param polynomial:
            The (optional) degree used to generate polynomial features. If not None, creates a scikit learn `Pipeline`
            object containing a `PolynomialFeatures` preprocessor (with given degree and no bias) at the beginning,
            followed by the given model.

        :param mask:
            The (optional) masking value used to mask the original targets.

        :param x_scaler:
            The (optional) scaler for the input data, or a string representing the default scaling method.

        :param y_scaler:
            The (optional) scaler for the output data, or a string representing the default scaling method.

        :param stats:
            Either a boolean value indicating whether or not to log statistics, or a list of parameters whose
            statistics must be logged.

        :param model_kwargs:
            Additional arguments to be passed to a sklearn.ensemble.GradientBoostingClassifier instance.
        """
        m = ens.GradientBoostingClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, **model_kwargs)
        super(GradientBoostingClassifier, self).__init__(model=m,
                                                         task=task,
                                                         polynomial=polynomial,
                                                         mask=mask,
                                                         x_scaler=x_scaler,
                                                         y_scaler=y_scaler,
                                                         stats=stats)
