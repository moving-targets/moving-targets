import random
from typing import Callable

import numpy as np
import sklearn.ensemble as ens
import sklearn.linear_model as lm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from moving_targets.learners import Learner
from moving_targets.learners.scikit_learners import LinearRegression, LogisticRegression, RandomForestRegressor, \
    RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from test.learners.abstract import TestLearners


class TestScikitLearners(TestLearners):

    @staticmethod
    def _custom_learner(learner_class: Callable, **kwargs) -> Learner:
        return learner_class(polynomial=3, x_scaler='std', **kwargs)

    @staticmethod
    def _custom_pipeline(model_class: Callable, **kwargs) -> Pipeline:
        return Pipeline([
            ('std', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)),
            ('model', model_class(**kwargs))
        ])

    @staticmethod
    def _random_state():
        random.seed(0)
        np.random.seed(0)

    def _reference(self, learner, x, y, sample_weight: np.ndarray) -> np.ndarray:
        if isinstance(learner, Pipeline):
            learner.fit(x, y, model__sample_weight=sample_weight)
        else:
            learner.fit(x, y, sample_weight=sample_weight)
        return learner.predict_proba(x)[:, 1].squeeze() if hasattr(learner, 'predict_proba') else learner.predict(x)

    def test_linear_regression_default(self):
        self._test(mt_learner=LinearRegression(), ref_learner=lm.LinearRegression(), classification=False)

    def test_linear_regression_custom(self):
        self._test(mt_learner=self._custom_learner(LinearRegression),
                   ref_learner=self._custom_pipeline(lm.LinearRegression), classification=False)

    def test_logistic_regression_default(self):
        self._test(mt_learner=LogisticRegression(), ref_learner=lm.LogisticRegression(), classification=True)

    def test_logistic_regression_custom(self):
        self._test(mt_learner=self._custom_learner(LogisticRegression),
                   ref_learner=self._custom_pipeline(lm.LogisticRegression), classification=True)

    def test_random_forest_regressor_default(self):
        self._test(mt_learner=RandomForestRegressor(), ref_learner=ens.RandomForestRegressor(), classification=False)

    def test_random_forest_regressor_custom(self):
        self._test(mt_learner=self._custom_learner(RandomForestRegressor),
                   ref_learner=self._custom_pipeline(ens.RandomForestRegressor), classification=False)

    def test_random_forest_classifier_default(self):
        self._test(mt_learner=RandomForestClassifier(), ref_learner=ens.RandomForestClassifier(), classification=True)

    def test_random_forest_classifier_custom(self):
        self._test(mt_learner=self._custom_learner(RandomForestClassifier),
                   ref_learner=self._custom_pipeline(ens.RandomForestClassifier), classification=True)

    def test_gradient_boosting_regressor_default(self):
        self._test(mt_learner=GradientBoostingRegressor(), ref_learner=ens.GradientBoostingRegressor(),
                   classification=False)

    def test_gradient_boosting_regressor_custom(self):
        self._test(mt_learner=self._custom_learner(GradientBoostingRegressor),
                   ref_learner=self._custom_pipeline(ens.GradientBoostingRegressor), classification=False)

    def test_gradient_boosting_classifier_default(self):
        self._test(mt_learner=GradientBoostingClassifier(), ref_learner=ens.GradientBoostingClassifier(),
                   classification=True)

    def test_gradient_boosting_classifier_custom(self):
        self._test(mt_learner=self._custom_learner(GradientBoostingClassifier),
                   ref_learner=self._custom_pipeline(ens.GradientBoostingClassifier), classification=True)
