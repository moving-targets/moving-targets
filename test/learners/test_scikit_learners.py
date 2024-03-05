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
from test.learners.test_learners import TestLearners


class TestScikitLearners(TestLearners):

    @staticmethod
    def _custom_learner(learner_class: Callable, **kwargs) -> Learner:
        return learner_class(polynomial=3, x_scaler='std', **kwargs)

    @staticmethod
    def _custom_pipeline(model_class: Callable, **kwargs) -> Pipeline:
        return Pipeline([
            ('poly', PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)),
            ('std', StandardScaler()),
            ('model', model_class(**kwargs))
        ])

    @classmethod
    def _random_state(cls):
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)

    def _reference(self, learner, x, y, sample_weight: np.ndarray) -> np.ndarray:
        if isinstance(learner, Pipeline):
            learner.fit(x, y, model__sample_weight=sample_weight)
        else:
            learner.fit(x, y, sample_weight=sample_weight)
        return learner.predict_proba(x)[:, 1].squeeze() if hasattr(learner, 'predict_proba') else learner.predict(x)

    def test_linear_regression_default(self):
        self._test(mt_learner=lambda: LinearRegression(),
                   ref_learner=lambda: lm.LinearRegression(),
                   classification=False)

    def test_linear_regression_custom(self):
        self._test(mt_learner=lambda: self._custom_learner(LinearRegression),
                   ref_learner=lambda: self._custom_pipeline(lm.LinearRegression),
                   classification=False)

    def test_logistic_regression_default(self):
        self._test(mt_learner=lambda: LogisticRegression(),
                   ref_learner=lambda: lm.LogisticRegression(),
                   classification=True)

    def test_logistic_regression_custom(self):
        self._test(mt_learner=lambda: self._custom_learner(LogisticRegression),
                   ref_learner=lambda: self._custom_pipeline(lm.LogisticRegression),
                   classification=True)

    def test_random_forest_regressor_default(self):
        self._test(mt_learner=lambda: RandomForestRegressor(),
                   ref_learner=lambda: ens.RandomForestRegressor(),
                   classification=False)

    def test_random_forest_regressor_custom(self):
        self._test(mt_learner=lambda: self._custom_learner(RandomForestRegressor),
                   ref_learner=lambda: self._custom_pipeline(ens.RandomForestRegressor),
                   classification=False)

    def test_random_forest_classifier_default(self):
        self._test(mt_learner=lambda: RandomForestClassifier(),
                   ref_learner=lambda: ens.RandomForestClassifier(),
                   classification=True)

    def test_random_forest_classifier_custom(self):
        self._test(mt_learner=lambda: self._custom_learner(RandomForestClassifier),
                   ref_learner=lambda: self._custom_pipeline(ens.RandomForestClassifier),
                   classification=True)

    def test_gradient_boosting_regressor_default(self):
        self._test(mt_learner=lambda: GradientBoostingRegressor(),
                   ref_learner=lambda: ens.GradientBoostingRegressor(),
                   classification=False)

    def test_gradient_boosting_regressor_custom(self):
        self._test(mt_learner=lambda: self._custom_learner(GradientBoostingRegressor),
                   ref_learner=lambda: self._custom_pipeline(ens.GradientBoostingRegressor),
                   classification=False)

    def test_gradient_boosting_classifier_default(self):
        self._test(mt_learner=lambda: GradientBoostingClassifier(),
                   ref_learner=lambda: ens.GradientBoostingClassifier(),
                   classification=True)

    def test_gradient_boosting_classifier_custom(self):
        self._test(mt_learner=lambda: self._custom_learner(GradientBoostingClassifier),
                   ref_learner=lambda: self._custom_pipeline(ens.GradientBoostingClassifier),
                   classification=True)
