import random

import sklearn.ensemble as ens
import sklearn.linear_model as lm

from moving_targets.learners.scikit_learners import *
from test.learners.abstract import TestLearners


class TestScikitLearners(TestLearners):
    @staticmethod
    def _random_state():
        random.seed(0)
        np.random.seed(0)

    def test_linear_regression(self):
        self._test(mt_learner=LinearRegression(),
                   ref_learner=lm.LinearRegression(),
                   classification=False,
                   random_state=self._random_state)

    def test_logistic_regression(self):
        self._test(mt_learner=LogisticRegression(),
                   ref_learner=lm.LogisticRegression(),
                   classification=True,
                   random_state=self._random_state)

    def test_random_forest_regressor(self):
        self._test(mt_learner=RandomForestRegressor(),
                   ref_learner=ens.RandomForestRegressor(),
                   classification=False,
                   random_state=self._random_state)

    def test_random_forest_classifier(self):
        self._test(mt_learner=RandomForestClassifier(),
                   ref_learner=ens.RandomForestClassifier(),
                   classification=True,
                   random_state=self._random_state)

    def test_gradient_boosting_regressor(self):
        self._test(mt_learner=GradientBoostingRegressor(),
                   ref_learner=ens.GradientBoostingRegressor(),
                   classification=False,
                   random_state=self._random_state)

    def test_gradient_boosting_classifier(self):
        self._test(mt_learner=GradientBoostingClassifier(),
                   ref_learner=ens.GradientBoostingClassifier(),
                   classification=True,
                   random_state=self._random_state)
