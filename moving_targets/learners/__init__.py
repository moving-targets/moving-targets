"""Interfaces and classes for Moving Targets Learners."""

from moving_targets.learners.curve_learners import CurveLearner, ScipyCurveFit
from moving_targets.learners.learner import Learner
from moving_targets.learners.scikit_learners import ScikitLearner, ScikitClassifier, LinearRegression, \
    LogisticRegression, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
from moving_targets.learners.tensorflow_learners import TensorflowLearner, TensorflowMLP
from moving_targets.learners.torch_learners import TorchLearner, TorchMLP
