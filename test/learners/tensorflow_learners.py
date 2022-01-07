import random

import numpy as np

from moving_targets.learners import MultiLayerPerceptron
from test.learners.abstract import TestLearners


# TENSORFLOW IS IMPORTED LAZILY TO AVOID CONFLICTS WITH DEPENDENCIES TESTS

class TestTensorflowLearners(TestLearners):
    @staticmethod
    def _random_state():
        import tensorflow as tf
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

    def test_regression_mlp(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        mt = MultiLayerPerceptron(loss='mse', output_units=1, output_activation=None)
        ref = Sequential([Dense(128, activation='relu'), Dense(1, activation=None)])
        ref.compile(optimizer='adam', loss='mse')
        self._test(mt_learner=mt, ref_learner=ref, classification=False, random_state=self._random_state)

    def test_classification_mlp(self):
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        mt = MultiLayerPerceptron(loss='binary_crossentropy', output_units=1, output_activation='sigmoid')
        ref = Sequential([Dense(128, activation='relu'), Dense(1, activation='sigmoid')])
        ref.compile(optimizer='adam', loss='binary_crossentropy')
        self._test(mt_learner=mt, ref_learner=ref, classification=True, random_state=self._random_state)
